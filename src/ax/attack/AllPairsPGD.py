import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


class SearchStrategy(enum.Enum):
    ALL = enum.auto()
    ONE_SIDED = enum.auto()
    TRAINING = enum.auto()
    TOPK = enum.auto()


class AllPairsPGD:
    def __init__(
        self,
        net1: nn.Module,
        net2: nn.Module,
        search_strategy: SearchStrategy,
        topk: int = 3,
        eps: float = 0.3,
        alpha: float = 2 / 255,
        steps: int = 40,
        random_start: bool = True,
        verbose: bool = False,
    ):
        assert steps > 0

        self.net1 = net1
        self.net2 = net2
        self.device = next(net1.parameters()).device

        self.search_strategy = search_strategy
        self.topk = topk

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.scaler = GradScaler()

        self.verbose = verbose

    def __call__(self, images: torch.Tensor, *_, **__) -> torch.Tensor:
        given_training1 = self.net1.training
        given_training2 = self.net2.training

        self.net1.eval()
        self.net2.eval()

        images = self.forward(images)

        if given_training1:
            self.net1.train()
        if given_training2:
            self.net2.train()

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)

        with autocast():
            with torch.no_grad():
                orig_logits1: torch.Tensor = self.net1(images)
                orig_logits2: torch.Tensor = self.net2(images)

        orig_preds1 = orig_logits1.argmax(-1)
        orig_preds2 = orig_logits2.argmax(-1)

        n_classes = orig_logits1.shape[-1]
        assert n_classes >= 2

        adv_images = images.clone().detach().to(self.device)
        preds_differ = torch.full(
            size=(images.shape[0],), fill_value=False, device=self.device
        )

        target_pairs = []
        if self.search_strategy is SearchStrategy.ALL:
            target_pairs = [
                (t1, t2)
                for t1 in range(n_classes)
                for t2 in range(n_classes)
                if t1 != t2
            ]
        elif self.search_strategy is SearchStrategy.ONE_SIDED:
            # TODO(tony): Optimize this codepath
            target_pairs = [(t, None) for t in range(n_classes)] + [
                (None, t) for t in range(n_classes)
            ]
        elif self.search_strategy is SearchStrategy.TOPK:
            assert self.topk <= n_classes
            target_pairs = [
                (t1, t2) for t1 in range(self.topk) for t2 in range(self.topk)
            ]
        elif self.search_strategy is SearchStrategy.TRAINING:
            target_pairs = [(None, None)]
        else:
            raise ValueError(self.search_strategy)

        it = tqdm(target_pairs) if self.verbose else target_pairs
        for target1, target2 in it:
            labs1 = (
                orig_preds1
                if target1 is None
                else torch.full_like(orig_preds1, fill_value=target1)
            )
            labs2 = (
                orig_preds2
                if target2 is None
                else torch.full_like(orig_preds2, fill_value=target2)
            )

            if self.search_strategy is SearchStrategy.TOPK:
                labs1 = orig_logits1.argsort(dim=-1, descending=True)[
                    :, target1
                ]
                labs2 = orig_logits1.argsort(dim=-1, descending=True)[
                    :, target2
                ]

            if self.search_strategy is SearchStrategy.TRAINING:
                mask = torch.rand(size=labs1.shape, device=labs1.device)
                sm0 = mask < 0.6  # 50% substrategy 0
                sm1 = (0.6 <= mask) & (mask < 0.8)  # 20% substrategy 1
                sm2 = 0.8 <= mask  # 20% substrategy 2

                # Substrategy 0 - topk
                y1 = (
                    orig_logits1.argsort(dim=-1, descending=True)
                    .gather(
                        dim=-1,
                        index=torch.randint_like(
                            labs1, low=0, high=self.topk
                        ).reshape(-1, 1),
                    )
                    .flatten()
                )
                y2 = (
                    orig_logits2.argsort(dim=-1, descending=True)
                    .gather(
                        dim=-1,
                        index=torch.randint_like(
                            labs2, low=0, high=self.topk
                        ).reshape(-1, 1),
                    )
                    .flatten()
                )
                labs1[sm0] = y1[sm0]
                labs1[sm0] = y2[sm0]

                # Substrategy 1 - change labs1 uniformly randomly
                y1 = (
                    labs2 + torch.randint_like(labs2, low=1, high=n_classes - 1)
                ) % n_classes
                labs1[sm1] = y1[sm1]

                # Substrategy 2 - change labs2 uniformly randomly
                y2 = (
                    labs1 + torch.randint_like(labs1, low=1, high=n_classes - 1)
                ) % n_classes
                labs2[sm2] = y2[sm2]

                # Fallback strategy - completely random
                # Around 1 / topk^2 entries will fail substrategy 0.
                # This fallback strategies handles those failures.
                still_eq = labs1 == labs2
                y1 = torch.randint_like(labs1, low=0, high=n_classes)
                y2 = (
                    y1 + torch.randint_like(y1, low=1, high=n_classes - 1)
                ) % n_classes
                labs1[still_eq] = y1[still_eq]
                labs2[still_eq] = y2[still_eq]

                assert (labs1 == labs2).int().sum() == 0

            cur_adv_images = self.targeted_attack(images, labs1, labs2)

            with autocast():
                with torch.no_grad():
                    preds1 = self.net1(cur_adv_images).argmax(dim=-1)
                    preds2 = self.net2(cur_adv_images).argmax(dim=-1)

            cur_preds_differ = preds1 != preds2

            adv_images[cur_preds_differ] = cur_adv_images[cur_preds_differ]
            preds_differ |= cur_preds_differ

            if self.verbose:
                assert isinstance(it, tqdm)
                it.set_description(
                    f"acc: {1 - preds_differ.float().mean().item()}"
                )

        return adv_images

    def targeted_attack(
        self,
        images: torch.Tensor,
        labs1: torch.Tensor,
        labs2: torch.Tensor,
    ) -> torch.Tensor:
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # Accelerating forward propagation
            with autocast():
                logits1 = self.net1(adv_images)
                logits2 = self.net2(adv_images)

                cost1 = -F.cross_entropy(input=logits1, target=labs1)
                cost2 = -F.cross_entropy(input=logits2, target=labs2)

                cost = cost1 + cost2

            # Update adversarial images with gradient scaler applied
            scaled_loss = self.scaler.scale(cost)
            grad = torch.autograd.grad(
                scaled_loss, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(
                adv_images - images, min=-self.eps, max=self.eps
            )
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
