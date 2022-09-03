import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm


class AllPairsPGD:
    def __init__(
        self,
        net1: nn.Module,
        net2: nn.Module,
        eps: float = 0.3,
        alpha: float = 2 / 255,
        steps: int = 40,
        random_start: bool = True,
        verbose: bool = False,
    ):
        self.net1 = net1
        self.net2 = net2
        self.device = next(net1.parameters()).device

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
        with torch.no_grad():
            with autocast():
                n_classes = self.net1(images[:1]).shape[-1]

        adv_images = images.clone().detach().to(self.device)
        preds_differ = torch.full(
            size=(images.shape[0],), fill_value=False, device=self.device
        )

        target_pairs = [
            (t1, t2)
            for t1 in range(n_classes)
            for t2 in range(n_classes)
            if t1 != t2
        ]
        it = tqdm(target_pairs) if self.verbose else target_pairs
        for target1, target2 in it:
            cur_adv_images = self.targeted_attack(images, target1, target2)

            with torch.no_grad():
                with autocast():
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
        target1: int,
        target2: int,
    ) -> torch.Tensor:
        labs1 = torch.full(
            size=(images.shape[0],),
            fill_value=target1,
            dtype=torch.int64,
            device=self.device,
        )
        labs2 = torch.full(
            size=(images.shape[0],),
            fill_value=target2,
            dtype=torch.int64,
            device=self.device,
        )

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
