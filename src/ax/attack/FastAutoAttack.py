import time

from src.ax.attack.FastAPGD import FastAPGD
from src.ax.attack.FastMultiAttack import FastMultiAttack
from torchattacks.attack import Attack


class FastAutoAttack(Attack):
    def __init__(self, model, eps=0.3, seed=None):
        super().__init__("FastAutoAttack", model)
        self.eps = eps
        self.seed = seed
        self._supported_mode = ["default"]

        self.autoattack = FastMultiAttack(
            [
                FastAPGD(model, eps=eps, seed=self.get_seed(), loss="ce"),
                FastAPGD(model, eps=eps, seed=self.get_seed(), loss="dlr"),
            ]
        )

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed
