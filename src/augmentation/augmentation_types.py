from dataclasses import dataclass

from torchvision import transforms


@dataclass(frozen=True)
class AugmentationTransforms:
    train: transforms.Compose
    val: transforms.Compose
