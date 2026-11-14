from augmentation.augmentation_types import AugmentationTransforms
from augmentation.transforms_utils import create_transforms


def weak_augmentation(image_size) -> AugmentationTransforms:
    return AugmentationTransforms(
        train=create_transforms(image_size),
        val=create_transforms(image_size),
    )
