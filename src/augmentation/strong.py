from torchvision import transforms
from augmentation.transforms_utils import create_transforms


def strong_augmentation(image_size):
    return {
        'train': create_transforms(image_size, [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.RandomAffine(degrees=3, scale=(0.97, 1.03)),

            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.05,
                    hue=0.02
                )
            ], p=0.5),

            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.1))
            ], p=0.2)
        ]),
        'val': create_transforms(image_size)
    }