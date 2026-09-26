from torchvision import transforms
from augmentation.transforms_utils import create_transforms


def strong_augmentation(image_size, pad=False):
    return {
        'train': create_transforms(image_size, pad, [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation((180, 180))], p=0.5),

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.25,
                hue=0.05
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.5)),
        ]),
        'val': create_transforms(image_size, pad)
    }