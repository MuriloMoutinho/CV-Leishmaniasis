from torchvision import transforms

from augmentation.transforms_utils import create_transforms

def medium_augmentation(image_size, pad=False):
    return {
        'train': create_transforms(image_size, pad, [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]),
        'val': create_transforms(image_size, pad)
    }