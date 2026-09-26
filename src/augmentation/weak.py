from augmentation.transforms_utils import create_transforms


def weak_augmentation(image_size):
    return {
        'train': create_transforms(image_size),
        'val': create_transforms(image_size)
    }