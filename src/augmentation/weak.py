from augmentation.transforms_utils import create_transforms


def weak_augmentation(image_size, pad=False):
    return {
        'train': create_transforms(image_size, pad),
        'val': create_transforms(image_size, pad)
    }