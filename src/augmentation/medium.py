from torchvision import transforms

from augmentation.transforms_utils import create_transforms

def medium_augmentation(image_size):
    return {
        'train': create_transforms(image_size, [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            transforms.RandomAffine(degrees=3, scale=(0.97, 1.03)),
        ]),
        'val': create_transforms(image_size)
    }