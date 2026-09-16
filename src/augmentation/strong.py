from torchvision import transforms

def strong_augmentation(image_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return {
        'train': transforms.Compose([
            transforms.Resize(image_size),

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

            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),

            transforms.ToTensor(),
            normalize
        ])
    }