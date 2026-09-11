from torchvision import transforms

def get_images_transformations():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_size = (683, 512) # mantem 4:3: 512x384 / 683x512 /768x576

    return {
        'train': transforms.Compose([
            transforms.Resize(image_size),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(360),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=(0.1, 2.0)
            ),

            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),

            transforms.ToTensor(),
            normalize
        ])
    }