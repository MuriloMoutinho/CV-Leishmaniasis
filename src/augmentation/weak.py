from torchvision import transforms

def weak_augmentation():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image_size = (683, 512) # mantem 4:3: 512x384 / 683x512 /768x576

    return {
        'train': transforms.Compose([
            transforms.Resize(image_size),

            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),

            transforms.ToTensor(),
            normalize
        ])
    }