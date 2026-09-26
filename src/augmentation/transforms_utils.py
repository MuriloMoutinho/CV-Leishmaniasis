from PIL import ImageOps
from torchvision import transforms

def create_transforms(image_size, pad, compose=None):
    if compose is None:
        compose = []

    pad_transform = []
    if pad is True:
        pad_transform = [transforms.Lambda(lambda img: pad_to_square(img, image_size))]

    return transforms.Compose([
        transforms.Resize(image_size),
        *pad_transform,

        *compose,

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def pad_to_square(img, image_size):
    return ImageOps.pad(
        img,
        image_size,
        color=(255, 255, 255),
        centering=(0.5, 0.5)
    )
