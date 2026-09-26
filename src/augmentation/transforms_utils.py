from PIL import ImageOps
from torchvision import transforms

def create_transforms(image_size, compose=None):
    if compose is None:
        compose = []

    return transforms.Compose([
        *get_resize_transform(image_size, False),

        *compose,

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_resize_transform(image_size, padding):
    if padding is True:
        return [transforms.Lambda(lambda img: pad_to_square(img, image_size))]
    else:
        return [transforms.Resize(image_size)]

def pad_to_square(img, image_size):
    return ImageOps.pad(
        img,
        image_size[::-1],
        color=(255, 255, 255),
        centering=(0.5, 0.5)
    )
