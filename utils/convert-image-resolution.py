from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

SET_DATASET = "DeepLeish/Positive"
IMAGE_DIR = Path("../datasets/raw/" + SET_DATASET)
EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }

total_images = 0
converted_images = 0
errors = 0

def pad_to_4_3_reflect(img):
    width, height = img.size
    target_ratio = 4 / 3
    current_ratio = width / height

    if current_ratio > target_ratio:
        # imagem muito larga, padding em cima/baixo
        new_height = round(width / target_ratio)

        pad_total = new_height - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        arr = np.array(img)
        arr = np.pad(arr,((pad_top, pad_bottom), (0, 0), (0, 0)), mode='reflect')

    else:
        # imagem muito alta, padding nas laterais
        new_width = round(height * target_ratio)

        pad_total = new_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        arr = np.array(img)
        arr = np.pad(arr, ((0, 0), (pad_left, pad_right), (0, 0)), mode='reflect')

    return Image.fromarray(arr)


for image_path in IMAGE_DIR.iterdir():

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:

            total_images += 1

            width, height = img.size
            current_ratio = width / height
            target_ratio = 4 / 3

            if abs(current_ratio - target_ratio) < 0.001:
                continue

            img_4_3 = pad_to_4_3_reflect(img)
            img_4_3.save(image_path)

            converted_images += 1

            print(f"{image_path.name}: {width}x{height} -> {img_4_3.size}")

    except Exception as e:
        errors += 1
        print(f"Erro ao ler: {image_path.name}")
        print(f"  {e}")

print("RESULTADO")
print("=" * 50)

print(f"Total de imagens: {total_images}")
print(f"Imagens convertidas: {converted_images}")

if errors > 0:
    print(f"\nImagens com erro: {errors}")


