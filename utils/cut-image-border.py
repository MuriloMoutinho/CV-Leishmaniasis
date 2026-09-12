from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

SET_DATASET = "DeepLeish/Positive"
IMAGE_DIR = Path("../datasets/raw/" + SET_DATASET)
EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }

total_images = 0
converted_images = 0
errors = 0

PIXELS_TO_REMOVE = 50

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

            cropped = img.crop((0, 0, width, height - PIXELS_TO_REMOVE))

            cropped.save(image_path)
            converted_images += 1

            print(f"{image_path.name}: {width}x{height} -> {cropped.size}")

    except Exception as e:
        errors += 1
        print(f"Erro ao ler: {image_path.name}")
        print(f"  {e}")

print("RESULTADO")
print("=" * 50)

print(f"Total de imagens: {total_images}")
print(f"Imagens cortadas: {converted_images}")

if errors > 0:
    print(f"\nImagens com erro: {errors}")


