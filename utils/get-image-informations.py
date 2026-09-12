from pathlib import Path
from PIL import Image
import numpy as np

SET_DATASET = "AIR_LEISH/Set2/Images"
IMAGE_DIR = Path("../datasets/raw/" + SET_DATASET)
EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }

resolutions = set()
channels = set()

total_images = 0
errors = 0

alpha_images = 0
useful_alpha_images = 0
transparent_images = 0

for image_path in IMAGE_DIR.iterdir():

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:

            width, height = img.size

            channels.add(img.mode)
            resolutions.add((width, height))

            total_images += 1

            if "A" in img.getbands():

                alpha_images += 1

                alpha = np.array(img.getchannel("A"))

                alpha_min = alpha.min()
                alpha_max = alpha.max()

                # Existe pelo menos algum pixel transparente
                if alpha_min < 255:
                    transparent_images += 1

                # O alpha realmente varia
                if alpha_min != alpha_max:
                    useful_alpha_images += 1

    except Exception as e:
        errors += 1
        print(f"Erro ao ler: {image_path.name}")
        print(f"  {e}")


print("\n" + "=" * 50)
print("RESULTADO")
print("=" * 50)

print(f"Total de imagens: {total_images}")
print(f"Resoluções diferentes: {len(resolutions)}")
print(f"Canais/modos diferentes: {len(channels)}")

print(f"\nImagens com canal Alpha: {alpha_images}")
print(f"Imagens com transparência: {transparent_images}")
print(f"Imagens com Alpha variável: {useful_alpha_images}")

print("\nResoluções encontradas:")

for width, height in sorted(resolutions):
    print(f"{width} x {height}")

print("\nCanais encontrados:")

for channel in sorted(channels):
    print(channel)

if errors > 0:
    print(f"\nImagens com erro: {errors}")