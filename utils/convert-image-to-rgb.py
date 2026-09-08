from pathlib import Path
from PIL import Image
import numpy as np

SET_DATASET = "AIR_LEISH/Set1"
IMAGE_DIR = Path("../datasets/" + SET_DATASET + "/Images")

OUTPUT_DIR = Path("visualizacoes/" + SET_DATASET + "/rgb/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENSIONS = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp" }
total_images = 0
converted_images = 0
errors = 0

for image_path in IMAGE_DIR.iterdir():

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:

            total_images += 1

            if "A" in img.getbands():
                alpha = np.array(img.getchannel("A"))

                alpha_min = alpha.min()
                alpha_max = alpha.max()

                if alpha_min != alpha_max or alpha_min < 255:
                    continue

                final_path = OUTPUT_DIR / image_path.name

                rgb_image = img.convert('RGB')
                rgb_image.save(final_path)

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