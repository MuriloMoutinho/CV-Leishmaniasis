import json
import shutil
from pathlib import Path

IMAGE_DIR = Path("./images_4_3/")
JSON_FILE = Path("./_annotations1.coco.json")

POSITIVE_OUTPUT_DIR = Path("../../datasets/AIR_LEISH/Positive")
NEGATIVE_OUTPUT_DIR = Path("../../datasets/AIR_LEISH/Negative")

POSITIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

AM_IDS = {
    category["id"]
    for category in data["categories"]
    if category["name"].upper() == "AM"
}
print("IDs considerados como AM:", AM_IDS)

images = {
    image["id"]: image["file_name"]
    for image in data["images"]
}

images_with_am = set()

for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    if category_id in AM_IDS:
        images_with_am.add(image_id)

images_with_amastigotes = []
images_without_amastigotes = []

for image_id, filename in images.items():
    if image_id in images_with_am:
        images_with_amastigotes.append(filename)
    else:
        images_without_amastigotes.append(filename)


def copy_images(filenames, output_dir):
    copied = 0
    not_found = 0

    for filename in filenames:
        source = IMAGE_DIR / filename
        destination = output_dir / filename

        if not source.exists():
            not_found += 1
            continue

        shutil.copy2(source, destination)
        copied += 1

    return copied, not_found


positive_copied, positive_not_found = copy_images(
    images_with_amastigotes,
    POSITIVE_OUTPUT_DIR
)

negative_copied, negative_not_found = copy_images(
    images_without_amastigotes,
    NEGATIVE_OUTPUT_DIR
)

print("\n" + "=" * 60)
print("CÓPIA FINALIZADA")
print("=" * 60)

print(f"Total de imagens:              {len(images)}")
print(f"Positivas copiadas:       {positive_copied}")
print(f"Positivas não encontradas: {positive_not_found}")

print(f"Negativas copiadas:       {negative_copied}")
print(f"Negativas não encontradas: {negative_not_found}")