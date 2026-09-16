import json
from pathlib import Path

import cv2
import numpy as np

SET_DATASET = "../../datasets/AIR_LEISH_dataset_v1/Set2/"
IMAGE_DIR = Path(SET_DATASET + "Images")
JSON_FILE = Path(SET_DATASET + "_annotations.coco.json")
OUTPUT_DIR = Path("./visualizacoes/masks/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# image_id - dados da imagem
images = {
    image["id"]: image
    for image in data["images"]
}

# category_id - nome da classe
categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

annotations_by_image = {}

for annotation in data["annotations"]:
    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)

COLORS = {
    0: (255, 0, 0),       # azul
    1: (0, 255, 0),       # verde
    2: (0, 0, 255),       # vermelho
    3: (255, 255, 0),     # amarelo
}

for image_id, image_info in images.items():

    filename = image_info["file_name"]
    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERRO] Imagem não encontrada: {image_path}")
        continue

    result = image.copy()
    overlay = image.copy()

    annotations = annotations_by_image.get(image_id, [])

    print(f"Processando {filename} ({len(annotations)} annotations)")

    for annotation in annotations:
        category_id = annotation["category_id"]

        category_name = categories.get(category_id, f"classe_{category_id}")
        color = COLORS.get(category_id, (255, 0, 255))

        segmentation = annotation.get("segmentation", [])

        if isinstance(segmentation, dict):
            print(f"  [AVISO] Annotation {annotation['id']} está em RLE. Ignorando.")
            continue

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(result, [points], isClosed=True, color=color, thickness=4, lineType=cv2.LINE_AA)

            x, y = points[0]

            cv2.putText(
                result,
                category_name,
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3,
                cv2.LINE_AA
            )

    result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)

    output_path = OUTPUT_DIR / filename
    cv2.imwrite(str(output_path), result)


print("\nConcluído!")
print(f"Visualizações salvas em: {OUTPUT_DIR}")