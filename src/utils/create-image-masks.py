import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================
SET_DATASET = "AIR_LEISH/Set2/"
IMAGE_DIR = Path("../datasets/" + SET_DATASET + "Images")
JSON_FILE = Path("../datasets/" + SET_DATASET + "_annotations.coco.json")
OUTPUT_DIR = Path("visualizacoes/" + SET_DATASET + "/masks/")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARREGAR JSON
# ============================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ============================================================
# MAPAS
# ============================================================

# image_id -> dados da imagem
images = {
    image["id"]: image
    for image in data["images"]
}

# category_id -> nome da classe
categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}


# ============================================================
# AGRUPAR ANNOTATIONS POR IMAGEM
# ============================================================

annotations_by_image = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)


# ============================================================
# CORES
# ============================================================

COLORS = {
    0: (255, 0, 0),       # azul
    1: (0, 255, 0),       # verde
    2: (0, 0, 255),       # vermelho
    3: (255, 255, 0),     # amarelo
}


# ============================================================
# PROCESSAR TODAS AS IMAGENS
# ============================================================

for image_id, image_info in images.items():

    filename = image_info["file_name"]

    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERRO] Imagem não encontrada: {image_path}")
        continue

    # Imagem original para desenhar
    result = image.copy()

    # Camada para as máscaras transparentes
    overlay = image.copy()

    annotations = annotations_by_image.get(image_id, [])

    print(
        f"Processando {filename} "
        f"({len(annotations)} annotations)"
    )

    # ========================================================
    # DESENHAR ANNOTATIONS
    # ========================================================

    for annotation in annotations:

        category_id = annotation["category_id"]

        category_name = categories.get(
            category_id,
            f"classe_{category_id}"
        )

        color = COLORS.get(
            category_id,
            (255, 0, 255)
        )

        segmentation = annotation.get(
            "segmentation",
            []
        )

        # ----------------------------------------------------
        # COCO pode ter segmentation em formato RLE.
        # No seu caso, estamos trabalhando com polígonos.
        # ----------------------------------------------------

        if isinstance(segmentation, dict):
            print(
                f"  [AVISO] Annotation {annotation['id']} "
                f"está em RLE. Ignorando."
            )
            continue

        # ----------------------------------------------------
        # Uma annotation pode possuir vários polígonos
        # ----------------------------------------------------

        for polygon in segmentation:

            if len(polygon) < 6:
                continue

            # Converter:
            #
            # [x1,y1,x2,y2,x3,y3,...]
            #
            # para:
            #
            # [[x1,y1],
            #  [x2,y2],
            #  [x3,y3], ...]
            points = np.array(
                polygon,
                dtype=np.float32
            ).reshape(-1, 2)

            points = np.round(points).astype(np.int32)

            # ------------------------------------------------
            # Máscara preenchida
            # ------------------------------------------------

            cv2.fillPoly(
                overlay,
                [points],
                color
            )

            # ------------------------------------------------
            # Contorno
            # ------------------------------------------------

            cv2.polylines(
                result,
                [points],
                isClosed=True,
                color=color,
                thickness=4,
                lineType=cv2.LINE_AA
            )

            # ------------------------------------------------
            # Nome da classe
            # ------------------------------------------------

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

    # ========================================================
    # MISTURAR MÁSCARA + IMAGEM
    # ========================================================

    result = cv2.addWeighted(
        overlay,
        0.35,
        result,
        0.65,
        0
    )

    # ========================================================
    # SALVAR
    # ========================================================

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(
        str(output_path),
        result
    )


print("\nConcluído!")
print(f"Visualizações salvas em: {OUTPUT_DIR}")