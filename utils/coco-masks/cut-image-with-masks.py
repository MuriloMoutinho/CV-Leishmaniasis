import json
from pathlib import Path

import cv2
import numpy as np

SET_DATASET = Path("../../datasets/AIR_LEISH_dataset_v1/Set1/")
IMAGE_DIR = SET_DATASET / "Images"
JSON_FILE = SET_DATASET / "_annotations.coco.json"

OUTPUT_DIR = Path("images_4_3")

TARGET_CATEGORY_ID = 1

ORIGINAL_WIDTH = 1844
ORIGINAL_HEIGHT = 2709

CROP_WIDTH = 1844
CROP_HEIGHT = 2458

ROTATE_CLOCKWISE = True

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


images = {
    image["id"]: image
    for image in data["images"]
}


annotations_by_image = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)

def get_category_bbox(annotations, category_id):

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    found = False

    for annotation in annotations:

        if annotation["category_id"] != category_id:
            continue

        segmentation = annotation.get(
            "segmentation",
            []
        )

        if isinstance(segmentation, list):

            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                min_x = min(
                    min_x,
                    np.min(points[:, 0])
                )

                min_y = min(
                    min_y,
                    np.min(points[:, 1])
                )

                max_x = max(
                    max_x,
                    np.max(points[:, 0])
                )

                max_y = max(
                    max_y,
                    np.max(points[:, 1])
                )

                found = True

        elif isinstance(segmentation, dict):

            bbox = annotation.get("bbox")

            if bbox is None:

                print(
                    f"  [AVISO] Annotation "
                    f"{annotation['id']} sem bbox."
                )

                continue

            x, y, w, h = bbox

            min_x = min(
                min_x,
                x
            )

            min_y = min(
                min_y,
                y
            )

            max_x = max(
                max_x,
                x + w
            )

            max_y = max(
                max_y,
                y + h
            )

            found = True

    if not found:
        return None

    return (
        min_x,
        min_y,
        max_x,
        max_y
    )


def calculate_class_area_in_region(
    annotations,
    category_id,
    x1,
    y1,
    x2,
    y2
):
    """
    Calcula quantos pixels da categoria existem
    dentro da região especificada.

    Região:
        x1 <= x < x2
        y1 <= y < y2
    """

    width = max(
        0,
        int(x2 - x1)
    )

    height = max(
        0,
        int(y2 - y1)
    )

    if width == 0 or height == 0:
        return 0

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    found = False

    for annotation in annotations:

        if annotation["category_id"] != category_id:
            continue

        segmentation = annotation.get(
            "segmentation",
            []
        )

        if isinstance(segmentation, list):

            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                points = np.array(
                    polygon,
                    dtype=np.float32
                ).reshape(-1, 2)

                # Transladar coordenadas para a região
                points[:, 0] -= x1
                points[:, 1] -= y1

                points = np.round(
                    points
                ).astype(np.int32)

                cv2.fillPoly(
                    mask,
                    [points],
                    1
                )

                found = True

        elif isinstance(segmentation, dict):

            # Para RLE, usamos o bbox como aproximação.
            bbox = annotation.get("bbox")

            if bbox is None:
                continue

            bx, by, bw, bh = bbox

            bx1 = max(
                int(np.floor(bx)),
                int(x1)
            )

            by1 = max(
                int(np.floor(by)),
                int(y1)
            )

            bx2 = min(
                int(np.ceil(bx + bw)),
                int(x2)
            )

            by2 = min(
                int(np.ceil(by + bh)),
                int(y2)
            )

            if bx1 < bx2 and by1 < by2:

                mask[
                    by1 - int(y1):
                    by2 - int(y1),

                    bx1 - int(x1):
                    bx2 - int(x1)
                ] = 1

                found = True

    if not found:
        return 0

    return int(
        np.sum(mask)
    )

def calculate_crop_y(
    image_height,
    crop_height,
    annotations,
    category_id,
    image_path
):
    """
    Escolhe a posição vertical do crop.

    Como precisamos remover:

        image_height - crop_height

    pixels, existem duas regiões principais:

        TOP:
            pixels removidos do topo

        BOTTOM:
            pixels removidos da parte inferior

    O lado que possuir MENOS área da categoria
    será removido.
    """

    pixels_to_remove = (
        image_height - crop_height
    )

    if pixels_to_remove < 0:

        raise ValueError(
            f"{image_path}: crop maior que a imagem."
        )

    if pixels_to_remove == 0:
        return 0

    top_area = calculate_class_area_in_region(
        annotations,
        category_id,
        0,
        0,
        ORIGINAL_WIDTH,
        pixels_to_remove
    )

    bottom_y1 = image_height - pixels_to_remove

    bottom_area = calculate_class_area_in_region(
        annotations,
        category_id,
        0,
        bottom_y1,
        ORIGINAL_WIDTH,
        image_height
    )

    if top_area < bottom_area:

        crop_y = pixels_to_remove

        side = "TOP"

    elif bottom_area < top_area:

        crop_y = 0

        side = "BOTTOM"

    else:

        # Empate:
        # mantém o crop centralizado
        crop_y = (
            image_height - crop_height
        ) // 2

        side = "EMPATE"

    print(
        f"  Área classe {category_id}: "
        f"topo={top_area} | "
        f"baixo={bottom_area} | "
        f"corte={side}"
    )

    return crop_y

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for image_id, image_info in images.items():

    filename = image_info["file_name"]

    image_path = (
        IMAGE_DIR / filename
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"[ERRO] Imagem não encontrada: "
            f"{image_path}"
        )

        continue

    height, width = image.shape[:2]

    print(f"\nProcessando: {filename}")

    if (
        width != ORIGINAL_WIDTH
        or height != ORIGINAL_HEIGHT
    ):

        print(
            "  [AVISO] Dimensão diferente "
            "do esperado!"
        )

        print(
            f"  Encontrado: "
            f"{width} x {height}"
        )

    annotations = annotations_by_image.get(
        image_id,
        []
    )

    bbox = get_category_bbox(
        annotations,
        TARGET_CATEGORY_ID
    )

    if bbox is None:

        print(
            f"  [AVISO] Não existe "
            f"categoria {TARGET_CATEGORY_ID}."
        )

        # Sem a classe:
        # simplesmente centraliza.

        crop_y = (
            height - CROP_HEIGHT
        ) // 2

    else:

        min_x, min_y, max_x, max_y = bbox

        print(
            f"  Categoria {TARGET_CATEGORY_ID}: "
            f"X={min_x:.1f}->{max_x:.1f} "
            f"Y={min_y:.1f}->{max_y:.1f}"
        )

        try:

            crop_y = calculate_crop_y(
                height,
                CROP_HEIGHT,
                annotations,
                TARGET_CATEGORY_ID,
                image_path
            )

        except ValueError as e:

            print(
                f"  [ERRO] {e}"
            )

            continue


    crop_y2 = (
        crop_y +
        CROP_HEIGHT
    )

    cropped = image[
        crop_y:crop_y2,
        0:CROP_WIDTH
    ]

    if ROTATE_CLOCKWISE:

        rotated = cv2.rotate(
            cropped,
            cv2.ROTATE_90_CLOCKWISE
        )

    else:

        rotated = cv2.rotate(
            cropped,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )


    output_path = (
        OUTPUT_DIR / filename
    )

    success = cv2.imwrite(
        str(output_path),
        rotated
    )

    if not success:

        print(
            f"  [ERRO] Não foi possível salvar: "
            f"{output_path}"
        )

        continue

    print(
        f"  [OK] Salvo: {output_path}"
    )


print("\n" + "=" * 60)
print("CONCLUÍDO")
print("=" * 60)

print(
    f"Imagens salvas em: "
    f"{OUTPUT_DIR}"
)

print(
    f"Tamanho final: "
    f"{CROP_HEIGHT} x {CROP_WIDTH}"
)

print(
    f"Proporção: "
    f"{CROP_WIDTH / CROP_HEIGHT:.4f}:1"
)