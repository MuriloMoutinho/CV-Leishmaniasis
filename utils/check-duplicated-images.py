from pathlib import Path
from PIL import Image
import hashlib
import imagehash

DATASET_DIR = Path("../datasets/raw/AIR_LEISH")

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

PHASH_THRESHOLD = 5

images = [
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in EXTENSIONS
]

print("=" * 70)
print("ANÁLISE DO DATASET")
print("=" * 70)

print(f"\nDataset: {DATASET_DIR}")
print(f"Total de imagens: {len(images)}")


print("\n" + "=" * 70)
print("1. DUPLICATAS EXATAS")
print("=" * 70)

md5_hashes = {}

for path in images:

    file_hash = hashlib.md5(
        path.read_bytes()
    ).hexdigest()

    md5_hashes.setdefault(
        file_hash,
        []
    ).append(path)


duplicate_groups = [
    paths
    for paths in md5_hashes.values()
    if len(paths) > 1
]


print(
    f"\nGrupos de duplicatas exatas: "
    f"{len(duplicate_groups)}"
)


if duplicate_groups:

    for group_number, paths in enumerate(
        duplicate_groups,
        start=1
    ):

        print(f"\nGrupo {group_number}:")

        for path in paths:
            print(f"  {path}")

else:

    print("Nenhuma duplicata exata encontrada.")

print("\n" + "=" * 70)
print("2. CALCULANDO PHASH")
print("=" * 70)

phashes = {}
errors = []

for index, path in enumerate(
    images,
    start=1
):

    try:

        with Image.open(path) as image:

            image_hash = imagehash.phash(image)

        phashes[path] = image_hash

    except Exception as error:

        errors.append(
            (path, str(error))
        )

    if index % 100 == 0 or index == len(images):

        print(
            f"Processadas: "
            f"{index}/{len(images)}"
        )


print(
    f"\nImagens com pHash calculado: "
    f"{len(phashes)}"
)


if errors:

    print(
        f"Erros ao abrir imagens: "
        f"{len(errors)}"
    )

    for path, error in errors:
        print(f"  {path}: {error}")


print("\n" + "=" * 70)
print("3. ANALISANDO SIMILARIDADE")
print("=" * 70)

paths = list(phashes.keys())

all_pairs = []

for i in range(len(paths)):

    path1 = paths[i]
    hash1 = phashes[path1]

    for j in range(i + 1, len(paths)):

        path2 = paths[j]
        hash2 = phashes[path2]

        distance = hash1 - hash2

        all_pairs.append(
            (
                path1,
                path2,
                distance
            )
        )


print(
    f"\nTotal de comparações: "
    f"{len(all_pairs):,}"
)


all_pairs.sort(
    key=lambda x: x[2]
)

print("\n" + "=" * 70)
print("20 PARES MAIS SEMELHANTES")
print("=" * 70)

for path1, path2, distance in all_pairs[:20]:

    print("\n" + "-" * 70)

    print(
        f"Distância pHash: {distance}"
    )

    print(f"Imagem 1:")
    print(f"  {path1}")

    print(f"Imagem 2:")
    print(f"  {path2}")


similar_pairs = [
    pair
    for pair in all_pairs
    if pair[2] <= PHASH_THRESHOLD
]


print("\n" + "=" * 70)
print(
    f"PARES COM DISTÂNCIA <= "
    f"{PHASH_THRESHOLD}"
)
print("=" * 70)

print(
    f"\nTotal: {len(similar_pairs)}"
)


for path1, path2, distance in similar_pairs:

    print("\n" + "-" * 70)

    print(
        f"Distância pHash: {distance}"
    )

    print(f"Imagem 1:")
    print(f"  {path1}")

    print(f"Imagem 2:")
    print(f"  {path2}")


print("\n" + "=" * 70)
print("RESUMO")
print("=" * 70)

print(
    f"Total de imagens:              "
    f"{len(images)}"
)

print(
    f"Grupos de duplicatas exatas:   "
    f"{len(duplicate_groups)}"
)

print(
    f"Pares com pHash <= "
    f"{PHASH_THRESHOLD}:             "
    f"{len(similar_pairs)}"
)

print(
    f"Threshold pHash:                "
    f"{PHASH_THRESHOLD}"
)

if errors:

    print(
        f"Imagens com erro:               "
        f"{len(errors)}"
    )

print("=" * 70)
