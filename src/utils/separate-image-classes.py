import json
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

SET_DATASET = "AIR_LEISH/Set2/"
JSON_FILE = Path("../datasets/" + SET_DATASET + "_annotations.coco.json")
OUTPUT_DIR = Path("visualizacoes/" + SET_DATASET)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CARREGAR JSON
# ============================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# IDENTIFICAR OS IDs DA CLASSE AM
# ============================================================

# No JSON existem:
#
# id 0 -> AM
# id 1 -> AM
# id 2 -> HC
# id 3 -> NU
#
# Como existem dois IDs chamados AM, pegamos ambos.

AM_IDS = {
    category["id"]
    for category in data["categories"]
    if category["name"].upper() == "AM"
}

print("IDs considerados como AM:", AM_IDS)


# ============================================================
# CRIAR MAPA DAS IMAGENS
# ============================================================

images = {
    image["id"]: image["file_name"]
    for image in data["images"]
}


# ============================================================
# VERIFICAR QUAIS IMAGENS POSSUEM AMASTIGOTAS
# ============================================================

images_with_am = set()

for annotation in data["annotations"]:

    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    # Se a annotation for AM
    if category_id in AM_IDS:
        images_with_am.add(image_id)


# ============================================================
# SEPARAR COM E SEM AM
# ============================================================

images_with_amastigotes = []
images_without_amastigotes = []

for image_id, filename in images.items():

    if image_id in images_with_am:
        images_with_amastigotes.append(filename)
    else:
        images_without_amastigotes.append(filename)


# ============================================================
# RESULTADO
# ============================================================

total_images = len(images)

total_with = len(images_with_amastigotes)
total_without = len(images_without_amastigotes)


print("\n" + "=" * 60)
print("RESULTADO")
print("=" * 60)

print(f"Total de imagens:              {total_images}")
print(f"Imagens COM amastigotas (AM):  {total_with}")
print(f"Imagens SEM amastigotas (AM):  {total_without}")

print("=" * 60)

# ============================================================
# SALVAR LISTAS EM ARQUIVOS TXT
# ============================================================

filename = "amastigotas.txt"
output_path = OUTPUT_DIR / filename
print(str(OUTPUT_DIR))

with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Imagens COM amastigotas (AM):  {total_with}" + "\n")
    f.write(f"Imagens SEM amastigotas (AM):  {total_without}" + "\n")

    f.write("COM AMASTIGOTAS" + ("=" * 55) + "\n")
    for filename in images_with_amastigotes:
        f.write(filename + "\n")

    f.write("SEM AMASTIGOTAS" + ("=" * 55) + "\n")
    for filename in images_without_amastigotes:
        f.write(filename + "\n")