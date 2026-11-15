import random
import shutil
import sys
from pathlib import Path

#AIR_LEISH
#DeepLeish
#DLB

PASTA_ORIGEM = Path("../datasets-cross/DLB")
PASTA_SAIDA = Path("../datasets/DLB")
PCT_TREINO = 0.8
SEED = 42
MOVER = False
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def listar_imagens(pasta: Path):
    return sorted(
        p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES
    )

def transferir(arquivos, destino: Path):
    destino.mkdir(parents=True, exist_ok=True)
    for arq in arquivos:
        alvo = destino / arq.name
        if MOVER:
            shutil.move(str(arq), str(alvo))
        else:
            shutil.copy2(arq, alvo)


def main():
    if not PASTA_ORIGEM.is_dir():
        sys.exit(f"Pasta '{PASTA_ORIGEM}' não encontrada.")
    if not 0 < PCT_TREINO < 1:
        sys.exit("PCT_TREINO deve estar entre 0 e 1 (ex: 0.7).")
    if PASTA_SAIDA.exists() and any(PASTA_SAIDA.iterdir()):
        sys.exit(f"A pasta de saída '{PASTA_SAIDA}' já existe e não está vazia.")

    classes = [d for d in sorted(PASTA_ORIGEM.iterdir()) if d.is_dir()]
    if not classes:
        sys.exit(f"Nenhuma subpasta (classe) encontrada em '{PASTA_ORIGEM}'.")

    rng = random.Random(SEED)
    total_train = total_test = 0

    for classe in classes:
        imagens = listar_imagens(classe)
        if not imagens:
            print(f"[aviso] '{classe.name}' não tem imagens, ignorando.")
            continue

        rng.shuffle(imagens)
        n_train = round(len(imagens) * PCT_TREINO)
        treino, teste = imagens[:n_train], imagens[n_train:]

        transferir(treino, PASTA_SAIDA / "train" / classe.name)
        transferir(teste, PASTA_SAIDA / "test" / classe.name)

        total_train += len(treino)
        total_test += len(teste)
        print(f"{classe.name}: {len(imagens)} imagens -> train={len(treino)}, test={len(teste)}")

    print(f"\nTotal: train={total_train}, test={total_test}")
    print(f"Saída em: {PASTA_SAIDA.resolve()}")

if __name__ == "__main__":
    main()