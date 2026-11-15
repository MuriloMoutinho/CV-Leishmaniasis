import copy
from pathlib import Path

from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

FOLDS_ROOT = Path("../datasets")

BASE_DATASETS = ("AIR_LEISH", "DeepLeish", "DLB")

CROSS_DATASETS = {
    "CROSS_AIR_LEISH": "AIR_LEISH",
    "CROSS_DeepLeish": "DeepLeish",
    "CROSS_DLB": "DLB",
}

COMBINED_DATASETS = {
    "DLB+Deep": ("CROSS_DLB", "CROSS_DeepLeish"),
    "AIR+Deep": ("CROSS_AIR_LEISH", "CROSS_DeepLeish"),
    "DLB+AIR": ("CROSS_DLB", "CROSS_AIR_LEISH"),
}

DATASET_NAMES = (*CROSS_DATASETS, *COMBINED_DATASETS, *BASE_DATASETS)

def load_dataset(name: str, keep_sources: bool = True):
    if name in BASE_DATASETS:
        return ImageFolder(root=str(FOLDS_ROOT / name / "train"))

    if name in CROSS_DATASETS:
        base = CROSS_DATASETS[name]
        return merge_image_folders([
            ImageFolder(root=str(FOLDS_ROOT / base / "train")),
            ImageFolder(root=str(FOLDS_ROOT / base / "test")),
        ])

    if name in COMBINED_DATASETS:
        parts = [load_dataset(part) for part in COMBINED_DATASETS[name]]
        return ConcatDataset(parts) if keep_sources else merge_image_folders(parts)

    raise KeyError(f"Dataset desconhecido: {name}. Opções: {DATASET_NAMES}")

def load_test_dataset(name: str):
    if name not in BASE_DATASETS:
        raise KeyError(f"Dataset desconhecido {name}. Opções: {BASE_DATASETS}")
        
    return ImageFolder(root=str(FOLDS_ROOT / name / "test"))

def merge_image_folders(datasets: list[ImageFolder]) -> ImageFolder:
    reference = datasets[0].class_to_idx
    for ds in datasets[1:]:
        if ds.class_to_idx != reference:
            raise ValueError(f"Classes incompatíveis entre os datasets: {reference} != {ds.class_to_idx}")

    samples = sorted(
        (sample for ds in datasets for sample in ds.samples),
        key=lambda sample: (sample[1], Path(sample[0]).name),
    )

    merged = copy.copy(datasets[0])
    merged.samples = samples
    merged.imgs = samples
    merged.targets = [target for _, target in samples]
    return merged
