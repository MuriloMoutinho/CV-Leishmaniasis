import gc
from pathlib import Path

import torch.cuda
import torchvision

from training_config import KFoldConfig, TrainingConfig
from data import save_kfold_result
from train import train_validate_kfold

DATASET_ROOT = Path("../datasets-folds")

DATASETS = {
    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_ROOT / "DLB" / "train",
}

def load_dataset(name: str):
    root = DATASETS[name]
    return torchvision.datasets.ImageFolder(root=str(root))

kfold_config = KFoldConfig(
    n_splits=5,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="f1",
)


configs = [
    ("DLB", TrainingConfig(model_name="resnet50", fine_tuning="last_block", dropout=0,
                           loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.003,
                           weight_decay=0.01, batch_size=16, epochs=30, scheduler_name="warmup+cosine",
                           augmentation_level="weak")),
]

if __name__ == "__main__":
    for dataset_name, config in configs:
        fold_results, fold_histories = train_validate_kfold(
            load_dataset(dataset_name),
            config=config,
            kfold_config=kfold_config
        )
        torch.cuda.empty_cache()
        gc.collect()

        save_kfold_result(
            fold_results=fold_results,
            config=config,
            dataset_name=dataset_name,
            kfold_config=kfold_config,
            filename="experiments/kfold.csv"
        )

