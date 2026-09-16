import gc, torch
from pathlib import Path

import torchvision

from config import TrainingConfig, HoldoutConfig
from data import save_holdout_result
from train import train_validate_holdout

DATASET_ROOT = Path("../datasets")

DATASETS = {
    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_ROOT / "DLB" / "train",
}

def load_dataset(name: str):
    root = DATASETS[name]
    return torchvision.datasets.ImageFolder(root=str(root))

holdout_config = HoldoutConfig(
    val_ratio=25,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="f1",
)

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
#verificar linear scaling rule

configs = [
    ("DLB", TrainingConfig(
        model_name="resnet50",
        fine_tuning="last_two_blocks",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.0003,
        weight_decay=0.01,
        batch_size=16,
        epochs=50,
        scheduler_name="warmup+cosine",
        augmentation_level="strong",
    )),
]

if __name__ == "__main__":
    for dataset_name, config in configs:
        fold_results, fold_histories = train_validate_holdout(
            load_dataset(dataset_name),
            config=config,
            holdout_config=holdout_config
        )
        torch.cuda.empty_cache()
        gc.collect()

        save_holdout_result(
            model_result=fold_results,
            config=config,
            holdout_config=holdout_config,
            dataset_name=dataset_name,
            filename="experiments/holdout.csv"
        )

