import gc
from dataclasses import replace
from pathlib import Path

import torch.cuda
import torchvision
from torch.utils.data import ConcatDataset

from data import save_kfold_result
from training import train_validate_kfold, KFoldConfig, TrainingConfig

DATASET_ROOT = Path("../datasets-folds")
DATASET_CROSS_ROOT = Path("../datasets-cross")

DATASETS = {
    "CROSS_AIR_LEISH": DATASET_CROSS_ROOT / "AIR_LEISH",
    "CROSS_DeepLeish": DATASET_CROSS_ROOT / "DeepLeish",
    "CROSS_DLB": DATASET_CROSS_ROOT / "DLB",

    "DLB+Deep": ["CROSS_DLB", "CROSS_DeepLeish"],
    "AIR+Deep": ["CROSS_AIR_LEISH", "CROSS_DeepLeish"],
    "DLB+AIR": ["CROSS_DLB", "CROSS_AIR_LEISH"],

    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_ROOT / "DLB" / "train",
}

def load_dataset(name: str):
    root = DATASETS[name]

    if isinstance(root, list):
        datasets = [
            torchvision.datasets.ImageFolder(root=str(DATASETS[dataset_name]))
            for dataset_name in root
        ]

        reference_classes = datasets[0].class_to_idx

        for ds in datasets[1:]:
            if ds.class_to_idx != reference_classes:
                raise ValueError(f"Classes incompatíveis entre os datasets: {reference_classes} != {ds.class_to_idx}")

        return ConcatDataset(datasets)

    return torchvision.datasets.ImageFolder(root=str(root))

kfold_config = KFoldConfig(
    n_splits=3,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="mcc",
)

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None, epochs=100,
                                batch_size=24, training_seed=42, fine_tuning="last_block", resolution=(576, 432),
                                augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

configs = [
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),

#("AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("AIR_LEISH",replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+AIR", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
("AIR+Deep", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+AIR", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.001, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.5, learning_rate=0.0007, weight_decay=0.01)),
("DLB+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
("AIR+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.001, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.5, learning_rate=0.0007, weight_decay=0.01)),
("DLB+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=0.0001)),
("AIR+Deep", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=0.0001)),
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
            filename="experiments/loso2/pre-results-loso/kfold/kfold.csv"
        )

