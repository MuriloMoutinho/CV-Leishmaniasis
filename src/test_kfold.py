import gc
from dataclasses import replace
from pathlib import Path

import torch.cuda
import torchvision

from data import save_kfold_result
from training import train_validate_kfold, set_seed, KFoldConfig, TrainingConfig

DATASET_FOLDS_ROOT = Path("../datasets-folds")
DATASET_CROSS_ROOT = Path("../datasets-cross")
DATASET_CROSSTRAIN_ROOT = Path("../datasets-crosstrain")

DATASETS = {
    "AIR_LEISH": DATASET_FOLDS_ROOT / "AIR_LEISH" / "train",
    "DeepLeish": DATASET_FOLDS_ROOT / "DeepLeish" / "train",
    "DLB": DATASET_FOLDS_ROOT / "DLB" / "train",

    "CROSS_AIR_LEISH": DATASET_CROSS_ROOT / "AIR_LEISH",
    "CROSS_DeepLeish": DATASET_CROSS_ROOT / "DeepLeish",
    "CROSS_DLB": DATASET_CROSS_ROOT / "DLB",

    "DLB+Deep": DATASET_CROSSTRAIN_ROOT / "DLB+Deep",
    "AIR+Deep": DATASET_CROSSTRAIN_ROOT / "AIR+Deep",
    "DLB+AIR": DATASET_CROSSTRAIN_ROOT / "DLB+AIR",
}

def load_dataset(name: str):
    root = DATASETS[name]
    return torchvision.datasets.ImageFolder(root=str(root))

kfold_config = KFoldConfig(
    n_splits=3,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="f1_macro",
)

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None, epochs=100,
                                batch_size=24, training_seed=42, fine_tuning="last_block", resolution=(576, 432),
                                augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

resnet50 = replace(default_config, model_name="resnet50", dropout=0.4, learning_rate=0.002, weight_decay=0.000001)
densenet121 = replace(default_config, model_name="densenet121", dropout=0, learning_rate=0.002, weight_decay=0.001)
efficientnetb0 = replace(default_config, model_name="efficientnetb0", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)

configs = [
    #("DLB", replace(resnet50, augmentation_level="weak")),
    #("AIR_LEISH", replace(resnet50, augmentation_level="weak")),
    #("DeepLeish", replace(resnet50, augmentation_level="weak")),
    #("DLB", replace(resnet50, augmentation_level="medium")),
    #("AIR_LEISH", replace(resnet50, augmentation_level="medium")),
    #("DeepLeish", replace(resnet50, augmentation_level="medium")),
    #("DLB", replace(resnet50, augmentation_level="strong")),
    #("AIR_LEISH", replace(resnet50, augmentation_level="strong")),
    #("DeepLeish", replace(resnet50, augmentation_level="strong")),

    #("DLB", replace(densenet121, augmentation_level="weak")),
    #("AIR_LEISH", replace(densenet121, augmentation_level="weak")),
    #("DeepLeish", replace(densenet121, augmentation_level="weak")),
    #("DLB", replace(densenet121, augmentation_level="medium")),
    #("AIR_LEISH", replace(densenet121, augmentation_level="medium")),
    #("DeepLeish", replace(densenet121, augmentation_level="medium")),
    #("DLB", replace(densenet121, augmentation_level="strong")),
    #("AIR_LEISH", replace(densenet121, augmentation_level="strong")),
    #("DeepLeish", replace(densenet121, augmentation_level="strong")),

    #("DLB", replace(efficientnetb0, augmentation_level="weak")),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="weak")),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="weak")),
    #("DLB", replace(efficientnetb0, augmentation_level="medium")),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="medium")),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="medium")),
    #("DLB", replace(efficientnetb0, augmentation_level="strong")),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="strong")),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="strong")),

#CROSS
    #("CROSS_DLB", replace(resnet50, augmentation_level="weak")),
    #("CROSS_AIR_LEISH", replace(resnet50, augmentation_level="weak")),
    #("CROSS_DeepLeish", replace(resnet50, augmentation_level="weak")),
    #("CROSS_DLB", replace(resnet50, augmentation_level="medium")),
    #("CROSS_AIR_LEISH", replace(resnet50, augmentation_level="medium")),
    #("CROSS_DeepLeish", replace(resnet50, augmentation_level="medium")),
    #("CROSS_DLB", replace(resnet50, augmentation_level="strong")),
    #("CROSS_AIR_LEISH", replace(resnet50, augmentation_level="strong")),
    #("CROSS_DeepLeish", replace(resnet50, augmentation_level="strong")),

    #("CROSS_DLB", replace(densenet121, augmentation_level="weak")),
    #("CROSS_AIR_LEISH", replace(densenet121, augmentation_level="weak")),
    #("CROSS_DeepLeish", replace(densenet121, augmentation_level="weak")),
    #("CROSS_DLB", replace(densenet121, augmentation_level="medium")),
    #("CROSS_AIR_LEISH", replace(densenet121, augmentation_level="medium")),
    #("CROSS_DeepLeish", replace(densenet121, augmentation_level="medium")),
    #("CROSS_DLB", replace(densenet121, augmentation_level="strong")),
    #("CROSS_AIR_LEISH", replace(densenet121, augmentation_level="strong")),
    #("CROSS_DeepLeish", replace(densenet121, augmentation_level="strong")),

    #("CROSS_DLB", replace(efficientnetb0, augmentation_level="weak")),
    #("CROSS_AIR_LEISH", replace(efficientnetb0, augmentation_level="weak")),
    #("CROSS_DeepLeish", replace(efficientnetb0, augmentation_level="weak")),
    #("CROSS_DLB", replace(efficientnetb0, augmentation_level="medium")),
    #("CROSS_AIR_LEISH", replace(efficientnetb0, augmentation_level="medium")),
    #("CROSS_DeepLeish", replace(efficientnetb0, augmentation_level="medium")),
    #("CROSS_DLB", replace(efficientnetb0, augmentation_level="strong")),
    #("CROSS_AIR_LEISH", replace(efficientnetb0, augmentation_level="strong")),
    #("CROSS_DeepLeish", replace(efficientnetb0, augmentation_level="strong")),

#POLL CROSS
    #("DLB+Deep", replace(resnet50, augmentation_level="weak")),
    #("AIR+Deep", replace(resnet50, augmentation_level="weak")),
    #("DLB+AIR", replace(resnet50, augmentation_level="weak")),
    #("DLB+Deep", replace(densenet121, augmentation_level="weak")),
    #("AIR+Deep", replace(densenet121, augmentation_level="weak")),
    #("DLB+AIR", replace(densenet121, augmentation_level="weak")),
    #("DLB+Deep", replace(efficientnetb0, augmentation_level="weak")),
    #("AIR+Deep", replace(efficientnetb0, augmentation_level="weak")),
    #("DLB+AIR", replace(efficientnetb0, augmentation_level="weak")),

    #("DLB+Deep", replace(resnet50, augmentation_level="medium")),
    #("AIR+Deep", replace(resnet50, augmentation_level="medium")),
    #("DLB+AIR", replace(resnet50, augmentation_level="medium")),
    #("DLB+Deep", replace(densenet121, augmentation_level="medium")),
    #("AIR+Deep", replace(densenet121, augmentation_level="medium")),
    #("DLB+AIR", replace(densenet121, augmentation_level="medium")),
    #("DLB+Deep", replace(efficientnetb0, augmentation_level="medium")),
    #("AIR+Deep", replace(efficientnetb0, augmentation_level="medium")),
    #("DLB+AIR", replace(efficientnetb0, augmentation_level="medium")),

    #("DLB+Deep", replace(resnet50, augmentation_level="strong")),
    #("AIR+Deep", replace(resnet50, augmentation_level="strong")),
    #("DLB+AIR", replace(resnet50, augmentation_level="strong")),
    ("DLB+Deep", replace(densenet121, augmentation_level="strong")),
    ("AIR+Deep", replace(densenet121, augmentation_level="strong")),
    ("DLB+AIR", replace(densenet121, augmentation_level="strong")),
    ("DLB+Deep", replace(efficientnetb0, augmentation_level="strong")),
    ("AIR+Deep", replace(efficientnetb0, augmentation_level="strong")),
    ("DLB+AIR", replace(efficientnetb0, augmentation_level="strong")),
]

if __name__ == "__main__":
    set_seed()
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

