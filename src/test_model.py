import gc, torch
from dataclasses import replace
from pathlib import Path

import torchvision

from data import save_training_result
from training import train_validate_model, TrainingConfig

DATASET_ROOT = Path("../datasets-folds")

DATASETS = {
    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH",
    "DeepLeish": DATASET_ROOT / "DeepLeish",
    "DLB": DATASET_ROOT / "DLB",
}

def load_datasets(name: str):
    root = DATASETS[name]
    train_dataset = torchvision.datasets.ImageFolder(root=str(root / "train"))
    test_dataset = torchvision.datasets.ImageFolder(root=str(root / "test"))
    return train_dataset, test_dataset

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", resolution=(768, 576), batch_size=16,
                                epochs=50, scheduler_name=None, augmentation_level="weak", fine_tuning="last_two_blocks",
                                model_name="", dropout=0, learning_rate=0, weight_decay=0)

resnet50 = replace(default_config, model_name="resnet50", dropout=0, learning_rate=0.002, weight_decay=1e-06)
densenet121 = replace(default_config, model_name="densenet121", dropout=0, learning_rate=0.002, weight_decay=1e-06)
efficientnetb0 = replace(default_config, model_name="efficientnetb0", dropout=0.5, learning_rate=0.002, weight_decay=0.01)

configs = [
    #("DLB", replace(resnet50, augmentation_level="weak", epochs=5)),
    #("AIR_LEISH", replace(resnet50, augmentation_level="weak", epochs=7)),
    #("DeepLeish", replace(resnet50, augmentation_level="weak", epochs=6)),

    #("DLB", replace(densenet121, augmentation_level="weak", epochs=8)),
    #("AIR_LEISH", replace(densenet121, augmentation_level="weak", epochs=5)),
    #("DeepLeish", replace(densenet121, augmentation_level="weak", epochs=8)),

    #("DLB", replace(efficientnetb0, augmentation_level="weak", epochs=5)),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="weak", epochs=6)),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="weak", epochs=7)),

    #("DLB", replace(resnet50, augmentation_level="medium", epochs=5)),
    #("AIR_LEISH", replace(resnet50, augmentation_level="medium", epochs=6)),
    #("DeepLeish", replace(resnet50, augmentation_level="medium", epochs=5)),

    #("DLB", replace(densenet121, augmentation_level="medium", epochs=6)),
    #("AIR_LEISH", replace(densenet121, augmentation_level="medium", epochs=9)),
    #("DeepLeish", replace(densenet121, augmentation_level="medium", epochs=8)),

    #("DLB", replace(efficientnetb0, augmentation_level="medium", epochs=8)),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="medium", epochs=8)),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="medium", epochs=10)),

    #("DLB", replace(resnet50, augmentation_level="strong", epochs=6)),
    #("AIR_LEISH", replace(resnet50, augmentation_level="strong", epochs=5)),
    #("DeepLeish", replace(resnet50, augmentation_level="strong", epochs=7)),

    #("DLB", replace(densenet121, augmentation_level="strong", epochs=4)),
    #("AIR_LEISH", replace(densenet121, augmentation_level="strong", epochs=7)),
    #("DeepLeish", replace(densenet121, augmentation_level="strong", epochs=9)),

    #("DLB", replace(efficientnetb0, augmentation_level="strong", epochs=7)),
    #("AIR_LEISH", replace(efficientnetb0, augmentation_level="strong", epochs=9)),
    #("DeepLeish", replace(efficientnetb0, augmentation_level="strong", epochs=11)),
]

if __name__ == "__main__":
    for dataset_name, config in configs:
        train_dataset, test_dataset = load_datasets(dataset_name)

        history = train_validate_model(
            train_dataset,
            test_dataset,
            config
        )

        torch.cuda.empty_cache()
        gc.collect()

        save_training_result(
            model_result=history,
            config=config,
            dataset_name=dataset_name,
            filename='experiments/training.csv')