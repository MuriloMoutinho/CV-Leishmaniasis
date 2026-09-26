import gc, torch
from pathlib import Path

import torchvision

from training_config import TrainingConfig
from data import save_training_result
from training import train_validate_model

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

configs = [
    ("DLB", TrainingConfig(model_name="resnet50", fine_tuning=None, dropout=0,
                           loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.003,
                           weight_decay=0.01, batch_size=16, epochs=10, scheduler_name="warmup+cosine",
                           augmentation_level="strong")),
    ("DLB", TrainingConfig(model_name="densenet121", fine_tuning=None, dropout=0,
                           loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                           weight_decay=0.01, batch_size=16, epochs=10, scheduler_name="warmup+cosine",
                           augmentation_level="strong")),
    ("DLB", TrainingConfig(model_name="efficientnetb0", fine_tuning=None, dropout=0,
                           loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                           weight_decay=0.01, batch_size=16, epochs=10, scheduler_name="warmup+cosine",
                           augmentation_level="strong")),

    ("AIR_LEISH", TrainingConfig(model_name="resnet50", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.003,
                                 weight_decay=0.01, batch_size=16, epochs=14, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),
    ("AIR_LEISH", TrainingConfig(model_name="densenet121", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                                 weight_decay=0.01, batch_size=16, epochs=22, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),
    ("AIR_LEISH", TrainingConfig(model_name="efficientnetb0", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                                 weight_decay=0.01, batch_size=16, epochs=13, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),

    ("DeepLeish", TrainingConfig(model_name="resnet50", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.003,
                                 weight_decay=0.01, batch_size=16, epochs=22, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),
    ("DeepLeish", TrainingConfig(model_name="densenet121", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                                 weight_decay=0.01, batch_size=16, epochs=29, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),
    ("DeepLeish", TrainingConfig(model_name="efficientnetb0", fine_tuning=None, dropout=0,
                                 loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
                                 weight_decay=0.01, batch_size=16, epochs=21, scheduler_name="warmup+cosine",
                                 augmentation_level="strong")),
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