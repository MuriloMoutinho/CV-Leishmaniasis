import gc, torch
from dataclasses import replace
from pathlib import Path

import torchvision

from data import save_training_result, save_npz, save_model, load_model, config_tag
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

def model_path(dataset_name: str, config: TrainingConfig, seed: int) -> str:
    return (f"experiments/loso/results/intra/models/"
            f"{dataset_name}_{config.model_name}_{config.augmentation_level}_{config_tag(config)}_{seed}.pth")

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None,
                                batch_size=24, fine_tuning="last_block", resolution=(576, 432),
                                epochs=0, augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

configs = [
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=9)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06, epochs=12)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=5)),
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=10)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06, epochs=11)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=10)),
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=13)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06, epochs=16)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=13)),

#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=6)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=16)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=7)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=15)),
#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=8)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=13)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=12)),

#("AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=12)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=23)),
#("AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=8)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=17)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=17)),
#("AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=19)),
]

if __name__ == "__main__":
    seeds = [41, 42, 43, 44, 45]

    for dataset_name, config in configs:
        train_dataset, test_dataset = load_datasets(dataset_name)
        historys = []

        for seed in seeds:
            print(f"\n{'=' * 50}")
            print(f"Treinando com seed {seed}")
            print(f"{'=' * 50}")

            path = model_path(dataset_name, config, seed)
            loaded_model = load_model(config, path)

            history, model = train_validate_model(
                train_dataset,
                test_dataset,
                replace(config, training_seed=seed),
                model=loaded_model
            )
            historys.append(history)

            if loaded_model is None:
                save_model(model, path)

            del model, loaded_model
            torch.cuda.empty_cache()
            gc.collect()

        experiment_id = save_training_result(
            model_result=historys,
            config=config,
            dataset_name=dataset_name,
            filename="experiments/loso/results/intra/traning.csv",
            seeds=seeds)
        save_npz(historys, f"experiments/loso/results/intra/labels-training/{experiment_id}", file_prefix="model")