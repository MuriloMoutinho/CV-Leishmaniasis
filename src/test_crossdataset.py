from dataclasses import replace
from pathlib import Path

import gc
import torch
import torchvision

from data import save_cross_dataset_result
from training import train_validate_crossdataset, TrainingConfig

DATASET_CROSS_ROOT = Path("../datasets-cross")
DATASET_CROSSTRAIN_ROOT = Path("../datasets-crosstrain")

DATASETS = {
    "CROSS_AIR_LEISH": DATASET_CROSS_ROOT / "AIR_LEISH",
    "CROSS_DeepLeish": DATASET_CROSS_ROOT / "DeepLeish",
    "CROSS_DLB": DATASET_CROSS_ROOT / "DLB",

    "DLB+Deep": DATASET_CROSSTRAIN_ROOT / "DLB+Deep",
    "AIR+Deep": DATASET_CROSSTRAIN_ROOT / "AIR+Deep",
    "DLB+AIR": DATASET_CROSSTRAIN_ROOT / "DLB+AIR",
}

def load_dataset(dataset_name: str):
    root = DATASETS[dataset_name]
    return torchvision.datasets.ImageFolder(root=str(root))


default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None,
                                batch_size=24, fine_tuning="last_block", resolution=(576, 432),
                                epochs=0, augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

configs = [
#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=5)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=5)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=12)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=12)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=7)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=7)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=9)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=9)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=5)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=5)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=6)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=6)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=11)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.003, weight_decay=0.0001, epochs=11)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=9)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=9)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=6)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=6)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=24)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=24)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=21)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=21)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=5)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=5)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=25)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=25)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=10)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=10)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=8)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=8)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=20)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=20)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=15)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=15)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=8)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=8)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=19)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=19)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=10)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=10)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=14)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=14)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=19)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=19)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=6)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=6)),

#("CROSS_AIR_LEISH", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=13)),
#("CROSS_AIR_LEISH", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=13)),
#("CROSS_DeepLeish", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=14)),
#("CROSS_DeepLeish", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=14)),
#("CROSS_DLB", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=10)),
#("CROSS_DLB", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=10)),

#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=11)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=12)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=22)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=19)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=12)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=18)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=17)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06, epochs=8)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=11)),

#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.2, learning_rate=0.001, weight_decay=0.0001, epochs=17)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.5, learning_rate=0.0007, weight_decay=0.01, epochs=28)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=23)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.001, weight_decay=0.0001, epochs=24)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.5, learning_rate=0.0007, weight_decay=0.01, epochs=20)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=19)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.001, weight_decay=0.0001, epochs=30)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.5, learning_rate=0.0007, weight_decay=0.01, epochs=24)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06, epochs=24)),

#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=23)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=13)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="weak", dropout=0.4, learning_rate=0.002, weight_decay=0.0001, epochs=17)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=15)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=14)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=0.0001, epochs=28)),
#("AIR+Deep", "CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001, epochs=15)),
#("DLB+AIR", "CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06, epochs=10)),
#("DLB+Deep", "CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=0.0001, epochs=15)),
]

if __name__ == "__main__":
    seeds = [41, 42, 43, 44, 45]

    for name_train_dataset, name_test_dataset, config in configs:
        train_dataset = load_dataset(name_train_dataset)
        test_dataset = load_dataset(name_test_dataset)
        historys = []

        for seed in seeds:
            print(f"\n{'=' * 50}")
            print(f"Treinando com seed {seed}")
            print(f"{'=' * 50}")
            print(f"\nTreinado em {name_train_dataset} | Testado em {name_test_dataset} | {config.model_name} | aug={config.augmentation_level} ")

            checkpoint_path = Path(
                f"experiments/loso2/results/cross/models/{name_train_dataset}_{config.model_name}_{config.augmentation_level}_{seed}.pth")

            #tag = hashlib.md5(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:8]
            #checkpoint_path = Path(f".../{train}_{config.model_name}_{config.augmentation_level}_{tag}_{seed}.pth")

            metrics = train_validate_crossdataset(
                train_dataset,
                test_dataset,
                replace(config, training_seed=seed),
                checkpoint_path
            )
            historys.append(metrics)

            torch.cuda.empty_cache()
            gc.collect()

        save_cross_dataset_result(
            model_result=historys,
            trained_on=name_train_dataset,
            tested_on=name_test_dataset,
            config=replace(config, training_seed=None),  # resultado agrega várias seeds
            filename="experiments/loso2/results/cross/cross_training.csv",
        )
