from dataclasses import replace
from pathlib import Path
from time import time

import gc
import torch
import torchvision

from handler.training_factorys import create_augmentation, create_binary_model, create_loss_function
from data import save_cross_dataset_result
from training import validate_model, train_model, set_seed, TrainingConfig
from training_config import create_dataloader

DATASET_ROOT = Path("../datasets-cross")

DATASETS = {
    "AIR_LEISH": DATASET_ROOT / "AIR_LEISH",
    "DeepLeish": DATASET_ROOT / "DeepLeish",
    "DLB": DATASET_ROOT / "DLB",
}

test_transform = create_augmentation("weak", (768, 576))["val"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_train_dataset(dataset_name: str):
    root = DATASET_ROOT / dataset_name
    return torchvision.datasets.ImageFolder(root=str(root))

def load_test_loader(dataset_name: str, batch_size: int = 16):
    root = DATASET_ROOT / dataset_name
    test_dataset = torchvision.datasets.ImageFolder(root=str(root), transform=test_transform)
    return create_dataloader(test_dataset, batch_size, False)

def load_trained_model(config: TrainingConfig, checkpoint_path: Path, device):
    if checkpoint_path.is_file():
        model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    return None


default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None,
                                batch_size=24, fine_tuning="last_block", resolution=(576, 432),
                                epochs=0, augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

resnet50 = replace(default_config, model_name="resnet50", dropout=0.4, learning_rate=0.002, weight_decay=0.000001)
densenet121 = replace(default_config, model_name="densenet121", dropout=0, learning_rate=0.002, weight_decay=0.001)
efficientnetb0 = replace(default_config, model_name="efficientnetb0", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)

configs = [

]

if __name__ == "__main__":
    set_seed()
    for name_train_dataset, name_test_dataset, config in configs:

        print(f"\nTreinado em {name_train_dataset} | Testado em {name_test_dataset} | {config.model_name} | aug={config.augmentation_level} ")
        checkpoint_path = Path(f"experiments/models/total/{name_train_dataset}_{config.model_name}_{config.augmentation_level}.pth")
        model = load_trained_model(config, checkpoint_path, device)

        if model is None:
            print(f"Modelo não encontrado. Treinando {config.model_name} com aug {config.augmentation_level} em {name_train_dataset}.")
            train_dataset = load_train_dataset(name_train_dataset)
            model = train_model(train_dataset, config, checkpoint_path)

        model.to(device)
        model.eval()

        test_loader = load_test_loader(name_test_dataset)
        loss_function = create_loss_function(config.loss_name, pos_weight=None)

        start = time()
        metrics = validate_model(model, test_loader, loss_function, device)
        metrics['time'] = time() - start
        print(metrics)

        torch.cuda.empty_cache()
        gc.collect()

        save_cross_dataset_result(
            model_result=metrics,
            trained_on=name_train_dataset,
            tested_on=name_test_dataset,
            config=config,
            filename="experiments/cross_dataset_results2.csv",
        )