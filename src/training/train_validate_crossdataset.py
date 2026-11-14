import time
from pathlib import Path

import torch

from handler import create_binary_model, create_loss_function, create_augmentation
from training_config import create_dataloader
from training import validate_model, train_model, TrainingConfig


def train_validate_crossdataset(
    train_dataset,
    test_dataset,
    config: TrainingConfig,
    checkpoint_path: Path | None = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(config, checkpoint_path, device)
    if model is None:
        print(f"Modelo não encontrado. Treinando {config.model_name} com aug {config.augmentation_level}.")
        model = train_model(train_dataset, config, checkpoint_path)

    model.to(device)
    model.eval()

    test_dataset.transform = create_augmentation(config.augmentation_level, config.resolution)["val"]
    test_loader = create_dataloader(test_dataset, config.batch_size, False)

    loss_function = create_loss_function(config.loss_name, pos_weight=None)

    start = time.time()
    metrics = validate_model(model, test_loader, loss_function, device)
    metrics['time'] = time.time() - start

    return metrics


def load_trained_model(config: TrainingConfig, path: Path | None, device):
    if path is None or not Path(path).is_file():
        return None

    model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
