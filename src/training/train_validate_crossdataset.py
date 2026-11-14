import time

import torch

from handler import create_loss_function, create_augmentation
from training_config import create_dataloader
from training import validate_model, train_model, set_seed, TrainingConfig


def train_validate_crossdataset(
    train_dataset,
    test_dataset,
    config: TrainingConfig,
    model=None
):
    set_seed(config.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = train_model(train_dataset, config)

    model.to(device)
    model.eval()

    test_dataset.transform = create_augmentation(config.augmentation_level, config.resolution)["val"]
    test_loader = create_dataloader(test_dataset, config.batch_size, False)

    loss_function = create_loss_function(config.loss_name, pos_weight=None)

    start = time.time()
    metrics = validate_model(model, test_loader, loss_function, device)
    metrics['time'] = time.time() - start

    return metrics, model
