from pathlib import Path

import torch, time
from torch.utils.data import DataLoader

from config import TrainingConfig, create_binary_model, create_optimizer, create_loss_function, create_scheduler, \
    create_augmentation, get_optimizer_param_groups, compute_pos_weight
from train import train_one_epoch, validate_model

def train_validate_model(
    train_dataset,
    test_dataset,
    config: TrainingConfig,
    filename=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformations = create_augmentation(config.augmentation_level)
    train_dataset.transform = transformations["train"]
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    test_dataset.transform = transformations["val"]
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)

    param_groups = get_optimizer_param_groups(model, config.learning_rate, config.weight_decay, config.fine_tuning)
    optimizer = create_optimizer(config.optimizer_name, param_groups)

    pos_weight = compute_pos_weight(train_dataset, range(len(train_dataset.samples)))
    pos_weight = pos_weight.to(device)
    loss_function = create_loss_function(config.loss_name, pos_weight)

    scheduler = create_scheduler(config.scheduler_name, optimizer, len(train_loader), config.epochs) \
        if config.scheduler_name is not None else None

    start = time.time()

    for epoch in range(config.epochs):

        start_epoch = time.time()
        print(f"\nÉpoca {epoch + 1}/{config.epochs}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device, scheduler)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

        end_epoch = time.time()
        print(f"Tempo época: {end_epoch - start_epoch:.2f}s")

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), filename)

    end = time.time()
    print(f"Tempo final: {end - start:.2f}s")

    val_metrics = validate_model(model, test_loader, loss_function, device)
    val_metrics['time'] = end - start

    return val_metrics