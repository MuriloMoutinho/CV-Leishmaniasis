import torch, time
from torch.utils.data import DataLoader

from config import TrainingConfig, create_binary_model, create_optimizer, create_loss_function, create_scheduler, \
    create_augmentation, get_optimizer_param_groups
from train import train_one_epoch, validate_model

def train_validate_model(
    train_dataset,
    teste_dataset,
    config: TrainingConfig,
    filename='model'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformations = create_augmentation(config.augmentation_level)
    train_dataset.transform = transformations["train"]
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    teste_dataset.transform = transformations["val"]
    test_loader = DataLoader(teste_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)
    param_groups = get_optimizer_param_groups(model, config.learning_rate, config.fine_tuning)
    optimizer = create_optimizer(config.optimizer_name, param_groups, config.weight_decay)
    loss_function = create_loss_function(config.loss_name)
    scheduler = create_scheduler(config.scheduler_name, optimizer) if config.scheduler_name is not None else None

    start = time.time()

    for epoch in range(config.epochs):

        start_epoch = time.time()
        print(f"\nÉpoca {epoch + 1}/{config.epochs}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device, scheduler)
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

        end_epoch = time.time()
        print(f"Tempo época: {end_epoch - start_epoch:.2f}s")

    if filename is not None:
        torch.save(model, filename)

    end = time.time()
    print(f"Tempo final: {end - start:.2f}s")

    val_metrics = validate_model(model, test_loader, loss_function, device)
    val_metrics['time'] = end - start

    return val_metrics