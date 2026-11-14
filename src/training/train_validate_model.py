import torch, time

from handler import create_loss_function, create_augmentation
from training_config import compute_pos_weight, create_dataloader, set_transform
from training import validate_model, train_model, set_seed, TrainingConfig

def train_validate_model(
    train_dataset,
    test_dataset,
    config: TrainingConfig,
    model=None
):
    set_seed(config.training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    if model is None:
        model = train_model(train_dataset, config)
    train_time = time.time() - start

    model.to(device)
    model.eval()

    set_transform(test_dataset, create_augmentation(config.augmentation_level, config.resolution).val)
    test_loader = create_dataloader(test_dataset, config.batch_size, False)

    pos_weight = compute_pos_weight(train_dataset, range(len(train_dataset))).to(device)
    loss_function = create_loss_function(config.loss_name, pos_weight)

    val_metrics = validate_model(model, test_loader, loss_function, device)
    val_metrics.time = train_time

    return val_metrics, model
