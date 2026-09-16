import torch, time, copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from config import TrainingConfig, create_binary_model, create_optimizer, create_loss_function, \
    create_scheduler, HoldoutConfig, create_augmentation, get_optimizer_param_groups, compute_pos_weight
from train import train_one_epoch, validate_model


def train_validate_holdout(
    dataset,
    config: TrainingConfig,
    holdout_config: HoldoutConfig
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformations = create_augmentation(config.augmentation_level)
    train_loader, val_loader, pos_weight = create_data_loaders(dataset, transformations, config, holdout_config)

    model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)

    param_groups = get_optimizer_param_groups(model, config.learning_rate, config.weight_decay, config.fine_tuning)
    optimizer = create_optimizer(config.optimizer_name, param_groups)

    pos_weight = pos_weight.to(device)
    loss_function = create_loss_function(config.loss_name, pos_weight)

    scheduler = create_scheduler(config.scheduler_name, optimizer, len(train_loader), config.epochs) \
        if config.scheduler_name is not None else None

    history, best_metrics_epoch = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_num=config.epochs,
        patience_early_stopping=holdout_config.patience_early_stopping,
        metric_to_monitor=holdout_config.metric_to_monitor
    )

    print(f"\nResultados melhor época: {best_metrics_epoch['epoch']:.4f}")
    print(f"Accuracy : {best_metrics_epoch['accuracy']:.4f}")
    print(f"Precision: {best_metrics_epoch['precision']:.4f}")
    print(f"Recall   : {best_metrics_epoch['recall']:.4f}")
    print(f"F1       : {best_metrics_epoch['f1']:.4f}")
    print(f"AUC      : {best_metrics_epoch['auc']:.4f}")
    print(f"Tempo total: {best_metrics_epoch['time']:.2f}s")

    return best_metrics_epoch, history

def create_data_loaders(dataset, transformations, config, holdout_config):
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=holdout_config.val_ratio / 100,
        stratify=labels,
        random_state=holdout_config.val_split_seed,
    )

    train_dataset = copy.copy(dataset)
    train_dataset.transform = transformations["train"]
    train_fold = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_fold, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    val_dataset = copy.copy(dataset)
    val_dataset.transform = transformations["val"]
    val_fold = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_fold, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    pos_weight = compute_pos_weight(dataset, train_indices)

    return train_loader, val_loader, pos_weight

def train_and_validate(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    epoch_num,
    patience_early_stopping,
    metric_to_monitor
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    history = []

    best_monitored_metric = -float("inf")
    best_metrics_epoch  = None

    epochs_without_improvement = 0

    start = time.time()

    for epoch in range(epoch_num):

        start_epoch = time.time()
        print(f"\nÉpoca {epoch + 1}/{epoch_num}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device, scheduler)
        val_metrics = validate_model(model, val_loader, loss_function, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics['loss'],
            "train_accuracy": train_metrics['accuracy'],
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "val_auc": val_metrics['auc']
        })

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} "
            f"Val Pre: {val_metrics['precision']:.4f} | Val Rec: {val_metrics['recall']:.4f} "
            f"Val F1: {val_metrics['f1']:.4f} | Val Auc: {val_metrics['auc']:.4f}"
        )

        current_monitored_metric = val_metrics[metric_to_monitor]

        # EARLY STOPPING
        # se for necessário usar loss como métrica observável, o código precisa ser ajustado
        if current_monitored_metric > best_monitored_metric:
            best_monitored_metric = current_monitored_metric
            epochs_without_improvement = 0

            best_metrics_epoch = val_metrics.copy()
            best_metrics_epoch['epoch'] = epoch + 1
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience_early_stopping:
                break

        end_epoch = time.time()
        print(f"Tempo época: {end_epoch - start_epoch:.2f}s")

    end = time.time()
    best_metrics_epoch['time'] = end - start

    return history, best_metrics_epoch