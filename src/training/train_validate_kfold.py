from dataclasses import replace

import optuna
import torch, time, numpy, copy, dataclasses
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, ConcatDataset

from handler import create_binary_model, create_optimizer, create_loss_function, create_scheduler, create_augmentation
from training_config import get_optimizer_param_groups, compute_pos_weight, create_dataloader
from training import train_one_epoch, validate_model, set_seed, TrainingConfig, KFoldConfig, ValidationMetrics

def train_validate_kfold(
    dataset,
    config: TrainingConfig,
    kfold_config: KFoldConfig,
    trial=None
) -> list[ValidationMetrics]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels, dataset_sources = get_dataset_labels_and_sources(dataset)
    labels_arr = numpy.zeros(len(labels))

    stratification_labels = numpy.array([
        f"{label}_{source}"
        for label, source in zip(labels, dataset_sources)
    ])

    skf = StratifiedKFold(n_splits=kfold_config.n_splits, shuffle=True, random_state=kfold_config.val_split_seed)

    fold_results = []

    transformations = create_augmentation(config.augmentation_level, config.resolution)

    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_arr, stratification_labels), start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{kfold_config.n_splits}")
        print("=" * 60)

        set_seed(config.training_seed)

        train_loader, val_loader, pos_weight = create_data_loaders_kfold(dataset, transformations, config, train_idx, val_idx)

        fold_model = create_binary_model(config.model_name, config.dropout, config.fine_tuning)

        param_groups = get_optimizer_param_groups(fold_model, config.learning_rate, config.weight_decay, config.fine_tuning)
        optimizer = create_optimizer(config.optimizer_name, param_groups)

        pos_weight = pos_weight.to(device)
        loss_function = create_loss_function(config.loss_name, pos_weight)

        scheduler = create_scheduler(config.scheduler_name, optimizer, len(train_loader), config.epochs) \
            if config.scheduler_name is not None else None

        best_metrics_epoch = train_and_validate_fold(
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=config.epochs,
            patience_early_stopping=kfold_config.patience_early_stopping,
            metric_to_monitor=kfold_config.metric_to_monitor,
            trial=trial,
            enable_pruning=(fold == 1)
        )

        fold_results.append(best_metrics_epoch)

        print(f"\nResultados Fold {fold}:" )
        print(f"Resultados melhor época: {best_metrics_epoch.epoch:.4f}")
        print(f"Accuracy : {best_metrics_epoch.accuracy:.4f}")
        print(f"Precision: {best_metrics_epoch.precision:.4f}")
        print(f"Recall   : {best_metrics_epoch.recall:.4f}")
        print(f"F1       : {best_metrics_epoch.f1_class1:.4f}")
        print(f"Tempo fold: {best_metrics_epoch.time:.2f}s")

    return fold_results

from torch.utils.data import ConcatDataset
import numpy


def get_dataset_labels_and_sources(dataset):
    if isinstance(dataset, ConcatDataset):
        labels = numpy.concatenate([numpy.array(ds.targets) for ds in dataset.datasets])

        sources = numpy.concatenate([
            numpy.full(len(ds), i, dtype=int)
            for i, ds in enumerate(dataset.datasets)
        ])

    else:
        labels = numpy.array(dataset.targets)
        sources = numpy.zeros(len(dataset), dtype=int)

    return labels, sources

def create_data_loaders_kfold(dataset, transformations, config, train_idx, val_idx):
    train_dataset = copy.deepcopy(dataset)
    if isinstance(train_dataset, ConcatDataset):
        for ds in train_dataset.datasets:
            ds.transform = transformations.train
    else:
        train_dataset.transform = transformations.train

    train_fold = Subset(train_dataset, train_idx)
    train_loader = create_dataloader(train_fold, config.batch_size, True)

    val_dataset = copy.deepcopy(dataset)
    if isinstance(val_dataset, ConcatDataset):
        for ds in val_dataset.datasets:
            ds.transform = transformations.val
    else:
        val_dataset.transform = transformations.val

    val_fold = Subset(val_dataset, val_idx)
    val_loader = create_dataloader(val_fold, config.batch_size, False)

    pos_weight = compute_pos_weight(dataset, train_idx)

    return train_loader, val_loader, pos_weight

def train_and_validate_fold(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    epoch_num,
    patience_early_stopping,
    metric_to_monitor,
    trial=None,
    enable_pruning=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_monitored_metric = -float("inf")
    best_metrics_epoch  = None

    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(epoch_num):

        start_epoch = time.time()
        print(f"\nÉpoca {epoch + 1}/{epoch_num}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device, scheduler)
        val_metrics = validate_model(model, val_loader, loss_function, device)

        print(
            f"Train Loss: {train_metrics.loss:.4f} | Train Acc: {train_metrics.accuracy:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | Val Acc: {val_metrics.accuracy:.4f}"
        )

        current_monitored_metric = val_metrics.get(metric_to_monitor)

        if trial is not None and enable_pruning:
            trial.report(float(current_monitored_metric), step=epoch)

            if trial.should_prune():
                print(f"Trial {trial.number} podado na época {epoch + 1}.")
                raise optuna.TrialPruned()

        # EARLY STOPPING
        # se for necessário usar loss como métrica observável, o código precisa ser ajustado
        if current_monitored_metric > best_monitored_metric:
            best_monitored_metric = current_monitored_metric
            epochs_without_improvement = 0

            best_metrics_epoch = replace(val_metrics, epoch=epoch + 1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience_early_stopping:
                break

        end_epoch = time.time()
        print(f"Tempo época: {end_epoch - start_epoch:.2f}s")

    end = time.time()
    best_metrics_epoch.time = end - start

    return best_metrics_epoch