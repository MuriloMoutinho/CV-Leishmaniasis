import torch, time, numpy, copy
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from config import TrainingConfig, KFoldConfig, create_binary_model, create_optimizer, create_loss_function
from train import train_one_epoch, validate_model


def run_stratified_kfold(
    dataset,
    transformations,
    config: TrainingConfig,
    kfold_config: KFoldConfig
):

    labels = numpy.array(dataset.targets)
    labels_arr = numpy.zeros(len(labels))
    skf = StratifiedKFold(n_splits=kfold_config.n_splits ,shuffle=True, random_state=42)

    fold_results = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_arr, labels), start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{kfold_config.n_splits}")
        print("=" * 60)
        start = time.time()

        train_dataset = copy.copy(dataset)
        train_dataset.transform = transformations["train"]
        train_fold = Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_fold, batch_size=config.batch_size, shuffle=True)

        val_dataset = copy.copy(dataset)
        val_dataset.transform = transformations["val"]
        val_fold = Subset(val_dataset, val_idx)
        val_loader = DataLoader(val_fold, batch_size=config.batch_size, shuffle=False)

        fold_model = create_binary_model(config.model_name)
        optimizer = create_optimizer(config.optimizer_name, fold_model, config.learning_rate, config.weight_decay)
        loss_function = create_loss_function(config.loss_name)

        history, best_metrics_epoch = train_and_validate_fold(
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            epoch_num=config.epochs,
            patience_early_stopping=kfold_config.patience_early_stopping,
            metric_to_monitor=kfold_config.metric_to_monitor
        )

        fold_results.append(best_metrics_epoch)
        fold_histories.append(history)

        print(f"\nResultados Fold {fold}:")
        print(f"Accuracy : {best_metrics_epoch['accuracy']:.4f}")
        print(f"Precision: {best_metrics_epoch['precision']:.4f}")
        print(f"Recall   : {best_metrics_epoch['recall']:.4f}")
        print(f"F1       : {best_metrics_epoch['f1']:.4f}")
        print(f"AUC      : {best_metrics_epoch['auc']:.4f}")

        end = time.time()
        print(f"Tempo fold: {end - start:.2f}s")

    return fold_results, fold_histories

def train_and_validate_fold(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    epoch_num,
    patience_early_stopping,
    metric_to_monitor
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = []

    best_monitored_metric = -float("inf")
    best_metrics_epoch  = None

    epochs_without_improvement = 0

    for epoch in range(epoch_num):

        start = time.time()
        print(f"\nÉpoca {epoch + 1}/{epoch_num}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device)
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
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        current_monitored_metric = val_metrics[metric_to_monitor]

        # EARLY STOPPING
        if current_monitored_metric > best_monitored_metric:
            best_monitored_metric = current_monitored_metric
            epochs_without_improvement = 0

            best_metrics_epoch = val_metrics.copy()
            best_metrics_epoch['epoch'] = epoch + 1
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience_early_stopping:
                break

        end = time.time()
        print(f"Tempo época: {end - start:.2f}s")

    return history, best_metrics_epoch