import torch, time, numpy,  torchvision
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from models import create_binary_model
from augmentation import get_images_transformations
from src import hyperparameters
from train import train_one_epoch, validate

def get_datasets(image_transforms):
    dataset_path = r"../datasets/test/train"

    train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=image_transforms["train"])
    val_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=image_transforms["val"])

    return train_dataset, val_dataset

def run_stratified_kfold(
    train_dataset,
    val_dataset,
    model_fn,
    loss_function_fn,
    optimizer_fn,
    n_splits=5,
    batch_size=8,
    epochs=50,
    patience_early_stopping=10,
    metric_to_monitor="f1"
):

    labels = numpy.array(train_dataset.targets)
    skf = StratifiedKFold(n_splits=n_splits ,shuffle=True, random_state=42)

    fold_results = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(numpy.zeros(len(labels)), labels), start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}/{n_splits}")
        print("=" * 60)

        train_fold = Subset(train_dataset, train_idx)
        val_fold = Subset(val_dataset, val_idx)

        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)

        fold_model = model_fn()
        optimizer = optimizer_fn(fold_model)
        loss_function = loss_function_fn()

        history, best_metrics = train_and_validate(
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            epoch_num=epochs,
            patience_early_stopping=patience_early_stopping,
            metric_to_monitor=metric_to_monitor
        )

        fold_results.append(best_metrics)
        fold_histories.append(history)

        print(f"\nResultados Fold {fold}:")
        print(f"Accuracy : {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall   : {best_metrics['recall']:.4f}")
        print(f"F1       : {best_metrics['f1']:.4f}")
        print(f"AUC      : {best_metrics['auc']:.4f}")

    return fold_results, fold_histories

def train_and_validate(
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
    best_metrics  = None

    epochs_without_improvement = 0

    for epoch in range(epoch_num):

        start = time.time()
        print(f"\nÉpoca {epoch + 1}/{epoch_num}")

        train_metrics = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_metrics = validate(model, val_loader, loss_function, device)

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

            best_metrics = val_metrics.copy()

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience_early_stopping:
                break

        end = time.time()
        print(f"Tempo: {end - start:.2f}s")

    return history, best_metrics

train_dataset, val_dataset = get_datasets(get_images_transformations())

fold_results, fold_histories = run_stratified_kfold(
    train_dataset,
    val_dataset,
    model_fn=create_binary_model,
    loss_function_fn=hyperparameters.create_loss_function,
    optimizer_fn=hyperparameters.create_optimizer,
    n_splits=5,
    batch_size=hyperparameters.batch_size,
    epochs=hyperparameters.epoch_num,
    patience_early_stopping=10,
    metric_to_monitor="f1"
)

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc"
]:
    values = [result[metric]for result in fold_results]

    print(
        f"{metric}: "
        f"{numpy.mean(values):.4f} ± "
        f"{numpy.std(values):.4f}"
    )