import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix
)
import numpy as np

def validate_model(model, data_loader, loss_function, device):
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for images_batch, labels in data_loader:

            images_batch = images_batch.to(device)
            labels = labels.float().to(device)

            outputs = model(images_batch).squeeze(1) 

            loss = loss_function(outputs, labels)
            total_loss += loss.item() * images_batch.size(0)

            probabilities = torch.sigmoid(outputs) # Probabilidade da classe 1, 0 ate 1
            predictions = (probabilities >= 0.5).float() # Classe prevista, 0 ou 1

            all_labels.extend(labels.cpu().numpy().ravel())
            all_probabilities.extend(probabilities.cpu().numpy().ravel())
            all_predictions.extend(predictions.cpu().numpy().ravel())

    average_loss = total_loss / len(data_loader.dataset)

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probabilities)
    y_pred = np.array(all_predictions).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_class0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc_class1": average_precision_score(y_true, y_prob),
        "pr_auc_class0": average_precision_score(1 - y_true, 1 - y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "labels": y_true, "probabilities": y_prob,
    }
    return metrics