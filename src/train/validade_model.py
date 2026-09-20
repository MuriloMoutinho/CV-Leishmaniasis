import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)

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
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    pr_auc = average_precision_score(all_labels, all_probabilities)

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    return metrics