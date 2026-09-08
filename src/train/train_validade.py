import numpy
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    model.to(device)

    total_loss = 0.0

    all_labels = []
    all_predictions = []

    for images_batch, labels in data_loader:

        images_batch = images_batch.to(device)
        labels = labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images_batch)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images_batch.size(0)

        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy().ravel())
        all_predictions.extend(predictions.detach().cpu().numpy().ravel())

    all_labels = numpy.array(all_labels)
    all_predictions = numpy.array(all_predictions)

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return {
        "loss": average_loss,
        "accuracy": accuracy
    }

def validate(model, data_loader, loss_function, device):
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for images_batch, labels in data_loader:

            images_batch = images_batch.to(device)
            labels = labels.float().to(device)

            outputs = model(images_batch).squeeze(1) # Logits [12312, 124123]

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
    auc = roc_auc_score(all_labels, all_probabilities)

    metrics = {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    return metrics