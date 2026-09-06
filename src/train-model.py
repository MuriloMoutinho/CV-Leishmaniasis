import torch, time, os, numpy
from torchvision import datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optmin
from models import create_binary_model
from augmentation import get_images_transformations
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def get_transformed_dataset(image_transforms):
    dataset_path = r'../datasets/test/'
    folder_train = os.path.join(dataset_path, 'train')
    folder_val = os.path.join(dataset_path, 'val')

    return {
        'train': datasets.ImageFolder(root=folder_train, transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=folder_val, transform=image_transforms['val']),
    }

def train_and_validate(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    epoch_num=20,
    metric_to_monitor="f1"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    history = []
    best_metric = -float("inf")

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
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        current_metric = val_metrics[metric_to_monitor]

        # stop early
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model, 'models/melhor_modelo.pt')

        end = time.time()
        print(f"Tempo: {end - start:.2f}s")

    return history


def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()

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

    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    all_predictions = []

    with torch.no_grad():
        for images_batch, labels in data_loader:

            images_batch = images_batch.to(device)
            labels = labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images_batch)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images_batch.size(0)

            probabilities = torch.sigmoid(outputs) # Probabilidade da classe 1
            predictions = (probabilities >= 0.5).float() # Classe prevista

            all_labels.extend(labels.cpu().numpy().ravel())
            all_probabilities.extend(probabilities.cpu().numpy().ravel())
            all_predictions.extend(predictions.cpu().numpy().ravel())

    all_labels = numpy.array(all_labels)
    all_probabilities = numpy.array(all_probabilities)
    all_predictions = numpy.array(all_predictions)

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


#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
batch_size = 16 #vai pegar x imagens por vez do dataset e enviá-las para a rede.
epoch_num = 40
learning_rate = 0.001 #verificar linear scaling rule (valor padrão)

model_train = create_binary_model()
error_function = nn.BCEWithLogitsLoss() #Binary Cross-Entropy
optimizer = optmin.AdamW(model_train.parameters(), lr=learning_rate)

images = get_transformed_dataset(get_images_transformations())
train_loader = DataLoader(images["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(images["val"], batch_size=batch_size, shuffle=False)

model_history = train_and_validate(model_train, train_loader, val_loader, error_function, optimizer, epoch_num)


def create_loss_history_graph(history):
    epochs = range(1, len(history) + 1)

    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.plot(epochs, train_loss, label="Loss treino")
    plt.plot(epochs, val_loss, label="Loss validação")

    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def create_metrics_history_graph(history):
    epochs = range(1, len(history) + 1)

    val_accuracy = [h["val_accuracy"] for h in history]
    val_precision = [h["val_precision"] for h in history]
    val_recall = [h["val_recall"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    plt.plot(epochs, val_accuracy, label="Accuracy")
    plt.plot(epochs, val_precision, label="Precision")
    plt.plot(epochs, val_recall, label="Recall")
    plt.plot(epochs, val_f1, label="F1")
    plt.plot(epochs, val_auc, label="AUC")

    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

create_loss_history_graph(model_history)
create_metrics_history_graph(model_history)
