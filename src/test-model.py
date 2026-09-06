import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from augmentation import get_images_transformations
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_prob = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1) # Logits [12312, 124123]
            probabilities = torch.sigmoid(outputs) # Probabilidade da classe 0-1
            predictions = (probabilities >= 0.5).long() # Classe prevista 0 ou 1

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


folder_test = r'../datasets/test/test'
model_path = r'./models/melhor_modelo.pt'

test_dataset = datasets.ImageFolder(root=folder_test, transform=get_images_transformations()['val'])
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # mudar o batch size não deve mudar as previsões do modelo

model = torch.load(model_path, weights_only=False)

metrics = evaluate_model(model, test_data_loader)
print(metrics)
