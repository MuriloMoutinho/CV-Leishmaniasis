import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from augmentation import weak_augmentation
from config import create_loss_function
from train import validate_model

folder_test = r'../datasets/test/test'
model_path = r'experiments/melhor_modelo.pt'

test_dataset = datasets.ImageFolder(root=folder_test, transform=weak_augmentation()['val'])
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # mudar o batch size não deve mudar as previsões do modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path, weights_only=False)
# model.load_state_dict(torch.load("results/melhor_modelo.pt"))

metrics = validate_model(model, test_data_loader, create_loss_function("cross_entropy"), device)
print(metrics)
