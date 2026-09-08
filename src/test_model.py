import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import hyperparameters
from augmentation import get_images_transformations
from train import validate

folder_test = r'../datasets/test/test'
model_path = r'./results/melhor_modelo.pt'

test_dataset = datasets.ImageFolder(root=folder_test, transform=get_images_transformations()['val'])
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # mudar o batch size não deve mudar as previsões do modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path, weights_only=False)

metrics = validate(model, test_data_loader, hyperparameters.create_loss_function(), device)
print(metrics)
