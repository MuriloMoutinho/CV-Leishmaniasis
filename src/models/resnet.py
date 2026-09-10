from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def create_binary_resnet(dropout):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    # trava o treinamento da rede

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, 1)
    )
    # Essa linha substitui a ultima camada.
    # define que a rede neural irá acabar em 1 nós (possui 2 classes, porém o resultado virá em um só float)
    # Define o droupout ao ser treinado

    return model