from torchvision import models
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights

def create_binary_efficientnet(dropout):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier[1].in_features, 1)
    )

    return model