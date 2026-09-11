from torchvision import models
import torch.nn as nn
from torchvision.models import DenseNet121_Weights

def create_binary_densenet(dropout):
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier.in_features, 1)
    )

    return model