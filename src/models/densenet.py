from torchvision import models
import torch.nn as nn
from torchvision.models import DenseNet121_Weights

def create_binary_densenet(dropout, fine_tuning=None):
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    if fine_tuning not in (None, "last_block", "last_two_blocks"):
            raise ValueError("Fine_tuning inválido")

    if fine_tuning in ["last_block", "last_two_blocks"]:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

    if fine_tuning == "last_two_blocks":
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier.in_features, 1)
    )

    return model