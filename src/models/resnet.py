from torchvision import models
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def create_binary_resnet(dropout, fine_tuning=None):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    if fine_tuning in ["last_block", "last_two_blocks"]:
        for param in model.layer4.parameters():
            param.requires_grad = True

    if fine_tuning == "last_two_blocks":
        for param in model.layer3.parameters():
            param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, 1)
    )
    return model