from torchvision import models
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights

def create_binary_efficientnet(dropout, fine_tuning=None):
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    if fine_tuning in ["last_block", "last_two_blocks"]:
        for param in model.features[-1].parameters():
            param.requires_grad = True

    if fine_tuning == "last_two_blocks":
        for param in model.features[-2].parameters():
            param.requires_grad = True

    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier[1].in_features, 1)
    )

    return model