from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from models.binary_model import BinaryModel


class BinaryResNet50(BinaryModel):

    def _build_backbone(self) -> nn.Module:
        return models.resnet50(weights=ResNet50_Weights.DEFAULT)

    def _replace_classifier(self, dropout: float):
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features, 1)
        )

    @property
    def classifier(self) -> nn.Module:
        return self.backbone.fc

    @property
    def last_block(self) -> nn.Module:
        return self.backbone.layer4

    @property
    def second_last_block(self) -> nn.Module:
        return self.backbone.layer3
