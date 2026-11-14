from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from models.binary_model import BinaryModel


class BinaryEfficientNetB0(BinaryModel):

    def _build_backbone(self) -> nn.Module:
        return models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    def _replace_classifier(self, dropout: float):
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.classifier[1].in_features, 1)
        )

    @property
    def classifier(self) -> nn.Module:
        return self.backbone.classifier

    @property
    def last_block(self) -> nn.Module:
        return self.backbone.features[-1]

    @property
    def second_last_block(self) -> nn.Module:
        return self.backbone.features[-2]
