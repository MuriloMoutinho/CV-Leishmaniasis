from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from models.binary_model import BinaryModel


class BinaryDenseNet121(BinaryModel):

    def _build_backbone(self) -> nn.Module:
        return models.densenet121(weights=DenseNet121_Weights.DEFAULT)

    def _replace_classifier(self, dropout: float):
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.classifier.in_features, 1)
        )

    @property
    def classifier(self) -> nn.Module:
        return self.backbone.classifier

    @property
    def last_block(self) -> nn.Module:
        return self.backbone.features.denseblock4

    @property
    def second_last_block(self) -> nn.Module:
        return self.backbone.features.denseblock3
