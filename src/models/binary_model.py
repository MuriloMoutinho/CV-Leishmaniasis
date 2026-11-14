from abc import ABC, abstractmethod

from torch import nn


class BinaryModel(nn.Module, ABC):
    FINE_TUNING_OPTIONS = (None, "last_block", "last_two_blocks")

    def __init__(self, dropout: float = 0.0, fine_tuning: str | None = None):
        super().__init__()

        if fine_tuning not in self.FINE_TUNING_OPTIONS:
            raise ValueError(f"fine_tuning inválido {fine_tuning}. Opções: {self.FINE_TUNING_OPTIONS}")

        self.fine_tuning = fine_tuning
        self.backbone = self._build_backbone()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for block in self.fine_tuned_blocks():
            for param in block.parameters():
                param.requires_grad = True

        self._replace_classifier(dropout)

    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        ...

    @abstractmethod
    def _replace_classifier(self, dropout: float):
        ...

    @property
    @abstractmethod
    def classifier(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def last_block(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def second_last_block(self) -> nn.Module:
        ...

    def fine_tuned_blocks(self) -> list[nn.Module]:
        if self.fine_tuning == "last_block":
            return [self.last_block]
        if self.fine_tuning == "last_two_blocks":
            return [self.last_block, self.second_last_block]
        return []

    def forward(self, x):
        return self.backbone(x)

    def state_dict(self, *args, **kwargs):
        return self.backbone.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.backbone.load_state_dict(state_dict, strict=strict, assign=assign)
