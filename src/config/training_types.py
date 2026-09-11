from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str
    dropout: float
    loss_name: str
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    scheduler_name: str | None = None

@dataclass
class KFoldConfig:
    n_splits: int = 5
    patience_early_stopping: int = 10
    metric_to_monitor: str = "f1"