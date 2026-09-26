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
    fine_tuning: str | None = None
    augmentation_level: str = 'medium'
    scheduler_name: str | None = None

@dataclass
class KFoldConfig:
    n_splits: int = 5
    val_split_seed: int = 42
    patience_early_stopping: int = 10
    metric_to_monitor: str = "f1"

@dataclass
class HoldoutConfig:
    val_ratio: int = 20
    val_split_seed: int = 42
    patience_early_stopping: int = 10
    metric_to_monitor: str = "f1"