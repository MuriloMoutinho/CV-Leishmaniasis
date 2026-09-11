import torchvision

from augmentation import get_images_transformations
from config import KFoldConfig, TrainingConfig
from data import save_experiment_result
from train import train_validate_kfold

full_dataset = torchvision.datasets.ImageFolder(root=r"../datasets/test/train")

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
#verificar linear scaling rule

kfold_config = KFoldConfig(
    n_splits=5,
    patience_early_stopping=10,
    metric_to_monitor="f1",
)

configs = [
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=16,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.01,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.006,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0.15,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0.25,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0.50,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.005,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.0001,
        weight_decay=0.005,
        batch_size=32,
        epochs=50,
    ),
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        scheduler_name="cosine",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=32,
        epochs=50,
    ),
]

for config in configs:
    fold_results, fold_histories = train_validate_kfold(
        full_dataset,
        get_images_transformations(),
        config=config,
        kfold_config=kfold_config
    )

    save_experiment_result(
        fold_results=fold_results,
        config=config,
        kfold_config=kfold_config,
        filename="experiments/experimentos.csv"
    )

