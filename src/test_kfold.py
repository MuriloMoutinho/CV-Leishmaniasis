import torchvision

from augmentation import get_images_transformations
from config import KFoldConfig, TrainingConfig
from data import save_kfold_result
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
    # ==========================================
    # 2. LEARNING RATE
    # ==========================================
    TrainingConfig(
        "resnet50", 0.0, "cross_entropy",
        "adamw", 0.004, 0.01, 16, 50
    ),
    TrainingConfig(
        "resnet50", 0.0, "cross_entropy",
        "adamw", 0.006, 0.01, 16, 50
    ),
    # ==========================================
    # 3. WEIGHT DECAY
    # ==========================================
    TrainingConfig(
        "resnet50", 0.0, "cross_entropy",
        "adamw", 0.005, 0.001, 16, 50
    ),
    TrainingConfig(
        "resnet50", 0.0, "cross_entropy",
        "adamw", 0.005, 0.005, 16, 50
    ),
    TrainingConfig(
        "resnet50", 0.0, "cross_entropy",
        "adamw", 0.005, 0.02, 16, 50
    ),
    # ==========================================
    # 4. DROPOUT
    # ==========================================
    TrainingConfig(
        "resnet50", 0.25, "cross_entropy",
        "adamw", 0.005, 0.01, 16, 50
    ),
    TrainingConfig(
        "resnet50", 0.5, "cross_entropy",
        "adamw", 0.005, 0.01, 16, 50
    ),
]

for config in configs:
    fold_results, fold_histories = train_validate_kfold(
        full_dataset,
        get_images_transformations(),
        config=config,
        kfold_config=kfold_config
    )

    save_kfold_result(
        fold_results=fold_results,
        config=config,
        kfold_config=kfold_config,
        filename="experiments/kfold.csv"
    )

