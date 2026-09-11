import torchvision

from augmentation import get_images_transformations
from config import TrainingConfig, HoldoutConfig
from data import save_holdout_result
from train import train_validate_holdout

full_dataset = torchvision.datasets.ImageFolder(root=r"../datasets/test/train")

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
#verificar linear scaling rule

holdout_config = HoldoutConfig(
    val_ratio=20,
    val_split_seed=42,
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

]

for config in configs:
    fold_results, fold_histories = train_validate_holdout(
        full_dataset,
        get_images_transformations(),
        config=config,
        holdout_config=holdout_config
    )

    save_holdout_result(
        model_result=fold_results,
        config=config,
        holdout_config=holdout_config,
        filename="experiments/holdout.csv"
    )

