import gc, torch
import torchvision

from config import TrainingConfig, HoldoutConfig
from data import save_holdout_result
from train import train_validate_holdout

full_dataset = torchvision.datasets.ImageFolder(root=r"../datasets/test/train")

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
#verificar linear scaling rule

holdout_config = HoldoutConfig(
    val_ratio=25,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="f1",
)

configs = [
    TrainingConfig(
        model_name="resnet50",
        fine_tuning="last_two_blocks",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.0003,
        weight_decay=0.01,
        batch_size=16,
        epochs=50,
        scheduler_name="warmup+cosine",
        augmentation_level="strong",
    ),
]

if __name__ == "__main__":
    for config in configs:
        fold_results, fold_histories = train_validate_holdout(
            full_dataset,
            config=config,
            holdout_config=holdout_config
        )
        torch.cuda.empty_cache()
        gc.collect()

        save_holdout_result(
            model_result=fold_results,
            config=config,
            holdout_config=holdout_config,
            filename="experiments/holdout.csv"
        )

