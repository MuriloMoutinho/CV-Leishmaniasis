import gc

import torch.cuda
import torchvision

from config import KFoldConfig, TrainingConfig
from data import save_kfold_result
from train import train_validate_kfold

full_dataset = torchvision.datasets.ImageFolder(root=r"../datasets/test/train")

#estimativa do gradiente mais "ruidosa" para batchs menores. Batchs maiores tras uma média melhor, mas pode piorar generalização
#verificar linear scaling rule

kfold_config = KFoldConfig(
    n_splits=5,
    val_split_seed=42,
    patience_early_stopping=8,
    metric_to_monitor="f1",
)

configs = [
    TrainingConfig(model_name="resnet50", fine_tuning="last_block", dropout=0,
        loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.003,
        weight_decay=0.01, batch_size=16, epochs=30, scheduler_name="warmup+cosine",
        augmentation_level="weak"),
    # ============ DENSENET121 ============
    TrainingConfig(model_name="densenet121", fine_tuning="last_block", dropout=0,
        loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
        weight_decay=0.01, batch_size=16, epochs=30, scheduler_name="warmup+cosine",
        augmentation_level="weak"),
    # ============ EFFICIENTNETB0 ============
    TrainingConfig(model_name="efficientnetb0", fine_tuning="last_two_blocks", dropout=0,
        loss_name="cross_entropy", optimizer_name="adamw", learning_rate=0.005,
        weight_decay=0.01, batch_size=16, epochs=30, scheduler_name="warmup+cosine",
        augmentation_level="weak"),
]

if __name__ == "__main__":
    for config in configs:
        fold_results, fold_histories = train_validate_kfold(
            full_dataset,
            config=config,
            kfold_config=kfold_config
        )
        torch.cuda.empty_cache()
        gc.collect()

        save_kfold_result(
            fold_results=fold_results,
            config=config,
            kfold_config=kfold_config,
            filename="experiments/kfold.csv"
        )

