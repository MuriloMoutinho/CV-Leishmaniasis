import gc
from dataclasses import replace
from pathlib import Path

import torch.cuda

from data import save_kfold_result, save_npz, load_dataset
from training import train_validate_kfold, KFoldConfig, TrainingConfig

kfold_config = KFoldConfig(
    n_splits=3,
    val_split_seed=42,
    patience_early_stopping=10,
    metric_to_monitor="mcc",
)

default_config = TrainingConfig(loss_name="cross_entropy", optimizer_name="adamw", scheduler_name=None, epochs=100,
                                batch_size=24, training_seed=42, fine_tuning="last_block", resolution=(576, 432),
                                augmentation_level="", model_name="", dropout=0, learning_rate=0, weight_decay=0)

configs = [
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.0007, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),

#("AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("AIR_LEISH",replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DeepLeish", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DLB", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),

#("CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_AIR_LEISH", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
#("CROSS_DeepLeish", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
#("CROSS_DLB", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+AIR", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="densenet121", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
("AIR+Deep", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+AIR", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.2, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="densenet121", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.2, learning_rate=0.001, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.5, learning_rate=0.0007, weight_decay=0.01)),
("DLB+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),
("AIR+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.2, learning_rate=0.001, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.5, learning_rate=0.0007, weight_decay=0.01)),
("DLB+Deep", replace(default_config, model_name="efficientnetb0", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=1e-06)),

("AIR+Deep", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="resnet50", augmentation_level="medium", dropout=0.4, learning_rate=0.002, weight_decay=0.0001)),
("AIR+Deep", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.003, weight_decay=0.0001)),
("DLB+AIR", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.1, learning_rate=0.002, weight_decay=1e-06)),
("DLB+Deep", replace(default_config, model_name="resnet50", augmentation_level="strong", dropout=0.4, learning_rate=0.002, weight_decay=0.0001)),
]

if __name__ == "__main__":
    for dataset_name, config in configs:
        fold_results = train_validate_kfold(
            load_dataset(dataset_name),
            config=config,
            kfold_config=kfold_config
        )
        torch.cuda.empty_cache()
        gc.collect()

        experiment_id = save_kfold_result(
            fold_results=fold_results,
            config=config,
            dataset_name=dataset_name,
            kfold_config=kfold_config,
            filename="experiments/loso2/pre-results-loso/kfold/kfold.csv"
        )
        save_npz(fold_results, f"experiments/loso2/pre-results-loso/kfold/labels-kfold/{experiment_id}", file_prefix="fold")

