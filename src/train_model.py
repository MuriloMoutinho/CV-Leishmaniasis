import gc, torch
import torchvision

from config import TrainingConfig
from data import save_training_result
from train import train_validate_model

train_dataset = torchvision.datasets.ImageFolder(root=r'../datasets/test/train')
test_dataset = torchvision.datasets.ImageFolder(root=r'../datasets/test/test')

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
        epochs=37,
        scheduler_name="warmup+cosine",
        augmentation_level="strong",
    ),
]

if __name__ == "__main__":
    for config in configs:

        history = train_validate_model(
            train_dataset,
            test_dataset,
            config)
        torch.cuda.empty_cache()
        gc.collect()

        save_training_result(history, config, 'experiments/training.csv')


