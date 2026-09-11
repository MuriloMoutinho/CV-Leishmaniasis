import torchvision

from config import TrainingConfig
from data import save_training_result
from augmentation import get_images_transformations
from train import train_validate_model

train_dataset = torchvision.datasets.ImageFolder(root=r'../datasets/test/train')
test_dataset = torchvision.datasets.ImageFolder(root=r'../datasets/test/test')

configs = [
    TrainingConfig(
        model_name="resnet50",
        dropout=0,
        loss_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.005,
        weight_decay=0.01,
        batch_size=32,
        epochs=10,
    ),
]

for config in configs:

    history = train_validate_model(
        train_dataset,
        test_dataset,
        get_images_transformations(),
        config,
        'experiments/melhor_modelo.pt')

    save_training_result(history, config, 'experiments/treino.csv')


