import torchvision

from config import TrainingConfig
from data import create_loss_history_graph
from augmentation import get_images_transformations
from train import train_complete_model

transformations = get_images_transformations()
dataset = torchvision.datasets.ImageFolder(root=r'../datasets/test/train', transform=transformations['train'])

config = TrainingConfig(
    model_name="resnet50",
    dropout=0.0,
    loss_name="cross_entropy",
    optimizer_name="adamw",
    learning_rate=0.001,
    weight_decay=0.01,
    batch_size=32,
    epochs=50,
)

model_history = train_complete_model(
    dataset,
    config,
    'experiments/melhor_modelo.pt')


create_loss_history_graph(model_history, 'experiments/melhor_modelo_train')
