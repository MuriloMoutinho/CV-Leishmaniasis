import torch, time, os, torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models import create_binary_model
from augmentation import get_images_transformations
from src import hyperparameters
from train import train_one_epoch

def get_transformed_dataset(image_transforms):
    dataset_path = r'../datasets/test/'
    folder_train = os.path.join(dataset_path, 'train')

    return torchvision.datasets.ImageFolder(root=folder_train, transform=image_transforms['train'])

def train(
    model,
    train_loader,
    loss_function,
    optimizer,
    epoch_num=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    history = []

    for epoch in range(epoch_num):

        start = time.time()
        print(f"\nÉpoca {epoch + 1}/{epoch_num}")

        train_metrics = train_one_epoch(model, train_loader, loss_function, optimizer, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics['loss'],
            "train_accuracy": train_metrics['accuracy'],
        })

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        end = time.time()
        print(f"Tempo: {end - start:.2f}s")

    torch.save(model, 'results/melhor_modelo.pt')
    # model.load_state_dict(torch.load("results/melhor_modelo.pt"))

    return history

model_train = create_binary_model()
loss_function = hyperparameters.create_loss_function()
optimizer = hyperparameters.create_optimizer(model_train)

images = get_transformed_dataset(get_images_transformations())
train_loader = DataLoader(images, batch_size=hyperparameters.batch_size, shuffle=True)

model_history = train(model_train, train_loader, loss_function, optimizer, 15) #hyperparameters.epoch_num

def create_loss_history_graph(history):
    epochs = range(1, len(history) + 1)

    train_loss = [h["train_loss"] for h in history]
    train_accuracy = [h["train_accuracy"] for h in history]

    plt.plot(epochs, train_loss, label="Loss treino")
    plt.plot(epochs, train_accuracy, label="Acuracia treino")

    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.legend()
    plt.show()

create_loss_history_graph(model_history)
