import torch, time
from torch.utils.data import DataLoader

from config import TrainingConfig, create_binary_model, create_optimizer, create_loss_function
from train import train_one_epoch

def train_complete_model(
    dataset,
    config: TrainingConfig,
    filename='model'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = create_binary_model(config.model_name)
    optimizer = create_optimizer(config.optimizer_name, model, config.learning_rate, config.weight_decay)
    loss_function = create_loss_function(config.loss_name)

    history = []
    start = time.time()

    for epoch in range(config.epochs):

        start_epoch = time.time()
        print(f"\nÉpoca {epoch + 1}/{config.epochs}")

        train_metrics = train_one_epoch(train_loader, model, loss_function, optimizer, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics['loss'],
            "train_accuracy": train_metrics['accuracy'],
        })

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        end_epoch = time.time()
        print(f"Tempo época: {end_epoch - start_epoch:.2f}s")

    torch.save(model, filename)
    end = time.time()
    print(f"Tempo final: {end - start:.2f}s")

    # model.load_state_dict(torch.load("results/melhor_modelo.pt"))

    return history