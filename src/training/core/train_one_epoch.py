import torch
from sklearn.metrics import accuracy_score
from torch import nn

def train_one_epoch(data_loader, model, loss_function, optimizer, device, scheduler=None):
    model.train()
    model = model.to(device)
    freeze_bn_in_frozen_layers(model)

    total_loss = 0.0
    n_seen = 0

    all_labels = []
    all_predictions = []

    for images_batch, labels in data_loader:

        images_batch = images_batch.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images_batch).squeeze(1)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images_batch.size(0)
        total_loss += loss.item() * batch_size
        n_seen += batch_size

        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy().ravel())
        all_predictions.extend(predictions.detach().cpu().numpy().ravel())

    average_loss = total_loss / n_seen
    accuracy = accuracy_score(all_labels, all_predictions)

    return {
        "loss": average_loss,
        "accuracy": accuracy
    }

def freeze_bn_in_frozen_layers(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()