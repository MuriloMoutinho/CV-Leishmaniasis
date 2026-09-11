import torch
from models import create_binary_resnet, create_binary_densenet, create_binary_efficientnet

def create_optimizer(optimizer_name, model, learning_rate, weight_decay):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return None

def create_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()

    return None

def create_binary_model(model_name, dropout=0):
    if model_name == "resnet50":
        return create_binary_resnet(dropout)
    elif model_name == "densenet121":
        return create_binary_densenet(dropout)
    elif model_name == "efficientnetb0":
        return create_binary_efficientnet(dropout)

    return None

def create_scheduler(scheduler_name, optimizer):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    return None