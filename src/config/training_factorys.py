import torch
from models import create_binary_resnet

def create_optimizer(optimizer_name, model, learning_rate, weight_decay):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return None

def create_loss_function(loss_name):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()

    return None

def create_binary_model(model_name, dropout=0):
    if model_name == "resnet50":
        return create_binary_resnet(dropout)

    return None