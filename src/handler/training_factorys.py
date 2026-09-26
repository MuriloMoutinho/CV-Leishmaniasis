import torch

from augmentation import weak_augmentation, medium_augmentation, strong_augmentation
from training_config.scheduler import get_warmup_cosine
from models import create_binary_resnet, create_binary_densenet, create_binary_efficientnet

def create_optimizer(optimizer_name, param_group):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_group)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_group)

    raise TypeError("Otimizador incorreto")

def create_loss_function(loss_name, pos_weight):
    if loss_name == "cross_entropy":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise TypeError("Loss function incorreta")

def create_binary_model(model_name, dropout=0, fine_tuning=None):
    if model_name == "resnet50":
        return create_binary_resnet(dropout, fine_tuning)
    elif model_name == "densenet121":
        return create_binary_densenet(dropout, fine_tuning)
    elif model_name == "efficientnetb0":
        return create_binary_efficientnet(dropout, fine_tuning)

    raise TypeError("Modelo incorreto")

def create_scheduler(scheduler_name, optimizer, steps_per_epoch, epochs):
    if scheduler_name == "warmup+cosine":
        return get_warmup_cosine(optimizer, steps_per_epoch, epochs)

    raise TypeError("Scheduler incorreto")

def create_augmentation(augmentation_level_name):
    image_size = (576, 768)
    pad = False

    if augmentation_level_name == "weak":
        return weak_augmentation(image_size, pad)
    elif augmentation_level_name == "medium":
        return medium_augmentation(image_size, pad)
    elif augmentation_level_name == "strong":
        return strong_augmentation(image_size, pad)

    raise TypeError("Augmentation incorreto")