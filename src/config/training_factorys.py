import torch

from augmentation import weak_augmentation, medium_augmentation, strong_augmentation
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

def create_binary_model(model_name, dropout=0, fine_tuning=None):
    if model_name == "resnet50":
        return create_binary_resnet(dropout, fine_tuning)
    elif model_name == "densenet121":
        return create_binary_densenet(dropout, fine_tuning)
    elif model_name == "efficientnetb0":
        return create_binary_efficientnet(dropout, fine_tuning)

    return None

def create_scheduler(scheduler_name, optimizer):
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    return None

def create_augmentation(augmentation_level_name):
    image_size = (512, 683)

    #if dataset_name.upper() == "DLB":
#        image_size = (512, 683) # mantem 4:3: 512x384 / 683x512 /768x576
#    elif dataset_name.upper() == "AIR":
#        image_size = (465, 683)
#    elif dataset_name.upper() == "DEEP":
#        image_size = (512, 683)

    if augmentation_level_name == "weak":
        return weak_augmentation(image_size)
    elif augmentation_level_name == "medium":
        return medium_augmentation(image_size)
    elif augmentation_level_name == "strong":
        return strong_augmentation(image_size)

    return None