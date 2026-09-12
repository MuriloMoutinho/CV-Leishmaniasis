import torch
from torchvision import models

from augmentation import weak_augmentation, medium_augmentation, strong_augmentation
from models import create_binary_resnet, create_binary_densenet, create_binary_efficientnet

def create_optimizer(optimizer_name, param_group, weight_decay):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_group, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_group, weight_decay=weight_decay)

    return None

def get_optimizer_param_groups(model, learning_rate, fine_tuning=None):
    layers = get_model_layers(model)
    param_groups = []

    if fine_tuning == "last_two_blocks":
        param_groups.append({ "params": layers["second_last_block"], "lr": learning_rate * 0.1 })

    if fine_tuning in ["last_block","last_two_blocks"]:
        param_groups.append({ "params": layers["last_block"], "lr": learning_rate * 0.3 })

    param_groups.append({ "params": layers["classifier"], "lr": learning_rate })

    return param_groups

def get_model_layers(model):
    if isinstance(model, models.ResNet):
        return {
            "classifier": list(model.fc.parameters()),
            "last_block": list(model.layer4.parameters()),
            "second_last_block": list(model.layer3.parameters()),
        }

    elif isinstance(model, models.EfficientNet):
        return {
            "classifier": list(model.classifier.parameters()),
            "last_block": list(model.features[-1].parameters()),
            "second_last_block": list(model.features[-2].parameters()),
        }

    elif isinstance(model, models.DenseNet):
        return {
            "classifier": list(model.classifier.parameters()),
            "last_block": list(model.features.denseblock4.parameters()),
            "second_last_block": list(model.features.denseblock3.parameters()),
        }

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
    if scheduler_name == "warmup+cosine":
        warmup_steps = 1000
        total_steps = 10000

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )


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