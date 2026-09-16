import torch
from torch import nn
from torchvision import models

from augmentation import weak_augmentation, medium_augmentation, strong_augmentation
from models import create_binary_resnet, create_binary_densenet, create_binary_efficientnet

def create_optimizer(optimizer_name, param_group):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_group)
    elif optimizer_name == "adam":
        return torch.optim.Adam(param_group)

    return None

def get_optimizer_param_groups(model, learning_rate, weight_decay, fine_tuning=None):
    layers = get_model_layers(model)
    param_groups = []

    def add_block(modules, lr):
        decay, no_decay = split_decay_params(modules)
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    if fine_tuning == "last_two_blocks":
        add_block(layers["second_last_block"], learning_rate * 0.1)

    if fine_tuning in ["last_block", "last_two_blocks"]:
        add_block(layers["last_block"], learning_rate * 0.3)

    add_block(layers["classifier"], learning_rate)

    return param_groups

def split_decay_params(module):
    decay_params = []
    no_decay_params = []

    for submodule in module.modules():
        for name, param in submodule.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            if isinstance(submodule, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                no_decay_params.append(param)
            elif name == "bias":
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return decay_params, no_decay_params


def get_model_layers(model):
    if isinstance(model, models.ResNet):
        return {
            "classifier": model.fc,
            "last_block": model.layer4,
            "second_last_block": model.layer3,
        }

    elif isinstance(model, models.EfficientNet):
        return {
            "classifier": model.classifier,
            "last_block": model.features[-1],
            "second_last_block": model.features[-2],
        }

    elif isinstance(model, models.DenseNet):
        return {
            "classifier": model.classifier,
            "last_block": model.features.denseblock4,
            "second_last_block": model.features.denseblock3,
        }

    return None

def compute_pos_weight(dataset, indices):
    labels = torch.tensor([dataset.samples[i][1] for i in indices])
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    return torch.tensor([n_neg / n_pos])

def create_loss_function(loss_name, pos_weight):
    if loss_name == "cross_entropy":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return None

def create_binary_model(model_name, dropout=0, fine_tuning=None):
    if model_name == "resnet50":
        return create_binary_resnet(dropout, fine_tuning)
    elif model_name == "densenet121":
        return create_binary_densenet(dropout, fine_tuning)
    elif model_name == "efficientnetb0":
        return create_binary_efficientnet(dropout, fine_tuning)

    return None

def create_scheduler(scheduler_name, optimizer, steps_per_epoch, epochs):
    if scheduler_name == "warmup+cosine":
        warmup_ratio = 0.1

        total_steps = steps_per_epoch * epochs

        warmup_steps = int(total_steps * warmup_ratio)
        warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, total_steps - 1)

        cosine_steps = total_steps - warmup_steps

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0,)

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    return None

def create_augmentation(augmentation_level_name):
    image_size = (576, 768)

    #224, 299
    #384, 512	4x	~5-7,5 px
    #576, 768	2,67x	~7,5-11,3 px
    #768, 1024	2x	~10-15 px
    #960, 1280	1,6x	~12,5-18,75 px
    #1152, 1536	1,33x	~15-22,5 px

    if augmentation_level_name == "weak":
        return weak_augmentation(image_size)
    elif augmentation_level_name == "medium":
        return medium_augmentation(image_size)
    elif augmentation_level_name == "strong":
        return strong_augmentation(image_size)

    return None