from torch import nn

from models import BinaryModel


def get_optimizer_param_groups(model: BinaryModel, learning_rate, weight_decay):
    param_groups = []

    def add_block(modules, lr):
        decay, no_decay = split_decay_params(modules)
        if decay:
            param_groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

    if model.fine_tuning == "last_two_blocks":
        add_block(model.second_last_block, learning_rate * 0.1)

    if model.fine_tuning in ["last_block", "last_two_blocks"]:
        add_block(model.last_block, learning_rate * 0.3)

    add_block(model.classifier, learning_rate)

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
