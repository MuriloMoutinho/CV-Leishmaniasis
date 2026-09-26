from torchvision import models


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

    raise TypeError("Modelo incorreto")
