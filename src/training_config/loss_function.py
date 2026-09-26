import torch


def compute_pos_weight(dataset, indices):
    labels = torch.tensor([dataset.samples[i][1] for i in indices])
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    return torch.tensor([n_neg / n_pos])