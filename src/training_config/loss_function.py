import numpy
import torch
from torch.utils.data import ConcatDataset

def compute_pos_weight(dataset, indices):

    if isinstance(dataset, ConcatDataset):
        labels = numpy.concatenate([
            numpy.array(ds.targets)
            for ds in dataset.datasets
        ])
    else:
        labels = numpy.array(dataset.targets)

    labels = torch.tensor(labels[indices])

    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()

    return torch.tensor([n_neg / n_pos], dtype=torch.float32)