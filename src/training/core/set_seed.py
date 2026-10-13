import random
import numpy
import torch

SEED = 42

def set_seed():
    random.seed(SEED)
    numpy.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)