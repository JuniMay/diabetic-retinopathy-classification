from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import numpy as np
import torch
import random
import os


def set_seed():
    seed = 49297 # just a prime number
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.derterministic = True


def load_data(data_dir,
              transform,
              target_transform,
              batch_size=64,
              num_workers=5,
              shuffle=True):
    set_seed()
    data = ImageFolder(root=data_dir,
                       transform=transform,
                       target_transform=target_transform)
    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)
    return loader