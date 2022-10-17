from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(data_dir,
              transform,
              target_transform,
              batch_size=64,
              num_workers=5,
              shuffle=True):
    data = ImageFolder(root=data_dir,
                       transform=transform,
                       target_transform=target_transform)
    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)
    return loader