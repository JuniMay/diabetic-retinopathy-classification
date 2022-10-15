import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datetime import datetime

from model import Net

from utils import run, onehot_fn, load_data

def main():
    train_data_dir = 'data/imagenette2-320/train/'
    test_data_dir = 'data/imagenette2-320/val'

    batch_size = 128
    num_epochs = 50
    base_dir = 'tf-logs'
    learning_rate = 0.001
    label_smoothing = 0.3
    gamma = 0.9

    dim = 256
    depth = 8
    kernel_size = 9
    patch_size = 8
    drop = 0.2

    timestamp = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())

    data_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data_loader = load_data(train_data_dir, data_transforms, onehot_fn(10),
                                  batch_size, True)
    test_data_loader = load_data(test_data_dir, data_transforms, onehot_fn(10),
                                 batch_size, True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(dim=dim,
                depth=depth,
                kernel_size=kernel_size,
                patch_size=patch_size,
                num_classes=10,
                in_channels=3,
                drop=drop,).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    metadata = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'base_dir': base_dir,
        'learning_rate': learning_rate,
        'label_smoothing': label_smoothing,
        'gamma': gamma,
        'dim': dim,
        'depth': depth,
        'kernel_size': kernel_size,
        'patch_size': patch_size,
        'drop': drop
    }

    run('Imagenette',
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        device=device,
        base_dir=base_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        timestamp=timestamp,
        metadata=metadata)


if __name__ == '__main__':
    main()