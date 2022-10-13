import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from rich.progress import Progress
from rich.progress import TextColumn, BarColumn
from rich.progress import TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime

from model import Net


def one_hot(target):
    res = torch.eye(5)[target]
    return res


def load_data(data_dir,
              transform,
              target_transform,
              batch_size=64,
              shuffle=True):
    data = ImageFolder(root=data_dir,
                       transform=transform,
                       target_transform=target_transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def run(name,
        model,
        criterion,
        optimizer,
        scheduler,
        train_data_loader,
        test_data_loader,
        device,
        base_dir,
        batch_size=64,
        num_epochs=30,
        timestamp=None):

    if timestamp is None:
        timestamp = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())

    log_dir = f'{base_dir}/{name}/{timestamp}'
    writer = SummaryWriter(f'{log_dir}/')
    epoch = 0

    sample_input, _ = next(iter(train_data_loader))
    writer.add_graph(model, sample_input.to(device))

    def train():
        writer.add_scalar(f'lr', scheduler.get_last_lr()[-1], epoch)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} train",
                                     total=len(train_data_loader))
            model.train()
            correct = 0
            accuracy = 0

            for i, (x, y) in enumerate(train_data_loader):
                x = x.to(device)
                y = y.to(device)

                predict = model(x)
                loss = criterion(predict, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct += (predict.argmax(dim=1) == y.argmax(
                    dim=1)).float().sum()
                accuracy = correct / (i + 1) / batch_size

                writer.add_scalar(f'Accuracy/train', accuracy,
                                  epoch * len(train_data_loader) + i)
                writer.add_scalar(f'Loss/train', loss.item(),
                                  epoch * len(train_data_loader) + i)

                progress.advance(task)

            scheduler.step()

    def test():
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} test ",
                                     total=len(test_data_loader))

            model.eval()
            correct = 0
            accuracy = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(test_data_loader):
                    x = x.to(device)
                    y = y.to(device)
                    predict = model(x)

                    loss = criterion(predict, y)

                    correct += (predict.argmax(dim=1) == y.argmax(
                        dim=1)).float().sum()
                    accuracy = correct / (i + 1) / batch_size

                    progress.advance(task)

                writer.add_scalar(f'Accuracy/test', accuracy, epoch)
                writer.add_scalar(f'Loss/test', loss.item(), epoch)

    def save():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        checkpoint_path = f'{log_dir}/{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

    def load():
        nonlocal epoch
        max_epoch = -1
        for file in os.listdir(log_dir):
            if file.endswith('.pt'):
                max_epoch = max(max_epoch, int(file.rsplit('.', 1)[0]))

        if max_epoch == -1:
            return

        checkpoint = torch.load(f'{log_dir}/{max_epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint['epoch'] + 1

    load()
    while epoch < num_epochs:
        train()
        test()
        save()
        epoch += 1


def main():

    train_data_dir = 'data/DDR/train/'
    test_data_dir = 'data/DDR/test'
    valid_data_dir = 'data/DDR/valid'

    batch_size = 32
    num_epochs = 30
    base_dir = 'log'
    learning_rate = 0.001
    label_smoothing = 0.2

    data_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_data_loader = load_data(train_data_dir, data_transforms, one_hot,
                                  batch_size, True)
    test_data_loader = load_data(test_data_dir, data_transforms, one_hot,
                                 batch_size, True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(dim=128,
                depth=8,
                kernel_size=9,
                patch_size=7,
                num_classes=5,
                in_channels=3,
                dropout_rate=0.25).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    run('Net', model, criterion, optimizer, scheduler, train_data_loader,
        test_data_loader, device, base_dir, batch_size, num_epochs, None)


if __name__ == '__main__':
    main()