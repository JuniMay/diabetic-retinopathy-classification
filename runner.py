import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from model import Net
from utils import load_data
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from rich.progress import Progress
from rich.progress import TextColumn, BarColumn
from rich.progress import TimeElapsedColumn, TimeRemainingColumn


class Runner:
    def __init__(self) -> None:
        self.config = Config()
        self.metadata = self.config.configurate()
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   and not self.metadata['no_cuda'] else 'cpu')
        self.mode = 'valid' if self.metadata['valid'] else 'train'
        self.input_size = self.metadata['input_size']
        self.batch_size = self.metadata['batch_size']
        self.num_classes = self.metadata['num_classes']
        self.num_workers = self.metadata['num_workers']
        self.num_epochs = self.metadata['num_epochs']

        self.net = Net(dim=self.metadata['dim'],
                       depth=self.metadata['depth'],
                       kernel_size=self.metadata['kernel_size'],
                       patch_size=self.metadata['patch_size'],
                       num_classes=self.num_classes,
                       in_channels=3,
                       drop=self.metadata['drop']).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.metadata['lr'])
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.metadata['label_smoothing']).to(self.device)
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.metadata['gamma'])

        if self.metadata['timestamp'] is None:
            self.timestamp = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
        else:
            self.timestamp = self.metadata['timestamp']

        self.log_dir = self.metadata['log_dir'] + '/' + self.timestamp

        self.writer = SummaryWriter(self.log_dir)

        with open(f'{self.log_dir}/metadata.json', 'w') as f:
            f.write(json.dumps(self.metadata))

        self.dataset_dir = self.metadata['dataset_dir']
        self.train_data_dir = self.metadata['train_data_dir']
        self.valid_data_dir = self.metadata['valid_data_dir']

        if self.train_data_dir is None:
            self.train_data_dir = f'{self.dataset_dir}/train'

        if self.valid_data_dir is None:
            self.valid_data_dir = f'{self.dataset_dir}/valid'

        self.train_data_loader = None
        self.valid_data_loader = None

        self.data_transforms = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45, 45)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def run(self) -> None:
        self.load_data()
        if self.mode == 'train':
            if self.metadata['method'] == 'classification':
                for epoch in range(self.num_epochs):
                    self.train_classification_epoch(epoch)
                    self.valid_classification_epoch(epoch)
                    self.save_checkpoint(epoch)
            else:
                pass
        else:
            pass

    def load_data(self) -> None:
        self.train_data_loader = load_data(
            self.train_data_dir,
            transform=self.data_transforms,
            target_transform=lambda x: torch.eye(self.num_classes)[x],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)
        self.valid_data_loader = load_data(
            self.valid_data_dir,
            transform=self.data_transforms,
            target_transform=lambda x: torch.eye(self.num_classes)[x],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def train_classification_epoch(self, epoch):
        self.writer.add_scalar(f'lr', self.scheduler.get_last_lr()[-1], epoch)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} train",
                                     total=len(self.train_data_loader))
            self.net.train()
            correct = 0
            accuracy = 0

            for i, (x, y) in enumerate(self.train_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                predict = self.net(x)
                loss = self.criterion(predict, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                correct += (predict.argmax(dim=1) == y.argmax(
                    dim=1)).float().sum()
                accuracy = correct / (i + 1) / self.batch_size

                self.writer.add_scalar(f'Accuracy/train', accuracy,
                                       epoch * len(self.train_data_loader) + i)
                self.writer.add_scalar(f'Loss/train', loss.item(),
                                       epoch * len(self.train_data_loader) + i)

                progress.advance(task)

            self.scheduler.step()

    def valid_classification_epoch(self, epoch):
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} valid",
                                     total=len(self.valid_data_loader))

            self.net.eval()
            correct = 0
            accuracy = 0
            with torch.no_grad():
                for i, (x, y) in enumerate(self.valid_data_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    predict = self.net(x)

                    loss = self.criterion(predict, y)

                    correct += (predict.argmax(dim=1) == y.argmax(
                        dim=1)).float().sum()
                    accuracy = correct / (i + 1) / self.batch_size

                    progress.advance(task)

                self.writer.add_scalar(f'Accuracy/valid', accuracy, epoch)
                self.writer.add_scalar(f'Loss/valid', loss.item(), epoch)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        checkpoint_path = f'{self.log_dir}/{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        max_epoch = -1
        for file in os.listdir(self.log_dir):
            if file.endswith('.pt'):
                max_epoch = max(max_epoch, int(file.rsplit('.', 1)[0]))

        if max_epoch == -1:
            raise RuntimeError('cannot find saved checkpoint')

        checkpoint = torch.load(f'{self.log_dir}/{max_epoch}.pt')
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
