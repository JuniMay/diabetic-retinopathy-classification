import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from config import Config
from model import Net, PatchEmbedding
from utils import load_data, set_seed
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from rich.progress import Progress
from rich.progress import TextColumn, BarColumn
from rich.progress import TimeElapsedColumn, TimeRemainingColumn
from torchmetrics.classification import CohenKappa
from torchmetrics.classification import ConfusionMatrix


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
        self.dim = self.metadata['dim']
        self.pretrained = self.metadata['pretrained']

        self.best_acc = 0

        set_seed()

        if self.metadata['only_fc']:
            if self.metadata['backbone'] == 'densenet121':
                net = torchvision.models.densenet121(
                    pretrained=self.pretrained)
                net.classifier = nn.Linear(1024, self.num_classes)
                self.model = net.to(self.device)

            elif self.metadata['backbone'] == 'efficientnet':
                net = torchvision.models.efficientnet_b4(
                    pretrained=self.pretrained)
                net.classifier[1] = nn.Linear(1792, self.num_classes)
                self.model = net.to(self.device)

            elif self.metadata['backbone'] == 'resnet101':
                net = torchvision.models.resnet101(pretrained=self.pretrained)
                net.fc = nn.Linear(2048, self.num_classes)
                self.model = net.to(self.device)

            elif self.metadata['backbone'] == 'resnet152':
                net = torchvision.models.resnet152(pretrained=self.pretrained)
                net.fc = nn.Linear(2048, self.num_classes)
                self.model = net.to(self.device)

            elif self.metadata['backbone'] == 'convnext_base':
                net = torchvision.models.convnext_base(
                    pretrained=self.pretrained)
                net.classifier[2] = nn.Linear(1024, self.num_classes)
                self.model = net.to(self.device)

            else:
                raise RuntimeError("Backbone is not supported for fc-only net")
        else:
            if self.metadata['backbone'] == 'densenet121':
                net = torchvision.models.densenet121(
                    pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-1]))  # 1024
                self.dim = 1024
            elif self.metadata['backbone'] == 'densenet161':
                net = torchvision.models.densenet161(
                    pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-1]))  # 2208
                self.dim = 2208
            elif self.metadata['backbone'] == 'efficientnet':
                net = torchvision.models.efficientnet_b4(
                    pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-2]))  # 1792
                self.dim = 1792
            elif self.metadata['backbone'] == 'resnet101':
                net = torchvision.models.resnet101(pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-2]))  # 2048
                self.dim = 2048
            elif self.metadata['backbone'] == 'resnet152':
                net = torchvision.models.resnet152(pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-2]))  # 2048
                self.dim = 2048
            elif self.metadata['backbone'] == 'convnext_base':
                net = torchvision.models.convnext_base(
                    pretrained=self.pretrained)
                self.backbone = nn.Sequential(*(list(
                    net.children())[:-2]))  # 1024
                self.dim = 1024
            elif self.metadata['backbone'] == 'patch':
                self.backbone = PatchEmbedding(self.dim,
                                               self.metadata['patch_size'], 3)
            else:
                raise RuntimeError("Backbone is not supported")

            if self.metadata['freeze_backbone']:
                self.backbone.requires_grad_(False)

            self.model = nn.Sequential(
                self.backbone,
                Net(dim=self.dim,
                    depth=self.metadata['depth'],
                    kernel_size=self.metadata['kernel_size'],
                    num_classes=self.num_classes)).to(self.device)

        if self.metadata['backbone_lr'] is not None and not self.metadata[
                'freeze_backbone'] and not self.metadata['only_fc']:
            self.optimizer = optim.AdamW([{
                'params': self.model[0].parameters(),
                'lr': self.metadata['backbone_lr']
            }, {
                'params': self.model[1].parameters()
            }],
                                         lr=self.metadata['lr'])
        else:
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                                self.model.parameters()),
                                         lr=self.metadata['lr'])

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.metadata['label_smoothing']).to(self.device)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                          gamma=0.95)

        if self.metadata['timestamp'] is None:
            self.timestamp = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
        else:
            self.timestamp = self.metadata['timestamp']

        self.log_dir = self.metadata['log_dir'] + '/' + self.timestamp

        if self.mode == 'valid':
            if self.metadata['timestamp'] is None:
                raise RuntimeError(
                    'timestamp must be indicated for valid mode')
            curr_timestamp = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
            self.log_dir = f'{self.log_dir}-valid-{curr_timestamp}'

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

        self.train_transforms = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-45, 45)),
            transforms.TrivialAugmentWide(),
            transforms.ColorJitter(brightness=0.2, hue=0.1, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(0.2)
        ])
        self.valid_transforms = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, x) -> torch.Tensor:
        return self.model(x)

    def run(self) -> None:
        self.load_data()
        if self.mode == 'train':
            self.best_acc = 0
            for epoch in range(self.num_epochs):
                self.train_epoch(epoch)
                self.valid_epoch(epoch)
                self.save_checkpoint(epoch)

        elif self.mode == 'valid':
            self.load_checkpoint(self.metadata['log_dir'] + '/' +
                                 self.metadata['timestamp'])
            acc, loss, kappa = self.valid_epoch(0)
            for i in range(self.num_epochs):
                self.writer.add_scalar('Accuracy/valid', acc, i)
                self.writer.add_scalar('Loss/valid', loss, i)
                self.writer.add_scalar('Kappa/valid', kappa, i)

    def load_data(self) -> None:
        print(f'using {self.train_data_dir} as dataset for training.')
        print(
            f'using {self.valid_data_dir} as dataset for validating(testing).')
        self.train_data_loader = load_data(
            self.train_data_dir,
            transform=self.train_transforms,
            target_transform=lambda x: torch.eye(self.num_classes)[x],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)
        self.valid_data_loader = load_data(
            self.valid_data_dir,
            transform=self.valid_transforms,
            target_transform=lambda x: torch.eye(self.num_classes)[x],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

    def train_epoch(self, epoch):
        self.writer.add_graph(
            self.model,
            next(iter(self.train_data_loader))[0].to(self.device))
        self.writer.add_scalar(f'lr', self.scheduler.get_last_lr()[-1], epoch)

        if len(self.scheduler.get_last_lr()) > 1:
            self.writer.add_scalar(f'lr/backbone_lr',
                                   self.scheduler.get_last_lr()[-2], epoch)

        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} train",
                                     total=len(self.train_data_loader))

            self.model.train()
            correct = 0
            accuracy = 0
            target = torch.empty(0, dtype=torch.int)
            preds = torch.empty(0, dtype=torch.int)
            kappa_calc = CohenKappa(num_classes=self.num_classes)

            for i, (x, y) in enumerate(self.train_data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                predict = self.predict(x)
                loss = self.criterion(predict, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                local_correct = (predict.argmax(dim=1) == y.argmax(
                    dim=1)).float().sum()
                accuracy = local_correct / len(x)
                correct += local_correct
                target = torch.concat([target, y.argmax(dim=1).cpu()])
                preds = torch.concat([preds, predict.argmax(dim=1).cpu()])
                kappa = kappa_calc(
                    predict.argmax(dim=1).cpu(),
                    y.argmax(dim=1).cpu())

                self.writer.add_scalar(f'Accuracy/train', accuracy,
                                       epoch * len(self.train_data_loader) + i)
                self.writer.add_scalar(f'Loss/train', loss.item(),
                                       epoch * len(self.train_data_loader) + i)
                self.writer.add_scalar(f'Kappa/train', kappa,
                                       epoch * len(self.train_data_loader) + i)

                progress.advance(task)

            accuracy = correct / len(self.train_data_loader.dataset)
            kappa = kappa_calc(preds, target)
            self.writer.add_scalar(f'Accuracy/global', accuracy, epoch)
            self.writer.add_scalar(f'Kappa/global', kappa, epoch)

            self.scheduler.step()

    def valid_epoch(self, epoch):
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            task = progress.add_task(description=f"Epoch {epoch:>3} valid",
                                     total=len(self.valid_data_loader))

            self.model.eval()
            correct = 0
            accuracy = 0
            target = torch.empty(0, dtype=torch.int)
            preds = torch.empty(0, dtype=torch.int)
            with torch.no_grad():
                for i, (x, y) in enumerate(self.valid_data_loader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    predict = self.predict(x)

                    loss = self.criterion(predict, y)

                    correct += (predict.argmax(dim=1) == y.argmax(
                        dim=1)).float().sum()
                    target = torch.concat([target, y.argmax(dim=1).cpu()])
                    preds = torch.concat([preds, predict.argmax(dim=1).cpu()])

                    progress.advance(task)

                accuracy = correct / len(self.valid_data_loader.dataset)

                kappa = CohenKappa(num_classes=self.num_classes)(preds, target)

                confmat = ConfusionMatrix(num_classes=self.num_classes)(preds, target)

                fig = plt.figure(figsize=(9, 9))
                sns.heatmap(confmat, )
                plt.savefig(f'{self.log_dir}/confusion_matrix.pdf')
                plt.close(fig)

                if self.mode == 'train':
                    self.writer.add_scalar(f'Accuracy/valid', accuracy, epoch)
                    self.writer.add_scalar(f'Loss/valid', loss.item(), epoch)
                    self.writer.add_scalar(f'Kappa/valid', kappa, epoch)

                    if accuracy > self.best_acc:
                        torch.save(self.model.state_dict(),
                                f'{self.log_dir}/best_acc.pt')
                        self.best_acc = accuracy

        return accuracy, loss.item(), kappa

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        if os.path.exists(f'{self.log_dir}/{epoch - 5}.pt'):
            os.remove(f'{self.log_dir}/{epoch - 5}.pt')
        checkpoint_path = f'{self.log_dir}/{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, directory):
        max_epoch = -1
        for file in os.listdir(directory):
            if file.endswith('.pt'):
                if not file.rsplit('.', 1)[0].isnumeric():
                    continue
                max_epoch = max(max_epoch, int(file.rsplit('.', 1)[0]))

        if max_epoch == -1:
            raise RuntimeError('cannot find saved checkpoint')

        if os.path.exists(f'{directory}/best_acc.pt'):
            max_epoch = 'best_acc'

            checkpoint = torch.load(f'{directory}/{max_epoch}.pt')
            self.model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load(f'{directory}/{max_epoch}.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
