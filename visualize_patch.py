from model import Net, PatchEmbedding, CbamBlock
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision
from torchvision.transforms import InterpolationMode

metadata = {
    "timestamp": "patch-model",
    "valid": False,
    "num_classes": 5,
    "num_workers": 12,
    "num_epochs": 80,
    "batch_size": 128,
    "dataset_dir": "./data/DDR",
    "train_data_dir": None,
    "valid_data_dir": "data/DDR/valid",
    "log_dir": "tf-logs",
    "no_cuda": False,
    "lr": 0.001,
    "label_smoothing": 0.3,
    "input_size": 480,
    "dim": 384,
    "depth": 8,
    "kernel_size": 9,
    "patch_size": 8,
    "backbone": "patch",
    "pretrained": False,
    "only_fc": False,
    "freeze_backbone": False,
    "backbone_lr": None
}

# img_path = 'data/DDR/train/4/007-6942-400.jpg'
img_path = 'data/DDR/train/4/007-6921-400.jpg'

device = torch.device(
    'cuda' if torch.cuda.is_available() and not metadata['no_cuda'] else 'cpu')
backbone = PatchEmbedding(metadata['dim'], metadata['patch_size'], 3)
model = nn.Sequential(
    backbone,
    Net(dim=metadata['dim'],
        depth=metadata['depth'],
        kernel_size=metadata['kernel_size'],
        num_classes=metadata['num_classes'])).to(device)
model.eval()

model.load_state_dict(torch.load(f'log/tf-logs/patch-model/best_acc.pt'))

trans = transforms.Compose([
    transforms.Resize(metadata['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

x = trans(Image.open(img_path)).unsqueeze(0).to(device)

y = model[0](x)
w = model[1].attention.channel_attention(y)

w_s = model[1].attention.spatial_attention(w * y)

w = torch.reshape(w, (16, 24))

y = y[0].unsqueeze(1)

grid = torchvision.utils.make_grid(y, nrow=24)

print(grid.shape)
w = w.unsqueeze(0).unsqueeze(0)
print(w.shape)
heatmap = transforms.Resize((grid.shape[1], grid.shape[2]),
                            interpolation=InterpolationMode.BILINEAR)(w.cpu())
heatmap = heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()
img = transforms.ToPILImage()(grid)

plt.figure(figsize=(96, 64))
plt.imshow(img)
plt.imshow(heatmap, alpha=0.4, cmap='jet')
plt.axis('off')
plt.savefig('log/heatmap.png', bbox_inches='tight')
plt.figure(figsize=(96, 64))
plt.imshow(img)
plt.axis('off')
plt.savefig('log/patches.png', bbox_inches='tight')

heatmap = transforms.Resize(
    (480, 480), interpolation=InterpolationMode.BILINEAR)(w_s.cpu())
heatmap = heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()

plt.imshow(transforms.Resize((480, 480))(Image.open(img_path)))
plt.imshow(heatmap, alpha=0.4, cmap='jet')

plt.savefig('log/spatial-heatmap.png', bbox_inches='tight')
