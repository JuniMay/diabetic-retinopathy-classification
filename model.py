import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, dim, patch_size, in_channels=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.activation = nn.GELU()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(dim,
                                        dim,
                                        kernel_size,
                                        groups=dim,
                                        padding='same')
        self.activation1 = nn.GELU()
        self.norm1 = nn.BatchNorm2d(dim)
        self.pointwise_conv = nn.Conv2d(dim, dim, 1)
        self.activation2 = nn.GELU()
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise_conv(x)
        x = self.activation1(x)
        x = self.norm1(x)

        x += residual

        x = self.pointwise_conv(x)
        x = self.activation2(x)
        x = self.norm2(x)

        return x


class ClassificationHead(nn.Module):
    def __init__(self,
                 dim,
                 num_hiddens = None,
                 num_classes=5,
                 drop=0.25,):
        super().__init__()

        if num_hiddens is None:
            num_hiddens = dim * 2

        self.linear1 = nn.Linear(dim, num_hiddens)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(drop)
        self.linear2 = nn.Linear(num_hiddens, num_hiddens)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(drop)
        self.linear3 = nn.Linear(num_hiddens, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x


class Net(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 kernel_size,
                 patch_size,
                 num_classes=5,
                 in_channels=3,
                 drop=0.25):
        super().__init__()

        self.patch = PatchEmbedding(dim, patch_size, in_channels)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(BasicLayer(dim, kernel_size))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.head = ClassificationHead(dim, None, num_classes, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.patch(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)

        return x
