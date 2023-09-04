"""
Implementation of the GoogLeNet (InceptionNet) Architecture

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn  # nn modules


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes: int = 1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # order for inception block: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.max_pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        red_3x3: int,
        out_3x3: int,
        red_5x5: int,
        out_5x5: int,
        out_1x1_pool: int,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_3x3, kernel_size=1),
            ConvBlock(
                in_channels=red_3x3,
                out_channels=out_3x3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_5x5, kernel_size=1),
            ConvBlock(
                in_channels=red_5x5,
                out_channels=out_5x5,
                kernel_size=5,
                padding=2,
            ),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_1x1_pool,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return torch.cat(
            [
                self.branch1(x),
                self.branch2(x),
                self.branch3(x),
                self.branch4(x),
            ],
            1,
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ConvBlock, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(3, 3, 224, 224).to(device)
    model = GoogLeNet(in_channels=3, num_classes=1000).to(device)
    print(model(x).shape)
