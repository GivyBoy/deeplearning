"""
Implementation of the ResNet Architecture

By Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, identity_downsample=None, stride: int = 1):
        super(ResNetBlock, self).__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1,
                               stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResNetBlock, layers: list, image_channels: int, num_classes: int):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(ResNetBlock, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def _make_layer(self, ResNetBlock, num_residual_block: int, out_channels: int, stride: int):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels * 4),)

        layers.append(ResNetBlock(self.in_channels, out_channels, identity_downsample, stride))

        self.in_channels = out_channels * 4  # output from block above will be 256

        for i in range(num_residual_block - 1):
            layers.append(ResNetBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # * unpacks the list


def ResNet50(img_channels: int = 3, num_classes: int = 1000) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 4, 6, 3], image_channels=img_channels, num_classes=num_classes)


def ResNet101(img_channels: int = 3, num_classes: int = 1000) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 4, 23, 3], image_channels=img_channels, num_classes=num_classes)


def ResNet152(img_channels: int = 3, num_classes: int = 1000) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 8, 36, 3], image_channels=img_channels, num_classes=num_classes)

# Testing out
# def test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(10, 3, 224, 224).to(device)
#     model = ResNet152(img_channels=3, num_classes=1000).to(device)
#     print(model(x).shape)
#
# test()
