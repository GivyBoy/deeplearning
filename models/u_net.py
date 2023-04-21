"""
Implementation of the U-Net Architecture for Image Segmentation

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as fn


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel: int = 3, stride: int = 1,
                 num_classes: int = 2):
        super(UNet, self).__init__()

        self.channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.relu = nn.ReLU()

        self.conv1 = self._make_conv_layer(in_channels=in_channels, out_channels=out_channels, kernel=kernel,
                                           stride=stride
                                           )

        self.conv2 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.conv3 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.conv4 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.conv5 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.upsample1 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv6 = self._make_conv_layer(in_channels=self.channels, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.upsample2 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv7 = self._make_conv_layer(in_channels=self.channels, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.upsample3 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv8 = self._make_conv_layer(in_channels=self.channels, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.upsample4 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv9 = self._make_conv_layer(in_channels=self.channels, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.conv10 = nn.Conv2d(in_channels=self.channels, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_pool = self.pool(self.relu(conv1))

        # print(f"conv1: {conv1.shape}")

        conv2 = self.conv2(conv1_pool)
        conv2_pool = self.pool(self.relu(conv2))

        # print(f"conv2: {conv2.shape}")

        conv3 = self.conv3(conv2_pool)
        conv3_pool = self.pool(self.relu(conv3))

        # print(f"conv3: {conv3.shape}")

        conv4 = self.conv4(conv3_pool)
        conv4_pool = self.pool(self.relu(conv4))

        # print(f"conv4: {conv4.shape}")

        conv5 = self.conv5(conv4_pool)
        # conv5 = self.pool(self.relu(conv5))

        # print(f"conv5: {conv5.shape}")

        conv6 = self.upsample1(conv5)
        conv6 = self._crop_concat(conv6, conv4)
        conv6 = self.relu(self.conv6(conv6))

        # print(f"conv6: {conv6.shape}")

        conv7 = self.upsample2(conv6)
        conv7 = self._crop_concat(conv7, conv3)
        conv7 = self.relu(self.conv7(conv7))

        # print(f"conv7: {conv7.shape}")

        conv8 = self.upsample3(conv7)
        conv8 = self._crop_concat(conv8, conv2)
        conv8 = self.relu(self.conv8(conv8))

        # print(f"conv8: {conv8.shape}")

        conv9 = self.upsample4(conv8)
        conv9 = self._crop_concat(conv9, conv1)
        conv9 = self.relu(self.conv9(conv9))

        # print(f"conv9: {conv9.shape}")

        conv10 = self.conv10(conv9)

        # print(f"conv10: {conv10.shape}")

        return conv10

    def _crop_concat(self, tensor1, tensor2):
        crop = fn.center_crop(tensor2, output_size=tensor1.shape[-1])
        return torch.cat((crop, tensor1), 1)

    def _make_conv_layer(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                      stride=(stride, stride), padding=(0, 0)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel, kernel),
                      stride=(stride, stride), padding=(0, 0))
        )
        return layers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = torch.randn(1, 1, 572, 572).to(device=device)

model = UNet().to(device=device)
print(f"model output: {model(test_data).shape}")
