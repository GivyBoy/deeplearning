"""
Implementation of the U-Net++ Architecture for Image Segmentation

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as fn


class UNet_pp(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel: int = 3, stride: int = 1,
                 num_classes: int = 2):
        super(UNet_pp, self).__init__()

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

        self.conv2_upsample = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                                 kernel_size=(2, 2), stride=2
                                                 )

        self.conv3 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.conv3_upsample = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                                 kernel_size=(2, 2), stride=2
                                                 )

        self.conv4 = self._make_conv_layer(in_channels=self.channels, out_channels=self.channels * 2, kernel=kernel,
                                           stride=stride
                                           )
        self.channels *= 2

        self.conv4_upsample = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                                 kernel_size=(2, 2), stride=2
                                                 )

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

        self.conv7 = self._make_conv_layer(in_channels=self.channels*2, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.upsample3 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv8 = self._make_conv_layer(in_channels=1152, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )
        self.channels /= 2
        self.channels = int(self.channels)

        self.upsample4 = nn.ConvTranspose2d(in_channels=self.channels, out_channels=int(self.channels / 2),
                                            kernel_size=(2, 2), stride=2
                                            )

        self.conv9 = self._make_conv_layer(in_channels=1216, out_channels=int(self.channels / 2),
                                           kernel=kernel, stride=stride
                                           )

        self.channels /= 2
        self.channels = int(self.channels)

        self.conv10 = nn.Conv2d(in_channels=self.channels, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

        # ---- Auxiliary upsamples??
        self.conv2_1_upsample = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                                   kernel_size=(2, 2), stride=2
                                                   )
        self.conv1_1_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                                   kernel_size=(2, 2), stride=2
                                                   )
        self.conv1_2_upsample = nn.ConvTranspose2d(in_channels=640, out_channels=128,
                                                   kernel_size=(2, 2), stride=2
                                                   )

    def forward(self, x):
        conv0_0 = self.conv1(x)
        conv0_pool = self.pool(self.relu(conv0_0))

        ##############################################
        conv1_0 = self.conv2(conv0_pool)
        conv1_pool = self.pool(self.relu(conv1_0))

        ##############################################
        conv2_0 = self.conv3(conv1_pool)
        conv2_pool = self.pool(self.relu(conv2_0))

        ##############################################
        conv3_0 = self.conv4(conv2_pool)
        conv3_pool = self.pool(self.relu(conv3_0))

        ##############################################
        conv4_0 = self.conv5(conv3_pool)

        ##############################################
        conv3_1 = self.upsample1(conv4_0)
        conv3_1 = self._crop_concat(conv3_1, conv3_0)
        conv3_1 = self.relu(self.conv6(conv3_1))

        ##############################################
        conv2_2 = self.upsample2(conv3_1)
        conv2_2 = self._crop_concat(conv2_2, conv2_0)
        conv2_1 = self.conv4_upsample(conv3_0)
        conv2_1 = self._crop_concat(conv2_1, conv2_0)
        conv2_2 = self._crop_concat(conv2_2, conv2_1)
        conv2_2 = self.relu(self.conv7(conv2_2))

        ##############################################
        conv1_3 = self.upsample3(conv2_2)
        conv1_3 = self._crop_concat(conv1_3, conv1_0)
        conv1_1 = self.conv3_upsample(conv2_0)
        conv1_1 = self._crop_concat(conv1_1, conv1_0)
        conv1_3 = self._crop_concat(conv1_3, conv1_1)
        conv1_2 = self.conv2_1_upsample(conv2_1)
        conv1_2 = self._crop_concat(conv1_2, conv1_1)
        conv1_2 = self._crop_concat(conv1_2, conv1_0)
        conv1_3 = self._crop_concat(conv1_3, conv1_2)
        conv1_3 = self.relu(self.conv8(conv1_3))

        ##############################################
        conv0_4 = self.upsample4(conv1_3)
        conv0_4 = self._crop_concat(conv0_4, conv0_0)
        conv0_1 = self.conv2_upsample(conv1_0)
        conv0_1 = self._crop_concat(conv0_1, conv0_0)
        conv0_4 = self._crop_concat(conv0_4, conv0_1)
        conv0_2 = self.conv1_1_upsample(conv1_1)
        conv0_2 = self._crop_concat(conv0_2, conv0_1)
        conv0_2 = self._crop_concat(conv0_2, conv0_0)
        conv0_4 = self._crop_concat(conv0_4, conv0_2)
        conv0_3 = self.conv1_2_upsample(conv1_2)
        conv0_3 = self._crop_concat(conv0_3, conv0_2)
        conv0_3 = self._crop_concat(conv0_3, conv0_1)
        conv0_3 = self._crop_concat(conv0_3, conv0_0)
        conv0_4 = self._crop_concat(conv0_4, conv0_3)
        conv0_4 = self.relu(self.conv9(conv0_4))

        ##############################################
        conv10 = self.relu(self.conv10(conv0_4))

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

model = UNet_pp().to(device=device)
print(f"model output: {model(test_data).shape}")