"""
Implementation of the U-Net Architecture for Image Segmentation

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class ConvBlock(nn.Module):
    """
    Conv Block for the U-Net Architecture
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm1(self.conv1(x))
        x = self.batch_norm2(self.conv2(x))
        return self.relu(x)


class EncoderBlock(nn.Module):
    """
    Encoder Block for the Unet-Architecture
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool


class DecoderBlock(nn.Module):
    """
    Decoder Block for the Unet-Architecture
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x: torch.Tensor, skip_con: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diffY = skip_con.size()[2] - x.size()[2]
        diffX = skip_con.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip_con], axis=1)
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder for UNet Architecture
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        self.e1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.e2 = EncoderBlock(in_channels=64, out_channels=128)
        self.e3 = EncoderBlock(in_channels=128, out_channels=256)
        self.e4 = EncoderBlock(in_channels=256, out_channels=512)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skip1, u1 = self.e1(x)
        skip2, u2 = self.e2(u1)
        skip3, u3 = self.e3(u2)
        skip4, u4 = self.e4(u3)
        return [skip1, skip2, skip3, skip4], u4


class Decoder(nn.Module):
    """ "
    Decoder for the UNet Architecture
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.d1 = DecoderBlock(in_channels=in_channels, out_channels=in_channels // 2)
        self.in_channels //= 2
        self.d2 = DecoderBlock(in_channels=self.in_channels, out_channels=self.in_channels // 2)
        self.in_channels //= 2
        self.d3 = DecoderBlock(in_channels=self.in_channels, out_channels=self.in_channels // 2)
        self.in_channels //= 2
        self.d4 = DecoderBlock(in_channels=self.in_channels, out_channels=self.in_channels // 2)

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        d1 = self.d1(x, skip_connections[-1])
        d2 = self.d2(d1, skip_connections[2])
        d3 = self.d3(d2, skip_connections[1])
        d4 = self.d4(d3, skip_connections[0])
        return d4


class UNet(nn.Module):
    """
    UNet Architecture
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()

        """ ENCODER """
        # self.e1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        # self.e2 = EncoderBlock(in_channels=64, out_channels=128)
        # self.e3 = EncoderBlock(in_channels=128, out_channels=256)
        # self.e4 = EncoderBlock(in_channels=256, out_channels=512)
        self.encoder = Encoder(in_channels=in_channels)

        """ BOTTLENECK """
        self.b = ConvBlock(in_channels=512, out_channels=1024)

        """ DECODER """
        # self.d1 = DecoderBlock(in_channels=1024, out_channels=512)
        # self.d2 = DecoderBlock(in_channels=512, out_channels=256)
        # self.d3 = DecoderBlock(in_channels=256, out_channels=128)
        # self.d4 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder = Decoder(in_channels=1024)

        """ Classifier """
        self.output = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # skip1, u1 = self.e1(x)
        # skip2, u2 = self.e2(u1)
        # skip3, u3 = self.e3(u2)
        # skip4, u4 = self.e4(u3)
        skip_connections, encoded = self.encoder(x)

        b = self.b(encoded)

        # d1 = self.d1(b, skip4)
        # d2 = self.d2(d1, skip3)
        # d3 = self.d3(d2, skip2)
        # d4 = self.d4(d3, skip1)
        decoded = self.decoder(b, skip_connections)
        return self.output(decoded)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = torch.randn(1, 3, 572, 572).to(device=device)

model = UNet().to(device=device)
print(f"model output: {model(test_data).shape}")

summary(UNet(), (3, 572, 572), device="cuda")
