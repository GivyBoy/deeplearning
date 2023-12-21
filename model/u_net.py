"""
Implementation of the U-Net Architecture for Image Segmentation

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
from torch import nn
from torchsummary import summary

from utils.main_blocks import UNetConvBlock, UNetEncoder, UNetDecoder


class UNet(nn.Module):
    """
    UNet Architecture
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()

        """ ENCODER """
        self.encoder = UNetEncoder(in_channels=in_channels)

        """ BOTTLENECK """
        self.b = UNetConvBlock(in_channels=512, out_channels=1024)

        """ DECODER """
        self.decoder = UNetDecoder(in_channels=1024)

        """ Classifier """
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, encoded = self.encoder(x)
        b = self.b(encoded)
        decoded = self.decoder(b, skip_connections)
        return self.output(decoded)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.randn(1, 3, 572, 572).to(device=device)

    model = UNet().to(device=device)
    print(f"model output: {model(test_data).shape}")

    summary(UNet(), (3, 572, 572), device="cuda")
