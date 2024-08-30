""""
Implementation of the VGG16 Architecture

By Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn  # all the NN modules that we use
from torchsummary import summary

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, architecture: str, in_channels: int = 3, num_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        assert architecture in VGG_types.keys(), f"Not a valid architecture! Choose from: {list(VGG_types.keys())}"
        self.architecture = architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.max_pool_counter = 0
        self.shape_data_len = 0
        self.shape_data_width = 0
        self.conv_layers = self.create_conv_layers(architecture=VGG_types[self.architecture]).to(self.device)

        # use nn.Sequential to make the code more compact
        # self.fcs = nn.Sequential(
        #     nn.Linear(512 * int(self.shape_data_len / 2 ** self.max_pool_counter) *
        #               int(self.shape_data_width / 2 ** self.max_pool_counter), 4096),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, self.num_classes)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.shape_data_len = x.shape[2]
        self.shape_data_width = x.shape[-1]
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.create_fcs_layers(self.shape_data_len, self.shape_data_width).to(self.device)(x)

        return x

    def create_conv_layers(self, architecture: list) -> nn.Sequential:
        layers = []

        in_channels = self.in_channels

        for layer in architecture:
            if isinstance(layer, int):
                out_channels = layer

                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(layer),  # not included in the original paper,
                        # but included because it improves performance
                        nn.ReLU(),
                    ]
                )  # uses extend, because it is going to add each element of the list
                # individually, instead of as a single object (like that append does)
                """
                a = []
                a.append([1, 2, 3])
                a.extend([1, 2, 3])
                print(a)
                >> [[1, 2, 3], 1, 2, 3]
                """
                in_channels = layer
            elif layer == "M":
                self.max_pool_counter += 1
                layers.extend([nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))])

        return nn.Sequential(*layers)  # * unpacks the elements of the lists

    def create_fcs_layers(self, length, width):
        fcs_layers = [
            nn.Linear(
                512 * int(length / 2**self.max_pool_counter) * int(width / 2**self.max_pool_counter),
                4096,
            ),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
        ]

        return nn.Sequential(*fcs_layers)  # * unpacks the elements of the lists


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG("VGG16", in_channels=3, num_classes=1000).to(device)
x = torch.randn(1, 3, 32, 32).to(device=device)
print(model(x).shape)
summary(VGG("VGG16", in_channels=3, num_classes=1000), (3, 488, 488), device="cuda")
