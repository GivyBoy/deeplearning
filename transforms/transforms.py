__author__ = "Gorkem Can Ates"

__email__ = "gca45@miami.edu"

from custom_transforms import *
from PIL import Image
import numpy as np


class Transforms:

    def __init__(self, mode='torch', shape=(224, 224), transform=True, multi_dim=False) -> None:
        train_transforms = []

        val_transforms = []

        train_transforms.extend([Resize(shape=shape, mode=mode, multi_dim=multi_dim)])

        val_transforms.extend([Resize(shape=shape, mode=mode, multi_dim=multi_dim)])

        if transform:
            train_transforms.extend(

                [RandomRotation(angles=(-60, 60), p=0.1, mode=mode),

                 RandomHorizontalFlip(p=0.1, mode=mode),

                 RandomVerticalFlip(p=0.1, mode=mode),

                 # GaussianBlur(p=0.1),

                 # GrayScale(p=0.1)

                 ])

        self.train_transform = Compose(transforms=train_transforms)

        self.val_transform = Compose(transforms=val_transforms)
