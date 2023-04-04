"""
Implementation of Custom Datasets, using PyTorch

By Anthony Givans (anthonygivans@miami.edu)
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class CustomDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Args
        :param csv_file: Path to the csv file w/ annotations
        :param root_dir: Directory containing the images
        :param transform: Optional transform to be applied on sample
        """
        super(CustomDataset).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            img = self.transform(img)

        return img, y_label
