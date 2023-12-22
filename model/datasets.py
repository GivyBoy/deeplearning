import os

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def plot_img_mask(img, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.show()


class MitochondriaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = torch.tensor(
            np.array(
                Image.fromarray(cv2.imread(os.path.join(self.image_dir, self.img_list[idx]))).resize((768, 768))
            ),  # resize from 768 x 1024 to 768 x 768
            dtype=torch.float32,
        ).reshape(3, 768, 768)
        mask = torch.tensor(
            np.array(
                Image.fromarray(cv2.imread(os.path.join(self.mask_dir, self.img_list[idx]))).resize((768, 768))
            ),  # resize from 768 x 1024 to 768 x 768
            dtype=torch.float32,
        ).reshape(3, 768, 768)
        return image, mask


class CVC(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = torch.tensor(Image.open(os.path.join(self.image_dir, self.img_list[idx])), dtype=torch.float32)
        mask = torch.tensor(Image.open(os.path.join(self.mask_dir, self.img_list[idx])), dtype=torch.float32)
        return image, mask


class Kaggle(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = torch.tensor(
            np.array(Image.fromarray(cv2.imread(os.path.join(self.image_dir, self.img_list[idx])))), dtype=torch.float32
        ).reshape(3, 101, 101)
        mask = torch.tensor(
            np.array(Image.fromarray(cv2.imread(os.path.join(self.mask_dir, self.img_list[idx])))), dtype=torch.float32
        ).reshape(3, 101, 101)
        return image, mask


if __name__ == "__main__":
    mito = MitochondriaDataset(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/img/",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/mask/",
    )

    cvc = CVC(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/cvc/image/",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/cvc/mask/",
    )  # slight error that needs addressing - doesn't read the img but reads the mask

    kaggle = Kaggle(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/imgs",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/masks",
    )

    print(kaggle[70][0].shape, kaggle[70][1].shape)
