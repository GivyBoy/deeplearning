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


def split_data(data, split_pct):
    size = int(len(data) * (1 - split_pct))
    data, test_data = torch.utils.data.random_split(data, [size, len(data) - size])
    return data, test_data


class DATASET:
    def __init__(self, im_path, mask_path, experiment="Kaggle/") -> None:
        if experiment == "Kaggle/":
            dataset = Kaggle(im_path, mask_path)
            self.train_dataset, self.test_dataset = split_data(dataset, split_pct=0.2)
        else:
            raise ValueError("Dataset not found")

        print("Dataset loaded")


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
            np.array(Image.fromarray(cv2.imread(os.path.join(self.image_dir, self.img_list[idx]))).convert("RGB")),
            dtype=torch.float32,
        ).reshape(3, 101, 101)
        mask = torch.tensor(
            np.array(Image.fromarray(cv2.imread(os.path.join(self.mask_dir, self.img_list[idx]))).convert("RGB")),
            dtype=torch.float32,
        ).reshape(3, 101, 101)
        return image, mask


if __name__ == "__main__":
    mito = MitochondriaDataset(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/img/",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/mask/",
    )

    cvc = CVC(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/cvc/images/",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/cvc/masks/",
    )  # slight error that needs addressing - doesn't read the img but reads the mask

    kaggle = Kaggle(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/images",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/masks",
    )

    print(kaggle[70][0].shape, kaggle[70][1].shape)

    dataset = DATASET(
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/images",
        "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/masks",
        "Kaggle/",
    )
    print(len(dataset.train_dataset), len(dataset.test_dataset))
