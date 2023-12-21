""" Train UNet """
import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from u_net import UNet


def train(model, device, optimizer, dataloader):
    print("training")
    model.train()
    loss_train = 0
    for batch_idx, (img, mask) in enumerate(tqdm(dataloader)):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = nn.BCEWithLogitsLoss()(output, mask)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    print("done epoch")
    return loss_train / (batch_idx + 1)


def test(model, device, dataloader):
    print("testing")
    model.eval()
    loss_test = 0
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(tqdm(dataloader)):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = nn.BCEWithLogitsLoss()(output, mask)
            loss_test += loss.item()
    return loss_test / (batch_idx + 1)


class MitochondriaDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).reshape(3, 256, 256)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32).reshape(3, 256, 256)
        return image, mask


train_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/img/"
train_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/mask/"
test_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/img/"
test_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/mask/"


def tif_to_png(tif_path: str, filename: str, png_path: str):
    """
    Extracts each frame from a .tif file and saves it as a .png file
    """
    img = Image.open(tif_path + filename)
    for im in range(img.n_frames):
        img.seek(im)
        img.save(
            os.path.join(
                png_path,
                f"{filename.split('.')[0]}_frame_{im}.png",
            )
        )


def get_imgs(path: str, size: int = 256):
    """
    Returns a list of images from a given path
    """
    images = os.listdir(path)
    image_dataset = []
    for i, image_name in enumerate(images):
        if image_name.split(".")[1] == "png":
            image = cv2.imread(path + image_name, 1)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            image_dataset.append(np.array(image))
    return image_dataset


train_img = np.array(get_imgs(train_img_path)) / 255
train_mask = get_imgs(train_mask_path)
test_img = np.array(get_imgs(test_img_path)) / 255
test_mask = get_imgs(test_mask_path)

train_dataset = MitochondriaDataset(train_img, train_mask)
test_dataset = MitochondriaDataset(test_img, test_mask)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=3).to(device)

optimizer = optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-4)

loss_train_list = []
loss_test_list = []

for epoch in range(0, 20):  # change 100 to a larger number if necessary
    # -------- training --------------------------------
    loss_train = train(model, device, optimizer, train_dataloader)
    loss_train_list.append(loss_train)
    print("epoch", epoch, "training loss:", loss_train)
    # -------- validation --------------------------------
    loss_test = test(model, device, test_dataloader)
    loss_test_list.append(loss_test)
    print("epoch", epoch, "test loss:", loss_test)
    # # --------save model-------------------------
    # result = (loss_train_list, acc_train_list, acc_val_list, other_val)

plt.title("loss v.s. epoch", fontsize=16)
plt.plot(loss_train_list, "-b", label="training loss")
plt.plot(loss_test_list, label="testing loss")
plt.xlabel("epoch", fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()
