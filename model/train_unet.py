""" Train UNet """
import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from u_net import UNet
from loss import DiceLoss
from metrics import IoU, DiceScore
from datasets import MitochondriaDataset, Kaggle


def train(model, device, optimizer, dataloader, writer):
    model.train()
    loss_train = 0
    dice = 0
    iou = 0
    step = 0
    for batch_idx, (img, mask) in enumerate(tqdm(dataloader)):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = DiceLoss()(output, mask)
        dice += DiceScore()(output, mask).item()
        iou += IoU()(output, mask).item()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Dice/train", dice, step)
        writer.add_scalar("IoU/train", iou, step)
        step += 1
    return (loss_train / (batch_idx + 1)), (dice / (batch_idx + 1)), (iou / (batch_idx + 1))


def test(model, device, dataloader, writer):
    model.eval()
    loss_test = 0
    dice = 0
    iou = 0
    step = 0
    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(tqdm(dataloader)):
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = DiceLoss()(output, mask)
            dice += DiceScore()(output, mask).item()
            iou += IoU()(output, mask).item()
            loss_test += loss.item()
            writer.add_scalar("Loss/test", loss.item(), step)
            writer.add_scalar("Dice/test", dice, step)
            writer.add_scalar("IoU/test", iou, step)
            step += 1
    return (loss_test / (batch_idx + 1)), (dice / (batch_idx + 1)), (iou / (batch_idx + 1))


# class MitochondriaDataset(Dataset):
#     def __init__(self, images, masks):
#         self.images = images
#         self.masks = masks

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = torch.tensor(self.images[idx], dtype=torch.float32).reshape(3, 256, 256)
#         mask = torch.tensor(self.masks[idx], dtype=torch.float32).reshape(3, 256, 256)
#         return image, mask


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
                f"frame_{im}.png",
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


def split_data(data, split_pct):
    size = int(len(data) * (1 - split_pct))
    data, test_data = torch.utils.data.random_split(data, [size, len(data) - size])
    return data, test_data


def get_mitochondria():
    train_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/img/"
    train_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/mask/"
    test_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/img/"
    test_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/mask/"
    train_dataset = MitochondriaDataset(train_img_path, train_mask_path)
    test_dataset = MitochondriaDataset(test_img_path, test_mask_path)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader


def get_kaggle(batch_size, val_pct):
    kaggle_img = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/imgs"
    kaggle_mask = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/masks"
    train_dataset = Kaggle(kaggle_img, kaggle_mask)
    train_dataset, test_dataset = split_data(train_dataset, split_pct=0.2)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset, val_data = split_data(test_dataset, split_pct=val_pct)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader, val_dataloader


# train_img = np.array(get_imgs(train_img_path)) / 255
# train_mask = get_imgs(train_mask_path)
# test_img = np.array(get_imgs(test_img_path)) / 255
# test_mask = get_imgs(test_mask_path)

# train_dataset = MitochondriaDataset(train_img_path, train_mask_path)
# test_dataset = MitochondriaDataset(test_img_path, test_mask_path)

# train_dataloader, test_dataloader = get_mitochondria()
train_dataloader, test_dataloader, val_dataloader = get_kaggle(batch_size=16, val_pct=0.3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=3).to(device)

optimizer = optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-4)

loss_train_list = []
loss_test_list = []
dice_train_list = []
dice_test_list = []
iou_train_list = []
iou_test_list = []

writer = SummaryWriter("tensor_board/kaggle/unet")

for epoch in range(0, 15):  # change 100 to a larger number if necessary
    # -------- training ---------------------------------
    loss_train, train_dice, train_iou = train(model, device, optimizer, train_dataloader, writer)
    loss_train_list.append(loss_train)
    dice_train_list.append(train_dice)
    iou_train_list.append(train_iou)
    # -------- validation -------------------------------
    loss_test, test_dice, test_iou = test(model, device, test_dataloader, writer)
    loss_test_list.append(loss_test)
    dice_test_list.append(test_dice)
    iou_test_list.append(test_iou)
    # -------- print metrics ----------------------------
    print(f"Epoch: {epoch} | Train Loss: {loss_train} | Test Loss: {loss_test}")
    print(f"Epoch: {epoch} | Train Dice: {train_dice} | Test Dice: {test_dice}")
    print(f"Epoch: {epoch} | Train IoU: {train_iou} | Test IoU: {test_iou}")


def plot_metric(train, test, metric: str) -> None:
    plt.plot(train, "-b", label=f"train {metric}")
    plt.plot(test, label=f"test {metric}")
    plt.title(f"{metric} vs epoch")
    plt.xlabel("epoch")
    plt.ylabel(f"{metric}")
    plt.grid(True)
    plt.legend()
    plt.show()


plot_metric(loss_train_list, loss_test_list, "loss")
plot_metric(dice_train_list, dice_test_list, "dice")
plot_metric(iou_train_list, iou_test_list, "iou")
