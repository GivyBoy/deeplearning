import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm  # used to create progress bars for for-loops
import matplotlib.pyplot as plt
from metrics import IoU, DiceScore
from loss import DiceLoss
from datasets import MitochondriaDataset, Kaggle
from u_net import UNet
from trans_unet import TransUNet

torch.manual_seed(17)  # computers a (pseudo) random, so specifying a seed allows for reproducibility


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim,
        device: torch.device,
        criterion: nn = DiceLoss(),
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def eval(self, model) -> None:
        train_loss = []
        test_loss = []
        train_dice = []
        test_dice = []
        train_iou = []
        test_iou = []
        for epoch in range(self.epochs):
            epoch_train_loss, epoch_train_dice, epoch_train_iou = self._train(model=model)
            train_loss.append(epoch_train_loss)
            train_dice.append(epoch_train_dice)
            train_iou.append(epoch_train_iou)
            epoch_test_loss, epoch_test_dice, epoch_test_iou = self._test(model=model)
            test_loss.append(epoch_test_loss)
            test_dice.append(epoch_test_dice)
            test_iou.append(epoch_test_iou)
            print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss} | Test Loss: {epoch_test_loss}")
            print(f"Epoch: {epoch} | Train Dice: {epoch_train_dice} | Test Dice: {epoch_test_dice}")
            print(f"Epoch: {epoch} | Train IoU: {epoch_train_iou} | Test IoU: {epoch_test_iou}")
        self._plot_metric(train=train_loss, test=test_loss, metric="loss")
        self._plot_metric(train=train_iou, test=test_iou, metric="IoU")
        self._plot_metric(train=train_dice, test=test_dice, metric="dice")

    def check_accuracy(self, model) -> None:
        if self.val_loader is None:
            print("No validation loader was provided. Cannot check accuracy.")
            return
        else:
            print("Checking accuracy on validation set")
            self._check_accuracy(model=model)

    def _train(self, model) -> float:
        model.train()
        train_loss = 0
        train_dice = 0
        train_iou = 0
        for batch_idx, (data, targets) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            # Get data to cuda if possible
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)
            # forward
            output = model(data)
            loss = self.criterion(output, targets)
            # backward
            loss.backward()
            # optimizer step
            self.optimizer.step()
            # update train loss and metrics
            train_loss += loss.item()
            train_dice += DiceScore()(output, targets).item()
            train_iou += IoU()(output, targets).item()
        return (train_loss / (batch_idx + 1)), (train_dice / (batch_idx + 1)), (train_iou / (batch_idx + 1))

    def _test(self, model) -> float:
        model.eval()
        test_loss = 0
        test_dice = 0
        test_iou = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.test_loader)):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                # forward
                output = model(data)
                loss = self.criterion(output, targets)
                # update test loss and metrics
                test_loss += loss.item()
                test_dice += DiceScore()(output, targets).item()
                test_iou += IoU()(output, targets).item()
        return (test_loss / (batch_idx + 1)), (test_dice / (batch_idx + 1)), (test_iou / (batch_idx + 1))

    def _check_accuracy(self, model) -> None:
        dice = 0
        iou = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.val_loader)):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                # forward
                output = model(data)
                dice += DiceScore()(output, targets).item()
                iou += IoU()(output, targets).item()

            print(f"VALIDATION | IoU: {iou} | Dice: {dice}")

    def _plot_metric(self, train, test, metric: str) -> None:
        plt.plot(train, "-b", label=f"train {metric}")
        plt.plot(test, label=f"test {metric}")
        plt.title(f"{metric} vs epoch")
        plt.xlabel("epoch")
        plt.ylabel(f"{metric}")
        plt.grid(True)
        plt.legend()
        plt.show()


def split_data(data, split_pct):
    size = int(len(data) * (1 - split_pct))
    data, test_data = torch.utils.data.random_split(data, [size, len(data) - size])
    return data, test_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/img/"
    train_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/train/mask/"
    test_img_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/img/"
    test_mask_path = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/mitochondria/test/mask/"

    kaggle_img = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/imgs"
    kaggle_mask = "/mnt/c/Users/givan/Desktop/deeplearning/datasets/segmentation/kaggle/masks"

    # Hyper-parameters
    in_channel = 3
    out_channels = 16
    num_classes = 10
    learning_rate = 3e-4
    batch_size = 4
    num_epochs = 10
    val_pct = 0.3  # % of test set that will be used for validation

    train_dataset = MitochondriaDataset(train_img_path, train_mask_path)
    test_dataset = MitochondriaDataset(test_img_path, test_mask_path)

    # train_dataset = Kaggle(kaggle_img, kaggle_mask)

    # train_dataset, test_dataset = split_data(train_dataset, split_pct=0.2)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_size = int(len(test_dataset) * (1 - val_pct))
    # test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])
    test_data, val_dataset = split_data(test_dataset, split_pct=val_pct)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=3).to(device)
    model_t = TransUNet(
        img_dim=768,
        in_channels=3,
        out_channels=3,
        head_num=4,
        mlp_dim=512,
        block_num=8,
        patch_dim=16,
        class_num=1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss = DiceLoss()

    trainer = Trainer(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        criterion=loss,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer.eval(model=model)
    trainer.check_accuracy(model=model)
