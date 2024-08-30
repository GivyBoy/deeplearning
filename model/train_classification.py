import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm  # used to create progress bars for for-loops
import matplotlib.pyplot as plt
from vit import ViT

torch.manual_seed(17)  # computers a (pseudo) random, so specifying a seed allows for reproducibility


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim,
        criterion: nn,
        device: torch.device,
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
        train_acc = []
        test_acc = []
        for epoch in range(self.epochs):
            epoch_train_loss, epoch_train_acc = self._train(model=model)
            train_loss.append(epoch_train_loss)
            train_acc.append(epoch_train_acc)
            epoch_test_loss, epoch_test_acc = self._test(model=model)
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)
            print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss} | Test Loss: {epoch_test_loss}")
            print(f"Epoch: {epoch} | Train Accuracy: {epoch_train_acc} | Test Accuracy: {epoch_test_acc}")
        self._plot_metric(train=train_loss, test=test_loss, metric="loss")
        self._plot_metric(train=train_acc, test=test_acc, metric="accuracy")

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
        train_accuracy = 0
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
            # calc accuracy
            accuracy = (output.argmax(dim=1) == targets).float().mean()
            # optimizer step
            self.optimizer.step()
            # update train loss
            train_loss += loss.item()
            train_accuracy += accuracy.item()
        return (train_loss / (batch_idx + 1)), (train_accuracy / (batch_idx + 1))

    def _test(self, model) -> float:
        model.eval()
        test_loss = 0
        train_accuracy = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(self.test_loader)):
                # Get data to cuda if possible
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                # forward
                output = model(data)
                loss = self.criterion(output, targets)
                # calc accuracy
                accuracy = (output.argmax(dim=1) == targets).float().mean()
                # update test loss
                test_loss += loss.item()
                train_accuracy += accuracy.item()
        return (test_loss / (batch_idx + 1)), (train_accuracy / (batch_idx + 1))

    def _check_accuracy(self, model) -> None:
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(self.val_loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            print(
                f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct) / float(num_samples) * 100:.2f}%"
            )

    def _plot_metric(self, train, test, metric: str) -> None:
        plt.plot(train, "-b", label=f"train {metric}")
        plt.plot(test, label=f"test {metric}")
        plt.title(f"{metric} vs epoch")
        plt.xlabel("epoch")
        plt.ylabel(f"{metric}")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    in_channel = 3
    out_channels = 16
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 15
    val_pct = 0.3  # % of test set that will be used for validation

    # Load Data
    train_dataset = datasets.CIFAR10(
        root="./datasets/",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.CIFAR10(
        root="./datasets/",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    test_size = int(len(test_dataset) * (1 - val_pct))
    test_data, val_data = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = ViT(in_channels=3, patch_size=8, emb_size=384, img_size=32, depth=8, n_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    trainer.eval(model=model)
    trainer.check_accuracy(model=model)
