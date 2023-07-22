"""
Implementation of a Convolutional Neural Network, using PyTorch

by Anthony Givans (anthonygivans@miami.edu)
"""

import sys
import os
import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
torch.manual_seed(17)  # computers a (pseudo) random, so specifying a seed allows for reproducibility

from tqdm import tqdm  # used to create progress bars for for-loops

file_name = os.path.basename(sys.argv[0]).split('.')[0]
PATH = f"deeplearning/saved_models/{file_name}_model.pt"


class CNN(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 16, num_classes: int = 10) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # same convolution - the size stays the same (ie, doesn't shrink)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # same convolution - the size stays the same (ie, doesn't shrink)

        # Fully connected layer w/ 784 values as input, and `num_classes` as output
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


def save_checkpoint(state: dict, file_name: str = PATH):
    """
    Saves the model at a specified point in the training process

    :param state: the current state/version of the model being saved
    :param file_name: Name of the saved model
    :return: None
    """
    print("Saving checkpoint")
    torch.save(state, file_name)


def load_checkpoint(checkpoint: dict) -> None:
    """
    Loads checkpoint of a trained model into a current model. This function assumes you only saved the model and
    optimizer states

    :param checkpoint: dictionary of specific state values to be loaded into the model
    :return: None
    """
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
in_channel = 1
out_channels = 16
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10
load_model = False

# Load Data
train_dataset = datasets.MNIST(root="./datasets/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="./datasets/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init network
model = CNN(in_channels=in_channel, out_channels=out_channels, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load(PATH))

# train the network
print("Training...\n")
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # send the data to cuda, if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward prop
        scores = model(data)
        loss = criterion(scores, targets)

        if (epoch == 0) and load_model:
            print(f"Loss at start: {loss:.2f}")

        # backward prop
        optimizer.zero_grad()  # MAKE SURE TO RESET THE GRADIENTS
        loss.backward()

        # Optimizer step
        optimizer.step()

    # saving the model
    if (epoch == num_epochs-1) and load_model:  # saves the model after the last epoch
        print(f"Loss at end: {loss:.2f}\n")
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

print("Done training, thankfully!\n")


def check_accuracy(loader: DataLoader, model: CNN):
    """
    Function to check the accuracy of the trained model

    :param loader: the datasets on which the model will be evaluated on
    :param model: the model that you would like to evaluate
    :return: None
    """
    if loader.dataset.train:
        print("Checking accuracy on training data\n")
    else:
        print("Checking accuracy on test data\n")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct)/float(num_samples) * 100:.2f}%")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
