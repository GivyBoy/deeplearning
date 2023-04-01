"""
Implementation of a Convolutional Neural Network, using PyTorch

by Anthony Givans (anthonygivans@miami.edu)
"""

import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm


class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # same convolution - the size stays the same (ie, doesn't shrink)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )  # same convolution - the size stays the same (ie, doesn't shrink)

        # Fully connected layer w/ 784 values as input, and `num_classes` as output
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# model = CNN()
# data = torch.rand(64, 1, 28, 28)
# print(model(data).shape)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Hyperparameters
in_channel = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 1

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# init network
model = CNN(in_channels=in_channel, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader, 1)):
        # send the data to cuda, if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward prop
        scores = model(data)
        loss = criterion(scores, targets)

        # backward prop
        optimizer.zero_grad()  # MAKE SURE TO RESET THE GRADIENTS
        loss.backward()

        # Optimizer step
        optimizer.step()


# check accuracy on  training and testing datasets to see model performance

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct)/float(num_samples) * 100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)