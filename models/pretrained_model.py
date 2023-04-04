"""
Exploring the use and fine-tuning of pretrained models offered by PyTorch

by Anthony Givans (anthonygivans@miami.edu)
"""

import sys
import torch
import torch.nn as nn  # neural network modules
import torch.optim as optim  # used for optimization libraries (SGD, Adam, etc)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from tqdm import tqdm  # used to create progress bars for for-loops

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
in_channel = 1
out_channels = 16
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.CIFAR10(root="../datasets/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root="../datasets/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Load pretrained model and modify it
model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

# Freeze the weights of the layers to make the training process faster - esp if the model is HUGE
for param in model.parameters():
    param.requires_grad = False

# layers that you add after freezing (and/or the ones you didn't freeze) will be trained

model.avgpool = Identity()  # changes the avgpool layer to one that returns itself
"Converts entire classifier into a single Linear layer. You could also specify changes to individual " \
    "layers in the classifier sequence, by indexing the specific layer (eg `model.classifier[0] ...`)"
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# print(model)
#
# sys.exit() # exits the script when called

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        # backward prop
        optimizer.zero_grad()  # MAKE SURE TO RESET THE GRADIENTS
        loss.backward()

        # Optimizer step
        optimizer.step()

print("Done training, thankfully!\n")


def check_accuracy(loader: DataLoader, model):
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

        print(
            f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct) / float(num_samples) * 100:.2f}%")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
