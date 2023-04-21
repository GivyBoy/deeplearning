import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import torch.optim as optim
import torchvision.transforms.functional as fn
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, identity_downsample=None, stride: int = 1):
        super(ResNetBlock, self).__init__()

        self.expansion = 4
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1,
                               stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResNetBlock, layers: list, image_channels: int, num_classes: int):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv1d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(ResNetBlock, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(ResNetBlock, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, layers[3], out_channels=512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        return x

    def _make_layer(self, ResNetBlock, num_residual_block: int, out_channels: int, stride: int):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv1d(self.in_channels, out_channels * 4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm1d(out_channels * 4), )

        layers.append(ResNetBlock(self.in_channels, out_channels, identity_downsample, stride))

        self.in_channels = out_channels * 4  # output from block above will be 256

        for i in range(num_residual_block - 1):
            layers.append(ResNetBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # * unpacks the list


def ResNet50(img_channels: int = 3, num_classes: int = 5) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 4, 6, 3], image_channels=img_channels, num_classes=num_classes)


def ResNet101(img_channels: int = 3, num_classes: int = 5) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 4, 23, 3], image_channels=img_channels, num_classes=num_classes)


def ResNet152(img_channels: int = 3, num_classes: int = 5) -> ResNet:
    return ResNet(ResNetBlock=ResNetBlock, layers=[3, 8, 36, 3], image_channels=img_channels, num_classes=num_classes)

def cal_accuracy(confusion):
    #input: confusion is the confusion matrix
    #output: acc is the standard classification accuracy
    M=confusion.copy().astype('float32')
    acc = M.diagonal().sum()/M.sum()
    sens=np.zeros(M.shape[0])
    prec=np.zeros(M.shape[0])
    for n in range(0, M.shape[0]):
        TP=M[n,n]
        FN=np.sum(M[n,:])-TP
        FP=np.sum(M[:,n])-TP
        sens[n]=TP/(TP+FN)
        prec[n]=TP/(TP+FP)
    return acc, sens, prec

def train(model, device, optimizer, dataloader, epoch):
    model.train()#set model to training mode
    loss_train=0
    acc_train =0
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()#clear grad of each parameter
        Z = model(X)#forward pass
        loss = nnF.cross_entropy(Z, Y)
        loss.backward()#backward pass
        optimizer.step()#update parameters
        loss_train+=loss.item()
        #do not need softmax
        Yp = Z.data.max(dim=1)[1]  # get the index of the max
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 100. * batch_idx / len(dataloader), loss.item()))
    loss_train/=len(dataloader)
    #due to upsampling, len(dataloader.dataset) != sample_count
    #acc_train/=len(dataloader.dataset)
    acc_train/=sample_count
    return loss_train, acc_train

def test(model, device, dataloader):
    model.eval()#set model to evaluation mode
    acc_test =0
    confusion=np.zeros((5,5))
    with torch.no_grad(): # tell Pytorch not to build graph in the with section
        for batch_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            Z = model(X)#forward pass
            #do not need softmax
            Yp = Z.data.max(dim=1)[1]  # get the index of the max
            acc_test+= torch.sum(Yp==Y).item()
            for i in range(0, 5):
                for j in range(0, 5):
                    confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
    acc, sens, prec=cal_accuracy(confusion)
    return acc, (confusion, sens, prec)

class MyDataset(torch_dataset):
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        #return the number of data points
        return self.X.shape[0]
    def __getitem__(self, idx):
        #we can use DatasetName[idx] to get a data point (x,y) with index idx
        x=torch.tensor(self.X[idx], dtype=torch.float32)
        y=torch.tensor(self.Y[idx], dtype=torch.int64)
        x=x.reshape(1,-1)  # if network is CNN
        return x, y


X=pd.read_csv('ECG_dataX.csv')
Y=pd.read_csv('ECG_dataY.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)

dataset_train=MyDataset(X_train, Y_train)
dataset_val=MyDataset(X_val, Y_val)
dataset_test=MyDataset(X_test, Y_test)

loader_train = torch_dataloader(dataset_train, batch_size=128, shuffle=True, num_workers=0)
loader_val = torch_dataloader(dataset_val, batch_size=128, shuffle=False, num_workers=0)
loader_test = torch_dataloader(dataset_test, batch_size=128, shuffle=False, num_workers=0)


model = ResNet152(img_channels=1, num_classes=5)
model.to(device)

optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-4)

loss_train_list=[]
acc_train_list=[]
acc_val_list=[]

for epoch in range(0, 100): #change 100 to a larger number if necessary
    #-------- training --------------------------------
    loss_train, acc_train =train(model, device, optimizer, loader_train, epoch)
    loss_train_list.append(loss_train)
    acc_train_list.append(acc_train)
    print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
    #-------- validation --------------------------------
    acc_val, other_val = test(model, device, loader_val)
    acc_val_list.append(acc_val)
    print('epoch', epoch, 'validation acc:', acc_val)
    #--------save model-------------------------
    result = (loss_train_list, acc_train_list,
              acc_val_list, other_val)

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].set_title('loss v.s. epoch',fontsize=16)
ax[0].plot(loss_train_list, '-b', label='training loss')
ax[0].set_xlabel('epoch',fontsize=16)
ax[0].legend(fontsize=16)
ax[0].grid(True)
ax[1].set_title('accuracy v.s. epoch',fontsize=16)
ax[1].plot(acc_train_list, '-b', label='training accuracy')
ax[1].plot(acc_val_list, '-g', label='validation accuracy')
ax[1].set_xlabel('epoch',fontsize=16)
ax[1].legend(fontsize=16)
ax[1].grid(True)

acc, (confusion, sens, prec) = test(model, device, loader_test)
print('Accuracy (average)', acc)
print('Sensitivity', sens)
print('Precision', prec)
print('Confusion \n', confusion)