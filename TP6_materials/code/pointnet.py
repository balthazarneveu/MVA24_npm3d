
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 21/02/2023
#

from ply import write_ply, read_ply
import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
HERE = Path(__file__).parent

# Import functions to read and write ply files


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta),      0],
                               [math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(), RandomNoise(), ToTensor()])


def test_transforms():
    return transforms.Compose([ToTensor()])


class PointCloudData_RAM(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.data = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    ply_path = new_dir+"/"+file
                    data = read_ply(ply_path)
                    sample = {}
                    sample['pointcloud'] = np.vstack((data['x'], data['y'], data['z'])).T
                    sample['category'] = self.classes[category]
                    self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.data[idx]['pointcloud'])
        return {'pointcloud': pointcloud, 'category': self.data[idx]['category']}


class MLP(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.flattening = torch.nn.Flatten()
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, classes)

        self.stack_lin_nonlin = nn.Sequential(
            self.flattening,
            self.fc1, self.bn1, self.relu1,
            self.fc2, self.bn2, self.relu2,
            self.dropout, self.fc3)

    def forward(self, input):
        return self.stack_lin_nonlin(input)


# class PointNetBasic(nn.Module):
#     def __init__(self, classes = 10):
#         super().__init__()
#         # YOUR CODE

#     def forward(self, input):
#         # YOUR CODE


# class Tnet(nn.Module):
#     def __init__(self, k=3):
#         super().__init__()
#         # YOUR CODE

#     def forward(self, input):
#         # YOUR CODE


# class PointNetFull(nn.Module):
#     def __init__(self, classes = 10):
#         super().__init__()
#         # YOUR CODE

#     def forward(self, input):
#         # YOUR CODE

def basic_loss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def pointnet_full_loss(outputs, labels, m3x3, alpha=0.001):
    criterion = torch.nn.CrossEntropyLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)


def train(model, device, train_loader, test_loader=None, epochs=250, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss = 0
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1, 2))
            # outputs, m3x3 = model(inputs.transpose(1,2))
            loss = basic_loss(outputs, labels)
            # loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        test_acc = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs = model(inputs.transpose(1, 2))
                    # outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' % (epoch+1, loss, test_acc))


if __name__ == '__main__':

    t0 = time.time()
    ROOT_DIR = HERE.parent/"__data"/"ModelNet10_PLY"
    # ROOT_DIR = HERE.parent/"__data"/"ModelNet40_PLY"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    train_ds = PointCloudData_RAM(ROOT_DIR.as_posix(), folder='train', transform=default_transforms())
    test_ds = PointCloudData_RAM(ROOT_DIR.as_posix(), folder='test', transform=test_transforms())

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    batch_size = 32
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)

    model = MLP(classes=len(train_ds.classes))
    # model = PointNetBasic()
    # model = PointNetFull()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)

    train(model, device, train_loader, test_loader, epochs=100, lr=1E-3)

    t1 = time.time()
    print("Total time for training : ", t1-t0)
