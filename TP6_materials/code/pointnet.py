
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

import matplotlib.pyplot as plt
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


class AnisotropicScale(object):
    """
    This applies slight random rescale in each axis
    
    """
    def __call__(self, pointcloud):
        scale_factor = 0.1
        scale = np.random.uniform(1-scale_factor, 1+scale_factor, (3))
        scale_matrix = np.array([[scale[0], 0, 0],
                                [0, scale[1], 0],
                                [0, 0, scale[2]]])
        scaled_pointcloud = scale_matrix.dot(pointcloud.T).T
        return scaled_pointcloud


class RandomRepeat(object):
    """
    This randomly removes a proportion of points, and duplicate
    others to keep the same number of points.
    
    
    """
    def __call__(self, pointcloud):
        ablation_rate = 0.1
        np.random.shuffle(pointcloud)
        repeat_size = np.random.randint(0, int(ablation_rate*1024))
        if repeat_size > 0:
            pointcloud[-repeat_size:] = pointcloud[:repeat_size]
        return pointcloud


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


def custom_transforms():
    return transforms.Compose([RandomRotation_z(), RandomNoise(), AnisotropicScale(), ToTensor()])


def custom_transforms_repeat():
    return transforms.Compose([RandomRepeat(), RandomRotation_z(), RandomNoise(), AnisotropicScale(), ToTensor()])


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
        return self.stack_lin_nonlin(input), None


class BaseBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.base_conv = torch.nn.Sequential(
            torch.nn.Conv1d(ch_in, ch_out, 1),
            torch.nn.BatchNorm1d(ch_out),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.base_conv(input)


class PointNetBasic(nn.Module):
    def __init__(self, classes: int = 10, ch_in: int = 3, h_dim: int = 64, tnet_in: bool = False):
        super().__init__()
        self.tnet_in = tnet_in
        if self.tnet_in:
            self.tnet = Tnet(ch_in=ch_in)
        self.l1 = BaseBlock(ch_in, h_dim)
        self.l2 = BaseBlock(h_dim, h_dim)
        self.l3 = BaseBlock(h_dim, h_dim)
        self.l4 = BaseBlock(h_dim, 2*h_dim)
        self.l5 = BaseBlock(2*h_dim, 1024)
        self.basic_pointnet = torch.nn.Sequential(
            self.l1, self.l2, self.l3, self.l4, self.l5
        )
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

        self.non_lin = torch.nn.ReLU()
        self.g_linear_1 = torch.nn.Linear(1024, 512)
        self.bn_1 = torch.nn.BatchNorm1d(512)
        self.g_linear_2 = torch.nn.Linear(512, 256)
        self.bn_2 = torch.nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.g_linear_3 = torch.nn.Linear(256, classes)
        # self.dropout, # after linear layer
        self.mlp = torch.nn.Sequential(
            self.g_linear_1, self.bn_1, self.non_lin,
            self.g_linear_2, self.bn_2, self.non_lin,
            self.dropout,  # after linear layer
            self.g_linear_3
        )

    def forward(self, input):
        if self.tnet_in:
            input, mat = self.tnet(input)
        else:
            mat = None
        out = self.basic_pointnet(input)
        out = self.pool(out).squeeze(-1)
        out = self.mlp(out)
        return out, mat


class Tnet(nn.Module):
    def __init__(self, ch_in=3, k=3):
        super().__init__()
        self.l1 = BaseBlock(ch_in, 64)
        self.l2 = BaseBlock(64, 128)
        self.l3 = BaseBlock(128, 1024)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)

        self.non_lin = torch.nn.ReLU()
        self.g_linear_1 = torch.nn.Linear(1024, 512)
        self.bn_1 = torch.nn.BatchNorm1d(512)
        self.g_linear_2 = torch.nn.Linear(512, 256)
        self.bn_2 = torch.nn.BatchNorm1d(256)
        self.g_linear_3 = torch.nn.Linear(256, k*k)
        self.extractor = torch.nn.Sequential(
            self.l1, self.l2, self.l3,
            self.pool
        )
        self.mlp = torch.nn.Sequential(
            self.g_linear_1, self.bn_1, self.non_lin,
            self.g_linear_2, self.bn_2, self.non_lin,
            self.g_linear_3
        )
        self.k = k

    def forward(self, input):
        out = self.extractor(input)
        out = self.mlp(out.squeeze(-1))
        mat = out.view(-1, self.k, self.k)
        out = torch.bmm(mat, input)
        return out, mat


# model = Tnet()
# out, mat = model(torch.rand(32, 3, 1024))
# print(out.shape, mat.shape)


class PointNetFull(PointNetBasic):
    def __init__(self, **kwargs):
        super().__init__(tnet_in=True, **kwargs)


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
    accuracies = []
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3 = model(inputs.transpose(1, 2))
            # outputs, m3x3 = model(inputs.transpose(1,2))
            
            if m3x3 is None:
                loss = basic_loss(outputs, labels)
            else:
                loss = pointnet_full_loss(outputs, labels, m3x3)
            
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
                    outputs, mat3x3 = model(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100. * correct / total
            accuracies.append(test_acc)
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' % (epoch+1, loss, test_acc))
    return accuracies


def train_(model, transforms, epochs, lr):
    train_ds = PointCloudData_RAM(ROOT_DIR.as_posix(), folder='train', transform=transforms)
    test_ds = PointCloudData_RAM(ROOT_DIR.as_posix(), folder='test', transform=test_transforms())
    
    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    batch_size = 512 # 64
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))
    model.to(device)

    accuracies = train(model, device, train_loader, test_loader, epochs=epochs, lr=lr)

    t1 = time.time()
    print("Total time for training : ", t1-t0)
    
    return accuracies
    

if __name__ == '__main__':

    t0 = time.time()
    n_classes = 10 # 10 or 40
    assert n_classes in [10, 40]
    
    
    ROOT_DIR = HERE.parent/"__data"/f"ModelNet{n_classes}_PLY"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    chosen_transforms = default_transforms()
    # chosen_transforms = custom_transforms_repeat()
    # chosen_transforms = custom_transforms()


    #%% Q1
    mlp_model = MLP(classes=n_classes)
    mlp_accuracies = train_(mlp_model, default_transforms(),
                            epochs=75, lr=1e-2)

    plt.figure()
    plt.plot(mlp_accuracies, label="mlp")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    
    torch.cuda.empty_cache()
    #%% Q2
    basic_pointnet_model = PointNetBasic(classes=n_classes)
    basic_pointnet_accuracies = train_(basic_pointnet_model, default_transforms(),
                            epochs=100, lr=1e-3)

    plt.figure()
    plt.plot(basic_pointnet_accuracies, label="basic Pointnet")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
  
    torch.cuda.empty_cache()
    #%% Q3
    full_pointnet_model = PointNetFull(classes=n_classes)
    full_pointnet_accuracies = train_(full_pointnet_model, default_transforms(),
                            epochs=100, lr=5e-3)
    plt.figure()
    plt.plot(full_pointnet_accuracies, label="full Pointnet")
    plt.plot(basic_pointnet_accuracies, label="basic Pointnet")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    
    torch.cuda.empty_cache()
    
    #%% Q4
    epochs = 100
    lr = 5e-3
    
    full_pointnet_model = PointNetFull(classes=n_classes)
    default_accuracies = train_(full_pointnet_model, default_transforms(),
                        epochs=epochs, lr=lr)
    torch.cuda.empty_cache()
    
    full_pointnet_model = PointNetFull(classes=n_classes)
    custom_accuracies = train_(full_pointnet_model, custom_transforms(),
                        epochs=epochs, lr=lr)
    torch.cuda.empty_cache()
    
    full_pointnet_model = PointNetFull(classes=n_classes)
    custom_repeat_accuracies = train_(full_pointnet_model, custom_transforms_repeat(),
                        epochs=epochs, lr=lr)
    
    
    #%% Q4 plot
    default_accuracies_ = np.array(default_accuracies) 
    custom_accuracies_ = np.array(custom_accuracies) 
    custom_repeat_accuracies_ = np.array(custom_repeat_accuracies)
    
    def smooth(x):
        
        x[1:-1] = 0.25 * (x[:-2] +
                          2 * x[1:-1] +
                          x[2:]
                          )
        return x
    
    default_accuracies_ = smooth(smooth(default_accuracies_))
    custom_accuracies_ = smooth(smooth(custom_accuracies_))
    custom_repeat_accuracies_ = smooth(smooth(custom_repeat_accuracies_))
    
    
    plt.figure()
    plt.plot(default_accuracies_, label="default")
    plt.plot(custom_accuracies_, label="rescale")
    plt.plot(custom_repeat_accuracies_, label="rescale + decimate")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    
    torch.cuda.empty_cache()
    
