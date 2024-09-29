import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# select cuda cores
print(device)#check device

#using MINIST dataset

input_size = 28*28 # image is 28x28 pixels
output_size = 10 # 10 classes - 0 to 9
norm_mean = 0.5
norm_std = 0.5
# download data
train_loader = data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((norm_mean,), (norm_std,))
                   ])), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((norm_mean,), (norm_std,))
                   ])), batch_size=1000, shuffle=True)


# now to print result
print(train_loader.dataset.__len__())# so pycharm is telling no such function exsists- guess not
print(test_loader.dataset.__len__())
#image shape
data_sample = train_loader.dataset.__getitem__(0)
print(f'data sample shape is: {data_sample[0].shape}, with the label: {data_sample[1]}')
print(data_sample[0].min(), data_sample[0].max())
