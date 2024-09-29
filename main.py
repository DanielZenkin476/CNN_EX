import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# select cuda cores
print(device)#check device