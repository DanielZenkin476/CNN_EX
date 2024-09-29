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
                   ])), batch_size=64, shuffle=True)# shuffle to randomize , batch size 64 as defualt
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((norm_mean,), (norm_std,))
                   ])), batch_size=1000, shuffle=True)


# now to print result
print(train_loader.dataset.__len__())# so pycharm is telling no such function exsists- guess not
print(test_loader.dataset.__len__())
#image shape
data_sample = train_loader.dataset.__getitem__(0)#gets first image
print(f'data sample shape is: {data_sample[0].shape}, with the label: {data_sample[1]}')
print(data_sample[0].min(), data_sample[0].max())

# now to vizualize the data
plt.figure(figsize=(16,6))
for i in(range(10)):
    plt.subplot(2,5,i+1)
    image, label = train_loader.dataset.__getitem__(i)
    plt.imshow(image.reshape(28,28), cmap='gray')#reshape data
    plt.axis('off')
plt.show()#show the plot

# now to permitate the data
perm = torch.randperm(input_size)
plt.figure(figsize=(16,12))
for i in(range(10)):
    #get image
    image, label = train_loader.dataset.__getitem__(i)
    #permutate pixels
    image_perm = image.view(input_size)# to row
    image_perm = image_perm[perm]# permutate pixels
    image_perm = image_perm.view(28,28)#change to 28x28
    plt.subplot(4,5,i+1)
    plt.imshow(image.reshape(28,28), cmap='gray')#reshape data
    plt.axis('off')
    plt.subplot(4, 5, i + 11)
    plt.imshow(image_perm, cmap='gray')# no need to reshape
    plt.axis('off')
plt.show()


