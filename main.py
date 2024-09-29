import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy
from torchsummary import summary
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# select cuda cores
print(device)#check device

#using MINIST dataset

input_size = 28*28 # image is 28x28 pixels
output_size = 10 # 10 classes - 0 to 9
norm_mean = 0.5
norm_std = 0.5
epochs = 3
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

# now onto models

#model 1 - fully connected , 3layer output is log softmax -

class FC2Layer( nn.Module):
    def __init__(self, input_size,n_hidden, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, n_hidden),# layer 1  ixn
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),# layer 2  nxn
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),  #layer 3 nxo
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):# forward path
        x= x.view(-1, self.input_size)#flatten data
        return self.network(x)# to network

n_hidden = 8
# param = n_hidden *(28*28 +1) = 6280

model_fnn = FC2Layer(input_size, n_hidden, output_size).to(device)# send to device
print(model_fnn)# print model
summary(model_fnn, input_size=(1,28*28))# summary of data

# second model - CNN(the good kind)
#kernal size is 5 , input channels (1 - greyscale)
class CNN(nn.Module):
    def __init__(self, input_size,n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature*4*4, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x): # layer 1
        x = self.conv1(x) # 28x 28 x1 -> 24(= 28-5+1)x24(= 28-5+1)xn_features
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2)# maxpool reduce size to 12x12xn_features
        # layer 2
        x = self.conv2(x)  # 12x12x1 -> 8(= 12-5+1)x8(= 12-5+1)xn_features
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)  # maxpool reduce size to 4x4xn_features

        x = x.view(-1,self.n_feature*4*4)# flatten data
        x = self.fc1(x)#fc layer 1
        x = F.relu(x)
        x = self.fc2(x)#fc layer 2
        x = F.log_softmax(x, dim=1)# Softmax layer
        return x

n_features = 6  # number of feature maps

model_cnn = CNN(input_size, n_features, output_size).to(device)
summary(model_cnn, input_size=(1, 28, 28))

# now that we have the modeles ,need to create traning function

criterion = torch.nn.NLLLoss()
def train(model, optimizer, perm = None):
    model.train()#train model
    for epoch in range(epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for batch_idx, (data, label) in enumerate(tepoch):
                tepoch.set_description(f'Epoch {epoch}')
                #send to device
                data ,label = data.to(device), label.to(device)
                #permutate pixels
                if perm is not None:
                    data = data.view(-1,28,28)
                    data = data[:,perm]
                    data = data.view(-1,1,28,28)

                optimizer.zero_grad()#zero optimizer gradients
                output = model(data)# get output
                loss = criterion(output,label)# get loss
                loss.backward()# backprop
                optimizer.step()#forward

                tepoch.set_postfix(
                    batch_loss=loss.item())  # Tensor.item() → number returns the value of this tensor as a standard Python number. This only works for tensors with one element.

def test(model):
    model.eval()
    test_loss = 0
    correct = 0

    with tqdm(test_loader, unit='batch') as loader:
        for data, label in loader:
             #send to device
            data,label = data.to(device),label.to(device)
             # permutate pixels
            if perm is not None:
                data = data.view(-1, 28, 28)
                data = data[:, perm]
                data = data.view(-1, 1, 28, 28)
            output = model(data)
            print(f"output shape: {output.shape}, label shape: {label.shape}")
            test_loss += F.nll_loss(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()# add if pred is corret

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(
            f'\nTest set: Average loss: {round(test_loss, 2)}, Accuracy: {correct}/{len(test_loader.dataset)} ({round(accuracy)}%)')
        return accuracy

results = {}
print("FNN:")
optimizer_fnn = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5)
train(model_fnn, optimizer_fnn)
results['NN image'] = test(model_fnn)
print("CNN")
optimizer_cnn = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
train(model_cnn, optimizer_cnn)
results['CNN image'] = test(model_cnn)
print(results)

