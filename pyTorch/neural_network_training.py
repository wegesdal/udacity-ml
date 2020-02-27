

#%%
# # Import necessary packages
%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch


import matplotlib.pyplot as plt


#%%

### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#%%

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

#%%

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

# %%
## Your solution
# Define the size of each layer in our network
n_input = 784     # Number of input units, must match number of input features
n_hidden = 256                    # Number of hidden units 
n_output = 10                    # Number of output units

## Solution
def activation(x):
    return 1/(1+torch.exp(-x))

# Flatten the input images
inputs = images.view(images.shape[0], -1)

# Create parameters
w1 = torch.randn(n_input, n_hidden)
b1 = torch.randn(n_hidden)

w2 = torch.randn(n_hidden, n_output)
b2 = torch.randn(n_output)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

# %%


# the list of values given by the softmax function.
def softmax(x):

    # the denominator needs to be shaped as a 64 x 1 so that division happens elementwise by row
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# Does it have the right shape?
print(probabilities.shape)
print(probabilities.sum(dim=1))

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.h1 = nn.Linear(784, 128)
        self.h2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)


    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        #Output layer with softmax activation
        x = F.relu(self.output(x), dim=1)
        
        return x

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Build a feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),  
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 10),
)

# Define our data loss
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainloader))

# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)

# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)

# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Build a feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),  
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1))

# Define our data loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainloader))

# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logps = model(images)


# %%

from torch import optim

#Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.01)

# %%

print('Initial weights - ', (model[0].weight))

i = 0
while i < 100:
    images, labels = next(iter(trainloader))

    images.resize_(64, 784)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # what would happen if you didn't zero them out? (momentum) we could divide these instead to keep some of the previous gradient

    # Forward pass, then backward pass, then update weights
    output = model.forward(images)
    loss = criterion(output, labels)
    loss.backward()
    print('Gradient -', model[0].weight.grad)

    optimizer.step()
    print('Updated weights - ', model[0].weight)

    i += 1


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))
                                ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Build a feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128), 
    nn.ReLU(),  
    nn.Linear(128, 64), 
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1))

# Define our data loss
criterion = nn.NLLLoss()

from torch import optim

#Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.03)

epochs = 5

for e in range(epochs):
    running_loss = 0


    for images, labels in trainloader:

        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()

        # what would happen if you didn't zero them out? (momentum) we could divide these instead to keep some of the previous gradient

        # Forward pass, then backward pass, then update weights
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# %%

