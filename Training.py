

import os # operating system, handle file mgmt
import torch #tensors and nn
import torch.nn as nn
import torch.optim as optim #choosing learning optimizations
import torchvision #recognitions
import torchvision.transforms as transforms # normalization


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#create transorm to convert data into tensors and normalize automatically (black and white pixels to -1 and 1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

#load dataset
#MNIST is a public data set of labled hand drawn numbers on a 28 by 28 pixel image
#train=true/false tells it to return either the training data or testing data
#batch size is to limit the incoming size, will be updating nn after each batch
training_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=64, shuffle=True)
# transform tells pytorch how to transform the data set, spesified transform above
testing_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=True)