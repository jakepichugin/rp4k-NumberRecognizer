import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #overriding the parents

        #first hidden layer, 28 x28 is our 784 pixels our input,
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512) #2 hidden layers of 512 neurons
        self.fc3 = nn.Linear(512, 10) # 10 outputs for each number from 0 to 9

    def forward(self, x):
        #flatten of the input to a 1D tensor
        x = x.view(-1, 28*28) # this shows that we are turning it from (64, 1, 28, 28) to (64, 784)

        #passing the flattened inputs through the first hidden layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        #final result of the output layer with no activation
        return self.fc3(x)