

import os # operating system, handle file mgmt
import torch #tensors and nn
import torch.nn as nn
import torch.optim as optim #choosing learning optimizations
import torchvision #recognitions
import torchvision.transforms as transforms # normalization

# make sur to install: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-cache-dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# torch.set_default_device(device) ||| causes problems



#create transorm to convert data into tensors and normalize automatically (black and white pixels to -1 and 1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

#load dataset
#MNIST is a public data set of labled hand drawn numbers on a 28 by 28 pixel image
#train=true/false tells it to return either the training data or testing data
#batch size is to limit the incoming size, will be updating nn after each batch
training_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=64, shuffle=True)
# transform tells pytorch how to transform the data set, spesified transform above
testing_data_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=64, shuffle=True)

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

model = Net().to(device)

criterion = nn.CrossEntropyLoss() # loss function using softmax, finds the answer percentage of each number in each batch, then compares to percentage of each number in actual answer batch
optimizer = optim.SGD(model.parameters(), lr=0.009, momentum=0.9) # momentum give the optimizer a chance to go past the smaller "valleys"

#training loop
for epoch in range(10):
    running_loss = 0.0 #summing up the loss for every batch and at the end of each epoch print average loss

    #60000 training images / batches of 64
    # training_data_loader is already a list of a list of 64 images of 784 pixels + their labels
    for images, labels in training_data_loader:
        images, labels = images.to(device), labels.to(device) #send to gpu

        optimizer.zero_grad()

        #forward pass, get what model thinks images are
        outputs = model(images)

        #calculate our loss by comparing model outputs with the labels
        loss = criterion(outputs, labels)

        #performing gradient descent and backpropagation
        loss.backward()

        #apply optimizer
        optimizer.step()

        #addition to show what the current loss is for the running loss of the epoch
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Average loss: {running_loss / len(training_data_loader)}")


correct , total = 0, 0
with torch.no_grad(): # no longer training thats why we use torch.no_grad
    for images, labels in testing_data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # pass test images to the model

        _, predicted = torch.max(outputs, 1)  # getting highest digit

        correct += (predicted == labels).sum().item()  # sum up the number of times predicted and labels match

        total += labels.size(0)  # we add the batch size to total

print(f'Accuracy on the 10000 test images: {100 * correct / total}%')