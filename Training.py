

import os # operating system, handle file mgmt
import torch #tensors and nn
import torch.nn as nn
import torch.optim as optim #choosing learning optimizations
import torchvision #recognitions
import torchvision.transforms as transforms # normalization
from Net import Net #importing from net.py
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



model = Net().to(device)

load_model = True # set to false if we want to train from new weights(scratch)

folder_path = "./models"
model_name = "MNIST_model.pth"
file_path = os.path.join(folder_path, model_name)

if load_model:
    state_dict = torch.load(file_path, weights_only=True)
    model.load_state_dict(state_dict)

torch.save(model.state_dict(), file_path) #saving the model using its dictionary

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
with torch.no_grad(): # no longer training that's why we use torch.no_grad
    for images, labels in testing_data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # pass test images to the model

        #torch.max gives us 2 outputs, we want only the 2nd one
        _, predicted = torch.max(outputs, 1)  # getting highest digit

        correct += (predicted == labels).sum().item()  # sum up the number of times predicted and labels match

        total += labels.size(0)  # we add the batch size to total

print(f'Accuracy on the 10000 test images: {100 * correct / total}%')


if not os.path.exists(folder_path):
    os.makedirs(folder_path)

torch.save(model.state_dict(), file_path) #saving the model using its dictionary