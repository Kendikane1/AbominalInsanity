#imports
import torch
import torch.nn as nn
import torch.optim as optim # all the optim algos like stochastic grad desc, adam, etc
import torch.nn.functional as F # all functions that dont have parameters such as relu,tanh, etc
from torch.utils.data import DataLoader # easier dataset management such as mini batches to train on
import torchvision.datasets as datasets # import pytorch's datasets
import torchvision.transforms as transforms # has transformations that can perform on dataset


# Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # (28x28) so 784 nodes is input size
        super(NN, self).__init__() # super calls initialisation method of parent class which is nn.module
        self.fc1 = nn.Linear(input_size, 70)
        self.fc2 = nn.Linear(70, 80)
        self.out = nn.Linear(80, num_classes) # so hidden layer is 70 and 80

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

'''
model = NN(784,10)
x = torch.randn(64 , 784)
print(model(x).shape) should output 64x10 for 64 images and probability for each digit 1-10 
'''

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10 # number of digit 0-9
learning_rate = 0.001
batch_size = 64
num_epochs = 15;'}}"'

# Load data
train_dataset = datasets.MNIST(root = 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset= train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root = 'dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size, shuffle=True)

# Initialise network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape (flatten it)
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimiser.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimiser.step()

# Check accuracy on training and test how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # lets pytorch know that you dont have to calc any gradients
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)



