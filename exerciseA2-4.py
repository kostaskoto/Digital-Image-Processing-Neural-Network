import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tabulate import tabulate
import matplotlib.pyplot as plt

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=0)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.avgpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Set the hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the neural network model
model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#Accuracy and Loss 
accT = []
lossT = []

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        # output: B x N --> argmax(dim=-1) : Nx1
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # Print training progress
        # if (i+1) % 100 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
    
    # Evaluate the model on the test dataset
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            # images.shape : B x C x H x W
            # B: batch size, C: channel, H: height, W: width
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
        acc = 100 * correct / total
        accT.append(acc)
        lossT.append(loss.item())
        print(f"Epoch [{epoch+1}/{num_epochs}]: test accuracy of the model on the test images: {acc} %")
        print(f"Epoch [{epoch+1}/{num_epochs}]: test loss of the model: {loss.item():.4f}")

# Print graphs of the accuracy and loss
plt.plot(accT)
plt.plot(lossT)
plt.legend(['Accuracy', 'Loss'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')

# Create the confusion matrix 
confusion_matrix = torch.zeros(10, 10)
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=-1)
        for i in range(len(labels)):
            confusion_matrix[labels[i]][predicted[i]] += 1

# Print the confusion matrix
print(tabulate(confusion_matrix, headers=[0,1,2,3,4,5,6,7,8,9], tablefmt='orgtbl'))
plt.show()
