import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda, device, __version__ as torch_version
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version
from torchvision import datasets, transforms

# Basic information
print("This is an experimental Hello World using ROCm Pytorch and DeepSpeed.")

# Print the versions of torch and torchvision
print(f"\tPyTorch Version: {torch_version}")
print(f"\tTorchvision Version: {torchvision_version}")

# Check if CUDA is available and set the device accordingly
device = device("cuda" if cuda.is_available() else "cpu")
print(f"\tUsing [cpu|cuda]: {device}")

# Check which actual device being utilized
gpu_name = cuda.get_device_name(0)
print(f"\tUsing device: {gpu_name}")

# Check how many devices are available
devices_available = cuda.device_count()
print(f"\tDevices Available {devices_available}")

# Define transformations and load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Create model, define loss function and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Evaluate the model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


import time

# Start time measurement
start_time = time.time()

# Run training and testing
for epoch in range(1, 11):
    train(epoch)
test()

# End time measurement
end_time = time.time()

# Calculate and print the total execution time
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# Save the model
torch.save(model.state_dict(), "saved_model/model_pytorch.pth")
