import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import cuda, device, __version__ as torch_version
from torchvision import __version__ as torchvision_version
import deepspeed

# Basic information
print("Experimental Hello World using ROCm PyTorch and DeepSpeed.")
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

# Create model and define loss function
model = Net()
criterion = nn.CrossEntropyLoss()

# DeepSpeed Configuration with ZeRO
ds_config = {
    "train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    # "zero_optimization": {
    #     "stage": 2,  # Adjust the ZeRO stage here (0, 1, 2, or 3)
    #     "allgather_partitions": True,
    #     "allgather_bucket_size": 2e8,
    #     "reduce_scatter": True,
    #     "reduce_bucket_size": 2e8,
    #     "overlap_comm": True,
    #     "contiguous_gradients": True,
    #     "cpu_offload": True  # Set to True if you want to enable CPU offloading
    # }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(args=None, model=model, model_parameters=model.parameters(), config_params=ds_config)

# Train the model using DeepSpeed
def train(epoch):
    model_engine.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(model_engine.local_rank), target.to(model_engine.local_rank)
        optimizer.zero_grad()
        output = model_engine(data)
        loss = criterion(output, target)
        model_engine.backward(loss)
        model_engine.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

import time

# Start time measurement
start_time = time.time()

# Run training and testing
for epoch in range(1, 11):
    train(epoch)

# End time measurement
end_time = time.time()

# Calculate and print the total execution time
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# Save the model using DeepSpeed's method
model_engine.save_checkpoint('./checkpoint')

# Save the model
torch.save(model.state_dict(), "saved_model/model_pytorch_deepspeed.pth")

# Note: The test function and evaluation should be adjusted for DeepSpeed if necessary
