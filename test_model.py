import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

label_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


# Define the same neural network architecture
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

# Load the trained model
model = Net()
# model.load_state_dict(torch.load("saved_model/model_pytorch.pth")) # Pytorch Normal
model.load_state_dict(torch.load("saved_model/model_pytorch_deepspeed.pth")) # DeepSpeed
model.eval()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transformations and load test data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Function to predict the image
def predict_image(img, model):
    img_batch = img.unsqueeze(0).to(device)
    outputs = model(img_batch)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Choose a random image from the test dataset
random_index = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[random_index]

# Predict the label
predicted_label = predict_image(image, model)

# Plot the image with actual and predicted labels
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label_map[label]}, Predicted: {label_map[predicted_label]}")
plt.show()
