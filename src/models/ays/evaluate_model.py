# File: src\models\ays\evaluate_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# -------------------------------
# Parameters
# -------------------------------
img_height, img_width = 64, 64
batch_size = 32

# -------------------------------
# Paths relative to this script
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

data_dir = os.path.join(PROJECT_ROOT, "data", "val")
model_path = os.path.join(PROJECT_ROOT, "outputs", "dnn_model", "dnn_model.pth")

# -------------------------------
# Check dataset and model
# -------------------------------
if not os.path.exists(data_dir):
    print(f"❌ ERROR: Dataset folder not found at {data_dir}")
    sys.exit(1)

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    sys.exit(1)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------------------
# Data preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

val_dataset = datasets.ImageFolder(data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

num_classes = len(val_dataset.classes)
print(f"✅ Classes detected: {val_dataset.classes}")

# -------------------------------
# Define the 3-layer DNN
# -------------------------------
class SimpleDNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * img_height * img_width, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
        return x

# -------------------------------
# Load trained model
# -------------------------------
model = SimpleDNN(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded from {model_path}")

# -------------------------------
# Evaluation
# -------------------------------
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"✅ Validation Accuracy: {accuracy:.4f}")
