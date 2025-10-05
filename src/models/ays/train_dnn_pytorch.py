# File: src/train_dnn_pytorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys

# -------------------------------
# Parameters
# -------------------------------
img_height, img_width = 64, 64
batch_size = 32
epochs = 3

# Use absolute path to avoid relative path issues
data_dir = r"D:\tea-leaf-disease-detection\data\train"
output_dir = r"D:\tea-leaf-disease-detection\outputs\dnn_model"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# -------------------------------
# Dataset check
# -------------------------------
if not os.path.exists(data_dir):
    print(f"❌ ERROR: Dataset folder not found at {data_dir}")
    sys.exit(1)

# -------------------------------
# Data preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(data_dir, transform=transform)

# Train/validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

num_classes = len(train_dataset.classes)
print("✅ Classes found:", train_dataset.classes)

# -------------------------------
# 3-Layer Fully Connected DNN
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

model = SimpleDNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
    train_acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader.dataset):.4f} - Acc: {train_acc:.4f}")

# -------------------------------
# Save model
# -------------------------------
model_path = os.path.join(output_dir, "dnn_model.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at {model_path}")
