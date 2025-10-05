# File: src/models/ays/predict_image.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys

# Choose model type: 'dnn' or 'cnn'
model_type = 'dnn'

# -------------------------------
# Parameters
# -------------------------------
img_height, img_width = 64, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Paths (fixed to project root)
# -------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.path.join(PROJECT_ROOT, "data", "train")

if model_type == 'dnn':
    from train_dnn_pytorch import SimpleDNN
    model_path = os.path.join(PROJECT_ROOT, "outputs", "dnn_model", "dnn_model.pth")
else:
    from train_cnn_pytorch import SimpleCNN
    model_path = os.path.join(PROJECT_ROOT, "outputs", "dnn_model", "cnn_model.pth")

# -------------------------------
# Load class names
# -------------------------------
if not os.path.exists(data_dir):
    print(f"❌ ERROR: Train folder not found at {data_dir}")
    sys.exit(1)

from torchvision import datasets
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
classes = dataset.classes
num_classes = len(classes)
print(f"✅ Classes: {classes}")

# -------------------------------
# Load model
# -------------------------------
model = (SimpleDNN(num_classes) if model_type == 'dnn' else SimpleCNN(num_classes)).to(device)

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model file not found at {model_path}")
    sys.exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded from {model_path}")

# -------------------------------
# Load and preprocess image
# -------------------------------
img_path = input("Enter path to image: ").strip()
if not os.path.exists(img_path):
    print("❌ File not found.")
    sys.exit(1)

image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0).to(device)

# -------------------------------
# Predict
# -------------------------------
with torch.no_grad():
    output = model(image)
    pred_class = output.argmax(1).item()

print(f"✅ Predicted Class: {classes[pred_class]}")
