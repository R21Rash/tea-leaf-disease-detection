
from typing import Tuple
import torch.nn as nn
from torchvision import models
from .baseline_cnn import BaselineCNN

def build_model(name: str, num_classes: int, img_size: int = 224, freeze_backbone: bool = True):
    name = name.lower()
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.features.parameters():
                p.requires_grad = False
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in m.layer1.parameters(): p.requires_grad = False
            for p in m.layer2.parameters(): p.requires_grad = False
            for p in m.layer3.parameters(): p.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in m.features.parameters():
                p.requires_grad = False
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    elif name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
