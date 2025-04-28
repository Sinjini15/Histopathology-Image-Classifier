# src/model.py

import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
