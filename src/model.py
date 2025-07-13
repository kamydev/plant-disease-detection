import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(PlantDiseaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # (3, 224, 224) -> (32, 224, 224)
        self.pool1 = nn.MaxPool2d(2, 2)                           # (32, 224, 224) -> (32, 112, 112)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, 112, 112)
        self.pool2 = nn.MaxPool2d(2, 2)                           # -> (64, 56, 56)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)# -> (128, 56, 56)
        self.pool3 = nn.MaxPool2d(2, 2)                           # -> (128, 28, 28)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class PlantDiseaseResNet(nn.Module):
    def __init__(self, num_classes=15):
        super(PlantDiseaseResNet, self).__init__()
        
        self.base_model = models.resnet18(pretrained=True)
        
        for param in self.base_model.parameters():
            param.requires_grad = False  # freeze all layers

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
