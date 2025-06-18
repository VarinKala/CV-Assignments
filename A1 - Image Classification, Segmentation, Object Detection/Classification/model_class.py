import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Conv Layer:
        # 32 filters, 3x3 kernel, stride=1, padding=1
        # Max Pool (4x4, stride=4)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Second Conv Layer:
        # 64 filters, 3x3 kernel, stride=1, padding=1
        # Max Pool (2x2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Conv Layer:
        # 128 filters, 3x3 kernel, stride=1, padding=1
        # Max Pool (2x2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(128 * 14 * 14, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool1(out)

        out = self.relu(self.conv2(out))
        out = self.pool2(out)

        out = self.relu(self.conv3(out))
        out = self.pool3(out)

        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        for params in self.model.fc.parameters():
            params.requires_grad = True

    def forward(self, x):
        return self.model(x)