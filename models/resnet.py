import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model_ft = models.resnet18(pretrained=True)
        num_feature = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_feature, num_classes)

    def forward(self, inputs):
        return self.model_ft(inputs)
