import torch.nn as nn
import torch.nn.functional as F
import torch


def SoftLabelCrossEntropy(inputs, targets, T):
    inputs = F.softmax(inputs / T, dim=1)
    targets = F.softmax(targets / T, dim=1)
    return -torch.sum(torch.log(inputs) * targets) / inputs.size()[0]
