import torch
from torch import nn


class KLLoss(nn.Module):
    def __init__(self, num_classes):
        super(KLLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        torch.
        return loss + high_cost