import torch
from torch import nn


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(torch.sum(torch.square(torch.log(target) - output)))
