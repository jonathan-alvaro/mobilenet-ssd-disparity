import torch
from torch import nn
import numpy as np


class CustomRMSE(nn.Module):
    def __init__(self, weights: np.ndarray):
        super().__init__()
        self.weights = weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        diff = prediction - target
        diff = diff ** 2

        weights = torch.Tensor(self.weights)
        pixel_weights = weights[target.flatten().long()]
        if prediction.is_cuda:
            pixel_weights = pixel_weights.cuda()

        diff = diff.flatten() * pixel_weights
        diff = diff.sum() / len(diff)
        return torch.sqrt(diff)
