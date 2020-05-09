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

        pixel_weights = torch.zeros_like(target.flatten())

        for i in range(len(self.weights)):
            pixel_weights[target.flatten() >= i] = self.weights[i]

        diff = diff.flatten() * pixel_weights
        diff = diff.sum() / len(diff)
        return torch.sqrt(diff)
