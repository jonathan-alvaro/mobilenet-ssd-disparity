import torch
from torch import nn


class BerHuLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.squeeze()
        target[target > 50] = 50

        diff = prediction - target
        abs_diff = diff.abs()

        c = abs_diff.max().item() / 5
        cutoff_mask = (abs_diff > c)
        abs_diff[cutoff_mask] += c ** 2
        abs_diff[cutoff_mask] /= 2 * c
        abs_diff = abs_diff[target != 0]

        return abs_diff.flatten().mean()
