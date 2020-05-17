import torch
from torch import nn


class BerHuLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.squeeze()

        diff = prediction - target
        abs_diff = diff.abs()

        c = abs_diff.max() / 5
        cutoff_mask = (abs_diff <= c)
        l2_loss = (diff ** 2 + c ** 2) / (2 * c)
        l2_loss[cutoff_mask] = 0
        abs_diff[~cutoff_mask] = 0

        print(l2_loss)
        print(abs_diff)

        loss = l2_loss + abs_diff
        return loss.flatten().mean()