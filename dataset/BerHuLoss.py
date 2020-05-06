import torch
from torch import nn


class BerHuLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        diff = prediction - target
        abs_diff = diff.abs()

        c = abs_diff.flatten(start_dim=1).max(dim=1)[0] / 5
        c = c.view((-1, 1, 1))
        c = c.repeat(diff.shape)
        cutoff_mask = (abs_diff <= c)
        l2_loss = (diff ** 2 + c ** 2) / (2 * c)
        l2_loss[cutoff_mask] = 0

        loss = l2_loss + abs_diff
        return loss.flatten().mean()
