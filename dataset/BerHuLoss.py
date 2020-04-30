import torch
from torch import nn


class BerHuLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        diff = prediction - target
        abs_diff = diff.abs()

        c = abs_diff.max() / 5
        cutoff_mask = (abs_diff <= c)
        l2_loss = (diff ** 2 + c ** 2) / (2 * c)

        loss = cutoff_mask.long().float() * abs_diff + (~cutoff_mask).long().float() * l2_loss
        return loss / loss.flatten().shape[0]
