import torch
from torch import nn


class BerHuLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = prediction.squeeze()

        diff = prediction - target
        abs_diff = diff.abs()

        c = abs_diff.max().item() / 5
        cutoff_mask = (abs_diff > c)
        weights = torch.ones(abs_diff.shape)
        weights[(target > 10) * (target < 50)] *= 2
        #weights[(target > 10) & (target < 50)] *= 2
        if prediction.is_cuda:
            weights = weights.cuda()
        abs_diff[cutoff_mask] += c ** 2
        abs_diff[cutoff_mask] /= 2 * c
        abs_diff *= weights
        # abs_diff[target >= 50] /= 4
        # abs_diff[target <= 10] /= 5

        return abs_diff.flatten().sum() / weights.sum()
