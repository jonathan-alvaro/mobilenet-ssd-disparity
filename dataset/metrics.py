import torch


def relative_absolute_error(prediction: torch.Tensor, target: torch.Tensor):
    abs_diff = (prediction - target).abs()
    abs_diff /= prediction

    return abs_diff.mean()


def pixel_miss_error(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 3.0):
    diff = (prediction - target).abs()
    error_count = (diff > threshold).long().sum()

    return error_count / prediction.flatten().shape[0]



