import torch
import torch.nn.functional as F
from torch import nn

from network.box_utils import hard_negative_mining, convert_locations_to_boxes


class MultiBoxLoss(nn.Module):
    def __init__(self, iou_threshold: float, background_label: int,
                 mining_ratio: float, config: dict):
        """
        mining_ratio: ratio between negative and positive instances
        """
        super().__init__()
        self._num_classes = config['num_classes']
        self._threshold = iou_threshold
        self._background_label = background_label
        self._neg_pos_ratio = mining_ratio

    def forward(self, confidence, locations, labels, gt_locations):
        num_classes = confidence.size(2)

        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self._neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(
            confidence.reshape(-1, num_classes), labels[mask], reduction='none'
        )

        class_weights = torch.Tensor([1, 2, 1])
        if confidence.is_cuda:
            class_weights = class_weights.cuda()
        loss_weights = class_weights[labels[mask]]
        classification_loss = classification_loss * loss_weights
        classification_loss = classification_loss.sum()

        pos_mask = labels > 0
        locations = locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)

        mean_weights = class_weights[labels[pos_mask]]

        return smooth_l1_loss / num_pos, classification_loss / mean_weights.sum(), mask
