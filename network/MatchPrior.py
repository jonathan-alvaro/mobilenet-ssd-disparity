import torch

from network.box_utils import center_to_corner, iou, corner_to_center, convert_boxes_to_locations, \
    convert_locations_to_boxes


class MatchPrior:
    def __init__(self, priors: torch.Tensor, config: dict):
        """
        priors: Generated prior boxes with format [cx, cy, w, h]
        config: Network config such as in mobilenet_ssd_config.py
        """
        self._priors = priors
        self._corner_priors = center_to_corner(priors)
        self._center_variance = config['variance'][0]
        self._size_variance = config['variance'][1]
        self._threshold = config['iou_threshold']

    def __call__(self, gt_boxes: torch.Tensor,
                 gt_labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        boxes, labels = self._assign_priors(gt_boxes, gt_labels,
                                            self._corner_priors, self._threshold)
        boxes = corner_to_center(boxes)
        locations = convert_boxes_to_locations(
            boxes, self._priors, self._center_variance, self._size_variance
        )

        return locations, labels

    def _assign_priors(self, gt_boxes: torch.Tensor,
                       gt_labels: torch.Tensor,
                       corner_priors: torch.Tensor,
                       threshold: float):
        """
        Assigns ground truth boxes to each prior box
        """
        # Size: num_priors * num_targets
        ious = iou(gt_boxes.unsqueeze(0), self._corner_priors.unsqueeze(1))

        # Size: num_priors
        best_target_per_prior, best_target_per_prior_index = ious.max(1)

        # Size: num_targets
        best_prior_per_target, best_prior_per_target_index = ious.max(0)

        # Give best prior for each target to ensure each target has at least 1 prior
        for target_index, prior_index in enumerate(best_prior_per_target_index):
            best_target_per_prior_index[prior_index] = target_index
        best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

        labels = gt_labels[best_target_per_prior_index]
        labels[best_target_per_prior < threshold] = 0  # Mark all matches below threshold as background
        boxes = gt_boxes[best_target_per_prior_index]
        return boxes, labels
