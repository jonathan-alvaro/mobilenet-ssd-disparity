import math
from itertools import product

import torch


def generate_priors(config: dict) -> torch.Tensor:
    priors = []

    for layer_idx, map_size in enumerate(config['feature_size']):
        for row, col in product(range(map_size), repeat=2):
            width_shrinkage = config['width'] / config['shrink_factor'][layer_idx]
            height_shrinkage = config['height'] / config['shrink_factor'][layer_idx]

            # Calculate center for box in position (row, col)
            cx = (col + 0.5) / width_shrinkage
            cy = (row + 0.5) / height_shrinkage

            # Create small box
            w = config['min_size'][layer_idx] / config['width']
            h = config['min_size'][layer_idx] / config['height']
            priors.append([cx, cy, w, h])

            # Create boxes with pre-defined aspect ratios
            for ratio in config['aspect_ratio'][layer_idx]:
                factor = math.sqrt(ratio)
                priors.append([cx, cy, w * factor, h / factor])
                priors.append([cx, cy, w / factor, h * factor])

            # Create big box
            size = math.sqrt(config['max_size'][layer_idx] * config['min_size'][layer_idx])
            w = size / config['width']
            h = size / config['height']
            priors.append([cx, cy, w, h])

            for ratio in config['aspect_ratio'][layer_idx]:
                factor = math.sqrt(ratio)
                priors.append([cx, cy, w * factor, h / factor])
                priors.append([cx, cy, w / factor, h * factor])

    priors = torch.tensor(priors).view(-1, 4)
    # Ensure all prior boxes lie within the image
    torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def area(left_top: torch.Tensor, right_bottom: torch.Tensor) -> torch.Tensor:
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a and b should be in corner form, that is [min_x, min_y, max_x, max_y]
    """
    left_top_overlap = torch.max(a[..., :2], b[..., :2])
    right_bottom_overlap = torch.min(a[..., 2:], b[..., 2:])

    overlap_area = area(left_top_overlap, right_bottom_overlap)
    area_a = area(a[..., :2], a[..., 2:])
    area_b = area(b[..., :2], b[..., 2:])
    return overlap_area.float() / (area_a.float() + area_b.float() - overlap_area.float())


def center_to_corner(boxes):
    """
    Convert boxes from [cx, cy, w, h] to [left, top, right, bottom]
    """
    return torch.cat(
        [(boxes[..., :2] - boxes[..., 2:] / 2),  # centers - half of width and height
         (boxes[..., :2] + boxes[..., 2:] / 2)],  # centers + half of width and height
        boxes.dim() - 1
    )


def corner_to_center(boxes):
    """
    Convert boxes from [left, top, right, bottom] to [cx, cy, w, h]
    """
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2,
         (boxes[..., 2:] - boxes[..., :2])],
        boxes.dim() - 1
    )


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def convert_locations_to_boxes(locations, priors, center_var, size_var):
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    return torch.cat([
        locations[..., :2] * center_var * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_var) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def hard_negative_mining(loss: torch.Tensor, labels: torch.Tensor, ratio: float):
    """
    Because the identified objects will mostly be negatives, this function suppresses most of the negatives
    to ensure a balanced ratio between negatives and positives in the loss calculation
    """

    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = ratio * num_pos

    loss[pos_mask] = -math.inf
    _, indices = loss.sort(dim=1, descending=True)
    _, orders = indices.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def nms(boxes: torch.Tensor, probs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Non-Max Suppression for prediction results
    """

    _, prob_ranking = probs.sort(descending=True)

    picked_boxes = []

    while prob_ranking.shape[0] > 0:
        current_index = prob_ranking[0]
        picked_boxes.append(current_index.item())

        current_box = boxes[current_index, :]
        prob_ranking = prob_ranking[1:]
        left_boxes = boxes[prob_ranking, :]

        ious = iou(left_boxes, current_box.unsqueeze(0))
        prob_ranking = prob_ranking[ious <= iou_threshold]

    return torch.tensor(picked_boxes).long()
