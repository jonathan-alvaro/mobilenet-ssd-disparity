import torch
import torchvision

from network.box_utils import nms
from network.ssd import SSD


class Predictor:
    def __init__(self, net: SSD, data_transform: torchvision.transforms.Compose,
                 iou_threshold: float = 0.5):
        self._net = net
        self._data_transform = data_transform
        self._threshold = iou_threshold

    def predict(self, image, prob_threshold=0.5):
        height, width, _ = image.shape
        image = self._data_transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            conf, boxes = self._net.forward(image)
            boxes = boxes[0]
            conf = conf[0]

        picked_boxes = []
        picked_probs = []
        picked_labels = []

        for class_index in range(1, conf.size(-1)):
            probs = conf[..., class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue

            boxes_subset = boxes[mask, ...]

            chosen_indices = nms(boxes_subset, probs, self._threshold)

            picked_boxes.append(boxes_subset[chosen_indices, ...])
            picked_probs.append(probs[chosen_indices, class_index])
            picked_labels.extend([class_index] * chosen_indices.shape[0])

        if len(picked_boxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        picked_boxes = torch.cat(picked_boxes)
        picked_boxes[..., 0] *= width
        picked_boxes[..., 2] *= width
        picked_boxes[..., 1] *= height
        picked_boxes[..., 3] *= height

        return picked_boxes, torch.tensor(picked_labels), torch.cat(picked_probs)
