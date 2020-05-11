import torch
from torchvision import transforms
from PIL import Image

from network.box_utils import nms, convert_locations_to_boxes, generate_priors
from network.mobilenet_ssd_config import priors
from network.integratedmodel import IntegratedModel


class Predictor:
    def __init__(self, net: IntegratedModel,
                 iou_threshold: float = 0.5):
        self._net = net
        self._resize = transforms.Resize((400, 200))
        self._to_tensor = transforms.ToTensor()
        self._threshold = iou_threshold

    def predict(self, image: Image.Image, prob_threshold=0.25):
        width, height = image.size
        image = self._resize(image)
        image = self._to_tensor(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            conf, boxes, disparity = self._net.forward(image)
            boxes = boxes[0]
            conf = conf[0]
            disparity = disparity[0]

        picked_boxes = []
        picked_probs = []
        picked_labels = []
        indices = []

        prediction_labels = torch.argmax(conf, dim=conf.dim() - 1)

        for class_index in range(1, conf.size(-1)):
            probs = conf[..., class_index]
            mask = probs > prob_threshold
            class_mask = prediction_labels == class_index
            print(class_mask.shape)
            print(mask.shape)
            probs = probs[mask & class_mask]
            if probs.size(0) == 0:
                continue

            boxes_subset = boxes[mask, ...]

            chosen_indices = nms(boxes_subset, probs, self._threshold)

            picked_boxes.append(boxes_subset[chosen_indices, ...])
            picked_probs.append(conf[chosen_indices])
            picked_labels.extend([class_index] * chosen_indices.shape[0])
            indices.append(chosen_indices.clone())

        if len(picked_boxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), disparity, torch.tensor([])

        picked_boxes = torch.cat(picked_boxes)
        picked_boxes[..., 0] *= width
        picked_boxes[..., 2] *= width
        picked_boxes[..., 1] *= height
        picked_boxes[..., 3] *= height

        picked_boxes = picked_boxes.view((-1, 4))
        picked_labels = torch.tensor(picked_labels).view((-1, 1))
        picked_probs = torch.cat(picked_probs).view((-1, conf.shape[-1]))
        indices = torch.cat(indices).flatten()
        return picked_boxes, picked_labels, picked_probs, disparity, indices
