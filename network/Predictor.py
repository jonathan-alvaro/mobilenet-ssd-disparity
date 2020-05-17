import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from network.box_utils import nms, convert_locations_to_boxes, generate_priors
from network.mobilenet_ssd_config import priors
from network.integratedmodel import IntegratedModel
from network.transforms import *


class Predictor:
    def __init__(self, net: IntegratedModel,
            iou_threshold: float = 0.5, use_cuda: bool = False):
        self._net = net
        self._resize = transforms.Resize((200, 400))
        self._pre_transform = Compose([
            Resize(400, 200),
            Scale(),
            ToTensor()
        ])
        self._to_tensor = transforms.ToTensor()
        self._threshold = iou_threshold
        self._cuda = use_cuda

        if use_cuda:
            self._net = self._net.cuda()

    def predict(self, image: Image.Image, prob_threshold=0.5):
        width, height = image.size
        image = np.array(image)
        image, _, _, _ = self._pre_transform(image)
        image = image.unsqueeze(0)

        if self._cuda:
            image = image.cuda()

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
            probs = probs[class_mask & mask]
            if probs.size(0) == 0:
                print("Class {} has no valid predictions".format(class_index))
                print("Max prob is {}".format(conf[..., class_index].max()))
                continue

            boxes_subset = boxes[class_mask & mask, ...]

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
