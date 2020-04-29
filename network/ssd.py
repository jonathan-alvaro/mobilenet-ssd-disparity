from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import generate_priors, convert_locations_to_boxes, corner_to_center, center_to_corner
from .mobilenet import MobileNet


class SSD(nn.Module):
    def __init__(self, num_classes: int, source_layer_indices: List[int], extractor: MobileNet,
                 extras: nn.ModuleList, location_headers: nn.ModuleList, class_headers: nn.ModuleList,
                 config: dict, is_test: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.source_layers = source_layer_indices
        self.extractor = extractor
        self.extras = extras
        self.location_headers = location_headers
        self.class_headers = class_headers
        self.image_size = config['image_size']
        self._test = is_test
        self._config = config

        if is_test:
            self._priors = generate_priors(self._config)

        self.extras.apply(_xavier_init_)
        self.class_headers.apply(_xavier_init_)
        self.location_headers.apply(_xavier_init_)

    def forward(self, x):
        sources = []

        for i, net_layer in enumerate(self.extractor.get_layers()):
            x = net_layer(x)
            if i in self.source_layers:
                sources.append(x)

        for layer_index, layer in enumerate(self.extras):
            x = layer(x)
            sources.append(x)

        confidences = []
        locations = []
        for i, s in enumerate(sources):
            confidences.append(self.compute_confidence(s, i))
            locations.append(self.compute_location(s, i))

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self._test:
            confidences = F.softmax(confidences, dim=2)
            boxes = convert_locations_to_boxes(locations, self._priors,
                                               self._config['variance'][0], self._config['variance'][1])
            boxes = center_to_corner(boxes)

            return confidences, boxes

        return confidences, locations

    def compute_confidence(self, x, i):
        confidence = self.class_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.shape[0], -1, self.num_classes)
        return confidence

    def compute_location(self, x, i):
        location = self.location_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.shape[0], -1, 4)
        return location


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
