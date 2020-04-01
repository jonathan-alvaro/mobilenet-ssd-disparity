from typing import List

import torch
import torch.nn as nn
from .mobilenet import MobileNet


class SSD(nn.Module):
    def __init__(self, num_classes: int, source_layer_indices: List[int], extractor: MobileNet,
                 location_headers: nn.ModuleList, class_headers: nn.ModuleList):
        super().__init__()

        self.num_classes = num_classes
        self.source_layers = source_layer_indices
        self.extractor = extractor
        self.location_headers = location_headers
        self.class_headers = class_headers

    def forward(self, x):
        start_layer = 0

        confidences = []
        locations = []

        for feature_index, feature_layer in enumerate(self.source_layers):
            for net_layer in self.extractor[start_layer:feature_layer]:
                x = net_layer(x)

                confidence = self.compute_confidence(x, feature_index)
                confidences.append(confidence)

                location = self.compute_location(x, feature_index)
                locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

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
