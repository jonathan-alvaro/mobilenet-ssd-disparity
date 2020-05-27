from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import convert_locations_to_boxes, center_to_corner
from .mobilenet import MobileNet
from .mobilenet_ssd_config import priors


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        out_channels = int(in_channels / 2)

        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.intermediate_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride= 1, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.intermediate_conv(x)
        x = self.point_conv(x)
        return x


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight)
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)


class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample1 = UpsamplingBlock(1024)
        self.upsample2 = UpsamplingBlock(512)
        self.upsample3 = UpsamplingBlock(256)
        self.upsample4 = UpsamplingBlock(128)
        self.upsample5 = UpsamplingBlock(64)

        self.pred_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.upsample1.apply(weights_init)
        self.upsample2.apply(weights_init)
        self.upsample3.apply(weights_init)
        self.upsample4.apply(weights_init)
        self.upsample5.apply(weights_init)
        self.pred_layer.apply(weights_init)

    def __call__(self, features: List[torch.Tensor]):
        """
        Performs multi-scale upsampling to produce a depth map
        """
        disparity1 = self.upsample1(features[0])
        disparity1 = F.interpolate(disparity1, size=(19, 19), mode='nearest')

        disparity2 = self.upsample2(disparity1)
        disparity2 = F.interpolate(disparity2, scale_factor=2, mode='nearest')
        disparity2 = disparity2 + features[1]

        disparity3 = self.upsample3(disparity2)
        disparity3 = F.interpolate(disparity3, size=(75, 75), mode='nearest')
        disparity3 = disparity3 + features[2]

        disparity4 = self.upsample4(disparity3)
        disparity4 = F.interpolate(disparity4, scale_factor=2, mode='nearest')
        disparity4 = disparity4 + features[3]

        disparity5 = self.upsample5(disparity4)
        disparity5 = F.interpolate(disparity5, scale_factor=2, mode='nearest')

        return self.pred_layer(disparity5)


class IntegratedModel(nn.Module):
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
        self.depth_source_layers = [1, 3, 5, 13]
        self.upsampling = DepthNet()
        self._test = is_test
        self._config = config

        if is_test:
            self._priors = priors.cuda()

        self.upsampling = self.upsampling

        self.extras.apply(_xavier_init_)
        self.class_headers.apply(_xavier_init_)
        self.location_headers.apply(_xavier_init_)

    def forward(self, x):
        sources = []
        depth_sources = []

        for i, net_layer in enumerate(self.extractor.get_layers()):
            x = net_layer(x)
            if i in self.source_layers:
                sources.append(x)
            if i in self.depth_source_layers:
                depth_sources.append(x)

        depth_sources = list(reversed(depth_sources))

        depth_sources = self.upsampling(depth_sources)

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
            confidences = F.softmax(confidences, dim=confidences.dim() - 1)
            boxes = convert_locations_to_boxes(locations, self._priors,
                                               self._config['variance'][0], self._config['variance'][1])
            boxes = center_to_corner(boxes)

            return confidences, boxes, depth_sources[-1]

        return confidences, locations, depth_sources

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
