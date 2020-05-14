from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import generate_priors, convert_locations_to_boxes, corner_to_center, center_to_corner
from .mobilenet import MobileNet
from .mobilenet_ssd_config import priors


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, expand_factor: int = 2, is_test: bool = False):
        super().__init__()
        self.expand = nn.PixelShuffle(2)

        out_channels = int(in_channels / 4)

        self.conv1 = nn.Conv2d(out_channels, 6 * out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(6 * out_channels, 6 * out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(6 * out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsampling1 = UpsamplingBlock(1024, 2)
        self.upsampling2 = UpsamplingBlock(768, 2)
        self.upsampling3 = UpsamplingBlock(448, 2)

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.bottleneck3 = nn.Sequential(
            nn.Conv2d(112, 224, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(224, 224, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(224, 112, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.prediction1 = nn.Conv2d(256, 1, kernel_size=3, padding=1, bias=False, stride=1)
        self.prediction2 = nn.Conv2d(192, 1, kernel_size=3, padding=1, bias=False, stride=1)
        self.prediction3 = nn.Conv2d(112, 1, kernel_size=3, padding=1, bias=False, stride=1)

    def __call__(self, features: List[torch.Tensor]):
        """
        Performs multi-scale upsampling to produce a depth map
        """
        disparities = []

        disparity1 = self.upsampling1(features[0])
        disparity1 = disparity1[..., 1:, 1:]
        disparity1 = self.bottleneck1(disparity1)
        disparities.append(torch.sigmoid(self.prediction1(disparity1)))
        disparity1 = torch.cat([disparity1, features[1]], dim=1)

        disparity2 = self.upsampling2(disparity1)
        disparity2 = disparity2[..., 1:, :]
        disparity2 = self.bottleneck2(disparity2)
        disparities.append(torch.sigmoid(self.prediction2(disparity2)))
        disparity2 = torch.cat([disparity2, features[2]], dim=1)

        disparity3 = self.upsampling3(disparity2)
        disparity3 = self.bottleneck3(disparity3)
        disparities.append(torch.sigmoid(self.prediction3(disparity3)))

        return disparities


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
        self.depth_source_layers = [5, 11, 13]
        self.upsampling = DepthNet()
        if not is_test:
            self.upsampling = self.upsampling.cuda()
        self._test = is_test
        self._config = config

        if is_test:
            self._priors = priors

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
