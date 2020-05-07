from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import generate_priors, convert_locations_to_boxes, corner_to_center, center_to_corner
from .mobilenet import MobileNet
from .mobilenet_ssd_config import priors


def agg_node(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def smooth(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    )

def predict(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid(),
    )

def upshuffle(in_planes, out_planes, upscale_factor):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(upscale_factor),
        nn.ReLU()
    )


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
        self.image_size = config['image_size']

        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.agg1 = agg_node(256, 128)
        self.agg2 = agg_node(256, 128)
        self.agg3 = agg_node(256, 128)

        self.up1 = upshuffle(128, 128, 4)
        self.up2 = upshuffle(128, 128, 2)

        self.predict1 = smooth(384, 128)
        self.predict2 = predict(128, 1)

        self.depth_sources_indices = [3, 5, 7]

        if not is_test:
            self.upsampling = self.upsampling.cuda()
        self._test = is_test
        self._config = config

        if is_test:
            self._priors = priors

        self.extras.apply(_xavier_init_)
        self.class_headers.apply(_xavier_init_)
        self.location_headers.apply(_xavier_init_)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y


    def calc_disparity(self, depth_sources):
        top = self.toplayer(depth_sources[-1])
        lat1 = self._upsample_add(top, self.latlayer1(depth_sources[-2]))
        lat1 = self.smooth1(lat1)
        lat2 = self._upsample_add(lat1, self.latlayer2(depth_sources[-3]))
        lat2 = self.smooth2(lat2)

        d_top = self.up1(self.agg1(top))
        d_lat1 = self.up2(self.agg2(lat1))
        d_lat2 = self.agg3(lat2)
        print(d_lat2.shape)
        raise ValueError


    def forward(self, x):
        sources = []
        depth_sources = []

        for i, net_layer in enumerate(self.extractor.get_layers()):
            x = net_layer(x)
            if i in self.source_layers:
                sources.append(x)
            if i in self.depth_sources_indices:
                depth_sources.append(x)

        for layer_index, layer in enumerate(self.extras):
            x = layer(x)
            sources.append(x)

        disparity = self.calc_disparity(depth_sources)
        raise ValueError

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

            return confidences, boxes, disparity

        return confidences, locations, disparity

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
