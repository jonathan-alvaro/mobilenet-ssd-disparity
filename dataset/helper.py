from collections import namedtuple
from typing import List

import torch

Label = namedtuple(
    'Label',
    ['label', 'category', 'encoding']
)

labels = [
    # Label('unlabeled', 'void', 0),
    # Label('ego vehicle', 'void', 0),
    # Label('rectification border', 'void', 0),
    # Label('out of roi', 'void', 0),
    # Label('static', 'void', 0),
    # Label('dynamic', 'void', 0),
    # Label('ground', 'void', 0),
    # Label('road', 'flat', 1),
    # Label('sidewalk', 'flat', 1),
    # Label('parking', 'flat', 1),
    # Label('rail track', 'flat', 1),
    # Label('building', 'construction', 2),
    # Label('wall', 'construction', 2),
    # Label('fence', 'construction', 2),
    # Label('guard rail', 'construction', 2),
    # Label('bridge', 'construction', 2),
    # Label('tunnel', 'construction', 2),
    # Label('pole', 'object', 3),
    # Label('polegroup', 'object', 3),
    Label('traffic light', 'object', 1),
    Label('traffic sign', 'object', 1),
    # Label('vegetation', 'nature', 4),
    # Label('terrain', 'nature', 4),
    # Label('sky', 'sky', 5),
    Label('person', 'human', 2),
    Label('rider', 'human', 2),
    Label('car', 'vehicle', 3),
    Label('truck', 'vehicle', 3),
    Label('bus', 'vehicle', 3),
    Label('caravan', 'vehicle', 3),
    Label('trailer', 'vehicle', 3),
    Label('train', 'vehicle', 3),
    Label('motorcycle', 'vehicle', 3),
    Label('bicycle', 'vehicle', 3),
    # Label('license plate', 'vehicle', 7)
]

label_mapping = {}
category_encoding = {}

for label in labels:
    label_mapping[label.label] = label.category
    category_encoding[label.category] = label.encoding