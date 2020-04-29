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
    # Label('road', 'flat', 0),
    # Label('sidewalk', 'flat', 1),
    # Label('parking', 'flat', 0),
    # Label('rail track', 'flat', 1),
    # Label('building', 'construction', 0),
    # Label('wall', 'construction', 0),
    # Label('fence', 'construction', 0),
    # Label('guard rail', 'construction', 0),
    # Label('bridge', 'construction', 0),
    # Label('tunnel', 'construction', 0),
    # Label('pole', 'object', 0),
    # Label('polegroup', 'object', 0),
    Label('traffic light', 'object', 1),
    Label('traffic sign', 'object', 2),
    # Label('vegetation', 'nature', 0),
    # Label('terrain', 'nature', 0),
    # Label('sky', 'sky', 0),
    Label('person', 'human', 3),
    # Label('rider', 'human', 3),
    Label('car', 'vehicle', 4),
    Label('truck', 'vehicle', 5),
    # Label('bus', 'vehicle', 6),
    # Label('caravan', 'vehicle', 7),
    # Label('trailer', 'vehicle', 8),
    # Label('train', 'vehicle', 9),
    # Label('motorcycle', 'vehicle', 10),
    # Label('bicycle', 'vehicle', 11),
    # Label('license plate', 'vehicle', 7)
]

label_mapping = {}
classes = set()

for label in labels:
    label_mapping[label.label] = label.encoding
    classes.add(label.encoding)

num_classes = len(classes)
if 0 in classes:
    num_classes -= 1