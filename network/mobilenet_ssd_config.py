from dataset.helper import num_classes
from network.box_utils import generate_priors

network_config = {
    'width': 1024,
    'height': 512,
    'num_classes': num_classes + 1,  # + 1 for background class
    'iou_threshold': 0.5,
    'variance': [0.1, 0.2],
    'feature_size': [64, 32, 16, 8, 4, 2],
    'min_size': [120, 210, 300, 390, 480, 570],
    'max_size': [210, 300, 390, 480, 570, 650],
    'aspect_ratio': [
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3],
        [2, 3]
    ]
}

priors = generate_priors(network_config)
