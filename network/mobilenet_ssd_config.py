from dataset.helper import num_classes
from network.box_utils import generate_priors

network_config = {
    'width': 300,
    'height': 300,
    'num_classes': num_classes + 1,  # + 1 for background class
    'iou_threshold': 0.5,
    'variance': [0.1, 0.2],
    'feature_size': [25, 13, 7, 4, 2, 1],
    'min_size': [60, 100, 140, 200, 250, 300],
    'max_size': [100, 140, 800, 250, 300, 380],
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
