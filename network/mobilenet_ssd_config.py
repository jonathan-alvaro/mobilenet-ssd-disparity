from dataset.helper import num_classes
from network.box_utils import generate_priors

network_config = {
    'width': 400,
    'height': 200,
    'num_classes': num_classes + 1,  # + 1 for background class
    'iou_threshold': 0.45,
    'variance': [0.1, 0.2],
    'feature_size': [19, 10, 5, 3, 2, 1],
    'min_size': [30, 50, 80, 120, 170, 230],
    'max_size': [150, 170, 200, 280, 300, 330],
    'shrink_factor': [16, 32, 64, 100, 150, 300],
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
