from dataset.helper import category_encoding

network_config = {
    'image_size': 300,
    'num_classes': len(category_encoding) + 1, # + 1 for background class
    'iou_threshold': 0.5,
    'variance': [0.1, 0.2],
    'feature_size': [19, 10, 5, 3, 2, 1],
    'min_size': [60, 105, 150, 195, 240, 285],
    'max_size': [105, 150, 195, 240, 285, 330],
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
