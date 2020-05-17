wanted_labels = [
    'person',
    'car'
]

wanted_labels = {label: i for i, label in enumerate(wanted_labels)}

label_to_category_map = {
    'unlabeled': 'void',
    'ego vehicle': 'void',
    'rectification border': 'void',
    'out of roi': 'void',
    'static': 'void',
    'dynamic': 'void',
    'ground': 'void',
    'road': 'flat',
    'sidewalk': 'flat',
    'parking': 'flat',
    'rail track': 'flat',
    'building': 'construction',
    'wall': 'construction',
    'fence': 'construction',
    'guard rail': 'construction',
    'bridge': 'construction',
    'tunnel': 'construction',
    'pole': 'object',
    'polegroup': 'object',
    'traffic light': 'object',
    'traffic sign': 'object',
    'vegetation': 'nature',
    'terrain': 'nature',
    'sky': 'sky',
    'person': 'human',
    'rider': 'human',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'caravan': 'vehicle',
    'trailer': 'vehicle',
    'train': 'vehicle',
    'motorcycle': 'vehicle',
    'bicycle': 'vehicle',
    'license plate': 'vehicle'
}
