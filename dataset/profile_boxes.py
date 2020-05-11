import json
import pathlib

import numpy as np

from helper import label_mapping


def load_bounding_boxes() -> np.ndarray:
    """
    Gathers all bounding boxes of objects to detect within train set
    :return: Numpy array with each row being a bounding box, shape (N, 4)
    """
    bb_folder = './train/bounding_boxes'

    bb_folder = pathlib.Path(bb_folder)
    bb_json_files = bb_folder.glob('./*json')

    bounding_boxes = []

    for fp in bb_json_files:
        with open(str(fp)) as f:
            json_data = json.load(f)

        objects = json_data['objects']

        for item in objects:
            if item['label'] not in label_mapping:
                continue

            bounding_boxes.append(np.array(item['bounding_box']))

    return np.clip(np.vstack(bounding_boxes), 0, None)


boxes = load_bounding_boxes()
centers = (boxes[:, :2] + boxes[:, 2:]) / 2
dimensions = (boxes[:, 2:] - boxes[:, :2])
ratios = dimensions[:, 0] / dimensions[:, 1]
ratios = ratios[ratios < 5.35] # Less than 200 data has ratio bigger than 5.35

cx = np.histogram(centers[:, 0], bins=20)
cy = np.histogram(centers[:, 1], bins=20)
width = np.histogram(dimensions[:, 0], bins=6)
height = np.histogram(dimensions[:, 1], bins=6)
ratios = np.histogram(ratios, bins=6)

cx = [list(item.astype(int)) for item in cx]
cy = [list(item.astype(int)) for item in cy]
width = [list(item.astype(int)) for item in width]
height = [list(item.astype(int)) for item in height]
ratio = [list(item) for item in ratios]

profile = {
        'cx':cx,
        'cy':cy,
        'width':width,
        'height':height,
        'ratio':ratio
}

class Int64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        else:
            return super().default(obj)

json.dump(profile, open('boxes_profile.json', 'w'), cls=Int64Encoder)
