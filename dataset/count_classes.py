import json
import pathlib

import numpy as np

from dataset.helper import label_mapping


def count_classes(annotation_dir: str) -> dict:
    annotation_dir = pathlib.Path(annotation_dir)

    annotation_fps = annotation_dir.glob("./*json")

    class_counts = {}

    for fp in annotation_fps:
        with fp.open() as f:
            objects = json.load(f)['objects']

        for item in objects:
            if item['label'] not in label_mapping:
                continue

            box = np.array(item['bounding_box'])
            wh = box[2:] - box[:2]
            area = np.prod(wh)

            if area < 6990.51:
                continue

            if label_mapping[item['label']] in class_counts:
                class_counts[label_mapping[item['label']]] += 1
            else:
                class_counts[label_mapping[item['label']]] = 1

    return class_counts


class_count = count_classes('val/bounding_boxes')