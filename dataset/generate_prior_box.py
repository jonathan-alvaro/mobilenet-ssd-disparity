from random import sample
import pathlib
import json

import numpy as np
from helper import label_mapping


def load_bounding_boxes() -> np.ndarray:
    """
    Gathers all bounding boxes of objects to detect within train set
    :return: Numpy array with each row being a bounding box
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


def iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculates iou between boxes

    :param boxes1: First set of boxes, shape(n1, 4)
    :param boxes2: Second set of boxes, shape(n2, 4)
    :return: Array of iou for every pair of boxes, shape(n1, n2)
    """
    boxes1 = np.expand_dims(boxes1, 1)
    boxes1 = np.repeat(boxes1, len(boxes2), axis=1)

    boxes2 = np.expand_dims(boxes2, 0)
    boxes2 = np.repeat(boxes2, len(boxes1), axis=0)

    if boxes1.shape != boxes2.shape:
        raise ValueError("Expanded shapes doesn't match")

    top_left = np.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
    bottom_right = np.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])

    intersection_dimensions = bottom_right - top_left
    invalid_intersections = np.any(intersection_dimensions <= 0, axis=intersection_dimensions.ndim - 1)

    intersection_area = np.prod(intersection_dimensions, axis=intersection_dimensions.ndim - 1)
    intersection_area[invalid_intersections] = 0

    boxes1_dims = boxes1[:, :, 2:] - boxes1[:, :, :2]
    boxes2_dims = boxes2[:, :, 2:] - boxes2[:, :, :2]

    boxes1_area = np.prod(boxes1_dims, axis=boxes1_dims.ndim - 1)
    boxes2_area = np.prod(boxes2_dims, axis=boxes2_dims.ndim - 1)

    if (boxes1_area.shape != boxes2_area.shape) or (boxes1_area.shape != intersection_area.shape):
        raise ValueError("Area matrix shapes don't match")

    union_area = boxes1_area + boxes2_area - intersection_area

    box_ious = intersection_area / union_area
    box_ious[box_ious == float('inf')] = 0

    return box_ious


def kmeans(boxes: np.ndarray, n_clusters: int = 6, max_iter: int = 600) -> np.ndarray:
    # Initialize centers
    centers = sample(list(boxes), n_clusters)
    centers = np.array(centers)

    distance = iou(boxes, centers)
    cluster_for_boxes = np.argmin(distance, axis=distance.ndim - 1)

    # Do first iteration
    prev_centers = centers.copy()
    valid_clusters = []
    for i, _ in enumerate(centers):
        boxes_in_cluster = boxes[cluster_for_boxes == i]
        if len(boxes_in_cluster) > 0:
            valid_clusters.append(i)
        else:
            continue

        centers[i] = np.mean(boxes_in_cluster, axis=0)
    centers = centers[valid_clusters]
    prev_centers = prev_centers[valid_clusters]

    iters = 1

    while iters < max_iter and np.any(centers != prev_centers):
        valid_clusters = []

        distance = iou(boxes, centers)
        cluster_for_boxes = np.argmax(distance, axis=distance.ndim - 1)

        prev_centers = centers.copy()
        for i, _ in enumerate(centers):
            boxes_in_cluster = boxes[cluster_for_boxes == i]
            if len(boxes_in_cluster) > 0:
                valid_clusters.append(i)
            else:
                continue

            centers[i] = np.mean(boxes_in_cluster, axis=0)

        centers = centers[valid_clusters]
        prev_centers = prev_centers[valid_clusters]

        iters += 1

    return centers


boxes = load_bounding_boxes()
boxes[..., 2:] -= boxes[..., :2]
boxes[..., :2] -= boxes[..., :2]
prior_boxes = kmeans(boxes)
print(prior_boxes)
