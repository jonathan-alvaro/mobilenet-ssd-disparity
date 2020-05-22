import numpy as np
import pandas as pd
import torch

from dataset.count_classes import class_count


def relative_absolute_error(prediction: torch.Tensor, target: torch.Tensor):
    abs_diff = (prediction - target).abs()
    abs_diff /= prediction

    return abs_diff.mean()


def pixel_miss_error(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 3.0):
    diff = (prediction - target).abs()
    error_count = (diff > threshold).long().sum()

    return error_count / prediction.flatten().shape[0]


def area(box: np.ndarray):
    dims = box[2:] - box[:2]
    dims = np.clip(dims, 0)
    return np.prod(dims)


def map_iou(row):
    p_box = np.array([row['p_left'], row['p_top'], row['p_right'], row['p_bottom']])
    t_box = np.array([row['t_left'], row['t_top'], row['t_right'], row['t_bottom']])
    intersection = np.zeros_like(p_box)
    intersection[:2] = np.maximum(p_box[:2], t_box[:2])
    intersection[2:] = np.minimum(p_box[2:], t_box[2:])

    return area(intersection) / (area(p_box) + area(t_box) - area(intersection))


def mean_accurate_precision(prediction_csv_file: str):
    predictions = pd.read_csv(prediction_csv_file)

    num_classes = len(pd.unique(predictions['p_label']))

    map_by_class = []

    for class_label in range(1, num_classes + 1):
        class_rows = predictions[predictions['p_label'] == class_label].copy()

        class_rows['iou'] = class_rows.apply(map_iou, axis=1)

        class_rows['tp'] = (class_rows['iou'] >= 0.5) & (class_rows['p_label'] == class_rows['t_label']) & (
                    class_rows['p_label'] == class_label)
        class_rows['fp'] = (class_rows['iou'] >= 0.5) & (class_rows['p_label'] == class_rows['t_label']) & (
                    class_rows['p_label'] == class_label)

        class_rows['tp'] = class_rows['tp'].cumsum()
        class_rows['fp'] = class_rows['fp'].cumsum()
        class_rows['precision'] = class_rows['tp'] / (class_rows['tp'] + class_rows['fp'])
        class_rows['recall'] = class_rows['tp'] / class_count[class_label]

        precision = class_rows['precision'].to_numpy()
        recall = class_rows['recall'].to_numpy()

        transitions = set()

        for i in range(len(precision), 0, -1):
            if precision[i - 1] < precision[i]:
                precision[i - 1] = precision[i]
            if recall[i - 1] == recall[i]:
                transitions.add(recall[i])

        transitions = sorted(list(transitions))
        block_start = 0

        class_map = 0

        for i, value in enumerate(transitions):
            block_end = np.where(recall == value)[0][0]

            class_map += precision[block_start] * (recall[block_end] - recall[block_start])

            block_start = np.where(recall == value)[0][-1]

        map_by_class.append(class_map)

    return map_by_class