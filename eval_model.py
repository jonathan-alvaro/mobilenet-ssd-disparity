from argparse import ArgumentParser

import torch
import numpy as np

from dataset.CityscapesDataset import CityscapesDataset
from dataset.metrics import relative_absolute_error, pixel_miss_error, mean_accurate_precision
from network import transforms
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.box_utils import iou
from network.mobilenet_ssd_config import network_config, priors
from train_utils import build_model

parser = ArgumentParser(description='Evaluate trained model on designated dataset')
parser.add_argument('N', type=int, help='Number of images to process')
parser.add_argument('--model_path', type=str, required=True, help='Path to .pth file containing model')
parser.add_argument('--data_folder', type=str, required=True, help='Path to root data folder')
args = parser.parse_args()


def create_predictor(model_path: str):
    model = build_model(network_config, is_test=True)
    model.load_state_dict(torch.load(model_path))
    model.train(False)

    model = Predictor(model, use_cuda=True)

    return model


def create_dataset(folder_path: str):
    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
        transforms.Resize(network_config['width'], network_config['height']),
        transforms.Scale(),
        transforms.ToTensor()
    ])

    dataset = CityscapesDataset(network_config, folder_path, None, data_transform, None, True)

    return dataset


def eval_model(dataset: CityscapesDataset, n: int, model: Predictor, csv_file: str):
    prediction_rows = []

    rae = []
    pixel_miss = []

    if n == -1:
        n = len(dataset)

    for i in range(n):
        img, gt_boxes, gt_labels, gt_disparity = dataset[i]
        gt_boxes = gt_boxes * 300

        boxes, labels, probs, disparity, indices = model.predict(img)

        disparity = disparity * 127
        gt_disparity = gt_disparity * 127

        rae.append(relative_absolute_error(disparity, gt_disparity))
        pixel_miss.append(pixel_miss_error(disparity, gt_disparity))

        if boxes.is_cuda:
            gt_boxes = gt_boxes.cuda()

        if boxes.shape[0] == 0:
            continue
        ious = iou(boxes.unsqueeze(1), gt_boxes.unsqueeze(0))

        best_iou_per_prediction = torch.argmax(ious, dim=ious.dim() - 1)

        for j, item in enumerate(boxes):
            target_index = best_iou_per_prediction[j]

            target_label = gt_labels[target_index]
            target_label = str(target_label.item())

            prediction_label = labels[j]
            prediction_label = str(prediction_label.item())

            prediction_prob = probs[j].max()
            prediction_prob = str(prediction_prob.item())

            target_box = gt_boxes[target_index].tolist()
            target_box = list(map(str, target_box))

            prediction_box = item.tolist()
            prediction_box = list(map(str, prediction_box))

            best_iou = ious[j].max()
            best_iou = str(best_iou.item())

            prediction_csv_form = ','.join([
                prediction_label, target_label, prediction_prob, *prediction_box, *target_box, best_iou
            ])
            prediction_rows.append(prediction_csv_form)

    column_headers = [
        'p_label', 't_label', 'p_prob', 'p_left', 'p_top', 'p_right', 'p_bottom',
        't_left', 't_top', 't_right', 't_bottom', 'iou'
    ]

    with open(csv_file, 'w') as f:
        f.write(','.join(column_headers))
        f.write('\n')
        f.write('\n'.join(prediction_rows))

    map_by_class = mean_accurate_precision(csv_file)

    return np.array(rae).mean(), np.array(pixel_miss).mean(), map_by_class


predictor = create_predictor(args.model_path)
dataset = create_dataset(args.data_folder)
print(eval_model(dataset, args.N, predictor, 'predictions.csv'))
