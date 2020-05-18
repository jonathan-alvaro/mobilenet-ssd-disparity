import torch
import json
from torch import nn

from network.box_utils import iou
from network.mobilenet import MobileNet
from network.integratedmodel import IntegratedModel


def forward_img(net: IntegratedModel, img: torch.Tensor):
    res = net(img)
    return res


def build_model(config: dict, is_test: bool = False) -> IntegratedModel:
    extra_layers = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    ])

    location_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1)
    ])

    mobilenet = MobileNet()
    mobilenet.load('mobilenetv1.pth')

    ssd = IntegratedModel(
        config['num_classes'], [11, 13], mobilenet, extra_layers,
        location_headers, classification_headers, config, is_test
    )
    return ssd


def calculate_map(pred_confidences: torch.Tensor, gt_labels: torch.Tensor,
                  pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> (float, torch.Tensor):
    num_classes = pred_confidences.shape[-1] - 1

    class_ap = []
    pred_labels = torch.argmax(pred_confidences[..., 1:], dim=pred_confidences.dim() - 1).view((-1, 1))
    
    for class_index in range(1, num_classes + 1):
        pred_class_probs = pred_confidences[:, class_index].view((-1, 1))
        _, prob_ranking = pred_class_probs.sort(descending=True)

        ious = iou(pred_boxes[prob_ranking], gt_boxes[prob_ranking])
        pred_class_labels = torch.argmax(pred_labels[prob_ranking], dim=pred_labels.dim()-1)
        class_gt_labels = gt_labels[prob_ranking].view((-1, 1))

        tp = ((ious >= 0.5).flatten()) & ((pred_class_labels == class_gt_labels).flatten()) & ((class_gt_labels == class_index).flatten())
        fp = (ious >= 0.5).flatten() & (pred_class_labels != class_gt_labels).flatten() & (class_gt_labels != class_index).flatten()
        fn = (class_gt_labels == class_index).flatten() & ((ious < 0.5) | (pred_class_labels != class_gt_labels)).flatten()

        precision = tp.long().float() / (tp.long().float() + fp.long().float())
        recall = tp.long().float() / (tp.long().float() + fn.long().float())

        precision[torch.isnan(precision)] = 0
        precision = precision.cumsum(dim=0)
        recall[torch.isnan(recall)] = 0
        recall = recall.cumsum(dim=0)

        for i, _ in enumerate(precision):
            precision[i] = precision[i:].max()

        class_ap.append((precision * recall).sum())

    return sum(class_ap) / len(class_ap)



def calculate_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    right_pointer = precision.shape[0]

    ap = 0

    change_points = recall[1:] != recall[:-1]
    return (precision[1:][change_points] * (recall[1:][change_points] - recall[:-1][change_points])).sum().item()

