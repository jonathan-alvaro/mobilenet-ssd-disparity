import torch
import json
from torch import nn

from network.box_utils import iou
from network.mobilenet import MobileNet
from network.ssd import SSD


def forward_img(net: SSD, img: torch.Tensor):
    res = net(img)
    return res


def build_ssd(config: dict, is_test: bool = False) -> SSD:
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

    ssd = SSD(
        config['num_classes'], [11, 13], mobilenet, extra_layers,
        location_headers, classification_headers, config, is_test
    )
    return ssd


def calculate_map(pred_confidences: torch.Tensor, gt_labels: torch.Tensor,
                  pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> (float, torch.Tensor):
    num_classes = pred_confidences.shape[-1]
    pred_labels = pred_confidences.argmax(dim=pred_confidences.dim() - 1)
    ious = iou(pred_boxes, gt_boxes)

    aps = torch.zeros((num_classes,))

    # Start from 1 because 0 is background
    for class_code in range(1, num_classes):
        class_mask = gt_labels == class_code

        _, pred_ranking = (pred_confidences[class_mask])[..., class_code].flatten().sort()
        sorted_pred_labels = pred_labels[class_mask].flatten()[pred_ranking]
        sorted_gt_labels = gt_labels[class_mask].flatten()[pred_ranking]
        sorted_ious = ious[class_mask].flatten()[pred_ranking]

        tp = (sorted_gt_labels == class_code) & (sorted_ious >= 0.5) & (sorted_pred_labels == class_code)
        fp = (sorted_gt_labels != class_code) & (sorted_ious >= 0.5) & (sorted_pred_labels == class_code)
        fn = (sorted_gt_labels == class_code) & (sorted_ious < 0.5)

        tp = tp.flatten().long().cumsum(0).float()
        fp = fp.flatten().long().cumsum(0).float()
        fn = fn.flatten().long().cumsum(0).float()

        precision = tp / (fp + tp)
        recall = tp / (tp + fn)

        precision[torch.isnan(precision)] = 0
        recall[torch.isnan(recall)] = 0

        # print(precision)
        # print(recall)
        # print(tp)
        # print(fp)

        aps[class_code] = calculate_ap(precision, recall)

    if (aps.mean().item()) > 1:
        print(aps)
        raise ValueError
    return aps.mean().item(), aps


def calculate_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    right_pointer = precision.shape[0]

    ap = 0

    change_points = recall[1:] != recall[:-1]
    return (precision[1:][change_points] * (recall[1:][change_points] - recall[:-1][change_points])).sum().item()

