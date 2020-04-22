from random import random, uniform, choice

from typing import Optional

import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as F
import numpy as np

from network.box_utils import iou


class CustomJitter:
    """
    Randomly changes brightness, contrast, hue and saturation using torchvision transforms
    Implemented in order to accommodate the boxes and labels used in object detection

    The lowest value for contrast, saturation and brightness factor is 0.3
    Factor for hue transformation is in the range of [-0.5, 0.5]

    Args:
        prob (float): Probability each transformation wil be applied
        max_brightness_factor (float): Maximum brightness will be multiplied by

    Example:
        With prob of 0.5 there is 50% change brightness will be adjusted, 50% contrast will be adjusted,
        and so on
    """

    def __init__(self, prob: float = 0.5, max_brightness_factor: float = 1,
                 max_contrast_factor: float = 1, max_hue_factor: float = 0.5,
                 max_saturation_factor: float = 1):
        self._prob = prob
        self._max_brightness_factor = max_brightness_factor
        self._max_contrast_factor = max_contrast_factor
        self._max_hue_factor = max_hue_factor
        self._max_saturation_factor = max_saturation_factor
        self._min = 0.3

    def __call__(self, img: Image.Image, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        if random() < self._prob:
            brightness_factor = uniform(self._min, self._max_brightness_factor)
            img = F.adjust_brightness(img, brightness_factor)

        contrast_factor = uniform(self._min, self._max_contrast_factor)
        hue_factor = uniform(-self._min, self._max_hue_factor)
        saturation_factor = uniform(self._min, self._max_saturation_factor)

        # 50% chance applying contrast followed by hue and saturation or vice versa
        if random() < 0.5:
            if random() < self._prob:
                img = F.adjust_contrast(img, contrast_factor)

            if random() < self._prob:
                img = img.convert("HSV")
                img = F.adjust_hue(img, hue_factor)
                img = img.convert("RGB")
                img = F.adjust_saturation(img, saturation_factor)

        else:
            if random() < self._prob:
                img = img.convert("HSV")
                img = F.adjust_hue(img, hue_factor)
                img = img.convert("RGB")
                img = F.adjust_saturation(img, saturation_factor)
                # img = img.convert("RGB")

            if random() < self._prob:
                img = F.adjust_contrast(img, contrast_factor)

        return img, boxes, labels


class RandomLightingNoise:
    """
    Changes the image by randomly swapping color channels
    """

    def __init__(self, prob: float = 0.5):
        self._perms = ((0, 1, 2), (0, 2, 1),
                       (1, 0, 2), (1, 2, 0),
                       (2, 0, 1), (2, 1, 0))
        self._prob = prob

    def __call__(self, img: Image.Image, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        if random() < self._prob:
            swap_mode = choice(self._perms)
            rgb = img.split()
            new_bands = []
            for idx in swap_mode:
                new_bands.append(rgb[idx])
            img = Image.merge("RGB", new_bands)
        return img, boxes, labels


class RandomExpand:
    def __init__(self, prob: float = 0.5, max_expand_ratio: int = 4):
        self._prob = prob
        self._max_ratio = max_expand_ratio

    def __call__(self, img: Image.Image, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        # noinspection PyTypeChecker
        img = np.array(img.convert("RGB"))

        if random() < self._prob:
            ratio = uniform(1, 4)

            height, width, depth = img.shape
            new_top = int(uniform(0, height * ratio - height))
            new_left = int(uniform(0, width * ratio - width))

            new_width = int(ratio * width)
            new_height = int(ratio * height)
            expanded_image = np.zeros((new_height, new_width, depth), dtype=img.dtype)
            expanded_image[new_top:new_top + height, new_left:new_left + width] = img
            img = expanded_image

            boxes[..., :2] += torch.tensor([new_left, new_top]).float()
            boxes[..., 2:] += torch.tensor([new_left, new_top]).float()

        return img, boxes, labels


class RandomCrop:
    def __init__(self, prob: float = 0.5, min_crop_ratio: float = 0.3):
        self._prob = prob
        self._sample_modes = ((None, None),
                              (0.1, None),
                              (0.3, None),
                              (0.5, None),
                              (0.7, None),
                              (0.9, None))
        self._min_crop = min_crop_ratio

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        if random() > self._prob:
            return img, boxes, labels

        min_iou, max_iou = choice(self._sample_modes)

        height, width, depth = img.shape

        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        # If after 30 trials no good crop is found, skip cropping
        for _ in range(30):
            crop_width = int(uniform(self._min_crop * width, width))
            crop_height = int(uniform(self._min_crop * height, height))

            crop_ratio = crop_width / crop_height

            if crop_ratio < 0.5 or crop_ratio > 2:
                continue

            crop_left = int(uniform(0, width - crop_width))
            crop_top = int(uniform(0, height - crop_height))

            crop_box = np.array([crop_left, crop_top, crop_left + width, crop_top + height])

            iou_scores = iou(torch.tensor(crop_box), boxes.long())

            if float(iou_scores.min()) < min_iou or float(iou_scores.max()) > max_iou:
                continue

            cropped_image = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]

            box_centers = (boxes[..., :2] + boxes[..., 2:]) / 2
            top_left_mask = (box_centers[..., 0] > crop_box[0]) & (box_centers[..., 1] > crop_box[1])
            bottom_right_mask = (box_centers[..., 0] < crop_box[2]) & (box_centers[..., 1] < crop_box[3])

            # Select boxes whose centers are within the cropped image
            box_selection_mask = top_left_mask & bottom_right_mask
            # If there's no valid box, try another crop
            if not box_selection_mask.any():
                continue

            valid_boxes = boxes[box_selection_mask, ...]
            valid_labels = labels[box_selection_mask, ...]

            # Adjust the boxes' corners into the crop's boundary
            valid_boxes[..., :2] = np.maximum(valid_boxes[..., :2], crop_box[:2])
            valid_boxes[..., 2:] = np.minimum(valid_boxes[..., 2:], crop_box[2:])

            # Adjust the corners to the new coordinates of the cropped image
            valid_boxes[..., :2] -= torch.tensor(crop_box[:2]).float()
            valid_boxes[..., 2:] -= torch.tensor(crop_box[:2]).float()

            return cropped_image, valid_boxes, valid_labels

        # Return original inputs if no tries succeed
        return img, boxes, labels


class RandomMirror:
    def __init__(self, prob: float = 0.5):
        self._prob = prob

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        _, width, _ = img.shape

        if random() < self._prob:
            img = img[:, ::-1]
            boxes = boxes.numpy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes = torch.Tensor(boxes)

        return img, boxes, labels


class ToRelativeBoxes:
    """
    Converts boxes from absolute pixel locations to locations relative to image size
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        height, width, depth = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return img, boxes, labels


class Resize:
    """
    Resize image to shape (size, size)
    """

    def __init__(self, size: int):
        self._size = size

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        img = cv2.resize(img, (self._size, self._size))
        return img, boxes, labels


class Scale:
    """
    Scale RGB values using min max scaling. This is meant to improve training performance
    by reducing the range of possible input values
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        return img / 255, boxes, labels


class ToTensor:
    """
    Converts image from numpy array to PyTorch Tensors
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        img = torch.tensor(img).float()
        img = img.permute((2, 0, 1))

        return img, boxes, labels


class Compose:
    """
    Pipeline of transformations to be applied to input
    """
    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, img: Image.Image, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        for op in self._transforms:
            img, boxes, labels = op(img, boxes, labels)

        return img, boxes, labels