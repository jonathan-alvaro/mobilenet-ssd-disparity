import pathlib
from random import random, uniform, choice
from typing import Optional, Union

import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from network.box_utils import iou
from network.mobilenet_ssd_config import network_config


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
                 labels: Optional[torch.Tensor] = None, disparity: Optional[torch.Tensor] = None):
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

            if random() < self._prob:
                img = F.adjust_contrast(img, contrast_factor)

        # noinspection PyTypeChecker
        img = np.array(img.convert("RGB"))

        disparity = disparity.numpy()

        return img, boxes, labels, disparity


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
                 labels: Optional[torch.Tensor] = None, disparity: Optional[torch.Tensor] = None):
        if random() < self._prob:
            swap_mode = choice(self._perms)
            rgb = img.split()
            new_bands = []
            for idx in swap_mode:
                new_bands.append(rgb[idx])
            img = Image.merge("RGB", new_bands)
        return img, boxes, labels, disparity


class RandomZoom:
    """
    Randomly zooms in or out the image
    """

    def __init__(self, prob: float = 0.5):
        self.zoom_out = Compose([
            RandomExpand(max_expand_ratio=2),
            RandomCrop()
        ])

        self.zoom_in = Compose([
            RandomCrop()
        ])

        self.prob = prob

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[torch.Tensor] = None):
        if random() < self.prob:
            return self.zoom_in(img, boxes, labels, disparity)
        else:
            return img, boxes, labels, disparity


class RandomExpand:
    def __init__(self, prob: float = 0.5, max_expand_ratio: int = 4):
        self._prob = prob
        self._max_ratio = max_expand_ratio

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[torch.Tensor] = None):

        if random() < self._prob:

            height, width, depth = img.shape

            for _ in range(30):
                ratio = uniform(1, self._max_ratio)

                new_top = int(uniform(0, height * ratio - height))
                new_left = int(uniform(0, width * ratio - width))

                new_width = int(ratio * width)
                new_height = int(ratio * height)

                boxes_dimensions = boxes[..., 2:] - boxes[..., :2]
                boxes_area = torch.prod(boxes_dimensions, dim=boxes_dimensions.dim() - 1)

                valid_boxes_area_threshold = new_width / network_config['width'] * new_height / network_config['height'] * 300
                valid_boxes = boxes[boxes_area >= valid_boxes_area_threshold]
                valid_labels = labels[boxes_area >= valid_boxes_area_threshold]

                if valid_boxes.shape[0] == 0:
                    continue

                expanded_image = np.zeros((new_height, new_width, depth), dtype=img.dtype)
                expanded_image.fill(127)
                expanded_disparity = np.zeros((new_height, new_width), dtype=disparity.dtype)

                expanded_image[new_top:new_top + height, new_left:new_left + width] = img
                expanded_disparity[new_top:new_top + height, new_left:new_left + width] = disparity

                img = expanded_image
                disparity = expanded_disparity

                valid_boxes[..., :2] += torch.tensor([new_left, new_top]).float()
                valid_boxes[..., 2:] += torch.tensor([new_left, new_top]).float()

                return img, valid_boxes, valid_labels, disparity

        return img, boxes, labels, disparity


class RandomCrop:
    def __init__(self, prob: float = 0.5, min_crop_ratio: float = 0.3):
        self._prob = prob
        self._min_crop = min_crop_ratio

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        if random() > self._prob:
            return img, boxes, labels, disparity

        height, width, depth = img.shape

        # If after 30 trials no good crop is found, skip cropping
        for _ in range(30):
            crop_width = int(uniform(self._min_crop * width, width))
            crop_height = int(uniform(self._min_crop * height, height))

            crop_ratio = crop_width / crop_height

            if crop_ratio < 0.5 or crop_ratio > 2:
                continue

            crop_left = int(uniform(0, width - crop_width))
            crop_top = int(uniform(0, height - crop_height))

            crop_box = np.array([crop_left, crop_top, crop_left + crop_width, crop_top + crop_height])

            new_top_left = torch.tensor([crop_left, crop_top]).float()
            new_bottom_right = torch.tensor([crop_left + crop_width, crop_top + crop_height]).float()

            new_boxes = boxes.clone()

            new_boxes[..., :2] = torch.max(new_boxes[..., :2], new_top_left)
            new_boxes[..., :2] = torch.min(new_boxes[..., :2], new_bottom_right)
            new_boxes[..., 2:] = torch.min(new_boxes[..., 2:], new_bottom_right)
            new_boxes[..., 2:] = torch.max(new_boxes[..., 2:], new_top_left)

            new_boxes_dimensions = new_boxes[..., 2:] - new_boxes[..., :2]
            new_boxes_area = torch.prod(new_boxes_dimensions, dim=new_boxes_dimensions.dim() - 1)

            boxes_dimensions = boxes[..., 2:] - boxes[..., :2]
            boxes_area = torch.prod(boxes_dimensions, dim=boxes_dimensions.dim() - 1)

            valid_area_threshold = crop_width / network_config['width'] * crop_height / network_config['height'] * 300

            # All boxes must have an area of 300 pixels and no less than 30% of its original area
            valid_boxes_by_area = (new_boxes_area >= valid_area_threshold) & ((new_boxes_area / boxes_area) >= 0.3)
            valid_boxes = new_boxes[valid_boxes_by_area]
            valid_labels = labels[valid_boxes_by_area]

            # Reset boxes origin to new top_left
            valid_boxes[..., :2] -= new_top_left
            valid_boxes[..., 2:] -= new_top_left

            # Cropped image must have at least 1 object
            if valid_boxes.shape[0] <= 0:
                continue

            cropped_image = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2], :]
            cropped_disparity = disparity[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

            return cropped_image, valid_boxes, valid_labels, cropped_disparity

        # Return original inputs if no tries succeed
        return img, boxes, labels, disparity


class RandomMirror:
    def __init__(self, prob: float = 0.5):
        self._prob = prob

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        _, width, _ = img.shape

        if random() < self._prob:
            img = img[:, ::-1]
            disparity = disparity[:, ::-1]
            boxes = boxes.numpy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            boxes = torch.tensor(boxes)

        return img, boxes, labels, disparity


class ToRelativeBoxes:
    """
    Converts boxes from absolute pixel locations to locations relative to image size
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        height, width, depth = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return img, boxes, labels, disparity


class Resize:
    """
    Resize image to shape (size, size)
    """

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        img = cv2.resize(img, (self._width, self._height))
        if disparity is not None:
            disparity = np.array(disparity)
            disparity = cv2.resize(disparity.astype(float), (300, 300))
        return img, boxes, labels, disparity


class Scale:
    """
    Scale RGB values using min max scaling. This is meant to improve training performance
    by reducing the range of possible input values
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        return (img - np.array([127, 127, 127])) / np.array([128]), boxes, labels, disparity


class ToTensor:
    """
    Converts image from numpy array to PyTorch Tensors
    """

    def __call__(self, img: np.ndarray, boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity: Optional[np.ndarray] = None):
        img = torch.from_numpy(img).float()
        if disparity is not None:
            disparity = torch.tensor(disparity).float()
        img = img.permute((2, 0, 1))

        return img, boxes, labels, disparity


class Compose:
    """
    Pipeline of transformations to be applied to input
    """

    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, img: Union[np.ndarray, Image.Image], boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity=None):
        for op in self._transforms:
            img, boxes, labels, disparity = op(img, boxes, labels, disparity)

        return img, boxes, labels, disparity


class ToOpenCV:
    def __call__(self, img: Union[Image.Image, torch.Tensor, np.ndarray],
                 boxes: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None, disparity=None):
        if type(img) == Image.Image:
            img = np.array(img.convert("RGB"))
        elif type(img) == torch.Tensor:
            img = img.numpy()

        return img, boxes, labels, np.array(disparity)
