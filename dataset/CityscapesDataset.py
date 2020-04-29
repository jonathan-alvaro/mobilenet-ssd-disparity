import json
import pathlib
from typing import Optional

import PIL
import cv2
import numpy as np
import torch
from PIL import Image

from network import transforms
from network.MatchPrior import MatchPrior
from .helper import label_mapping


class CityscapesDataset(torch.utils.data.Dataset):
    """
    Outputs images in tensor form (scaled), labels (integer, not one-hot encoded),
    and locations (not gt_boxes)
    """
    IMAGE_SUFFIX = "_leftImg8bit.png"
    IMAGE_FOLDER = "images"
    ANNOTATION_FOLDER = "bounding_boxes"
    ANNOTATION_SUFFIX = "_gtFine_boxes.json"

    def __init__(self, config: dict, root_dir: str, train_transform: Optional[transforms.Compose],
                 data_transform: Optional[transforms.Compose],
                 target_transform: Optional[MatchPrior], is_test: bool = False):
        self._root_dir = pathlib.Path(root_dir)
        self._image_ids = []
        self._train_transform = train_transform
        self._data_transform = data_transform
        self._target_transform = target_transform
        self._num_classes = config['num_classes']
        self._test = is_test

        for i, path in enumerate(sorted(self._root_dir.glob("**/images/*.png"))):
            image_filename = path.stem
            *id_components, filetype = image_filename.split('_')
            self._image_ids.append('_'.join(id_components))

            _, labels = self._load_annotations(len(self._image_ids) - 1)
            if labels.shape[0] == 0:
                self._image_ids.pop(len(self._image_ids) - 1)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        image = self.get_image(index)

        # Boxes are in corner form [cx, cy, w, h]
        gt_boxes, gt_labels = self._load_annotations(index)

        if self._train_transform:
            image, gt_boxes, gt_labels = self._train_transform(image, gt_boxes, gt_labels)

        if self._data_transform:
            image, gt_boxes, gt_labels = self._data_transform(image, gt_boxes, gt_labels)

        # TODO Cleanup
        # from vision.ssd.data_preprocessing import TrainAugmentation
        # transform = TrainAugmentation(300, np.array([127, 127, 127]), 128.0)
        # import random
        # image = cv2.imread('/home/jon/projects/uni/TA/pytorch-ssd/data/VOC2007/train/JPEGImages/00000.jpg')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # gt_boxes = gt_boxes.numpy()
        # random.seed(0)
        # np.random.seed(0)
        # image, gt_boxes, gt_labels = transform(image, gt_boxes, gt_labels)
        # gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        if self._target_transform and not self._test:
            boxes, labels = self._target_transform(gt_boxes, gt_labels)
        else:
            boxes = gt_boxes
            labels = gt_labels

        return image, boxes, labels

    def get_image(self, index: int) -> PIL.Image.Image:
        image_id = self._get_image_id(index)
        image_file = image_id + self.IMAGE_SUFFIX
        image_path = self._root_dir.joinpath(self.IMAGE_FOLDER, image_file)
        image = Image.open(str(image_path)).convert("RGB")

        return image

    def _load_annotations(self, index: int) -> (torch.Tensor, torch.Tensor):
        image_id = self._get_image_id(index)
        annotation_file = self._root_dir.joinpath(
            self.ANNOTATION_FOLDER, image_id + self.ANNOTATION_SUFFIX
        )

        with open(annotation_file) as f:
            annotation = json.load(f)

        boxes = []
        labels = []

        for item in annotation['objects']:
            if item['label'] in label_mapping:
                labels.append(item['label'])
                boxes.append(item['bounding_box'])

        boxes = torch.tensor(boxes, dtype=torch.float32).clamp(0)
        labels = torch.tensor([label_mapping[elem] for elem in labels])

        return boxes, labels

    def _get_image_id(self, index):
        image_id = self._image_ids[index]
        return image_id
