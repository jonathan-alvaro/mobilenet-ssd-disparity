import json
import pathlib
from typing import Union

import PIL
import torch
import torchvision
from PIL import Image

from network.MatchPrior import MatchPrior
from network.box_utils import corner_to_center
from .helper import label_mapping, category_encoding


class CityscapesDataset(torch.utils.data.Dataset):
    IMAGE_SUFFIX = "_leftImg8bit.png"
    IMAGE_FOLDER = "images"
    ANNOTATION_FOLDER = "bounding_boxes"
    ANNOTATION_SUFFIX = "_gtFine_boxes.json"

    def __init__(self, root_dir: str, data_transform: torchvision.transforms.Compose,
                 target_transform: Union[MatchPrior, type(None)], config: dict, is_test: bool = False):
        self._root_dir = pathlib.Path(root_dir)
        self._image_ids = []
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

        gt_boxes, gt_labels = self._load_annotations(index)

        # Convert from absolute coordinates to relative coordinates
        # TODO: Convert to composed transformations
        gt_boxes[:, 0] /= image.size[1]
        gt_boxes[:, 2] /= image.size[1]
        gt_boxes[:, 1] /= image.size[0]
        gt_boxes[:, 3] /= image.size[0]

        if self._data_transform:
            image = self._data_transform(image)

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

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([category_encoding[label_mapping[elem]] for elem in labels])

        return boxes, labels

    def _get_image_id(self, index):
        image_id = self._image_ids[index]
        return image_id
