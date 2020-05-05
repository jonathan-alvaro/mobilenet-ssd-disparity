import torch
from torchvision import transforms
from PIL import Image

from network.box_utils import nms, convert_locations_to_boxes, generate_priors
from network.mobilenet_ssd_config import priors
from network.integratedmodel import IntegratedModel


class Predictor:
    def __init__(self, net: IntegratedModel,
                 iou_threshold: float = 0.5):
        self._net = net
        self._resize = transforms.Resize((300, 300))
        self._to_tensor = transforms.ToTensor()
        self._threshold = iou_threshold

    def predict(self, image: Image.Image, prob_threshold=0.25):
        width, height = image.size
        image = self._resize(image)
        image = self._to_tensor(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            conf, boxes, disparity = self._net.forward(image)
            boxes = boxes[0]
            conf = conf[0]
            disparity = disparity[0]

        return conf, boxes, disparity
