import time

import cv2
import numpy as np
import torch
from PIL.ImageDraw import Draw

from dataset.CityscapesDataset import CityscapesDataset
from model_validation_utils import build_model
from network import transforms
from network.MatchPrior import MatchPrior
from network.val_predictor import Predictor
from network.mobilenet_ssd_config import network_config, priors


def eval(config: dict, model_path='checkpoints/model_epoch1.pth'):
    ssd = build_model(config, is_test=True)
    ssd.load_state_dict(torch.load(model_path))
    ssd.train(False)

    net = Predictor(ssd)

    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
        transforms.Resize(config['width'], config['height']),
        transforms.Scale(),
        transforms.ToTensor()
    ])

    target_transform = MatchPrior(priors, config)

    val_set = CityscapesDataset(config, 'dataset/train', None, data_transform, target_transform, True)

    test_image = val_set.get_image(0)
    
    for _ in range(2):
        start = time.time()
        boxes, labels, conf, _ = net.predict(test_image)
        print("Prediction time:", time.time() - start)
    
    drawer = Draw(test_image)
    
    for i in range(boxes.shape[0]):
        top_left = tuple(boxes[i][:2].numpy().flatten())
        bottom_right = tuple(boxes[i][2:].numpy().flatten())
        drawer.rectangle((top_left, bottom_right))

    print(labels)
    test_image.save("predict.jpg")

eval(network_config)
