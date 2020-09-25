import math
import cv2
import numpy as np

import torch
from PIL.ImageDraw import Draw
from PIL import Image
from network import transforms

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.mobilenet_ssd_config import network_config, priors
from train_utils import build_model, calculate_map


def eval(config: dict, model_path='checkpoints/to_reduce_lr/model_epoch14.pth'):
    ssd = build_model(config, is_test=True)
    ssd.load_state_dict(torch.load(model_path))
    ssd.train(False)

    net = Predictor(ssd)

    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
        transforms.Resize(config['image_size']),
        transforms.Scale(),
        transforms.ToTensor()
    ])

    target_transform = MatchPrior(priors, config)

    val_set = CityscapesDataset(config, 'dataset/val', None, data_transform, target_transform, True)

    test_image = val_set.get_image(3)
    original = np.array(test_image)
    cv2.imwrite('original.jpg', original)
    
    boxes, labels, conf, disparity, _ = net.predict(test_image)
    
    drawer = Draw(test_image)
    
    for i in range(boxes.shape[0]):
        top_left = tuple(boxes[i][:2].numpy().flatten())
        bottom_right = tuple(boxes[i][2:].numpy().flatten())
        drawer.rectangle((top_left, bottom_right), width=5)

    test_image.save("predict.jpg")
    disparity = disparity[0].cpu().numpy()
    gt_disparity = val_set.get_disparity(2)[0]
    gt_disparity = gt_disparity.numpy()
    gt_disparity = cv2.resize(gt_disparity, (94, 94))
    diff = gt_disparity - disparity

    cv2.imwrite('disparity.png', disparity.astype(np.uint8)) 
    cv2.imwrite('disparity-target.png', gt_disparity.astype(np.uint8))
    temp = disparity
    temp = temp * 126 / temp.max()
    cv2.imwrite('test.png', temp)

eval(network_config)
