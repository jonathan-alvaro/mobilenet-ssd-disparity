import math

import torch
from PIL.ImageDraw import Draw
from network import transforms

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.mobilenet_ssd_config import network_config, priors
from train_utils import build_model, calculate_map


def eval(config: dict, model_path='checkpoints/model_epoch30.pth'):
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

    test_image = val_set.get_image(0)
    
    boxes, labels, conf, _, _ = net.predict(test_image)
    
    drawer = Draw(test_image)
    
    for i in range(boxes.shape[0]):
        top_left = tuple(boxes[i][:2].numpy().flatten())
        bottom_right = tuple(boxes[i][2:].numpy().flatten())
        drawer.rectangle((top_left, bottom_right))

    test_image.save("predict.jpg")