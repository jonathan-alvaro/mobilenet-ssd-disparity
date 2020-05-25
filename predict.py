import sys
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL.ImageDraw import Draw
from PIL import Image
from network import transforms

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.mobilenet_ssd_config import network_config, priors
from train_utils import build_model, calculate_map


def eval(config: dict, model_path='checkpoints/model_epoch10.pth'):
    ssd = build_model(config, is_test=True)
    ssd.load_state_dict(torch.load(model_path))
    ssd.train(False)

    net = Predictor(ssd, use_cuda=True)

    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
        transforms.Resize(300, 300),
        transforms.Scale(),
        transforms.ToTensor()
    ])

    target_transform = MatchPrior(priors, config)

    val_set = CityscapesDataset(config, 'dataset/val', None, data_transform, target_transform, True)

    arg1 = int(sys.argv[1])

    for i in range(arg1):
        test_image = val_set.get_image(i)
        test_image = test_image.resize((300, 300))

        img,_, _, _ = val_set[i]
        img = img.cuda()
        
        boxes, labels, conf, disparity, _ = net.predict(img)
        boxes = boxes.cpu()
    
        drawer = Draw(test_image)
        
        for j in range(boxes.shape[0]):
            top_left = tuple(boxes[j][:2].numpy().flatten())
            bottom_right = tuple(boxes[j][2:].numpy().flatten())
            drawer.rectangle((top_left, bottom_right), width=5)

        test_image.save('prediction/{}.jpg'.format(i))

        disparity = disparity.squeeze().cpu().numpy() * 127
        print("PRediction Max:", disparity.max())
        print("Prediction Min:", disparity.min())
        print("Prediction MeaN:", disparity.mean())
        cv2.imwrite('prediction/{}_disparity_raw.png'.format(i), disparity)
        plt.imsave('prediction/{}_disparity.png'.format(i), disparity, vmin=0, vmax=126, cmap='magma')

        gt_disparity = val_set.get_disparity(i)[0]
        gt_disparity = gt_disparity.numpy()
        gt_disparity = cv2.resize(gt_disparity, (300, 300))
        cv2.imwrite('prediction/{}_target_raw.png'.format(i), gt_disparity)
        plt.imsave('prediction/{}_target.png'.format(i), gt_disparity, vmin=0, vmax=126, cmap='magma')
        

    print(labels)
    test_image.save("predict.jpg")
    print(disparity.shape)
    print("Cuda:", disparity.is_cuda)
    disparity = disparity[0].cpu().numpy() * 127
    gt_disparity = val_set.get_disparity(1)[0]
    gt_disparity = gt_disparity.numpy()
    gt_disparity = cv2.resize(gt_disparity, (76, 76))
    print("None zero gt disparity:", sum((gt_disparity == 0).flatten()))
    print("Mean normalized gt disparity:", (gt_disparity / 126).flatten().mean())
    print(gt_disparity.shape)
    print(disparity.shape)
    diff = gt_disparity - disparity
    print("Max prediction value:", disparity.max())
    print("Mean prediction value:", disparity.flatten().mean())
    gt_median = np.median(gt_disparity.flatten())
    print("GT median valueL:", gt_median)
    print("GT pixels above median:", (gt_disparity > gt_median).flatten().sum())
    print("GT mean value:", gt_disparity.flatten().mean())
    print("MAE non zero gt:", np.sqrt(diff[gt_disparity != 0] ** 2).flatten().mean())
    print("MAE difference:",abs(diff).flatten().mean())

    print(gt_disparity)
    print(disparity)

    cv2.imwrite('disparity.png', disparity.astype(np.uint8)) 
    cv2.imwrite('disparity-target.png', gt_disparity.astype(np.uint8))
    temp = disparity
    temp = temp * 126 / temp.max()
    cv2.imwrite('test.png', temp)

eval(network_config)
