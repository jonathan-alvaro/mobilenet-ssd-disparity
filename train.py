import itertools
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.BerHuLoss import BerHuLoss
from dataset.CityscapesDataset import CityscapesDataset
from dataset.CustomRMSE import CustomRMSE
from network import transforms
from network.MatchPrior import MatchPrior
from network.mobilenet_ssd_config import network_config, priors
from network.multibox_loss import MultiBoxLoss
from train_utils import build_model

torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

def train_ssd(start_epoch: int, end_epoch: int, config: dict, use_gpu: bool = True, model_name='model',
              checkpoint_folder='checkpoints',
              log_folder='log', redirect_output=True):
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    target_transform = MatchPrior(priors, config)
    train_transform = transforms.Compose([
        transforms.CustomJitter(),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.RandomMirror()
    ])
    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
        transforms.Resize(config['width'], config['height']),
        transforms.Scale(),
        transforms.ToTensor()
    ])
    train_set = CityscapesDataset(config, 'dataset/train', train_transform,
                                  data_transform, target_transform)

    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)

    ssd = build_model(config)
    if use_gpu:
        ssd = ssd.cuda()
    ssd.train(True)
    if os.path.isfile(os.path.join(checkpoint_folder, "{}_epoch{}.pth".format(model_name, start_epoch - 1))):
        ssd.load_state_dict(
            torch.load(os.path.join(checkpoint_folder, "{}_epoch{}.pth".format(model_name, start_epoch - 1))))

    criterion = MultiBoxLoss(0.5, 0, 3, config)
    disparity_criterion = torch.nn.MSELoss()

    ssd_params = [
        {'params': ssd.extractor.parameters()},
        {'params': ssd.extras.parameters(), 'lr': 0.001},
        {'params': ssd.class_headers.parameters(), 'lr': 0.001},
        {'params': ssd.location_headers.parameters(), 'lr': 0.00001},
        {'params': ssd.upsampling.parameters(), 'lr': 0.0005}
    ]

    optimizer = SGD(ssd_params, lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
    lr_scheduler = CosineAnnealingLR(optimizer, 60, eta_min=0, last_epoch=-1)
    if os.path.isfile(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))):
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))))

    for epoch in range(start_epoch, end_epoch):
        label_count = [0, 0, 0, 0, 0, 0, 0, 0]
        prediction_count = [0, 0, 0, 0, 0, 0, 0]

        lr_scheduler.step()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        running_disparity_loss = torch.Tensor([0, 0, 0])
        num_steps = len(train_loader)

        if redirect_output:
            sys.stdout = open(os.path.join(log_folder, 'train_epoch_{}.txt'.format(epoch)), 'w')

        for i, batch in enumerate(train_loader):
            images, gt_locations, labels, gt_disparity = batch

            if use_gpu:
                images = images.cuda()
                gt_locations = gt_locations.cuda()
                labels = labels.cuda()
                gt_disparity = gt_disparity.cuda()

            gt_disparity = gt_disparity
            optimizer.zero_grad()

            confidences, locations, disparities = ssd(images)

            regression_loss, classification_loss, mask = criterion.forward(confidences, locations, labels, gt_locations)
            with torch.no_grad():
                masked_labels = labels[mask]
                train_labels, train_counts = masked_labels.unique(return_counts=True)
                predictions = torch.argmax(confidences, dim=confidences.dim() - 1)
                prediction_labels, prediction_counts = predictions.unique(return_counts=True)
                for i, item in enumerate(train_labels):
                    label_count[item.item()] += train_counts[i].item()
                for i, item in enumerate(prediction_labels):
                    prediction_count[item.item()] += prediction_counts[i].item()

            disparity_losses = []
            for i in range(len(disparities)):
                disparity = disparities[i] * 223
                scale_gt_disparity = []
                for img in gt_disparity:
                    if i == len(disparities) - 1:
                        break
                    shape = disparity.shape[-2:]
                    scale_gt_disparity.append(cv2.resize(img.cpu().squeeze().numpy(), (shape[1], shape[0])))

                scale_gt_disparity = torch.from_numpy(np.array(scale_gt_disparity))
                if disparity.is_cuda:
                    scale_gt_disparity = scale_gt_disparity.cuda()
                    
                if i == len(disparities) - 1:
                    disparity_losses.append(disparity_criterion(disparity.squeeze(), gt_disparity))
                else:
                    disparity_losses.append(
                        disparity_criterion(disparity.squeeze(), scale_gt_disparity.squeeze())
                    )

                if (disparity.abs() > 1000).any():
                    print("Prediction{}:".format(i))
                    print(disparity)
                    print(images)
                    print(disparity_losses)
                    raise ValueError

            loss = regression_loss + 2 * classification_loss + sum(disparity_losses)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            running_disparity_loss += torch.Tensor(disparity_losses)

        avg_loss = running_loss / num_steps
        avg_reg_loss = running_regression_loss / num_steps
        avg_class_loss = running_classification_loss / num_steps
        avg_disp_loss = running_disparity_loss / num_steps

        print("Epoch {}".format(epoch))
        print("Average Loss: {:.2f}".format(avg_loss))
        print("Average Regression Loss: {:.2f}".format(avg_reg_loss))
        print("Average Classification Loss: {:.2f}".format(avg_class_loss))
        print("Average Disparity Loss: {}".format(avg_disp_loss))
        print("Training label: {}".format(label_count))

        torch.save(ssd.state_dict(), os.path.join(checkpoint_folder, "{}_epoch{}.pth".format(model_name, epoch)))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(epoch)))

    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


starting_epoch = int(input("Starting Epoch: "))
end_epoch = int(input("End Epoch: "))
train_ssd(starting_epoch, end_epoch, network_config, redirect_output=True)
