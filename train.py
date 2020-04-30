import itertools
import os
import sys

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.BerHuLoss import BerHuLoss
from dataset.CityscapesDataset import CityscapesDataset
from network import transforms
from network.MatchPrior import MatchPrior
from network.box_utils import generate_priors, convert_locations_to_boxes
from network.mobilenet_ssd_config import network_config
from network.multibox_loss import MultiBoxLoss
from train_utils import build_model, calculate_map

torch.set_default_dtype(torch.float32)


def train_ssd(start_epoch: int, end_epoch: int, config: dict, use_gpu: bool = True, model_name='mobilenet_ssd',
              checkpoint_folder='checkpoints',
              log_folder='log', redirect_output=True):
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    priors = generate_priors(config)

    target_transform = MatchPrior(priors, config)
    train_transform = transforms.Compose([
        transforms.CustomJitter(),
        transforms.ToOpenCV(),
        transforms.RandomMirror(),
        transforms.ToRelativeBoxes()
    ])
    data_transform = transforms.Compose([
        transforms.Resize(config['image_size']),
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
    disparity_criterion = BerHuLoss()

    ssd_params = [
        {'params': ssd.extractor.parameters()},
        {'params': ssd.extras.parameters()},
        {'params': itertools.chain(ssd.class_headers.parameters(),
                                   ssd.location_headers.parameters(),
                                   ssd.upsampling.parameters())}
    ]

    optimizer = SGD(ssd_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = CosineAnnealingLR(optimizer, 120, last_epoch= -1)
    if os.path.isfile(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))):
        print("Loading previous optimizer")
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))))

    for epoch in range(start_epoch, end_epoch):
        lr_scheduler.step()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        running_disparity_loss = 0.0
        num_steps = len(train_loader)
        aps = torch.zeros((config['num_classes'],))
        running_map = 0

        if redirect_output:
            sys.stdout = open(os.path.join(log_folder, 'train_epoch_{}.txt'.format(epoch)), 'w')

        for i, batch in enumerate(train_loader):
            images, gt_locations, labels, gt_disparity = batch

            if use_gpu:
                images = images.cuda()
                gt_locations = gt_locations.cuda()
                labels = labels.cuda()
                gt_disparity = gt_disparity.cuda()

            optimizer.zero_grad()

            confidences, locations, disparity = ssd(images)

            regression_loss, classification_loss = criterion.forward(confidences, locations, labels, gt_locations)
            disparity_loss = disparity_criterion.forward(disparity, gt_disparity)
            loss = regression_loss + classification_loss + disparity_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            running_disparity_loss += disparity_loss

            with torch.no_grad():
                boxes = convert_locations_to_boxes(locations, priors.cuda(), config['variance'][0],
                                                   config['variance'][1])
                gt_boxes = convert_locations_to_boxes(gt_locations, priors.cuda(), config['variance'][0],
                                                      config['variance'][1])
                batch_map, batch_ap = calculate_map(confidences, labels, boxes, gt_boxes)
                running_map += batch_map
                aps += batch_ap

        avg_loss = running_loss / num_steps
        avg_reg_loss = running_regression_loss / num_steps
        avg_class_loss = running_classification_loss / num_steps
        avg_disp_loss = running_disparity_loss / num_steps
        mean_ap = running_map / num_steps
        epoch_ap = aps / num_steps

        print("Epoch {}".format(epoch))
        print("Average Loss: {:.2f}".format(avg_loss))
        print("Average Regression Loss: {:.2f}".format(avg_reg_loss))
        print("Average Classification Loss: {:.2f}".format(avg_class_loss))
        print("Average Disparity Loss: {:.2f}".format(avg_disp_loss))
        print("Average mAP: {:.2f}".format(mean_ap))
        print("Average AP per class: {}".format(epoch_ap))

        torch.save(ssd.state_dict(), os.path.join(checkpoint_folder, "{}_epoch{}.pth".format(model_name, epoch)))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(epoch)))

    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


starting_epoch = int(input("Starting Epoch: "))
end_epoch = int(input("End Epoch: "))
train_ssd(starting_epoch, end_epoch, network_config, redirect_output=False)
