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


def train_ssd(start_epoch: int, end_epoch: int, config: dict, use_gpu: bool = True, model_name='model',
              checkpoint_folder='checkpoints',
              log_folder='log', redirect_output=True):
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    priors = generate_priors(config)

    target_transform = MatchPrior(priors, config)
    train_transform = transforms.Compose([
        transforms.CustomJitter(),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.RandomMirror()
    ])
    data_transform = transforms.Compose([
        transforms.ToRelativeBoxes(),
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

    disparity_criterion = torch.nn.MSELoss()

    ssd_params = [
        {'params': ssd.extractor.parameters()},
        {'params': ssd.upsampling.parameters()}
    ]

    optimizer = SGD(ssd_params, lr=0.001, momentum=0.1, weight_decay=0.0005)
    lr_scheduler = CosineAnnealingLR(optimizer, 120, last_epoch= -1)
    if os.path.isfile(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))):
        print("Loading previous optimizer")
        optimizer.load_state_dict(
            torch.load(os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(start_epoch - 1))))

    for epoch in range(start_epoch, end_epoch):
        lr_scheduler.step()
        running_loss = 0.0
        running_disparity_loss = 0.0
        num_steps = len(train_loader)

        if redirect_output:
            sys.stdout = open(os.path.join(log_folder, 'train_epoch_{}.txt'.format(epoch)), 'w')

        for i, batch in enumerate(train_loader):
            images, _, _, gt_disparity = batch

            if use_gpu:
                images = images.cuda()
                gt_disparity = gt_disparity.cuda()

            optimizer.zero_grad()
            features = images
            for l in ssd.extractor.get_layers():
                features = l(features)
            disparity = ssd.upsampling(features)
            disparity = disparity.squeeze()


            disparity_loss = torch.sqrt(disparity_criterion.forward(disparity, gt_disparity))
            loss = disparity_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_disparity_loss += disparity_loss

        avg_loss = running_loss / num_steps
        avg_disp_loss = running_disparity_loss / num_steps

        print("Epoch {}".format(epoch))
        print("Average Loss: {:.2f}".format(avg_loss))
        print("Average Disparity Loss: {:.2f}".format(avg_disp_loss))

        torch.save(ssd.state_dict(), os.path.join(checkpoint_folder, "{}_epoch{}.pth".format(model_name, epoch)))
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_folder, "optimizer_epoch{}.pth".format(epoch)))

    if sys.stdout != sys.__stdout__:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

train_ssd(0, 60, network_config, redirect_output=False)
