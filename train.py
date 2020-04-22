import itertools
import os
from time import time

import torch
from PIL.ImageDraw import Draw
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset.CityscapesDataset import CityscapesDataset
from network import transforms
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.box_utils import generate_priors
from network.mobilenet_ssd_config import network_config
from network.multibox_loss import MultiBoxLoss
from train_utils import build_ssd

torch.set_default_dtype(torch.float32)


def train_ssd(config: dict, use_gpu: bool = True):
    priors = generate_priors(config)

    target_transform = MatchPrior(priors, config)
    data_transform = transforms.Compose([
        transforms.CustomJitter(),
        transforms.RandomExpand(),
        transforms.RandomCrop(),
        transforms.RandomMirror(),
        transforms.ToRelativeBoxes(),
        transforms.Resize(config['image_size']),
        transforms.Scale(),
        transforms.ToTensor()
    ])
    train_set = CityscapesDataset('dataset/train', data_transform, target_transform, config)

    num_classes = config['num_classes']

    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)

    ssd = build_ssd(config)
    if use_gpu:
        ssd = ssd.cuda()
    ssd.train(True)
    if os.path.isfile('mobilenet_ssd.pth'):
        ssd.load_state_dict(torch.load('mobilenet_ssd.pth'))
    criterion = MultiBoxLoss(0.5, 0, 3, config)

    ssd_params = [
        {'params': ssd.extractor.parameters(), 'lr': 0.0003},
        {'params': ssd.extras.parameters(), 'lr': 0.0003},
        {'params': itertools.chain(ssd.class_headers.parameters(),
                                   ssd.location_headers.parameters()),
         'lr': 0.0003}
    ]

    optimizer = SGD(ssd_params, lr=0.0003)

    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num_steps = len(train_loader)

    for i, batch in enumerate(train_loader):
        images, boxes, labels = batch

        if use_gpu:
            images = images.cuda()
            boxes = boxes.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        confidences, locations = ssd(images)

        regression_loss, classification_loss = criterion.forward(confidences, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    avg_loss = running_loss / 10
    avg_reg_loss = running_regression_loss / 10
    avg_class_loss = running_classification_loss / 10

    print("Average Loss: {:.2f}".format(avg_loss))
    print("Average Regression Loss: {:.2f}".format(avg_reg_loss))
    print("Average Classification Loss: {:.2f}".format(avg_class_loss))

    torch.save(ssd.state_dict(), "mobilenet_ssd.pth")


def predict(config: dict):
    ssd = build_ssd(config, is_test=True)
    ssd.load_state_dict(torch.load("mobilenet_ssd.pth"))
    ssd.train(False)

    data_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    net = Predictor(ssd, data_transform)

    val_set = CityscapesDataset('dataset/val', data_transform, None, config, True)
    test_image = val_set.get_image(0)

    boxes, labels, conf = net.predict(test_image.unsqueeze(0))

    converter = transforms.ToPILImage()
    test_image = converter(test_image)
    drawer = Draw(test_image)

    for i in range(10):
        print(boxes[0][i])
        print(labels[0][i])
        drawer.rectangle(boxes[0][i].numpy(), fill=(0, 0, 0))

    test_image.show()


for epoch in range(50):
    start = time()
    train_ssd(network_config)
    print("Epoch time: {}".format(time() - start))

# with torch.no_grad():
#     predict(network_config)
