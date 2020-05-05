import math

import torch
from PIL.ImageDraw import Draw
from network import transforms

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
from network.box_utils import nms
from network.mobilenet_ssd_config import network_config, priors
from train_utils import build_model, calculate_map


def eval(config: dict, model_path='checkpoints/model_epoch40.pth'):
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

    for i, _ in enumerate(val_set):
        if i % 10 == 0:
            print("Image {}".format(i))

        image = val_set.get_image(i)

        probs, boxes, disparity = net.predict(image)
        labels = torch.argmax(probs, dim=probs.dim() - 1)

        chosen_indices = []

        for class_index in range(1, config['num_classes'] + 1):
            class_mask = labels == class_index

            # If there's no prediction in this class, skip the class
            if class_mask.long().sum() <= 0:
                continue

            class_probabilities = probs[class_mask, class_index]
            class_boxes = boxes[class_mask]

            class_indices = nms(class_boxes, class_probabilities, 0.5)
            chosen_indices.append(class_indices)

        chosen_indices = torch.cat(chosen_indices)

        probs = probs[chosen_indices]
        boxes = boxes[chosen_indices]

        box_drawer = Draw(image)

        for box in boxes:
            top_left = tuple(box[:2].numpy().flatten())
            bottom_right = tuple(box[2:].numpy().flatten())
            box_drawer.rectangle((top_left, bottom_right))

        image.save('result.jpg')

        # TODO change to all image evaluation
        break


with torch.no_grad():
    eval(network_config)


def eval_disparity(config: dict, model_path='checkpoints/model_epoch30.pth'):
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

    errors = []

    for i, _ in enumerate(val_set):
        image = val_set.get_image(i)
        gt_disparity = val_set.get_disparity(i)

        _, _, _, _, disparity = net.predict(image)

        error = ((gt_disparity - disparity) ** 2).flatten()

        errors.append(error)

    errors = torch.cat(errors)
    print("RMSE: {}".format(math.sqrt(errors.mean().item())))
