import torch
from PIL.ImageDraw import Draw
from network import transforms

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.Predictor import Predictor
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

    all_boxes = []
    all_preds = []
    all_probs = []
    all_disparity = []

    map_gt_boxes = []
    map_gt_labels = []
    map_gt_disparity = []

    for i, _ in enumerate(val_set):
        if i % 10 == 0:
            print("Image {}".format(i))
        _, gt_boxes, labels, gt_disparity = val_set[i]
        image = val_set.get_image(i)

        boxes, preds, probs, disparity, indices = net.predict(image)
        if len(boxes) <= 0:
            continue

        all_boxes.append(boxes)
        all_preds.append(preds)
        all_probs.append(probs)
        all_disparity.append(disparity)

        map_gt_boxes.append(gt_boxes[indices])
        map_gt_labels.append(labels[indices])
        map_gt_disparity.append(gt_disparity)

    all_probs = torch.cat(all_probs).view((-1, config['num_classes']))
    map_gt_labels = torch.cat(map_gt_labels).view((-1, 1))
    all_boxes = torch.cat(all_boxes).view((-1, 4))
    map_gt_boxes = torch.cat(map_gt_boxes).view((-1, 4))

    map = calculate_map(all_probs, map_gt_labels, all_boxes, map_gt_boxes)

    print("Mean Average Precision: {}".format(map))

    # test_image = val_set.get_image(0)
    #
    # boxes, labels, conf = net.predict(test_image)
    #
    # drawer = Draw(test_image)
    #
    # for i in range(boxes.shape[0]):
    #     top_left = tuple(boxes[i][:2].numpy().flatten())
    #     bottom_right = tuple(boxes[i][2:].numpy().flatten())
    #     drawer.rectangle((top_left, bottom_right))
    # test_image.show()


with torch.no_grad():
    eval(network_config)
