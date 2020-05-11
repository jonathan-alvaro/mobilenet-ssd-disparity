import json

from dataset.CityscapesDataset import CityscapesDataset
from network.MatchPrior import MatchPrior
from network.box_utils import generate_priors
from network.mobilenet_ssd_config import network_config
from network import transforms

priors = generate_priors(network_config)

target_transform = MatchPrior(priors, network_config)
# train_transform = transforms.Compose([
#     transforms.CustomJitter(),
#     transforms.RandomExpand(),
#     transforms.RandomCrop(),
#     transforms.RandomMirror()
# ])
data_transform = transforms.Compose([
    transforms.ToOpenCV(),
    transforms.ToRelativeBoxes(),
    transforms.Resize(network_config['image_size']),
    transforms.Scale(),
    transforms.ToTensor()
])
train_set = CityscapesDataset(network_config, 'dataset/train', None,
                              data_transform, target_transform)

labels = []

for i in range(len(train_set)):
    _, _, img_labels, _ = train_set[i]
    labels.append(img_labels.tolist())

json.dump(labels, open('prior_assignment.json', 'w'))
