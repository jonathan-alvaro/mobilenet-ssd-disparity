import numpy as np
import torch
from PIL import Image

from network import transforms
from torchvision.transforms import ToPILImage

transformer = transforms.Compose([
    transforms.CustomJitter(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.RandomMirror(),
    transforms.ToRelativeBoxes(),
    transforms.Resize(1000),
    transforms.Scale(),
    transforms.ToTensor()
])

test_image = Image.open('../dataset/train/images/aachen_000000_000019_leftImg8bit.png')
test_boxes = np.random.randint(30, 60, (2, 4))
test_labels = np.random.randint(0, 10, (2, 1))
res, _, _ = transformer(test_image, torch.tensor(test_boxes), torch.tensor(test_labels))

tensor_to_img = ToPILImage()
res_img = tensor_to_img(res)

test_image.show()
res_img.show()