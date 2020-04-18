import torch
from torch import nn

from network.mobilenet import MobileNet
from network.ssd import SSD


def forward_img(net: SSD, img: torch.Tensor):
    res = net(img)
    return res


def build_ssd(config: dict, is_test: bool = False) -> SSD:
    extra_layers = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    ])

    location_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1)
    ])

    classification_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=6 * config['num_classes'], kernel_size=3, padding=1)
    ])

    mobilenet = MobileNet()
    mobilenet.load('mobilenetv1.pth')

    ssd = SSD(
        config['num_classes'], [11, 13], mobilenet, extra_layers,
        location_headers, classification_headers, config, is_test
    )
    return ssd
