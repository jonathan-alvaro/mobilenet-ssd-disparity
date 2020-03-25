import torch
import torch.nn.functional as F
import torch.nn as nn


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def bn_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def dw_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.model = nn.Sequential(
            bn_conv(3, 32, 2),
            dw_conv(32, 64, 1),
            dw_conv(64, 128, 2),
            dw_conv(128, 128, 1),
            dw_conv(128, 256, 2),
            dw_conv(256, 256, 1),
            dw_conv(256, 512, 2),
            dw_conv(512, 512, 1),
            dw_conv(512, 512, 1),
            dw_conv(512, 512, 1),
            dw_conv(512, 512, 1),
            dw_conv(512, 512, 1),
            dw_conv(512, 1024, 2),
            dw_conv(1024, 1024, 1),
        )

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        return x

    def load(self, path):
        self.model.load_state_dict(torch.load(path))