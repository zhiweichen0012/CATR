import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as S


BN_MOMEMTUM = 0.01


class NEWConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(NEWConv, self).__init__()
        self.UConv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out, momentum=BN_MOMEMTUM),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.UConv(x)


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super(Up, self).__init__()

        self.conv = NEWConv(c_in, c_out)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class cnn(nn.Module):
    def __init__(self, c_in, n_classes, c_base=4):
        super(cnn, self).__init__()

        self.down = S(
            nn.MaxPool2d(2),
            NEWConv(c_in, c_base * 2),
        )
        self.up = Up(c_base * 2 + c_in, n_classes)

    def forward(self, x1):
        x2 = self.down(x1)
        x = self.up(x2, x1)
        return x
