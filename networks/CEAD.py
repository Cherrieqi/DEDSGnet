import torch.nn as nn


class CE(nn.Module):
    def __init__(self, in_ch, out_ch: list):
        super(CE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch[0], kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch[0]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch[0], out_ch[1], kernel_size=3, padding=1, groups=16),
                                   nn.BatchNorm2d(out_ch[1]),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_ch[1], out_ch[2], kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch[2]),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y = self.conv3(y2)

        return y


class DE(nn.Module):
    def __init__(self, in_ch, out_ch: list):
        super(DE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(out_ch[2], out_ch[1], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_ch[1]),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch[1], out_ch[0], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_ch[0]),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch[0], in_ch, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(in_ch),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        y = self.conv(x)

        return y

