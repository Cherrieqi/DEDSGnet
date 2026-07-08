import torch
import torch.nn as nn

from networks.CEAD import CE, DE
from networks.IFEH import IFEH
from networks.IFEM import IFEM

class DEDSGnet(nn.Module):
    def __init__(self, in_ch, out_ch, out_ch_cede, out_ch_ifem, class_num=4, slice_size=3):
        """
        :param in_ch: img输入波段数, 激励的波段数为其三倍
        :param out_ch: 分类器输出波段
        :param out_ch_cede: list cede的各阶段输出波段数
        :param out_ch_ifem: list IFEM的各阶段输出波段数
        :param class_num: 类别数
        :param slice_size: 激励 patch大小
        """
        super(DEDSGnet, self).__init__()

        self.ce = CE(in_ch, out_ch_cede)
        self.de = DE(in_ch, out_ch_cede)
        self.ifeh = IFEH(in_ch, in_ch)
        self.ifem = IFEM(in_ch, out_ch_ifem)

        self.classifier = nn.Sequential(nn.Conv2d(out_ch_ifem[3], out_ch[0], kernel_size=slice_size, padding=0),
                                        nn.Conv2d(out_ch[0], out_ch[1], kernel_size=1, padding=0),
                                        nn.Conv2d(out_ch[1], class_num, kernel_size=1, padding=0))


    def forward(self, x, mode='test'):

        if mode != 'test':

            x_ce = self.ce(x)
            x_ex = self.de(x_ce)
            x_ex_ce = self.ce(x_ex)

            x_img = self.ifeh(x)
            y, y1, y2, y3, y4 = self.ifem(x_img)
            x_ex = self.ifeh(torch.sigmoid(x_ex))
            y_ex, y1_ex, y2_ex, y3_ex, y4_ex = self.ifem(x_ex)

            # classifier
            y = self.classifier(y)
            y = y.squeeze(-1).squeeze(-1)

            y_ex = self.classifier(y_ex)
            y_ex = y_ex.squeeze(-1).squeeze(-1)

            return x_ce, x_ex_ce, \
                   y, y1, y2, y3, y4, \
                   y_ex, y1_ex, y2_ex, y3_ex, y4_ex

        else:
            x = self.ifeh(x)
            y, __, __, __, __ = self.ifem(x)
            y = self.classifier(y)
            y = y.squeeze(-1).squeeze(-1)

            return y



