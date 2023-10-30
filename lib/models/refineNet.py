import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from lib.models.CA import CoordAtt

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish,self).__init__()
        self.inplace = inplace
    def forward(self,x):
        if self.inplace == True:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x*torch.sigmoid(x)

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        channel_settings = [256,128,64,32]#w32
        # channel_settings = [384, 192, 96, 48]#w48
        self.channel_settings = channel_settings
        laterals = []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
        self.laterals = nn.ModuleList(laterals)
        self.down = nn.Upsample(scale_factor=2, mode='nearest')
        self.swish = Swish()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, input_size,
            kernel_size=1, stride=1, bias=False))
        layers.append(CoordAtt(input_size, input_size))
        layers.append(nn.BatchNorm2d(input_size))
        layers.append(Swish())

        return nn.Sequential(*layers)

    def forward(self, x):
        global_outs = []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[3-i])
            else:
                feature = self.laterals[i](x[3-i]) + up
            if i != len(self.channel_settings) - 1:
                BN = nn.BatchNorm2d(self.channel_settings[i]).cuda()
                up = BN(self.down(feature))
                change = nn.Conv2d(in_channels=self.channel_settings[i], out_channels=self.channel_settings[i+1], kernel_size=1).cuda()
                up = change(up)
                up = self.swish(up)

            global_outs.append(feature)

        return global_outs
