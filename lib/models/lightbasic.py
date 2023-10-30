import torch
import torch.nn as nn
BN_MOMENTUM = 0.1

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class light_basic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(light_basic, self).__init__()
        self.dconv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=inplanes)
        self.pconv1 = nn.Conv2d(planes, planes, kernel_size=1, groups=1)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.dconv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=planes)
        self.pconv2 = nn.Conv2d(planes, planes, kernel_size=1, groups=1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.inplanes = inplanes
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        residual = x

        out = self.dconv1(x)
        out = self.pconv1(out)
        out = self.bn1(out)
        # out = self.relu(out)

        out = channel_shuffle(out)

        out = self.dconv2(out)
        out = self.pconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu6(out)

        return out
if __name__ == '__main__':
    x = torch.randn(1, 8, 256, 256)
    y1,y2 = torch.split(x,4,dim=1)
    print(y1.shape)
    print(y2.shape)
    y = torch.cat((y1,y2),dim=1)
    print(y.shape)