import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, channel_sizes, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = channel_sizes[0]
        self.conv1 = nn.Conv2d(1, channel_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_sizes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channel_sizes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channel_sizes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channel_sizes[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_sizes[2], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



# They are specifies as '[size]_[tp]' Where allowed `size`s are {tiny, small, medium, large, huge}
# which correspond networks with {~68k, ~270k, ~1.1M, ~4.2M, ~17M} parameters. And `tp` say how
# the number of channel sizes increase through the network: `tp` = {const, incr, fincr}.
types = {'tiny_const': [20, 20, 20],
         'small_const': [40, 40, 40],
         'medium_const': [80, 80, 80],
         'large_const': [160, 160, 160],
         'tiny_incr': [8, 16, 32],  # 81,512
         'small_incr': [16, 32, 64],  # 297,536
         'medium_incr': [32, 64, 128],  # 1,134,320
         'large_incr': [64, 128, 256],  # 4,426,832
         }


def resnet8(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [1, 1, 1], channel_sizes, num_classes)


def resnet20(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [3, 3, 3], channel_sizes, num_classes)


def resnet32(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [5, 5, 5], channel_sizes, num_classes)


def resnet44(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [7, 7, 7], channel_sizes, num_classes)


def resnet56(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [9, 9, 9], channel_sizes, num_classes)


def resnet110(num_classes=400, tp='small_incr'):
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [18, 18, 18], channel_sizes, num_classes)


def resnet272(num_classes=400, tp='small_incr'):  # 4,380,608
    channel_sizes = types[tp]
    return ResNet(BasicBlock, [45, 45, 45], channel_sizes, num_classes)


if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser(description='Plot ilustrative samples of the task.')
    parser.add_argument('--arch', default='resnet20_small_incr', type=str,
                        help='type of neural network.')
    args, unk = parser.parse_known_args()

    p = 'resnet[0-9]+'
    name = args.arch[slice(*re.match(p, args.arch).regs[0])]
    print(name)
    tp = re.split(p + '_', args.arch)[1]
    net = resnet8(tp='small_incr')

    print(net)
    print('num of parameters = {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    x = torch.rand((10, 1, 32, 32))

    y = net(x)