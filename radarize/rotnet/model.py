#!/usr/bin/env python3

import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):  # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):  # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.channel = EfficientChannelAttention(
            planes
        )  # Efficient Channel Attention module

        self.shortcut = nn.Sequential()
        if (
            stride != 1 or in_planes != self.expansion * planes
        ):  # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        ECA_out = self.channel(out)
        out = out * ECA_out
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECAResNet18(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(ECAResNet18, self).__init__()
        self.in_planes = 64
        num_blocks = [2, 2, 2, 2]
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )  # conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, 32)
        self.fc = FcBlock(32, n_outputs)

        weight_init(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mid_dim=256, dropout=0.05):
        super(FcBlock, self).__init__()
        self.mid_dim = mid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # fc layers
        self.fcs = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim),
            nn.ReLU(True),
            nn.Linear(self.mid_dim, self.out_dim),
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class ResNet34(nn.Module):
    def __init__(self, n_channels, n_outputs):
        super(ResNet34, self).__init__()

        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet34.fc = nn.Linear(512, n_outputs)

    def forward(self, x):
        return self.resnet34(x)


class ResNet18(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = nn.Linear(512, n_outputs)

        # self.resnet18.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # self.resnet18.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, dilation=2, padding=(2,2))
        # self.resnet18.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))
        # self.resnet18.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=2, padding=(2,2))

        # print(self.resnet18)

        weight_init(self)

    def forward(self, x):
        out = self.resnet18(x)
        return out


class ResNet50(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet50, self).__init__()

        # CNN encoder for heatmaps
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet50.fc = nn.Linear(2048, n_outputs)

    def forward(self, x):
        out = self.resnet50(x)
        return out


class ResNet18Nano(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18Nano, self).__init__()

        # CNN encoder for48eatmaps
        resnet18 = models.resnet._resnet(
            "resnet18",
            models.resnet.BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=False,
        )
        resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:5]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(64, n_outputs)

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, x):
        x = self.enc(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18Micro(nn.Module):
    """Model to predict x and y flow from radar heatmaps."""

    def __init__(self, n_channels, n_outputs):
        super(ResNet18Micro, self).__init__()

        # CNN encoder for48eatmaps
        resnet18 = models.resnet._resnet(
            "resnet18",
            models.resnet.BasicBlock,
            [1, 1, 1, 1],
            pretrained=False,
            progress=False,
        )
        resnet18.conv1 = nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:6]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = self.enc(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

