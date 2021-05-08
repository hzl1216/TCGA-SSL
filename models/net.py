import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.nn.init as init

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_planes, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(places),
        nn.ReLU(inplace=True),
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=2):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(places),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm1d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=9, expansion=2, dropout=0.5):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=1, places=32)

        self.layer1 = self.make_layer(in_places=32, places=16, stride=2)
        self.layer2 = self.make_layer(in_places=32, places=16, stride=2)
        self.layer3 = self.make_layer(in_places=32, places=16, stride=2)
        self.layer4 = self.make_layer(in_places=32, places=16, stride=2)

        self.avgpool = nn.AvgPool1d(7, stride=7)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(896, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), 1, x.size(1))

        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        #        x = self.dropout(x)
        #print(x.shape)
        x = self.fc(x)
        return x


def ResNet50(num_classes):
    return ResNet(num_classes)
