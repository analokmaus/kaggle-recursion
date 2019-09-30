import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from .backbones import *
from .metrics import *
from .resnet_cbam import *
from .senet import *


class AngleSimpleLinear(nn.Module):
    '''
    Computes cos of angles between input vectors and weights vectors
    '''

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class XNetAngular(nn.Module):
    '''
    Deep Metric Learning with Any CNN model
    '''

    def __init__(self, base_model, in_channel=6, embedding_size=256,
                 num_classes=1108, feature=True, angular=True):
        super(XNetAngular, self).__init__()

        self.bn_first = nn.BatchNorm2d(in_channel)
        self.feature = feature
        self.model = base_model
        self.embedding_size = embedding_size

        if not self.feature:
            if angular:
                self.fc_angular = AngleSimpleLinear(
                    self.embedding_size, num_classes)
            else:
                self.fc_angular = nn.Linear(
                    self.embedding_size, num_classes)

    def forward(self, x):

        x = self.bn_first(x)
        x = self.model(x)

        # if self.feature or not self.training:
        if self.feature:
            return x

        x = x.view(x.size(0), -1)
        y = self.fc_angular(x)

        return x, y

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.


def get_model(model_type, num_classes, angular=True, feature=False):
    if model_type in ['baseline', 'alpha']:
        base_model = densenet_mod(
            models.densenet201, in_channel=6, num_classes=256, pretrained=True)
        model = XNetAngular(base_model, in_channel=6, embedding_size=256,
                            num_classes=num_classes, feature=feature, angular=angular)
    elif model_type == 'beta':
        base_model = enet_mod(model_size=4, in_channel=6, num_classes=256)
        model = XNetAngular(base_model, in_channel=6, embedding_size=256,
                            num_classes=num_classes, feature=feature, angular=angular)
    elif model_type == 'chi':
        base_model = xception_mod(
            in_channel=6, num_classes=1000, pretrained=True)
        model = XNetAngular(base_model, in_channel=6, embedding_size=1000,
                            num_classes=num_classes, feature=feature, angular=angular)
    elif model_type == 'rho':
        base_model = resnet_mod(
            resnet101_cbam, in_channel=6, num_classes=512, pretrained=True)
        model = XNetAngular(base_model, in_channel=6, embedding_size=512,
                            num_classes=num_classes, feature=feature, angular=angular)
    elif model_type == 'sigma':
        base_model = senet_mod(
            se_resnext101_32x4d, in_channel=6, num_classes=512, pretrained=True)
        model = XNetAngular(base_model, in_channel=6, embedding_size=512,
                            num_classes=num_classes, feature=feature, angular=angular)
    else:
        raise ValueError()

    return model
