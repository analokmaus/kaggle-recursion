'''
cosFace implementation gently borrowed from
https://github.com/grib0ed0v/face_recognition.pytorch
'''
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(input_values, gamma, mixup=False):
    '''
    Computes the focal loss
    '''
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    if mixup:
        return loss
    else:
        return loss.mean()


class AMSoftmaxLoss(nn.Module):
    '''
    Computes the AM-Softmax loss with cos or arc margin
    '''
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', gamma=0., m=0.5, s=30, t=1., mixup=False):
        super(AMSoftmaxLoss, self).__init__()
        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert m > 0
        self.m = m
        assert s > 0
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        assert t >= 1
        self.t = t
        self.mixup = mixup

    def forward(self, cos_theta, target):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.m
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - \
                sine * self.sin_m  # cos(theta+m)
            phi_theta = torch.where(
                cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            if self.mixup:
                return F.cross_entropy(self.s * output, target, reduction='none')
            else:
                return F.cross_entropy(self.s * output, target)

        if self.t > 1:
            h_theta = self.t - 1 + self.t * cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1,
                                                                    1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            return F.cross_entropy(self.s * output, target)

        return focal_loss(F.cross_entropy(self.s * output, target, reduction='none'), self.gamma, self.mixup)
