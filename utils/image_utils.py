import numpy as np
import pandas as pd
from pathlib import Path
import numbers
import random
import math

from PIL import Image
import cv2
from skimage import transform as skimage_transform
from skimage.util import random_noise

import torch
import torch.nn as nn
from torchvision import transforms as T


'''
Multichannel Tensor Transformer
'''
class CenterCrop(object):
    '''
    Center crop for torch.Tensor
    '''

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        assert len(self.size) == 2

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        c, w, h = img.shape
        c_x, c_y = int(w / 2), int(h / 2)
        b_w, b_h = int(self.size[0] / 2), int(self.size[1] / 2)
        if self.size[0] % 2 == 0:
            x1, x2 = c_x - b_w, c_x + b_w
        else:
            x1, x2 = c_x - b_w, c_x + b_w + 1

        if self.size[1] % 2 == 0:
            y1, y2 = c_y - b_w, c_y + b_w
        else:
            y1, y2 = c_y - b_w, c_y + b_w + 1

        return img[:, x1:x2, y1:y2]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    '''
    Random crop augmentation for torch.Tensor
    '''

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        assert len(self.size) == 2

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        c, w, h = img.shape
        x1, y1 = random.randint(
            0, w - self.size[0]), random.randint(0, h - self.size[1])
        x2, y2 = x1 + self.size[0], y1 + self.size[1]
        return img[:, x1:x2, y1:y2]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomFlip(object):
    '''
    Random flip augmentation for torch.Tensor
    axis: 1 or 2
    '''

    def __init__(self, p=0.5, axis=None):
        self.p = p
        self.axis = axis

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if self.axis is None:
            self.axis = random.randint(1, 2)
        if random.random() < self.p:
            return img.flip(self.axis)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__


class RandomRotate(object):
    def __init__(self, p=0.5, axis=1):
        self.p = p

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if random.random() < self.p:
            rad = random.randint(1, 3)
            if rad == 1:  # 90
                return img.transpose(1, 2).flip(2)
            elif rad == 2:  # 180
                return img.transpose(1, 2)
            elif rad == 3:  # 270
                return img.transpose(1, 2).flip(1)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__


class RandomRotate2(object):
    def __init__(self, p=0.5, angle=None, mode='reflect'):
        self.p = p
        self.angle = angle
        self.mode = mode

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if random.random() < self.p:
            if self.angle is None:
                angle = random.random() * 360.0
            else:
                angle = random.random() * self.angle

            img = img.numpy()
            for i in range(img.shape[0]):
                img[i] = skimage_transform.rotate(
                    img[i], angle, mode=self.mode)
            img = torch.from_numpy(img)
        return img

    def __repr__(self):
        return self.__class__.__name__


class CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        c, w, h = img.shape
        for ch in range(c):
            tmp_img = np.array(img[ch] * 255, dtype=np.uint8)
            img[ch] = torch.from_numpy(self.clahe.apply(tmp_img) / 255)
        return img

    def __repr__(self):
        return self.__class__.__name__


class RandomErase(object):
    '''
    Random erase augmentation for torch.Tensor
    '''

    def __init__(self, area=(0.02, 0.20), aspect=0.3, p=0.5):
        self.area = area
        self.aspect = aspect
        self.p = p

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        c, w, h = img.shape
        if random.random() < self.p:
            for i in range(100):  # try until box is inside img
                img_area = w * h

                target_area = random.uniform(*self.area) * img_area
                aspect_ratio = random.uniform(self.aspect, 1 / self.aspect)

                h_t = int(round(math.sqrt(target_area * aspect_ratio)))
                w_t = int(round(math.sqrt(target_area / aspect_ratio)))

                if w_t < w and h_t < h:
                    x1 = random.randint(0, h - h_t)
                    y1 = random.randint(0, h - w_t)
                    for ch in range(c):
                        tmp_img = np.array(img[ch])
                        c_max = tmp_img.max()
                        tmp_img[x1:x1 + h_t, y1:y1 +
                                w_t] = np.random.rand(h_t, w_t) * c_max
                        img[ch] = torch.from_numpy(tmp_img)
        return img

    def __repr__(self):
        return self.__class__.__name__


class RandomNoise(object):
    '''
    Random noise augmentation for torch.Tensor
    '''

    def __init__(self, p=0.5, mode='gaussian'):
        self.p = p
        self.mode = mode

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if random.random() < self.p:
            img = random_noise(img, mode=self.mode, var=1e-3)
            img = torch.from_numpy(img).float()
        return img

    def __repr__(self):
        return self.__class__.__name__


class RandomBrightness(object):
    def __init__(self, p=0.5, b_range=(0.75, 1.25)):
        self.p = p
        self.b_range = b_range

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if random.random() < self.p:
            img = img * random.uniform(*self.b_range)
        return img.clamp(0, 1)

    def __repr__(self):
        return self.__class__.__name__


class RandomResize:
    def __init__(self, p=0.5, ratio=None):
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        if random.random() < self.p:
            if self.ratio is None:
                ratio = random.uniform(1.0, 1.25)
            else:
                ratio = random.uniform(*self.ratio)

            img = img.numpy()
            h_resized = int(img.shape[1] * ratio)
            w_resized = int(img.shape[2] * ratio)

            img_resized = np.stack(
                [cv2.resize(img[i], (h_resized, w_resized)) for i in range(img.shape[0])])
            img = torch.from_numpy(img_resized)
        return img

    def __repr__(self):
        return self.__class__.__name__


def get_transform(aug_level=0, img_size=384, prep=None):
    if prep is not None:
        _img_t = [prep]
    else:
        _img_t = []

    if aug_level == 0:
        img_t = [CenterCrop(img_size)] + _img_t
        img_tt = img_t
    elif aug_level == 1:
        img_t = [RandomCrop(img_size), RandomFlip(), RandomRotate()] + _img_t
        img_tt = img_t
    elif aug_level == 2:
        img_t = [RandomRotate2(), RandomCrop(img_size), RandomFlip(),
                 RandomRotate()] + _img_t
        img_tt = img_t
    elif aug_level == 3:
        img_t = [RandomResize(), RandomCrop(img_size), RandomRotate2(), RandomFlip(),
                 RandomRotate(), RandomBrightness()] + _img_t
        img_tt = img_t
    elif aug_level == 4:
        img_t = [RandomResize(), RandomCrop(img_size), RandomRotate2(), RandomFlip(),
                 RandomRotate(), RandomBrightness(), RandomErase()] + _img_t
        img_tt = [RandomResize(), RandomCrop(img_size), RandomRotate2(), RandomFlip(),
                  RandomRotate(), RandomBrightness()] + _img_t  # no Random Erase

    return T.Compose(img_t), T.Compose(img_tt)
