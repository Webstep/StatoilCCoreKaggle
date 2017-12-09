# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import transform
from scipy import ndimage


class ImageTransformer:

    @staticmethod
    def normalize(pic):
        return (pic - pic.min()) / (pic.max() - pic.min())

    @staticmethod
    def log(pic):
        return np.log(pic ** 2)

    @staticmethod
    def square_root(pic):
        return (pic ** 2) ** 0.5

    @staticmethod
    def squared(pic):
        return pic ** 2

    @staticmethod
    def smooth(pic):
        return ndimage.gaussian_filter(pic, sigma=(1, 1), mode='wrap')

    @staticmethod
    def cos(pic):
        return np.cos(pic)

    def _transform(self, pic):
        layers = []

        layers.append(pic.copy())
        layers.append(self.normalize(pic.copy()))
        layers.append(self.log(pic.copy()))
        layers.append(self.square_root(pic.copy()))
        layers.append(self.squared(pic.copy()))
        layers.append(self.smooth(pic.copy()))
        layers.append(self.cos(pic.copy()))

        pic = np.dstack(layers)
        return pic

    def _preprocess(self, X):
        imgs = np.array([
            np.dstack([
                self._transform(img1.reshape(75, 75)),
                self._transform(img2.reshape(75, 75))
            ])
            for (img1, img2) in X.loc[:, ['band_1', 'band_2']].values
        ]).swapaxes(-1, -2).swapaxes(-2, -3)  # Swap to shape (-1, channels, 75, 75) while keeping original orientation
        return imgs


class DoubleDownBase(nn.Module, ImageTransformer):
    def __init__(self, input_channels):
        super(DoubleDownBase, self).__init__()

        torch.manual_seed(0)

        self.conv_out_shape = 14112

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input 1 branch
        self.input1_layer1_conv2d = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=3)
        self.input1_layer2_conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=3)
        self.input1_layer3_conv2d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=3)
        self.input1_layer4_conv2d = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=3)

        self.input1_layer5_fc = nn.Linear(in_features=self.conv_out_shape, out_features=1024)

        # Concatenated branch, fully connected stream
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)


    def forward(self, input1, training=False):

        input1 = (F.relu(self.input1_layer1_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.1, training=training)

        input1 = self.max_pool(F.relu(self.input1_layer2_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.1, training=training)

        input1 = (F.relu(self.input1_layer3_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.1, training=training)

        input1 = self.max_pool(F.relu(self.input1_layer4_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.1, training=training)

        input1 = F.relu(self.input1_layer5_fc(input1.view(-1, self.conv_out_shape)))
        input1 = F.dropout2d(input1, p=0.1, training=training)

        combined = (self.fc1(input1))
        combined = F.dropout2d(combined, p=0.1, training=training)

        combined = F.sigmoid(self.fc2(combined))
        return combined
