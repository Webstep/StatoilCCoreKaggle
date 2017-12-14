# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from skimage import transform
from scipy import ndimage
from sklearn.preprocessing import RobustScaler as Scaler
from sklearn.pipeline import Pipeline


class ImageTransformer:

    # sklearn preprocessing pipe
    pipe = Pipeline(steps=[
        ('sc', Scaler())
    ])

    @staticmethod
    def normalize(pic):
        return (pic - pic.min()) / (pic.max() - pic.min())

    @staticmethod
    def log(pic):
        return np.log(pic ** 2)

    @staticmethod
    def square_root(pic):
        return (pic ** 2) ** 0.25

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

        # Crop
        pic = transform.resize(pic[15:60, 15:60], output_shape=(75, 75), mode='wrap')

        layers.append(pic.copy())
        layers.append(self.normalize(pic.copy()))
        layers.append(self.log(pic.copy()))
        layers.append(self.square_root(pic.copy()))
        layers.append(self.squared(pic.copy()))
        layers.append(self.smooth(pic.copy()))
        layers.append(self.cos(pic.copy()))

        pic = np.dstack(layers)
        return pic

    def _preprocess(self, X: pd.DataFrame, y: np.ndarray=None):

        imgs = np.array([
            np.dstack([
                self._transform(img1.reshape(75, 75)),
                self._transform(img2.reshape(75, 75))
            ])
            for (img1, img2) in X.loc[:, ['band_1', 'band_2']].values
        ]).swapaxes(-1, -2).swapaxes(-2, -3)  # Swap to shape (-1, channels, 75, 75) while keeping original orientation

        if y is not None:
            y = y if not hasattr(y, 'values') else y.values
            # Augment by flipping images
            upside_down = np.concatenate([np.flipud(img).reshape(1, -1, 75, 75) for img in imgs])
            imgs = np.vstack((upside_down, imgs))
            y = np.concatenate((y, y))

        return imgs if y is None else (imgs, y)


class HighStackBase(nn.Module, ImageTransformer):

    def __init__(self, input_channels):
        super(HighStackBase, self).__init__()

        torch.manual_seed(0)

        self.conv_out_shape = 1152

        # Pooling layers
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv layers
        self.conv_1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=3)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=3)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=3)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=3)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=self.conv_out_shape, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=1)


    def forward(self, x, training=False):

        x = F.relu(self.conv_1(x))
        x = self.max_pool(x)
        x = F.dropout2d(x, p=0.1, training=training)

        x = F.relu(self.conv_2(x))
        x = self.max_pool(x)
        x = F.dropout2d(x, p=0.1, training=training)

        x = F.relu(self.conv_3(x))
        x = self.max_pool(x)
        x = F.dropout2d(x, p=0.1, training=training)

        x = F.relu(self.conv_4(x))
        x = self.max_pool(x)
        x = F.dropout2d(x, p=0.1, training=training)

        x = F.relu(self.fc1(x.view(-1, self.conv_out_shape)))
        x = F.dropout2d(x, p=0.1, training=training)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x
