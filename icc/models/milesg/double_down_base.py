# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class DoubleDownBase(nn.Module):
    def __init__(self):
        super(DoubleDownBase, self).__init__()

        torch.manual_seed(0)

        self.conv_out_shape = 2592

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input 1 branch
        self.input1_layer1_conv2d = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=3)
        self.input1_layer2_conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.input1_layer3_conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.input1_layer4_fc = nn.Linear(in_features=self.conv_out_shape, out_features=512)

        # Concatenated branch, fully connected stream
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, input1, training=False):
        input1 = self.max_pool(F.relu(self.input1_layer1_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.2, training=training)

        input1 = self.max_pool(F.relu(self.input1_layer2_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.2, training=training)

        input1 = self.max_pool(F.relu(self.input1_layer3_conv2d(input1)))
        input1 = F.dropout2d(input1, p=0.2, training=training)

        input1 = F.relu(self.input1_layer4_fc(input1.view(-1, self.conv_out_shape)))
        input1 = F.dropout2d(input1, p=0.2, training=training)

        combined = (self.fc1(input1))
        combined = F.dropout2d(combined, p=0.2, training=training)
        combined = self.fc2(combined)

        return combined

    def _preprocess(self, X: pd.DataFrame):
        """
        Preprocess X, same alterations for train and test; per image mods
        """
        X = self._transform(X)
        return X

    def _transform(sef, df):
        images = []
        for i, row in df.iterrows():
            band_1 = np.array(row['band_1']).reshape(75, 75)
            band_2 = np.array(row['band_2']).reshape(75, 75)
            band_3 = band_1 / band_2

            band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
            band_2_norm = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
            band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
            img = np.dstack((band_1_norm, band_2_norm, band_3_norm)) ** 2
            images.append(img)
        return np.array(images)

    def _augment(self, images):
        """Used for creating psuedo training images, DO NOT use for testing images."""
        image_mirror_lr = []
        image_mirror_ud = []
        for i in range(0, images.shape[0]):
            band_1 = images[i, :, :, 0]
            band_2 = images[i, :, :, 1]
            band_3 = images[i, :, :, 2]

            # mirror left-right
            band_1_mirror_lr = np.flip(band_1, 0)
            band_2_mirror_lr = np.flip(band_2, 0)
            band_3_mirror_lr = np.flip(band_3, 0)
            image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))

            # mirror up-down
            band_1_mirror_ud = np.flip(band_1, 1)
            band_2_mirror_ud = np.flip(band_2, 1)
            band_3_mirror_ud = np.flip(band_3, 1)
            image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))

        mirrorlr = np.array(image_mirror_lr)
        mirrorud = np.array(image_mirror_ud)
        images = np.concatenate((images, mirrorlr, mirrorud))
        return images
