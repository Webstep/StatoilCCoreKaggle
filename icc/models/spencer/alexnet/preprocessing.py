# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Preprocess(object):
    """Main class for preprocessing data fed into AlexNet."""

    def __init__(self):
        self.scaler = None

    def _basic_trainset(self, X: pd.DataFrame, y: pd.Series, test_size: float=0.15):
        """Preprocess data for training."""
        X = self.basic_dstack(X)
        y = np.asarray(y.tolist())

        # Randomize instances in set
        shuffle(X, y)
        
        # Normalize images between 0 and 1.
        self.scaler = FeatureScaling()
        X_scaled = self.scaler.fit_transform(X)

        # Splitting X into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=test_size, random_state=0)

        print("=> Train images shape: {}".format(X_train.shape))
        print("=> Train labels shape: {}\n".format(y_train.shape))
        print("=> Validation images shape: {}".format(X_val.shape))
        print("=> Validation labels shape: {}".format(y_val.shape))
        print("")
        print("=> Train ratio of icebergs-to-ships is {}:{}".format(len(y_train[y_train == 1]),
                                                   len(y_train[y_train == 0])))
        print("=> Validation ratio of icebergs-to-ships is {}:{}\n".format(len(y_val[y_val == 1]),
                                                   len(y_val[y_val == 0])))

        return [X_train, X_val, y_train, y_val]


    def _basic_testset(self, X: pd.DataFrame):
        """Preprocess testset.

        Args:
            X: test set.

        Returns: Rescaled set using training scaling params.
        """
        X = self.basic_dstack(X)
        return self.scaler.transform(X)


    def basic_dstack(self, X: pd.DataFrame):
        """Creates a deep stack of 3 layers to form RGB-like images.

        Args:
            X: must contain column names band_1 and band_2.

        Returns: np.array, shape=(n_samples, height, width, channels)
        """
        assert 'band_1' in X.columns, 'Column error: "band_1" not found in DataFrame'
        assert 'band_2' in X.columns, 'Column error: "band_2" not found in DataFrame'

        # Image dimensions
        width = 75
        height = 75
        n_channels = 3

        band1 = np.asarray(X["band_1"].tolist()).reshape(-1, height, width)
        band2 = np.asarray(X["band_2"].tolist()).reshape(-1, height, width)
        band3 = band1 / band2
        band3 = (band1 + band2) / 2

        band1 = band1[:,:,:,np.newaxis]
        band2 = band2[:,:,:,np.newaxis]
        band3 = band3[:,:,:,np.newaxis]

        return np.concatenate((band1, band2, band3), axis=n_channels)


class FeatureScaling(object):

    def __init__(self):
        self.scaler_param = {}

    def fit_transform(self, X: np.ndarray):
        """Rescaling the range of features to scale the range in [0,1].

        Args:
            X: np.ndarray, data shape=(n_examples, height, width, channels).

        Returns: np.ndarray, Rescaled features.
        """
        n_channels = X.shape[-1]
        self.scaler_param['min'] = np.zeros((n_channels,))
        self.scaler_param['max'] = np.zeros((n_channels,))

        for i in range(n_channels):
            x_min = X[:,:,:,i].min()
            x_max = X[:,:,:,i].max()
            X[:,:,:,i] = (X[:,:,:,i] - x_min) / (x_max - x_min)

            # Save params for test set.
            self.scaler_param['min'][i] = x_min
            self.scaler_param['max'][i] = x_max
        return X


    def transform(self, X: np.ndarray):
        """Use for transforming test set.
        It is assumed you have collected min/max values on a training set.

        Args:
            X: np.ndarray, data shape=(n_examples, height, width, channels).

        Returns: 
        """
        n_channels = X.shape[-1]

        for i in range(n_channels):
            x_min = self.scaler_param['min'][i]
            x_max = self.scaler_param['max'][i]
            X[:,:,:,i] = (X[:,:,:,i] - x_min) / (x_max - x_min)
        return X