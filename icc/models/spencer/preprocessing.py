# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


def basic_vstack_band1_band2(X: pd.DataFrame):
    """Vertically stack columns band_1 and band_2. 
    We leave out 'Inc_angle' data.

    Args:
        X: must contain column names band_1 and band_2.
    """
    assert 'band_1' in X.columns
    assert 'band_2' in X.columns

    band1 = np.asarray(X["band_1"].tolist())
    band2 = np.asarray(X["band_2"].tolist())
    return np.concatenate((band1, band2), axis=0) 


def reshape_grayscale_images(X: np.ndarray):
    """Reshape images from 1D to 3D for model consumption.

    Args:
        X: Images to be reshaped.
        format: Input images are grayscale.

    Returns:
        Formatted images as np.ndarray.
    """
    num_examples = X.shape[0]
    img_height = 75
    img_width = 75
    X = X.reshape(num_examples, img_height, img_width)
    
    # Add extra dim to satisfy channel input.
    return X[:,:,:, np.newaxis]


class Preprocess(object):

    def __init__(self):
        self.scaler_name = None
        self.scaler = None


    def _basic_trainset(self, X: pd.DataFrame, y: pd.Series):
        """Preprocess data for training."""

        X = basic_vstack_band1_band2(X)
        y = np.asarray(y.tolist() * 2)

        # Randomize instances in set
        shuffle(X, y)
        
        # Normalize images between 0 and 1. 
        # Note: Use the same scaler on the testset.
        self.scaler_name = StandardScaler.__name__
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        print("{} images in the train.json set".format(X.shape[0]))

        # Splitting X into training and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0)

        # Reshape images to (batch size, height, width, channels).
        X_train = reshape_grayscale_images(X_train)
        X_val = reshape_grayscale_images(X_val)

        print("{} images in the train set".format(X_train.shape[0]))
        print("{} images in the validation set".format(X_val.shape[0]))

        print("Train images shape: {}".format(X_train.shape))
        print("Train labels shape: {}".format(y_train.shape))
        print("Validation images shape: {}".format(X_val.shape))
        print("Validation labels shape: {}".format(y_val.shape))

        print("")
        print("Train ratio of icebergs-to-ships is {}:{}".format(len(y_train[y_train == 1]),
                                                   len(y_train[y_train == 0])))
        print("Validation ratio of icebergs-to-ships is {}:{}".format(len(y_val[y_val == 1]),
                                                   len(y_val[y_val == 0])))

        return [X_train, X_val, y_train, y_val]


    def _basic_testset(self, X: pd.DataFrame):
        """Preprocess data for testing."""
        X = basic_vstack_band1_band2(X)
        return self.scaler.transform(X)