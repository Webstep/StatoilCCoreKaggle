# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from scipy import ndimage
from skimage.transform import resize
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Imputer, RobustScaler as Scaler, FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost.sklearn import XGBClassifier


class BoostedGoodnessBase:
    """
    Model using XGBoost Classifier and preprocessing images by normalizing and centering images
    based on the coordinate of their max values
    """

    def __init__(self):
        """
        Model initialization
        """
        self.model = Pipeline(steps=[
            ('imputer', Imputer(strategy='median')),
            ('feature_union', FeatureUnion(n_jobs=1, transformer_list=[
                ('band_1_pipe', Pipeline(steps=[
                    ('band_1', FunctionTransformer(self.band1)),
                    ('band_1_standard_scale', Scaler()),
                ])),
                ('band_2_pipe', Pipeline(steps=[
                    ('band_2', FunctionTransformer(self.band2)),
                    ('band_2_standard_scale', Scaler()),
                ])),
                ('inc_angle', FunctionTransformer(self.angle))
            ])),
            ('xgboost', XGBClassifier(n_estimators=100))
        ])

    def _preprocess(self, X: np.ndarray):
        """
        Preprocess X, alters in a way which does not affect training vs validation sets.
        preprocessing is unique to the image itself without regard between sets.
        """
        # Exploded embedded images so each pixel has its own column, after applying image specific transforms
        for band in ['band_1', 'band_2']:
            X[band] = X[band].map(self.norm_by_pic)
            X[band] = X[band].map(self.transform)
            exploded = X[band].apply(pd.Series)
            exploded.columns = ['{}_pixel_{}'.format(band, col) for col in exploded.columns]
            X = X.join(exploded) \
                 .drop(columns=band)

        X['inc_angle_missing'] = X.inc_angle.map(lambda val: 1 if pd.isnull(val) else 0)
        self.COLUMNS = X.columns
        return X

    def band1(self, x):
        return x[:, [i for i in range(0, len(self.COLUMNS)) if self.COLUMNS[i].startswith('band_1')]]

    def band2(self, x):
        return x[:, [i for i in range(0, len(self.COLUMNS)) if self.COLUMNS[i].startswith('band_2')]]

    def angle(self, x):
        return x[:, [i for i in range(0, len(self.COLUMNS)) if self.COLUMNS[i] in ['inc_angle', 'inc_angle_missing']]]

    @staticmethod
    def norm_by_pic(pic: np.ndarray):
        """
        Normalize the image based on it's own values
        """
        pic = (pic - pic.min()) / (pic.max() - pic.min())
        return pic

    @staticmethod
    def transform(pic: np.ndarray):
        """
        Transform raw band img to centered, and gaussian filter transformation
        reshaped to 32x32 flattened
        """
        # Reshape and apply gaussian filter
        pic = pic.reshape(75, 75)
        pic = ndimage.gaussian_filter(pic, sigma=(2, 2), )

        # List of indexes in both directions to look for max value in image, used for centering
        indxs = np.linspace(5, 70, 60, dtype=int)

        # Find max in each direction, this intersection will be the new center
        x_idx = max([{'idx': i, 'val': pic[:, i].max()} for i in indxs], key=lambda k: k['val'])['idx']
        y_idx = max([{'idx': i, 'val': pic[i, :].max()} for i in indxs], key=lambda k: k['val'])['idx']

        # Check found idx is within bounds given buffer around center
        buffer = 20
        x_idx = x_idx if x_idx - buffer > 0 and x_idx + buffer < 75 else 37
        y_idx = y_idx if y_idx - buffer > 0 and y_idx + buffer < 75 else 37

        # Center based on idx and buffer size
        pic = pic[:, x_idx - buffer:x_idx + buffer]
        pic = pic[y_idx - buffer:y_idx + buffer, :]

        # Resize the image, since transformation above doesn't guarantee same sized outputs.
        pic = resize(pic, output_shape=(32, 32), mode='reflect')
        return pic.flatten()