# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer, Imputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from icc.ml_stack import StackedClassifier

try:
    # Try to import the compiled rust extension, if not, try to build it.
    from icc.models.milesg.rust.rust_models.rust import models
except ImportError:
    from subprocess import Popen, PIPE
    import warnings
    warnings.warn('Unable to import compiled rust extension, trying to build it!')
    dir = os.path.dirname(os.path.abspath(__file__))
    rust_setup_dir = os.path.join(dir, 'milesg', 'rust')
    with Popen(['bash', '-c', 'cd {} && python setup.py develop'.format(rust_setup_dir)], stdout=PIPE) as p:
        for line in p.stdout:
            print(line, end='')
    if p.returncode == 0:
        from icc.models.milesg.rust.rust_models.rust import models
    else:
        raise OSError('Unable to build rust extension!')


@StackedClassifier.register
class RustyLogistic(BaseEstimator):
    """
    Concept model, to demonstrate ability to connect to rust-lang model.
    """

    def __init__(self):
        """
        Interface to Rust logistic regression
        """

    def get_params(self, deep=True):
        return {}

    def _format_x(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Format x so that it represents unnested arrays of source data

        Parameters
        ----------
        X: pd.DataFrame - Data frame resulting form the icc.data_loader.DataLoader class

        Returns
        -------
        pd.DataFrame with each pixel from each band in it's own column
        """
        bands = ['band_1', 'band_2']
        for band in bands:
            unstacked = X[band].apply(pd.Series).stack().unstack()
            unstacked.columns = ['{}_{}'.format(band, i) for i in range(unstacked.shape[1])]
            X = X.join(unstacked)

        X.drop(columns=bands + ['inc_angle'], inplace=True)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit to training data
        """
        self.xTrain = self._format_x(X.copy())
        self.yTrain = y.copy()


        # Pipeline
        self.pipeline = Pipeline(steps=[
            ('qt', QuantileTransformer(output_distribution='normal')),
            ('pca', PCA(200))
        ])
        self.xTrain = self.pipeline.fit_transform(self.xTrain)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get classification predictions for X
        """
        probs = self.predict_proba(X)
        return np.array((1 if p[1] > 0.5 else 0 for p in probs))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability for X

        'rusty-machine' presently has no way to save a trained model, so we're sending over training with
        the testing data at once.
        """
        X = self._format_x(X.copy())
        X = self.pipeline.transform(X)
        output = models.logistic_regression(X, self.xTrain, self.yTrain.values)
        return np.array([[1-p, p] for p in output])
