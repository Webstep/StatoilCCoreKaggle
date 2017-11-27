# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator

from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class DumbModel(BaseEstimator):

    def fit(self, X, y):
        self.min = y.min()
        self.max = y.max()
        return self

    def predict(self, X):
        return np.random.randint(self.min, self.max, X.shape[0])

    def predict_proba(self, X):
        return np.random.random_sample(X.shape[0])