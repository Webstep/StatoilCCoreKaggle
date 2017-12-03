# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from icc.models.milesg.boosted_base import BoostedGoodnessBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class BoostedGoodness(BoostedGoodnessBase, BaseEstimator):

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        Fit model
        """
        X = self._preprocess(X.copy())
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict classification of X"""
        X = self._preprocess(X.copy())
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probability of X"""
        X = self._preprocess(X.copy())
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return {}
