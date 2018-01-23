# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.base import BaseEstimator

from icc.ml_stack import StackedClassifier
from icc.models.spencer.transfer_learning.keras_app_models import *
from icc.contrib.preprocessing.utils import *


@StackedClassifier.register
class TransferLearnModel(BaseEstimator):

    def __init__(self, basenet: str='VGG16', verbose=True, save_bottleneck_feats: bool=False,
        path_to_bottleneck_feats=None, path_to_top_weights=None, apply_transfer: bool=True, apply_finetune: bool=True):
        self._basenet = basenet
        self._verbose = verbose
        self._save_bnfeats = save_bottleneck_feats
        self._path_to_bottleneck_feats = path_to_bottleneck_feats
        self._path_to_top_weights = path_to_top_weights
        self._apply_finetune = apply_finetune
        self._apply_transfer = apply_transfer
        self.model = None


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train a new fully-connected top classifier on top of a convolutional model that has been trained with Imagenet."""
 
        # Preprocessing step
        self.prep = Preprocess()
        X_train, _, y_train, _ = self.prep._basic_trainset(X, y, how='deep', test_size=0, verbose=self._verbose)

        # Initialize Keras application object
        self.model = AppModel(basenet=self._basenet)

        # ?: Need to save those output features from the convolutional part of trained model. 
        if self._save_bnfeats:
            self.model.save_bottleneck_features(X_train, y_train, filename=self._path_to_bottleneck_feats)
            self._path_to_bottleneck_feats = self.model._path_to_bnfeats

        # Perform transfer learning on top part of classifer. Run only when you have bottleneck features saved.
        if self._apply_transfer:
            self.model.run_transferlearning(bottleneck_features_path=self._path_to_bottleneck_feats, 
                            top_weights_path=self._path_to_top_weights)
            self._path_to_top_weights = self.model._top_model_weights
            print('=> Transfer done.\n')

        if self._apply_finetune:
            # Now finetune your model.
            self.model.run_finetuning(X_train, y_train, top_weights_path=self._path_to_top_weights, epochs=10)
            print('=> Finetuning done.\n')

        return self


    def get_params(self, deep: bool=True):
        """Get parameters for this estimator.

        Returns:  If True, will return the parameters and subobjects that are estimators.
        """
        return {'basenet': self._basenet}


    def predict(self, X: pd.DataFrame, thresh: float=0.5):
        """Get binary prediction output.

        Args:
            X: data set.
            thresh: set float to sort instances into classes, default 0.5.

        Returns: np.ndarray, binary predictions for is_iceberg class.
        """
        probs = self.predict_proba(X)
        return np.array([1 if p > thresh else 0 for p in probs])


    def predict_proba(self, X: pd.DataFrame):
        """Compute probabilities for testset.

        Returns: probabilities computed using softmax.
        """
        X_scaled = self.prep._basic_testset(X)
        return self.model.stacked_model.predict(X_scaled, batch_size=200)




