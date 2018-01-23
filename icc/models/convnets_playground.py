# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from icc.contrib.preprocessing.utils import *
from icc.ml_stack import StackedClassifier

from sklearn.base import BaseEstimator


@StackedClassifier.register
class ConvnetsBox(BaseEstimator):

    def __init__(self, my_model, epochs: int=10, batch_size: int=24, learning_rate: float=1e-4, verbose: int=2,
     wdir='.'):
        self._init_model = my_model
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._wdir = wdir
        self._verbose = verbose


    def fit(self, X: pd.DataFrame, y: pd.Series):
        prep = Preprocess()
        X_filt, y_filt = prep.filter_angle(X, y)
        Xtr, Xval, Ytr, Yval = prep._basic_trainset(X_filt, y_filt, how='deep', test_size=0.1)

        self.model = self._init_model()

        CKPT = 'weights-VAcc_{val_acc:.4f}-TrAcc_{acc:.4f}-VLoss_{val_loss:.4f}-Ep{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(os.path.join(self._wdir, CKPT), monitor='val_acc', save_best_only=True, mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

        self.model.fit(Xtr, Ytr, epochs=self._epochs, verbose=self._verbose, batch_size=self._batch_size,
            validation_data=(Xval, Yval), callbacks=[reduce_lr_loss, checkpoint])

        score = self.model.evaluate(Xval, Yval, verbose=1)
        print('\n Val score:', score[0])
        print('\n Val accuracy:', score[1])

        return self


    def predict(self, X: pd.DataFrame, thresh: float=0.5):
        probabilities = self.predict_proba(X)
        return [1 if p >= thresh else 0 for p in probabilities]


    def predict_proba(self, X: pd.DataFrame):
        return model.predict(X)


    def get_params(self, deep: bool=True):
        return {'epochs': self._epochs, 
            'batch_size': self._batch_size, 
            'learning_rate': self._lr,
            'wdir': self._wdir
            }