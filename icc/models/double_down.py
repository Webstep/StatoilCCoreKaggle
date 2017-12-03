# -*- coding: utf-8 -*-

import os; os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
from sklearn.base import BaseEstimator
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from keras.preprocessing.image import ImageDataGenerator
from icc.models.milesg.double_down_base import DoubleDownBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class DoubleDown(DoubleDownBase, BaseEstimator):

    def __init__(self, n_epoch=35, batch_size=100):
        super().__init__()
        self.model = DoubleDownBase()
        self.model.cuda()
        self.n_epoch = n_epoch
        self.batch_size = batch_size

    def get_params(self, deep=True):
        return {'n_epoch': self.n_epoch, 'batch_size': self.batch_size}

    def fit(self, X, y):
        """
        Fit to X given y
        """
        X = self._preprocess(X)
        X = self._augment(X)
        y = np.concatenate((y, y, y))
        self._train(X, y)
        return self

    def predict_proba(self, X):
        """Predict probability for X"""
        X = self._preprocess(X)
        probabilities = np.zeros(X.shape[0])
        for batch_idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
            batch = Variable(torch.FloatTensor(X[batch_idx:batch_idx+self.batch_size].reshape(-1, 3, 75, 75)).cuda())
            probs = self.model(batch).data.cpu().numpy().squeeze()
            probabilities[batch_idx:batch_idx+self.batch_size] = probs
        probabilities = np.array([[1-p, p] for p in probabilities])
        return probabilities

    def predict(self, X):
        """Predict binary outcome of X"""
        probs = self.predict_proba(X)
        return np.array([1 if p > 0.5 else 0 for p in probs])

    def _train(self, X, y):
        """
        Train model given processed X and y
        """
        print('{}: NOTE, this uses pseudo image creation, so training loss is cray-cray. ;)'
              .format(self.__class__.__name__))
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)
        criterion = nn.BCEWithLogitsLoss()

        datagen_params = dict(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.1,
            channel_shift_range=0.01,
            fill_mode='wrap',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=None,
            preprocessing_function=None,
            data_format='channels_last'
        )

        datagen = ImageDataGenerator(**datagen_params)
        datagen.fit(X, augment=True, seed=123)

        for epoch in range(self.n_epoch):
            i = 0
            for img_batch, target in datagen.flow(X, y):

                trainImgs = Variable(torch.FloatTensor(img_batch.reshape(-1, 3, 75, 75)).cuda())
                target = Variable(torch.FloatTensor(target.astype(float).reshape(-1, 1)).cuda())

                # batch step
                optimizer.zero_grad()

                out = self.model(trainImgs, training=True)
                loss = criterion(out, target)
                loss.backward()

                optimizer.step()

                i += 1
                if i > 100:
                    print('{}: Epoch: {}, Train loss: {:.4f}'
                          .format(self.__class__.__name__, epoch + 1, loss.data.cpu().numpy()[0]))
                    break
