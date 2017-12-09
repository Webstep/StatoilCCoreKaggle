# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from icc.models.milesg.double_down_base import DoubleDownBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class DoubleDown(DoubleDownBase, BaseEstimator):

    def __init__(self, n_epoch=35, batch_size=100, lr=0.001, weight_decay=0.0):
        super().__init__(1)
        self.model = None

        # Model and training params
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def get_params(self, deep=True):
        return {
            'n_epoch': self.n_epoch,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }

    def fit(self, X, y):
        """
        Fit to X given y
        """
        X = self._preprocess(X)

        # Index 1 is the input channels X shape = (n_samples, channels, 75, 75)
        if self.model is None:
            self.model = DoubleDownBase(input_channels=X.shape[1])
            self.model.cuda()

        self._train(X, y)
        return self

    def predict_proba(self, X):
        """Predict probability for X"""
        X = self._preprocess(X)
        probabilities = np.zeros(X.shape[0])
        for batch_idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
            batch = Variable(torch.FloatTensor(X[batch_idx:batch_idx+self.batch_size].reshape(-1, X.shape[1], 75, 75)).cuda())
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()

        for epoch in range(self.n_epoch):

            X, y = shuffle(X, y)

            for i in range(0, X.shape[0] - self.batch_size, self.batch_size):
                img_batch = X[i:i+self.batch_size]
                target = y[i:i+self.batch_size].values

                trainImgs = Variable(torch.FloatTensor(img_batch.reshape(-1, X.shape[1], 75, 75)).cuda())
                target = Variable(torch.FloatTensor(target.astype(float).reshape(-1, 1)).cuda())

                # batch step
                optimizer.zero_grad()

                out = self.model(trainImgs, training=True)
                loss = criterion(out, target)
                loss.backward()

                optimizer.step()

                i += 1
                #if i > 100: break
            print('{}: Epoch: {}, Train loss: {:.4f}'
                  .format(self.__class__.__name__, epoch + 1, loss.data.cpu().numpy()[0]))
