# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from icc.models.milesg.high_stack_base import HighStackBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class HighStack(HighStackBase, BaseEstimator):

    def __init__(self, n_epoch=15, batch_size=20, lr=0.0001, weight_decay=0.0):
        super().__init__(1)

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

    def fit(self, X: pd.DataFrame, y: np.ndarray, xTest=None, yTest=None):
        """
        Fit to X given y
        """

        # Preprocess training data
        # y is not altered, just passed for augmenting purposes.
        X, y = self._preprocess(X, y=y)

        # Process testing data if passed.
        if xTest is None:
            print('Testing data not passed to fit(), will not output testing loss during training.')
        else:
            xTest = self._preprocess(xTest)
            yTest = yTest if not hasattr(yTest, 'values') else yTest.values

        # Index 1 is the input channels X shape = (n_samples, channels, 75, 75)
        self.model = HighStackBase(input_channels=X.shape[1])
        self.model.cuda()

        self._train(X, y, xTest, yTest)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability for X"""
        X = self._preprocess(X)
        probabilities = np.zeros(X.shape[0])
        for batch_idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
            batch = Variable(torch.FloatTensor(X[batch_idx:batch_idx+self.batch_size].reshape(-1, X.shape[1], 75, 75)).cuda())
            probs = self.model(batch).data.cpu().numpy().squeeze()
            probabilities[batch_idx:batch_idx+self.batch_size] = probs
        probabilities = np.array([[1-p, p] for p in probabilities])
        return probabilities

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcome of X"""
        probs = self.predict_proba(X)
        return np.array([1 if p > 0.5 else 0 for p in probs])

    def _train(self, X: np.ndarray, y: np.ndarray, xTest: np.ndarray=None, yTest: np.ndarray=None) -> None:
        """
        Train model given processed X and y
        """
        print('{}: Good day, let\'s begin!'.format(self.__class__.__name__))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()

        for epoch in range(self.n_epoch):

            X, y = shuffle(X, y)

            for i in range(0, X.shape[0] - self.batch_size, self.batch_size):
                img_batch = X[i:i+self.batch_size]
                target = y[i:i+self.batch_size]

                trainImgs = Variable(torch.FloatTensor(img_batch).cuda())
                target = Variable(torch.FloatTensor(target.astype(float).reshape(-1, 1)).cuda())

                # batch step
                optimizer.zero_grad()

                out = self.model(trainImgs, training=True)
                loss = criterion(out, target)
                loss.backward()

                optimizer.step()

                # Tranfer back to RAM incase training data is set, to make room, jic.
                trainImgs = trainImgs.cpu()
                target = target.cpu()

            # If testing data was passed with fit, record testing loss while fitting
            if xTest is not None and yTest is not None:
                _x = Variable(torch.FloatTensor(xTest).cuda(), requires_grad=False)
                _y = Variable(torch.FloatTensor(yTest.astype(float).reshape(-1, 1)).cuda(), requires_grad=False)
                _pred = self.model(_x)
                _loss = criterion(_pred, _y)
                _test_loss = _loss.data.cpu().numpy()[0]
                _x = _x.cpu()
                _y = _y.cpu()
            else:
                _test_loss = None

            print('{name}: Epoch: {n_epoch}, Train loss: {train_loss:.4f} {test_loss}'
                  .format(name=self.__class__.__name__,
                          n_epoch=epoch + 1,
                          train_loss=loss.data.cpu().numpy()[0],
                          test_loss='- Test loss: {:.4f}'.format(_test_loss) if _test_loss is not None else '')
                  )
