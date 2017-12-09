# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

from icc.models.milesg.gangsta_base import GangstaNetBase
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class GangstaNet(BaseEstimator):
    """
    Excuse the poorly named model. :)

    Net based on separating the two input images and angle features into their own
    network branches and combines them later on.
    """
    def __init__(self, n_epoch: int = 35, batch_size: int = 20):
        super().__init__()

        self.net = GangstaNetBase()
        if torch.cuda.is_available():
            self.net.cuda()

        self.n_epoch = n_epoch
        self.batch_size = batch_size

    def get_params(self, deep=True):
        return {'n_epoch': self.n_epoch, 'batch_size': self.batch_size}

    def _preprocess(self, X):
        """

        """
        # Fill missing values with the fitted value from training set
        tmp = X.copy()
        tmp['inc_angle'], tmp['inc_angle_was_null'] = zip(
            *X.inc_angle.map(lambda val: (float(val), 0) if pd.notnull(val) else (self.inc_fill_value, 1))
        )
        return tmp

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit model to X given y

        Parameters
        ----------
        X: pd.DataFrame - Dataframe which results directly from icc.data_loader.DataLoader.load_train()[0]
        y: pd.Series    - Series object which results directly from ....DataLoader.load_train()[1]
        """
        self.inc_fill_value = X.inc_angle.median()
        X = self._preprocess(X.copy())
        self._train(X, y)
        return self

    def predict(self, X: pd.DataFrame, thresh=0.5) -> np.ndarray:
        """
        Get binary prediction output
        """
        probabilities = self.predict_proba(X)
        return np.array([1 if p[1] > thresh else 0 for p in probabilities])

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict for X
        """
        X = self._preprocess(X)

        # Predict in batches to avoid memory error
        probabilities = np.zeros(shape=X.shape[0])

        for i in range(0, X.shape[0] - self.batch_size, self.batch_size):

            # Extract raw numpy arrays
            band1 = np.concatenate(X.iloc[i:i+self.batch_size]['band_1'].values, axis=0).reshape(-1, 1, 75, 75, )
            band2 = np.concatenate(X.iloc[i:i+self.batch_size]['band_2'].values, axis=0).reshape(-1, 1, 75, 75, )
            angle = X.iloc[i:i+self.batch_size][['inc_angle', 'inc_angle_was_null']].values.reshape(-1, 2)

            # Transfer to variable tensors
            band1 = Variable(torch.FloatTensor(band1).cuda(), requires_grad=False)
            band2 = Variable(torch.FloatTensor(band2).cuda(), requires_grad=False)
            angle = Variable(torch.FloatTensor(angle).cuda(), requires_grad=False)

            # Get predictions and assign to output array
            probs = self.net(band1, band2, angle).data.cpu().numpy().squeeze()
            probabilities[i:i+self.batch_size] = probs

        # Need to return two sided probabilities -> [n_samples, 2] shape
        return np.array([[1-p, p] for p in probabilities])

    def _train(self, X: pd.DataFrame, y: pd.Series):
        """
        Handle the actual model training
        """
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        np.random.seed(0)

        for epoch in range(self.n_epoch):

            X, y = shuffle(X, y)

            for batch_idx in range(0, X.shape[0] - self.batch_size, self.batch_size):

                band1 = np.concatenate(X.iloc[batch_idx:batch_idx+self.batch_size]['band_1'].values, axis=0).reshape(-1, 1, 75, 75, )
                band2 = np.concatenate(X.iloc[batch_idx:batch_idx+self.batch_size]['band_2'].values, axis=0).reshape(-1, 1, 75, 75, )
                angle = X.iloc[batch_idx:batch_idx+self.batch_size][['inc_angle', 'inc_angle_was_null']].values.reshape(-1, 2)

                band1 = Variable(torch.FloatTensor(band1).cuda())
                band2 = Variable(torch.FloatTensor(band2).cuda())
                angle = Variable(torch.FloatTensor(angle).cuda())
                target = Variable(torch.FloatTensor(y.iloc[batch_idx:batch_idx+self.batch_size].values.astype(float).reshape(-1, 1)).cuda())

                # batch step
                optimizer.zero_grad()

                out = self.net(band1, band2, angle)
                loss = criterion(out, target)
                loss.backward()

                optimizer.step()

            print('{}: Epoch: {} - Loss: {:.6f}'.format(self.__class__.__name__, epoch + 1, loss.data.cpu().numpy()[0]))



