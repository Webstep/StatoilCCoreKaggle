
import numpy as np
from sklearn.base import BaseEstimator
from icc.models.spencer.alexnet.preprocessing import *
from keras.utils import to_categorical
from keras.models import model_from_json
import os


class JBaseKerasModel(BaseEstimator):

    def __init__(self, epochs = 100, batch_size = 24, weights_path = None):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights_path = weights_path


    def fit(self, X, y):
        x_train, x_valid, y_train, y_valid = self._preprocess(X, y)
        
        # Convert labels to categorical one-hot encoding
        y_train = to_categorical(y_train, num_classes=2)
        y_valid = to_categorical(y_valid, num_classes=2)

        self.model = self.get_model(input_shape = (75, 75, 3), classes = 2)

        assert self.model != None

        self.model.compile( optimizer = self._get_optimizer(), 
                            loss = self._get_loss_func(), 
                            metrics = ['accuracy'])

        if self.weights_path is None:
            self.model.fit( x_train,
                            y_train,
                            epochs = self.epochs,
                            verbose = 2,
                            batch_size = self.batch_size,
                            validation_data = (x_valid, y_valid),
                            callbacks = self._get_callbacks())
        else:
            self.model.load_weights(self._path_to_weights())

        return self

    def predict(self, X, thresh = 0.5):
        """
        Binary prediction for X.
        """
        probs = self.predict_proba(X)
        return np.array([1 if p[1] > thresh else 0 for p in probs])

    def predict_proba(self, X):
        """
        Prediction for X.
        """
        X = self.prep._basic_testset(X)
        return self.model.predict(X)

    def get_model():
        """
        Returns the model.
        """
        return None

    def save_model_and_weights(self, path = 'model'):
        self._save_model(path)
        self._save_weights(path)

    def _save_model(self, path = 'model'):
        model_json = self.model.to_json()
        with open(path + '.json', 'w') as json_file:
            json_file.write(model_json)
    
    def _save_weights(self, path = 'model'):
        self.model.save_weights(path + '.h5')

    def load_model_and_weights(self, path = 'model'):
        self._load_model(path)
        self._load_weights(path)

    def _load_model(self, path = 'model'):
        with open(path + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)

    def _load_weights(self, path = 'model'):
        self.model.load_weights(path + '.h5')

    def _get_callbacks(self):
        """
        A list of callbacks that is added in fit.
        """
        return []

    def _preprocess(self, X, y):
        """
        Preprocess the data in X and y.
        returns (x_train, x_valid, y_train, y_valid)
        """
        self.prep = Preprocess()
        print(X.shape)
        preprocessed = self.prep._basic_trainset(X, y)
        return preprocessed

    def _get_loss_func(self):
        """
        The loss function to be used in fit.
        """
        return 'binary_crossentropy'

    def _get_optimizer(self):
        """
        The optimizer to be used in fit.
        """
        return Adam(lr = 0.001, epsilon = 1e-8)
