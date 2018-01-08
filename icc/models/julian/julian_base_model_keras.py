
import numpy as np
from sklearn.base import BaseEstimator



class JBaseKerasModel(BaseEstimator):

    def __init__(self):
        super().__init__()
        self.epochs = 10
        self.batch_size = 16

    def fit(self, X, y):
        self.prep = Preprocess()
        x_train, x_valid, y_train, y_valid = self.prep._basic_trainset(X, y)
        
        # Convert labels to categorical one-hot encoding
        y_train = to_categorical(y_train, num_classes=2)
        y_valid = to_categorical(y_valid, num_classes=2)

        self.model = self.get_model(input_shape = (75, 75, 3), classes = 2)

        assert self.model != None

        self.model.compile( optimizer = self._get_optimizer(), 
                            loss = self._get_loss_func(), 
                            metrics = ['accuracy'])

        filepath = 'weights.EPOCH{epoch:02d}-VAL_LOSS{val_loss:.2f}.hdf5'
        callbacks = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='auto', period=10)

        self.model.fit( x_train,
                        y_train,
                        epochs = self.epochs,
                        verbose = 2,
                        batch_size = self.batch_size,
                        validation_data = (x_valid, y_valid),
                        callbacks = [callbacks])
        return self

    def predict(self, X):
        """
        Binary prediction for X.
        """
        probs = self.predict_proba(X)
        return np.array([1 if p[1] > thresh else 0 for p in probs])

    def predict_proba(self, X):
        """
        Prediction for X
        """
        return self.model.predict(X)

    def get_model():
        return None
