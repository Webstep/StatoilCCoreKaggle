# -*- coding: utf-8 -*-

import os
import importlib
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier

from icc.data_loader import DataLoader
from icc.utils import DATA_DIR

# Create tensorflow config which allows growing memory
TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.25
TF_CONFIG.gpu_options.allow_growth = True

# For some reason, in the depths of AI and the wonders of why Shania Twain's hair is so red,
# you MUST make a tensorflow session BEFORE assigning a session to Keras, or even importing it at all.
TF_SESSION = tf.Session(config=TF_CONFIG)

from keras import backend as K
K.set_session(tf.Session(config=TF_CONFIG))


class StackedClassifier(DataLoader):
    """
    Decorator to register all ML models for the final StackingClassifier model
    """
    _models = []

    @classmethod
    def register(cls, model: object) -> object:
        """
        Register the model
        """
        # TODO: (Miles) Add model checks here, ensure they all play nicely together...
        cls()._models.append(model)
        return model

    def echo_registered_models(self) -> None:
        print('Running Stacking w/ Models: {}'.format([model for model in self._models if model.__name__ != 'DumbModel']))

    @classmethod
    def run(cls) -> StackingClassifier:
        """
        Run a Stacking Classifier using all registered models
        """
        sc = cls()
        X, y = sc.load_train()

        # Define the StackingClassifier using all models registered.
        classifiers = []
        for model in sc._models:
            if model.__name__ == 'DumbModel': continue
            classifiers.extend([model(), model()])

        clf = StackingClassifier(classifiers=classifiers,
                                 meta_classifier=LogisticRegression(),
                                 use_probas=True)

        # Run cross-val to get an idea of what to expect for final output
        scores = cross_val_score(clf, X.copy(), y.copy(), scoring='neg_log_loss', cv=2)

        print('\n---------\nCross validation (3) --> StackingClassifier - Avg Log Loss: {:.8f} - STD: {:.4f}\n---------'
              .format(scores.mean(), scores.std())
              )

        # Finally, refit clf to entire dataset
        print('Fitting Stacking Classifier to entire training dataset...')
        clf.fit(X.copy(), y.copy())
        return clf


def run_stack() -> None:
    """
    Run the StackedClassifier and any registered Models
    """
    # Run imports from everything in icc.models directory, where models are expected to be
    # this ensures that any stack.register calls are executed.
    for mod in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
        if mod == '__init__.py' or mod[-3:] != '.py':
            continue
        globals().update(importlib.import_module('icc.models.' + mod[:-3]).__dict__)

    # Debug, echo the registered models
    StackedClassifier().echo_registered_models()

    # Run the stacked classifier
    # This outputs each registered model with its log-loss score on cross validation
    # and trains the final StackingClassifier using all models.
    # this returns the trained StackingClassifier
    clf = StackedClassifier.run()

    # Run the stacked classifier on the test data
    # Then output 'submission.csv' to repo data directory
    print('Predicting and writing submission for test data...')
    X = DataLoader.load_test()
    out = clf.predict_proba(X[[col for col in X.columns if col != 'id']]).squeeze()
    results = pd.DataFrame({'id': X['id'], 'is_iceberg': out[:, 1]})
    results.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    print('done.')


if __name__ == '__main__':
    run_stack()
