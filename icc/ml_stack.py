# -*- coding: utf-8 -*-

import os
import importlib


class StackedClassifier:
    """
    Decorator to register all ML models for the final StackingClassifier model
    """
    _models = []

    @classmethod
    def register(cls, model):
        """
        Register the model
        """
        # TODO: (Miles) Add model checks here, ensure they all play nicely together...
        cls()._models.append(model)
        return model

    def echo_registered_models(self):
        print('Models: {}'.format(self._models))



def run():
    """
    Run the StackedClassifier and any registered Models
    """

    # Run imports from everything in icc.models directory, where models are expected to be
    # this ensures that any stack.register calls are executed.
    for mod in os.listdir(os.path.join(os.path.dirname(__file__), 'models')):
        if mod == '__init__.py' or mod[-3:] != '.py':
            continue
        print(mod)
        globals().update(importlib.import_module('icc.models.' + mod[:-3]).__dict__)

    # Debug, echo the registered models
    StackedClassifier().echo_registered_models()

    # TODO: Implement the StackedClassifier and model checking.


if __name__ == '__main__':
    run()
