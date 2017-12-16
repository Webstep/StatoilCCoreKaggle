# -*- coding: utf-8 -*-

import os
import importlib
import pytest


@pytest.fixture
def models():
    """
    PyTest fixture which returns a list of valid and registered models
    """
    from icc.ml_stack import StackedClassifier
    for mod in os.listdir(os.path.join(os.path.dirname(__file__), '..', 'models')):
        if mod == '__init__.py' or mod[-3:] != '.py':
            continue
        locals().update(importlib.import_module('icc.models.' + mod[:-3]).__dict__)
    return [model for model in StackedClassifier()._models if model.__name__ != 'DumbModel']


@pytest.mark.parametrize('model', models(), ids=lambda model: 'model={}'.format(model.__name__))
def test_model_has_proper_fit(model):
    """
    Test that all registered models via StackedClassifier.register implement a valid "fit" method
    """
    import inspect
    model = model()

    # Validate model implements a fit method
    assert hasattr(model, 'fit'), 'Model has no attribute "fit"'
    assert callable(model.fit), 'The model "fit" method is not callable'

    # Validate model takes X and y args in fit method
    # Get fit signature, and verify first two args are X, y
    params = inspect.signature(model.fit).parameters  # parameters.keys() is OrderedDict
    assert 'X' == list(params.keys())[0] and 'y' == list(params.keys())[1], \
        'Model fit method signature should have "X, y" as first positional args'

    # Assert all other args are not required, optional args are ok.
    assert all([params.get(param).default != inspect._empty for param in params.keys() if param not in ['X', 'y']]), \
        'All other args to "fit" should be optional aside from "X" and "y"\nfit parameters: {}'.format(params)


@pytest.mark.parametrize('model', models(), ids=lambda model: 'model={}'.format(model.__name__))
def test_model_has_proper_predict_proba(model):
    """
    Test that all registered models via StackedClassifier.register implement a valid "predict_proba" method
    """
    import inspect
    model = model()

    # Validate model implements a predict method
    assert hasattr(model, 'predict_proba'), \
        'Model should implement a "predict_proba" method'
    assert callable(model.predict_proba), \
        'The model "predict_proba" method should be callable'

    # Validate model takes X and y args in fit method
    # Get fit signature, and verify first two args are X, y
    params = inspect.signature(model.predict_proba).parameters  # parameters.keys() is OrderedDict
    assert 'X' in params, \
        'Model "predict" method should take an "X" parameter'

    # Assert all other args are not required, since predict_proba will only get passed X
    # ok, to have other optional args.
    assert all([params.get(param).default != inspect._empty for param in params.keys() if param != 'X']), \
        'All other args to "predict" method should be optional aside from "X"\npredict_proba parameters: {}'.format(params)



