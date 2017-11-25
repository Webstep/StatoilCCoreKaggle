# -*- coding: utf-8 -*-

import os
import json
import warnings
import pandas as pd

from subprocess import Popen, PIPE

from icc.utils import DATA_DIR


class DataLoader:
    """
    Load datasets

    Example:
    >>> DataLoader.load_train()  # Returns the raw data from training set.

    Available methods: load_train(), load_test(), load_sample_submission()
    Note: train and test are returned as JSON/dict and sample submission is returned as pandas.core.DataFrame
    """

    @staticmethod
    def _check_n_uncompress(path):
        """
        Check if file.7z has been processed, if not, process it.

        Parameters
        ----------
        path: str - Full path to uncompressed file, if it doesn't exist, expects compressed file to be in 'data' dir.
        """
        if not os.path.exists(path):
            # Hasn't been uncompressed..
            cpath = os.path.join(DATA_DIR, '{}.7z'.format(os.path.basename(path)))
            warnings.warn('{} was not found, uncompressing file at: {}'.format(path, cpath))
            process = Popen(['7z', 'e', cpath, '-o/code/data'])
            stdout, stderr = process.communicate()
            warnings.warn('STDOUT: {}\nSTDERR: {}\n'.format(stdout, stderr))
            if not os.path.exists(path):
                raise IOError('Unable to properly uncompress file at: {}'.format(cpath))

    def _load_json(self, path):
        """
        Load json file located at path.
        If uncompressed json doesn't exist, assumes that compressed .7z is in the same directory and uncompressed it
        7z should be installed on system in the latter case.
        """
        self._check_n_uncompress(path=path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    @classmethod
    def load_train(cls):
        return cls()._load_json(path=os.path.join(DATA_DIR, 'train.json'))

    @classmethod
    def load_test(cls):
        return cls()._load_json(path=os.path.join(DATA_DIR, 'test.json'))

    @classmethod
    def load_sample_submission(cls):
        cls()._check_n_uncompress(path=os.path.join(DATA_DIR, 'sample_submission.csv'))
        filename = os.path.join(DATA_DIR, 'sample_submission.csv')
        return pd.read_csv(filename, compression='xz')
