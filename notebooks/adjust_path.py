# -*- coding: utf-8 -*-

"""
Import this file to add the main 'icc' package to python path.
Then importing from icc will work as normal
"""

import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

if ROOT_DIR not in sys.path:
    print('Appending "{}" to path'.format(ROOT_DIR))
    sys.path.insert(0, ROOT_DIR)
