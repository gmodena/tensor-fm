"""
Second Order Factorization Machine implemented atop Tensorflow.
"""
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Borrowed from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/__init__.py
def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import numpy as np
    import random

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get('TENSORFM_SEED', None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * (2 ** 31 - 1)
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)