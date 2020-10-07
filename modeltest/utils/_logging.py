# -*- coding: utf-8 -*
"""
Some utility functions.
"""
# Authors: Tan Tingyi <5636374@qq.com>

import logging

logger = logging.getLogger(
    'model_test')  # one selection here used across model-test
logger.propagate = False  # don't propagate (in case of multiple imports)
