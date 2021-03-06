# Authors: Tan Tingyi <5636374@qq.com>

import sys
import os
import os.path as op
import tarfile

from tqdm.auto import tqdm

from .base import BaseDataset
from ..utils import get_config, set_config, _fetch_file, logger

VALID_DATASET = {'FM': ['Criteo CTR']}


def _get_path(path, key, name):
    """Get a dataset path."""
    # 1. Input
    if path is not None:
        if not isinstance(path, str):
            raise ValueError('path must be a string or None')
        return path
    # 2. get_config(key)
    # 3. get_config('MODEL_TEST_DATA')
    path = get_config(key, get_config('MODEL_TEST_DATA'))
    if path is not None:
        return path
    # 4. ~/modeltest (but use a fake home during testing so we don't
    #    unnecessarily create ~/modeltest)
    logger.info('Using default location ~/modeltest for %s...' % name)
    path = op.join(os.getenv('_MODEL_TEST_FAKE_HOME_DIR', op.expanduser("~")),
                   'modeltest_data')
    if not op.exists(path):
        logger.info('Creating ~/modeltest')
        try:
            os.mkdir(path)
        except OSError:
            raise OSError("User does not have write permissions "
                          "at '%s', try giving the path as an "
                          "argument to data_path() where user has "
                          "write permissions, for ex:data_path"
                          "('/home/xyz/me2/')" % (path))
    return path


def _do_path_update(path, update_path, key, name):
    """Update path."""
    path = op.abspath(path)
    identical = get_config(key, '', use_env=False) == path
    if not identical:
        if update_path is None:
            update_path = True
            if '--update-dataset-path' in sys.argv:
                answer = 'y'
            else:
                msg = ('Do you want to set the path:\n    %s\nas the default '
                       '%s dataset path in the modeltest config [y]/n? ' %
                       (path, name))
                answer = input(msg)
            if answer.lower() == 'n':
                update_path = False
        if update_path:
            set_config(key, path, set_env=False)
    return path


def _un_tar(file_path, target_path):
    """ untar zip file """
    with tarfile.open(file_path) as tar:
        names = tar.getnames()
        for name in tqdm(names, desc='Extract files', unit='files'):
            tar.extract(name, target_path)


def _valid_dataset(paradigm):
    return VALID_DATASET[paradigm].copy()


def _get_nunique(dataset, data):
    nunique = dict()
    for feature in dataset.sparse_features:
        nunique[feature] = data[feature].nunique()
    return nunique