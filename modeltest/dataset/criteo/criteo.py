# -*- coding: utf-8 -*
"""
Criteo CTR dataset.
"""
# Authors: Tan Tingyi <5636374@qq.com>

import os
import os.path as op

import pandas as pd

from sklearn.model_selection import train_test_split

from ..base import BaseDataset
from ..utils import _get_path, _do_path_update, _un_tar
from ...utils import _fetch_file, _url_to_local_path, get_config, set_config

BASE_URL = 'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'


class Criteo(BaseDataset):
    """Cirteo CTR Dataset.

    Criteo CTR dataset: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

    train.txt - The training set consists of a portion of Criteo's traffic over a period of 7 days. 
                Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives 
                (non-clicked) examples have both been subsampled at different rates in order to reduce 
                the dataset size. The examples are chronologically ordered.
    test.txt  - The test set is computed in the same way as the training set but for events on the day 
                following the training period.

    Label     - Target variable that indicates if an ad was clicked (1) or not (0).
    I1-I13    - A total of 13 columns of integer features (mostly count features).
    C1-C26    - A total of 26 columns of categorical features. The values of these features have been 
                hashed onto 32 bits for anonymization purposes. 
                
    The semantic of the features is undisclosed.
    When a value is missing, the field is empty.

    Parameters
    ----------
    References
    ----------

    """
    def __init__(self):
        super().__init__(code='Criteo CTR')
        self._paradigm = 'FM'
        self._sparse_features = ['C' + str(i) for i in range(1, 27)]
        self._dense_features = ['I' + str(i) for i in range(1, 14)]
        self._nsample = None
        self._test_size = 0.2
        self.nunique = None
        self.random = 2020
        self.target = ['label']

    @property
    def paradigm(self):
        return self._paradigm

    @property
    def sparse_features(self):
        return self._sparse_features.copy()

    @property
    def dense_features(self):
        return self._dense_features.copy()

    @property
    def feature_names(self):
        return self.dense_features + self.sparse_features

    @property
    def nsample(self):
        return self._nsample

    @nsample.setter
    def nsample(self, value):
        if not isinstance(value, int):
            raise ValueError('nsample must be an integer but get {}'.format(
                type(ivalue)))
        self._nsample = value

    @property
    def test_size(self):
        return self._test_size

    @test_size.setter
    def test_size(self, value):
        if not isinstance(value, float):
            raise ValueError('test_size must be an float but get {}'.format(
                type(value)))
        self._test_size = value

    def _get_nunique(self, data):
        nunique = dict()
        for feature in self.sparse_features:
            nunique[feature] = data[feature].nunique()
        return nunique

    def get_data(self):
        """Return data"""

        data = {}
        if get_config('MODEL_TEST_DATASETS_CRITEO_PATH') is None:
            set_config('MODEL_TEST_DATASETS_CRITEO_PATH',
                       op.join(op.expanduser("~"), "modeltest_data"))

        filename = self.load_data()
        columns_name = ['label'] + self.dense_features + self.sparse_features
        data = pd.read_table(filename, nrows=self.nsample, names=columns_name)
        self.nunique = self._get_nunique(data)
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random)
        return train_data, test_data

    def _data_path(self,
                   url=BASE_URL,
                   path=None,
                   force_update=False,
                   update_path=None):
        """Get path to local copy of CTR dataset URL.
        This is a low-level function useful for getting a local copy of a
        remote CTR dataset.
        Parameters
        ----------
        url : str                      
            The dataset to use.
        path : None | str
            Location of where to look for the EEGBCI data storing location.
            If None, the environment variable or config parameter
            ``MODEL_TEST_DATASETS_CRITEO_PATH`` is used. If it doesn't exist, the
            "~/modeltest" directory is used. If the Criteo dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MODEL_TEST_DATASETS_CRITEO_PATH in modeltest
            config to the given path. If None, the user is prompted.

        Returns
        -------
        path : str
            Local path to the given data file. 
        References
        ----------
        # TODO
        """
        key = 'MODEL_TEST_DATASETS_CRITEO_PATH'
        name = 'CRITEO'
        path = _get_path(path, key, name)
        destination = _url_to_local_path(url, op.join(path, 'criteo'))

        # Fetch the file
        if not op.isfile(destination) or force_update:
            if op.isfile(destination):
                os.remove(destination)
            if not op.isdir(op.dirname(destination)):
                os.makedirs(op.dirname(destination))
            _fetch_file(url, destination)

        # Offer to update the path
        _do_path_update(path, update_path, key, name)
        return destination

    def load_data(self,
                  url=BASE_URL,
                  path=None,
                  force_update=False,
                  update_path=None):
        """Get paths to local copies of Criteo dataset files.
        This will fetch data for the Criteo dataset.
        Parameters
        ----------
        url : str                                   
            The dataset to use.
        path : None | str
            Location of where to look for the Criteo data storing location.
            If None, the environment variable or config parameter
            ``MODEL_TEST_DATASETS_CRITEO_PATH`` is used. If it doesn't exist, the
            "~/modeltest" directory is used. If the Criteo dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MODEL_TEST_DATASETS_CRITEO_PATH in modeltest
            config to the given path. If None, the user is prompted.
        Returns
        -------
        filenames : list
            List of local data paths of the given type.
        References
        ----------
        # TODO
        """
        if get_config('MODEL_TEST_DATASETS_CRITEO_PATH') is None:
            set_config('MODEL_TEST_DATASETS_CRITEO_PATH',
                       op.join(op.expanduser("~"), "modeltest_data"))

        file = 'train.txt'
        path = self._data_path(url, path, force_update, update_path)
        filename = op.join(op.split(path)[0], file)

        # Unzip the file
        if not op.isfile(filename) or force_update:
            if op.isfile(filename):
                os.remove(filename)
        if not op.isfile(filename) or force_update:
            print('Unzipping the file ...')
            _un_tar(path, op.split(path)[0])

        return filename
