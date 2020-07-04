# -*- coding: utf-8 -*
"""
Criteo CTR dataset.
"""
# Authors: Tan Tingyi <5636374@qq.com>

import os
import os.path as op

import pandas as pd

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

    parameters
    ----------

    references
    ----------

    """
    def __init__(self):
        super().__init__(code='Criteo CTR')

    def get_data(self, train_size=None, test_size=None, usecols=None):
        """return data"""

        data = {}
        if get_config('MODEL_TEST_DATASETS_CRITEO_PATH') is None:
            set_config('MODEL_TEST_DATASETS_CRITEO_PATH',
                       op.join(op.expanduser("~"), "modeltest_data"))

        paths = self.load_data()
        filenames = _un_tar(paths[0], op.split(paths[0])[0], replace=False)

        names_train = [
            'label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
            'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
            'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25',
            'C26'
        ]
        names_test = [
            'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
            'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
            'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
            'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26'
        ]
        for name in filenames:
            filename = op.basename(name).split('.')[0]
            if filename is 'train':
                names = names_train
            elif filename is 'test':
                names = names_test
            else:
                continue
            data[filename] = pd.read_table(name,
                                           nrows=train_size,
                                           names=names,
                                           usecols=usecols)
        return data

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
            If True, set the MNE_DATASETS_EEGBCI_PATH in mne-python
            config to the given path. If None, the user is prompted.

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a list
            of length one, for compatibility.
        References
        ----------
        # TODO
        """
        key = 'MODEL_TEST_DATASETS_CRITEO_PATH'
        name = 'CRITEO'
        path = _get_path(path, key, name)
        destination = _url_to_local_path(url, op.join(path, 'criteo'))
        destinations = [destination]

        # Fetch the file
        if not op.isfile(destination) or force_update:
            if op.isfile(destination):
                os.remove(destination)
            if not op.isdir(op.dirname(destination)):
                os.makedirs(op.dirname(destination))
            _fetch_file(url, destination)

        # Offer to update the path
        _do_path_update(path, update_path, key, name)
        return destinations

    def load_data(
            self,
            url=BASE_URL,
            path=None,
            force_update=False,
            update_path=None,
    ):
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
        paths : list
            List of local data paths of the given type.
        References
        ----------
        # TODO
        """
        if get_config('MODEL_TEST_DATASETS_CRITEO_PATH') is None:
            set_config('MODEL_TEST_DATASETS_CRITEO_PATH',
                       op.join(op.expanduser("~"), "modeltest_data"))

        paths = self._data_path(url, path, force_update, update_path)
        return paths
