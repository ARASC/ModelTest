"""
Base class for a dataset
"""
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    """BaseDataset

    Parameters required for all datasets

    parameters
    ----------
    code: string
        Unique identifier for dataset.
    """
    def __init__(self, code):
        self.code = code

    @abc.abstractmethod
    def load_data(self, url, path=None, force_update=False, update_path=None):
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
        """
        pass

    @abc.abstractmethod
    def get_data(self,
                 train_size=None,
                 test_size=None,
                 chunksize=None,
                 usecols=None):
        """
        Return the data.
        The returned data is a dictionary with the folowing structure::

            data = {'train' : data_train,
                    'test'  : data_test
                    }

        parameters
        ----------
        train_size : int
            Number of train data to read. 
        test_size : int             
            Number of test data to read. 
        chunksize : int
            Return TextFileReader object for iteration. 
        usecols : list-like or callable
            A subset of the columns.
        returns
        -------
        data: Dict
            dict containing the raw data
        """
        pass

    @abc.abstractmethod
    def _data_path(self, url, path=None, force_update=False, update_path=None):
        """Get path to local copy of data.

        Parameters
        ----------
        url : str         
            The dataset to use.
        path : None | str             
        Location of where to look for the data storing location.
            If None, the environment variable or config parameter             
            ``MODEL_TEST_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the             
            "~/modeltest" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MODEL_TEST_DATASETS_(dataset)_PATH in modeltest
            config to the given path. If None, the user is prompted.

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a
            list of length one, for compatibility.
        """
        pass
