from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

from ..dataset.utils import _get_nunique


class BaseParadigm(metaclass=ABCMeta):
    """
    Base Paradigm.
    """
    def __init__(self):
        pass

    @abstractproperty
    def scoring(self):
        '''
        Property that defines scoring metric (e.g. ROC-AUC or accuracy
        or f-score), given as a sklearn-compatible string or a compatible
        sklearn scorer.

        '''
        pass

    @abstractproperty
    def datasets(self):
        '''Property that define the list of compatible datasets
        '''
        pass

    @abstractmethod
    def is_valid(self, dataset):
        """Verify the dataset is compatible with the paradigm.

        This method is called to verify dataset is compatible with the
        paradigm.

        This method should raise an error if the dataset is not compatible
        with the paradigm. This is for example the case if the
        dataset is an Movielens dataset for DIN paradigm, or if the
        dataset does not contain any of the required feature.

        Parameters
        ----------
        dataset : dataset instance
            The dataset to verify.
        """
        pass

    @abstractmethod
    def make_feature_cols(self, dataset, embedding_params):
        '''Return deepctr.feature_column.
        Parameters         
        ---------        
        dataset : dataset instance.
            a dataset instance.
        embedding_params : dict 
            dict containing embedding params for create feature colmns
            i.e. {embedding_dim: 8}
        Returns         
        ------        
        dnn_features : list
            list of feature_column instance for dnn inputs.
        linear_features : list
            list of feature_column instance for linear inputs.
        '''
        pass

    def _prepare_process(self, dataset):
        """Prepare processing of raw files

        This function allows to set parameter of the paradigm class prior to
        the preprocessing (process_raw). Does nothing by default and could be
        overloaded if needed.

        Parameters
        ----------

        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.
        """
        pass

    def _default_filling_rule(self, raw, dataset):
        raw[dataset.sparse_features] = raw[dataset.sparse_features].fillna(
            '-1')
        raw[dataset.dense_features] = raw[dataset.dense_features].fillna('0')
        return raw

    def _data_munging(self, raw, dataset):
        """
        Fill in missing values.

        Parameters
        ----------
        raw: DataFrame instance
            the raw data.
        dataset : dataset instance
            The dataset corresponding to the raw file. mainly use to access
            dataset specific information.

        Returns
        -------
        metadata: pd.DataFrame
            A dataframe containing the metadata.

        """

        # fill nan
        raw = self._default_filling_rule(raw, dataset)
        return raw

    def _feature_transform(self, raw, dataset):
        """
        Label encoding for sparse features, and do simple transformation for
        dense features
        """
        for feat in dataset.sparse_features:
            lbe = LabelEncoder()
            raw[feat] = lbe.fit_transform(raw[feat])
        dataset.nunique = _get_nunique(dataset, raw)

        mms = MinMaxScaler(feature_range=(0, 1))
        raw[dataset.dense_features] = mms.fit_transform(
            raw[dataset.dense_features])

        return raw

    def _process_raw(self, raw, dataset):
        """
        This function apply the preprocessing and return a dataframe.
        Data is a dataframe with as many row as the length of the data
        and labels.
        """
        raw = self._data_munging(raw, dataset)
        raw = self._feature_transform(raw, dataset)
        return raw

    def get_data(self, dataset):
        """
        Return data of the dataset.

        Parameters
        ----------
        dataset:
            dataset instance.
        Returns
        -------
        train : pd.DataFrame
            DataFrame containing train data.
        test : pd.DataFrame
            DataFrame containing test data.
        """

        if not self.is_valid(dataset):
            message = "Dataset {} is not valid for paradigm".format(
                dataset.code)
            raise AssertionError(message)

        # TODO generater case
        raw = dataset.get_data()
        self._prepare_process(dataset)

        raw = self._process_raw(raw, dataset)
        train_data, test_data = train_test_split(raw,
                                                 test_size=dataset.test_size,
                                                 random_state=dataset.random)
        return train_data, test_data
