from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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
        '''
        Property that define the list of compatible datasets
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
    def get_feature_cols(self, raw, dataset, embedding_dim):
        """ 
        Count unique features for each sparse field,and record
        dense feature field name.
       """

    def prepare_process(self, dataset):
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

    def data_munging(self, raw, dataset):
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

    def feature_transform(self, raw, dataset):
        """
        Label encoding for sparse features, and do simple transformation for
        dense features
        """
        for feat in dataset.sparse_features:
            lbe = LabelEncoder()
            raw[feat] = lbe.fit_transform(raw[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        raw[dataset.dense_features] = mms.fit_transform(
            raw[dataset.dense_features])

        return raw

    def process_raw(self, raw, dataset):
        """
        This function apply the preprocessing and return a dataframe.
        Data is a dataframe with as many row as the length of the data
        and labels.
        """
        raw = self.data_munging(raw, dataset)
        raw = self.feature_transform(raw, dataset)
        return raw

    def get_data(self, dataset, train_size=None, test_size=None):
        """
        Return feature columns and data of the dataset.

        Parameters
        ----------
        dataset:
            A dataset instance.
        train_size:
            The size of train set.
        test_size:
            The size of test set.
        Returns
        -------
        train: pd.DataFrame
            A DataFrame containing train data.
        test: pd.DataFrame
            A DataFrame containing test data.
        """

        if not self.is_valid(dataset):
            message = "Dataset {} is not valid for paradigm".format(
                dataset.code)
            raise AssertionError(message)

        # TODO generater case
        raw = dataset.get_data(train_size, test_size)
        self.prepare_process(dataset)

        train_proc = self.process_raw(raw['train'], dataset)
        test_proc = self.process_raw(raw['test'], dataset)

        return train_proc, test_proc
