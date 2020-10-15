""" FM """

from deepctr.feature_column import SparseFeat, DenseFeat
from .base import BaseParadigm
from ..dataset.utils import _valid_dataset


class BaseFM(BaseParadigm):
    @property
    def datasets(self):
        return _valid_dataset('FM')

    def is_valid(self, dataset):
        if not dataset.paradigm == 'FM':
            return False
        return dataset.code in self.datasets

    def make_feature_cols(self, dataset, embedding_dim):
        '''Return deepctr.feature_column.

        Parameters
        ----------
        dataset :
            A dataset instance.
        Returns
        -------
        dnn_features : 
            A list of feature_column instance for dnn inputs.
        linear_features :
            A list of feature_column instance for linear inputs.
            
        '''
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=dataset.nunique[feat],
                       embedding_dim=embedding_dim)
            for feat in dataset.sparse_features
        ]
        fixlen_feature_columns += [
            DenseFeat(feat, 1) for feat in dataset.dense_features
        ]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        return dnn_feature_columns, linear_feature_columns


class ClassificationFM(BaseFM):
    @property
    def scoring(self):
        return 'roc_auc'
