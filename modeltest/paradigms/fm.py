""" FM """

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
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


class CTRFM(BaseFM):
    def get_feature_cols(self, raw, dataset, embedding_dim):
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=raw[feat].nunique(),
                       embedding_dim=embedding_dim)
            for feat in dataset.sparse_features
        ]
        fixlen_feature_columns += [
            DenseFeat(feat, 1) for feat in dataset.dense_features
        ]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns +
                                          dnn_feature_columns)
        return feature_names, dnn_feature_columns, linear_feature_columns

    @property
    def scoring(self):
        return 'roc_auc'
