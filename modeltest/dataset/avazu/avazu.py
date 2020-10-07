import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from deepctr.inputs import SparseFeat, get_feature_names
from deepctr.models import DeepFM

SEED = 1024
np.random.seed(SEED)
random.seed(SEED)


def load_data(path, chunk_size=10000000):
    reader = pd.read_csv(path, iterator=True)
    loop = True
    chunks = []

    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
            print("Loading...")
        except StopIteration:
            loop = False
            print("Loading completed.")
    return pd.concat(chunks)


def feature_construct(path, embedding_dim=16, data_sample=100000, test_size=0.2):
    data = load_data(path)
    data = data.sample(data_sample, random_state=SEED)
    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])
    target = ['click']
    sparse_features = [
        'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
        'app_id', 'app_domain', 'app_category', 'device_id', 'device_model',
        'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18',
        'C19', 'C20', 'C21'
    ]
    field_info = dict(
        C14='user',
        C15='user',
        C16='user',
        C17='user',
        C18='user',
        C19='user',
        C20='user',
        C21='user',
        C1='user',
        device_model='user',
        device_type='user',
        device_id='user',
        banner_pos='context',
        site_id='context',
        site_domain='context',
        site_category='context',
        device_conn_type='context',
        hour='context',
        app_id='item',
        app_domain='item',
        app_category='item',
    )

    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])

    fixlen_feature_columns = [
        SparseFeat(feature,
                   data[feature].nunique(),
                   embedding_dim=embedding_dim,
                   group_name=field_info[feature])
        for feature in sparse_features
    ]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns +
                                      dnn_feature_columns)

    data_train, data_test = train_test_split(data, test_size=test_size)
    target_train = data_train[target].values
    target_test = data_test[target].values

    train_model_input = {
        name: data_train[name].values
        for name in feature_names
    }
    test_model_input = {name: data_test[name].values for name in feature_names}

    return (train_model_input, target_train), (
        test_model_input,
        target_test), linear_feature_columns, dnn_feature_columns


def train_model(train, test, linear_feature, dnn_feature):

    model = DeepFM(linear_feature, dnn_feature, task='binary')
    model.compile(
        "adam",
        "binary_crossentropy",
        metrics=['AUC'],
    )
    history = model.fit(
        *train,
        batch_size=512,
        epochs=5,
        verbose=2,
        validation_split=0.1,
    )
    pred_ans = model.predict(test[0], batch_size=512)
    print("test LogLoss", round(log_loss(test[1], pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[1], pred_ans), 4))


if __name__ == "__main__":
    path = 'train'
    data_train, data_test, linear_feature, dnn_feature = feature_construct(
        path)
    train_model(data_train, data_test, linear_feature, dnn_feature)