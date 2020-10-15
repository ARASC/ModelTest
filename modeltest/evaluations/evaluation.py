from time import time

from sklearn.metrics import log_loss, roc_auc_score

from .result import Results
from ..utils import logger
from ..paradigms.base import BaseParadigm
from ..dataset.base import BaseDataset


class Evaluation:
    '''Class that defines necessary operations for an evaluation.
    Evaluations determine what the train and test sets are and can implement
    additional data preprocessing steps for more complicated algorithms.
    Parameters
    ----------
    paradigm : Paradigm instance
        the paradigm to use.
    datasets : List of Dataset Instance.
        The list of dataset to run the evaluation. 
    overwrite: bool (defaul False)
        if true, overwrite the results.
    suffix: str
        suffix for the results file.
    '''
    def __init__(self,
                 paradigm,
                 datasets,
                 overwrite=False,
                 suffix='',
                 hdf5_path=None):
        self.hdf5_path = hdf5_path

        # check paradigm
        if not isinstance(paradigm, BaseParadigm):
            raise (ValueError("paradigm must be an Paradigm instance"))
        self.paradigm = paradigm

        if not isinstance(datasets, list):
            if isinstance(datasets, BaseDataset):
                datasets = [datasets]
            else:
                raise (ValueError("datasets must be a list or a dataset "
                                  "instance"))

        for dataset in datasets:
            if not (isinstance(dataset, BaseDataset)):
                raise (ValueError("datasets must only contains dataset "
                                  "instance"))
        rm = []
        for dataset in datasets:
            # fixme, we might want to drop dataset that are not compatible
            valid_for_paradigm = self.paradigm.is_valid(dataset)
            if not valid_for_paradigm:
                logger.warning(
                    f"{dataset} not compatible with "
                    "paradigm. Removing this dataset from the list.")
                rm.append(dataset)

        [datasets.remove(r) for r in rm]
        if len(datasets) > 0:
            self.datasets = datasets
        else:
            raise Exception("No datasets left after paradigm checks.")

        self.results = Results(type(paradigm), suffix, overwrite, hdf5_path)

    def process(self, model_creaters, model_params):
        '''Runs all models on all datasets.
        This function will apply all provided models and return a dataframe
        containing the results of the evaluation.
        Parameters
        ----------
        model_creaters : dict of deepctr.models function.
            dict containing function to create models to evaluate.
        model_params : dict of dict
            dict containing parameters for generate feature columns,
            create model and train.
            i.e. {
                'DeepFM': {
                    'dnn_hidden_units': (128, 128),
                    'task': 'binary'
                },
                'train': {
                    'batch_size': 1024,
                    'epochs': 10,
                    'valudation_split': 0.2
                },
                'compile': {
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy',
                    'metrics': ['AUC']
                },
                'embedding': {
                    'embedding_dim': 8
                }
            }
        Return
        ------
        results: pd.DataFrame
            A dataframe containing the results.
        '''

        # check type
        if not isinstance(model_creaters, dict):
            raise (ValueError(
                "model_creaters must be a dict but get a {}".format(
                    type(model_creaters))))

        # TODO check items of model_creaters

        if not isinstance(model_params, dict):
            raise (ValueError("model_params must be a int but get a {}".format(
                type(model_params))))

        for dataset in self.datasets:
            logger.info('Processing dataset: {}'.format(dataset.code))
            results = self.evaluate(dataset, model_creaters, model_params)
            for res in results:
                self._push_result(res, model_creaters)

        return self.results.to_dataframe(model_creaters)

    def _push_result(self, res, models):
        message = '{} | '.format(res['model'])
        message += '{} | '.format(res['dataset'].code)
        message += ': Score %.4f' % res['score']
        logger.info(message)
        self.results.add({res['model']: res}, models=models)

    def get_results(self):
        return self.results.to_dataframe()

    def get_train_test_data(self, dataset):
        '''Train test split.
        preprocess data and split it into train and test data.

        Parameters:
        ----------
        dataset : datset instance
            Mainly use to access dataset specific information.
        Returns:
        -------
        x_train : dict of pd.Series 
            Train data for model
        x_test : dict of pd.Series
            Test data for model
        y_train : pd.Dataframe
            Train label which has same rows as x_train
        y_test : pd.Dataframe
            Test label which has same rows as x_test
        '''

        train_data, test_data = self.paradigm.get_data(dataset)
        # generate input data for model
        x_train = {name: train_data[name] for name in dataset.feature_names}
        x_test = {name: test_data[name] for name in dataset.feature_names}
        y_train = train_data[dataset.target]
        y_test = test_data[dataset.target]

        return x_train, x_test, y_train, y_test

    def _make_models(self, dataset, model_creaters, model_params):
        '''Create tf.keras.model for evaluation.

        Parameters
        ----------
        dataset : dataset instance
            mainly use to access dataset specific information.
        model_creaters : dict of functions for create model
            function should be import from deepctr.models.
        model_params : dict of models' parameters
            containing the parameters for create models.
        Returns
        -------
        models : dict of tf.keras.model
            Compiled model.
        '''

        models = {}

        for name, model_creater in model_creaters.items():
            # get feature columns & names
            linear_feature_columns, dnn_feature_columns = self.paradigm.make_feature_cols(
                dataset, **model_params['embedding'])
            model = model_creater(linear_feature_columns, dnn_feature_columns,
                                  **model_params[name])
            model.compile(**model_params['compile'])
            models[name] = model

        return models

    def evaluate(self, dataset, model_creaters, params):
        '''Evaluate results on a single dataset.
        This method return a generator. each results item is a dict with
        the following convension::
            res = {
                    'time': Duration of the training ,
                    'dataset': dataset id,
                    'score': score,
                    'n_samples': number of training examples,
                    'model': model name
                   }

        Parameters
        ----------
        dataset : dataset instance
        model_creaters : dict of function
        params : dict
        '''

        # check if we already have result for this model
        run_models = self.results.not_yet_computed(model_creaters, dataset)
        if len(run_models) == 0:
            return

        x_train, x_test, y_train, y_test = self.get_train_test_data(dataset)
        models = self._make_models(dataset, run_models, params)
        # train -> predict -> evaluate
        for name, model in models.items():
            t_start = time()
            history = model.fit(x_train, y_train, **params['fit'])
            duration = time() - t_start
            pred_ans = model.predict(x_test,
                                     batch_size=params['fit']['batch_size'])
            roc_auc = round(roc_auc_score(y_test, pred_ans), 4)
            res = {
                'time': duration,
                'dataset': dataset,
                'score': roc_auc,
                'model': name,
                'nsamples': dataset.nsample
            }
            yield res
