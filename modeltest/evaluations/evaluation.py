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
    random_state:
        if not None, can guarantee same seed`
    overwrite: bool (defaul False)
        if true, overwrite the results.
    suffix: str
        suffix for the results file.
    '''
    def __init__(self,
                 paradigm,
                 datasets,
                 random_state=None,
                 overwrite=False,
                 error_score='raise',
                 suffix='',
                 hdf5_path=None):
        self.random_state = random_state
        self.error_score = error_score
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
            A dict containing function to create models to evaluate.
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
                self.push_result(res, model_creaters)

        return self.results.to_dataframe(model_creaters)

    def push_result(self, res, models):
        message = '{} | '.format(res['model'])
        message += '{} | '.format(res['dataset'].code)
        message += ': Score %.4f' % res['score']
        logger.info(message)
        self.results.add({res['model']: res}, models=models)

    def get_results(self):
        return self.results.to_dataframe()

    def get_train_test(self, dataset):
        ''' Train test split.
        '''
        # get data
        train_data, test_data = self.paradigm.get_data(dataset)
        # generate input data for model
        x_train = {name: train_data[name] for name in dataset.feature_names}
        x_test = {name: test_data[name] for name in dataset.feature_names}
        y_train = train_data[dataset.target]
        y_test = test_data[dataset.target]

        return x_train, x_test, y_train, y_test

    def create_model(self, dataset, model_creaters, model_params):
        '''Create tf.keras.model for evaluation.

        Parameters
        ----------
        dataset : dataset instance
            mainly use to access dataset specific information.
        model_creaters : dict of functions for create model
            function should be import from deepctr.models.
        model_params : dict of models' parameters
            containing the parameters for create models.
            i.e. {'deepfm': {'embedding_dim': 32}}
            # todo support more parameters 
        Returns
        -------
        models : dict of tf.keras.model
            return the compiled model.

        '''

        models = {}

        for name, model_creater in model_creaters.items():
            # get feature columns & names
            linear_feature_columns, dnn_feature_columns = self.paradigm.get_feature_cols(
                dataset, model_params[name])
            model = model_creater(linear_feature_columns,
                                  dnn_feature_columns,
                                  task='binary')
            model.compile('adam',
                          'binary_crossentropy',
                          metrics=['binary_crossentropy'])
            models[name] = model

        return models

    def evaluate(self, dataset, model_creaters, model_params):
        '''Evaluate results on a single dataset.
        This method return a generator. each results item is a dict with
        the following convension::
            res = {'time': Duration of the training ,
                   'dataset': dataset id,
                   'score': score,
                   'n_samples': number of training examples,
                   'model': model name}

        Parameters
        ----------
        inputs : tuple of train & test data for models.

        '''

        # check if we already have result for this model
        run_models = self.results.not_yet_computed(model_creaters, dataset)
        if len(run_models) == 0:
            return

        x_train, x_test, y_train, y_test = self.get_train_test(dataset)
        models = self.create_model(dataset, run_models, model_params)
        # train -> predict -> evaluate
        for name, model in models.items():
            t_start = time()
            history = model.fit(x_train,
                                y_train,
                                batch_size=256,
                                epochs=10,
                                verbose=2,
                                validation_split=0.2)
            duration = time() - t_start
            pred_ans = model.predict(x_test, batch_size=256)
            roc_auc = round(roc_auc_score(y_test, pred_ans), 4)
            res = {
                'time': duration,
                'dataset': dataset,
                'score': roc_auc,
                'model': name,
                'nsamples': dataset.nsample
            }
            yield res
