import re
import os
import h5py
import hashlib

import numpy as np
import pandas as pd

from datetime import datetime

from ..paradigms import BaseParadigm


def get_string_rep(obj):
    str_repr = repr(obj)
    str_no_addresses = re.sub('0x[a-z0-9]*', '0x__', str_repr)
    return str_no_addresses.replace('\n', '').encode('utf8')


def get_digest(obj):
    """Return hash of an object repr.
    If there are memory addresses, wipes them
    """
    return hashlib.md5(get_string_rep(obj)).hexdigest()


class Results:
    """Class to hold results.

    Appropriate test would be to ensure the result of 'evaluate' is
    consistent and can be accepted by 'results.add'
    
    Saves dataframe per model.
    
    """
    def __init__(self,
                 paradigm_class,
                 suffix='',
                 overwrite=False,
                 hdf5_path=None):
        """
        class that will abstract result storage
        """

        assert issubclass(paradigm_class, BaseParadigm)

        # defult path of result
        if hdf5_path is None:
            self.mod_dir = os.path.join(os.path.expanduser("~"),
                                        "modeltest_result")
        else:
            self.mod_dir = os.path.abspath(hdf5_path)

        self.filepath = os.path.join(self.mod_dir, paradigm_class.__name__,
                                     'results{}.hdf5'.format('_' + suffix))
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        # self.filepath = self.filepath

        if overwrite and os.path.isfile(self.filepath):
            os.remove(self.filepath)

        if not os.path.isfile(self.filepath):
            with h5py.File(self.filepath, 'w') as f:
                f.attrs['create_time'] = np.string_(
                    '{:%Y-%m-%d, %H:%M}'.format(datetime.now()))

    def add(self, results, models):
        """add results"""
        def to_list(res):
            if type(res) is dict:
                return [res]
            elif type(res) is not list:
                raise ValueError("Results are given as neither dict nor"
                                 "list but {}".format(type(res).__name__))
            else:
                return res

        with h5py.File(self.filepath, 'r+') as f:
            for name, data_dict in results.items():
                digest = get_digest(models[name])
                if digest not in f.keys():
                    # create pipeline main group if nonexistant
                    f.create_group(digest)

                model_grp = f[digest]
                model_grp.attrs['name'] = name
                model_grp.attrs['repr'] = repr(models[name])

                dlist = to_list(data_dict)
                d1 = dlist[0]  # FIXME: handle multiple session ?
                dname = d1['dataset'].code
                if dname not in model_grp.keys():
                    # create dataset subgroup if nonexistant
                    dset = model_grp.create_group(dname)
                    dt = h5py.special_dtype(vlen=str)
                    dset.create_dataset('id', (0, 2),
                                        dtype=dt,
                                        maxshape=(None, 2))
                    dset.create_dataset('data', (0, 3), maxshape=(None, 3))
                    dset.attrs.create('columns', ['score', 'time', 'nsamples'],
                                      dtype=dt)
                dset = model_grp[dname]
                for d in dlist:
                    # add id and scores to group
                    length = len(dset['id']) + 1
                    dset['id'].resize(length, 0)
                    dset['data'].resize(length, 0)
                    dset['data'][-1, :] = np.asarray(
                        [d['score'], d['time'], d['nsamples']])

    def to_dataframe(self, models):
        df_list = []

        # get the list of models hash
        digests = []
        if models is not None:
            digests = [get_digest(models[name]) for name in models]

        with h5py.File(self.filepath, 'r') as f:
            for digest, p_group in f.items():

                # skip if not in models list
                if (models is not None) & (digest not in digests):
                    continue

                name = p_group.attrs['name']
                for dname, dset in p_group.items():
                    array = np.array(dset['data'])
                    ids = np.array(dset['id'])
                    df = pd.DataFrame(array, columns=dset.attrs['columns'])
                    df['dataset'] = dname
                    df['model'] = name
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def not_yet_computed(self, models, dataset):
        """Check if a results has already been computed."""
        ret = {
            k: models[k]
            for k in models.keys()
            if not self._already_computed(models[k], dataset)
        }
        return ret

    def _already_computed(self, models, dataset):
        """Check if we have results for a current models
        """
        with h5py.File(self.filepath, 'r') as f:
            # get the digest from repr
            digest = get_digest(models)

            # check if digest present
            if digest not in f.keys():
                return False
            else:
                models_grp = f[digest]
                # if present, check for dataset code
                if dataset.code not in models_grp.keys():
                    return False
                else:
                    return True
