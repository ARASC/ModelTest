# -*- coding: utf-8 -*-
"""
The config functions.
"""
# Authors: Tan Tingyi <5636374@qq.com>

import atexit
from functools import partial
import os
import os.path as op
import shutil
import sys
import json
import tempfile

from ._logging import logger
from .check import _validate_type

_temp_home_dir = None

known_config_types = (
    'MODEL_TEST_DATA',
    'MODEL_TEST_CACHE_DIR',
    'MODEL_TEST_DATASETS_CRITEO_PATH',
)


def set_cache_dir(cache_dir):
    """Set the directory to be used for temporary file storage.
    This directory is used by joblib to store memmapped arrays,
    which reduces memory requirements and speeds up parallel
    computation.
    Parameters
    ----------
    cache_dir : str or None
        Directory to use for temporary file storage. None disables
        temporary file storage.
    """
    if cache_dir is not None and not op.exists(cache_dir):
        raise IOError('Directory %s does not exist' % cache_dir)

    set_config('MODEL_TEST_CACHE_DIR', cache_dir, set_env=False)


def _load_config(config_path, raise_error=False):
    """Safely load a config file."""
    with open(config_path, 'r') as fid:
        try:
            config = json.load(fid)
        except ValueError:
            # No JSON object could be decoded --> corrupt file?
            msg = ('The Model-Test config file (%s) is not a valid JSON '
                   'file and might be corrupted' % config_path)
            if raise_error:
                raise RuntimeError(msg)
            config = dict()
    return config


def get_config_path(home_dir=None):
    r"""Get path to standard modeltest config file.
    Parameters
    ----------
    home_dir : str | None
        The folder that contains the .modeltest config folder.
        If None, it is found automatically.
    Returns
    -------
    config_path : str
        The path to the modeltest configuration file. On windows, this
        will be '%USERPROFILE%\.modeltest\modeltest.json'. On every other
        system, this will be ~/.modeltest/modeltest.json.
    """
    val = op.join(_get_extra_data_path(home_dir=home_dir), 'modeltest.json')
    return val


def get_config(key=None,
               default=None,
               raise_error=False,
               home_dir=None,
               use_env=True):
    """Read Model-Test preferences from environment or config file.
    Parameters
    ----------
    key : None | str
        The preference key to look for. The os environment is searched first,
        then the modeltest config file is parsed.
        If None, all the config parameters present in environment variables or
        the path are returned. If key is an empty string, a list of all valid
        keys (but not values) is returned.
    default : str | None
        Value to return if the key is not found.
    raise_error : bool
        If True, raise an error if the key is not found (instead of returning
        default).
    home_dir : str | None
        The folder that contains the .modeltest config folder.
        If None, it is found automatically.
    use_env : bool
        If True, consider env vars, if available.
        If False, only use modeltest configuration file values.
    Returns
    -------
    value : dict | str | None
        The preference key value.
    See Also
    --------
    set_config
    """
    _validate_type(key, (str, type(None)), "key", 'string or None')

    if key == '':
        return known_config_types

    # first, check to see if key is in env
    if use_env and key is not None and key in os.environ:
        return os.environ[key]

    # second, look for it in modeltest config file
    config_path = get_config_path(home_dir=home_dir)
    if not op.isfile(config_path):
        config = {}
    else:
        config = _load_config(config_path)

    if key is None:
        # update config with environment variables
        if use_env:
            env_keys = (set(config).union(known_config_types).intersection(
                os.environ))
            config.update({key: os.environ[key] for key in env_keys})
        return config
    elif raise_error is True and key not in config:
        loc_env = 'the environment or in the ' if use_env else ''
        meth_env = ('either os.environ["%s"] = VALUE for a temporary '
                    'solution, or ' % key) if use_env else ''
        extra_env = (' You can also set the environment variable before '
                     'running python.' if use_env else '')
        meth_file = ('modeltest.utils.set_config("%s", VALUE, set_env=True) '
                     'for a permanent one' % key)
        raise KeyError(
            'Key "%s" not found in %s'
            'the modeltest config file (%s). '
            'Try %s%s.%s' %
            (key, loc_env, config_path, meth_env, meth_file, extra_env))
    else:
        return config.get(key, default)


def set_config(key, value, home_dir=None, set_env=True):
    """Set a Model-Test preference key in the config file and environment.
    Parameters
    ----------
    key : str
        The preference key to set.
    value : str |  None
        The value to assign to the preference key. If None, the key is
        deleted.
    home_dir : str | None
        The folder that contains the .mne config folder.
        If None, it is found automatically.
    set_env : bool
        If True (default), update :data:`os.environ` in addition to
        updating the Model-Test config file.
    See Also
    --------
    get_config
    """
    _validate_type(key, 'str', "key")
    # While JSON allow non-string types, we allow users to override config
    # settings using env, which are strings, so we enforce that here
    _validate_type(value, (str, 'path-like', type(None)), 'value')
    if value is not None:
        value = str(value)

    # Read all previous values
    config_path = get_config_path(home_dir=home_dir)
    if op.isfile(config_path):
        config = _load_config(config_path, raise_error=True)
    else:
        config = dict()
        logger.info('Attempting to create new modeltest configuration '
                    'file:\n%s' % config_path)
    if value is None:
        config.pop(key, None)
        if set_env and key in os.environ:
            del os.environ[key]
    else:
        config[key] = value
        if set_env:
            os.environ[key] = value

    # Write all values. This may fail if the default directory is not
    # writeable.
    directory = op.dirname(config_path)
    if not op.isdir(directory):
        os.mkdir(directory)
    with open(config_path, 'w') as fid:
        json.dump(config, fid, sort_keys=True, indent=0)


def _get_extra_data_path(home_dir=None):
    """Get path to extra data (config, tables, etc.)."""
    global _temp_home_dir
    if home_dir is None:
        home_dir = os.environ.get('_MODEL_TEST_FAKE_HOME_DIR')
    if home_dir is None:
        # this has been checked on OSX64, Linux64, and Win32
        if 'nt' == os.name.lower():
            if op.isdir(op.join(os.getenv('APPDATA'), '.modeltest')):
                home_dir = os.getenv('APPDATA')
            else:
                home_dir = os.getenv('USERPROFILE')
        else:
            # This is a more robust way of getting the user's home folder on
            # Linux platforms (not sure about OSX, Unix or BSD) than checking
            # the HOME environment variable. If the user is running some sort
            # of script that isn't launched via the command line (e.g. a script
            # launched via Upstart) then the HOME environment variable will
            # not be set.
            if os.getenv('MODEL_TEST_DONTWRITE_HOME', '') == 'true':
                if _temp_home_dir is None:
                    _temp_home_dir = tempfile.mkdtemp()
                    atexit.register(
                        partial(shutil.rmtree,
                                _temp_home_dir,
                                ignore_errors=True))
                home_dir = _temp_home_dir
            else:
                home_dir = os.path.expanduser('~')

        if home_dir is None:
            raise ValueError('modeltest config file path could '
                             'not be determined, please report this '
                             'error to modeltest developers')

    return op.join(home_dir, '.modeltest')
