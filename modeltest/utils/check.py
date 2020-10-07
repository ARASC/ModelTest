# -*- coding: utf-8 -*-
"""
The check functions.
"""
# Authors: Tan Tingyi <5636374@qq.com>

import operator
from pathlib import Path

import numpy as np


def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        # someone passing True/False is much more likely to be an error than
        # intentional usage
        if isinstance(x, bool):
            raise TypeError()
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be %s, got %s' % (name, must_be, type(x)))
    return x


def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.
    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'str', 'numeric', 'info', 'path-like'}.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "info":
        from mne.io import Info as types

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(
        ((type(None), ) if type_ is None else
         (type_, ) if not isinstance(type_, str) else _multi[type_]
         for type_ in types), ())
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = [
                'None' if cls_ is None else
                cls_.__name__ if not isinstance(cls_, str) else cls_
                for cls_ in types
            ]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        raise TypeError('%s must be an instance of %s, got %s instead' % (
            item_name,
            type_name,
            type(item),
        ))


class _IntLike(object):
    @classmethod
    def __instancecheck__(cls, other):
        try:
            _ensure_int(other)
        except TypeError:
            return False
        else:
            return True


int_like = _IntLike()


class _Callable(object):
    @classmethod
    def __instancecheck__(cls, other):
        return callable(other)


_multi = {
    'str': (str, ),
    'numeric': (np.floating, float, int_like),
    'path-like': (str, Path),
    'int-like': (int_like, ),
    'callable': (_Callable(), ),
}
