import functools

from hydra.utils import get_method

"""
TODO See hydra https://github.com/facebookresearch/hydra/issues/1283
"""


def partial(_partial_, *args, **kwargs):
    return functools.partial(get_method(_partial_), *args, **kwargs)
