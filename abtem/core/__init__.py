import os

from abtem.core import config

_ROOT = os.path.abspath(os.path.dirname(__file__))


def _set_path(path):
    """Internal function to set the parametrization data directory."""
    return os.path.join(_ROOT, '../data', path)
