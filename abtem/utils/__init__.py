from abtem.utils.convert import *
from abtem.utils.complex import *
from abtem.utils.utils import *

_ROOT = os.path.abspath(os.path.dirname(__file__))


def _set_path(path):
    """Internal function to set the parametrization data directory."""
    return os.path.join(_ROOT, '../data', path)


def _disc_meshgrid(r):
    """Internal function to return all indices inside a disk with a given radius."""
    cols = np.zeros((2 * r + 1, 2 * r + 1)).astype(np.int32)
    cols[:] = np.linspace(0, 2 * r, 2 * r + 1) - r
    rows = cols.T
    inside = (rows ** 2 + cols ** 2) <= r ** 2
    return rows[inside], cols[inside]