import os

import pytest
from ase.io import read

_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope='session')
def data_path():
    """Internal function to set the parametrization data directory."""
    return os.path.join(_ROOT, 'data')

@pytest.fixture(scope='session')
def graphene_atoms(data_path):
    return read(os.path.join(data_path, 'orthogonal_graphene.cif'))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpaw")
    config.addinivalue_line("markers", "gpu")
    config.addinivalue_line("markers", "hyperspy")


def pytest_addoption(parser):
    parser.addoption("--rungpaw", action="store_true", help="run gpaw tests")
    parser.addoption("--rungpu", action="store_true", help="run gpu tests")
    parser.addoption("--runhyperspy", action="store_true", help="run hyperspy tests")


def pytest_runtest_setup(item):
    if 'gpaw' in item.keywords and not item.config.getvalue("rungpaw"):
        pytest.skip("need --rungpaw option to run")

    if 'gpu' in item.keywords and not item.config.getvalue("rungpu"):
        pytest.skip("need --rungpu option to run")

    if 'hyperspy' in item.keywords and not item.config.getvalue("rungpu"):
        pytest.skip("need --runhyperspy option to run")
