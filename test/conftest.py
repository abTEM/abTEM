import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpaw")
    config.addinivalue_line("markers", "gpu")


def pytest_addoption(parser):
    parser.addoption("--rungpaw", action="store_true", help="run gpaw tests")
    parser.addoption("--rungpu", action="store_true", help="run gpu tests")


def pytest_runtest_setup(item):
    if 'gpaw' in item.keywords and not item.config.getvalue("rungpaw"):
        pytest.skip("need --rungpaw option to run")

    if 'gpu' in item.keywords and not item.config.getvalue("rungpu"):
        pytest.skip("need --rungpu option to run")
