import pytest


def pytest_addoption(parser):
    parser.addoption("--rungpaw", action="store_true", help="run gpaw tests")


def pytest_runtest_setup(item):
    if 'gpaw' in item.keywords and not item.config.getvalue("rungpaw"):
        pytest.skip("need --rungpaw option to run")
