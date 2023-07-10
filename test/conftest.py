import pytest
from hypothesis import settings, HealthCheck
from abtem import config

config.set({"local_diagnostics.progress_bar": False})

settings.register_profile("dev", max_examples=20, print_blob=True, deadline=None,
                          suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large))
settings.load_profile("dev")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
