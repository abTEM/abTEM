import warnings

import pytest
from hypothesis import HealthCheck, Phase, settings

from abtem import config

config.set({"diagnostics.progress_bar": False})

settings.register_profile(
    "dev",
    max_examples=10,
    print_blob=True,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much),
    phases=[Phase.generate],
)
settings.load_profile("dev")


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # Ignore specific warnings globally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )
    config.addinivalue_line(
        "markers",
        "multigpu: requires >=2 GPUs and dask-cuda; skipped otherwise",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _close_matplotlib_figures_after_test():
    """Figures created via matplotlib.pyplot (e.g. any test calling
    .show()) are retained by pyplot's global state until explicitly closed.
    Across a whole test session that accumulates past matplotlib's default
    figure.max_open_warning (20), which pytest can promote into a failure
    for whichever test happens to open the 21st one -- an innocent
    bystander unrelated to whatever actually leaked the open figures.
    Close everything after every test, regardless of outcome, so no test's
    figures can accumulate into another's failure."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
