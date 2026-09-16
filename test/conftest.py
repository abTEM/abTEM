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


# Metal is a single-precision backend -- torch refuses a float64 tensor on the
# MPS device outright -- so a test parametrized on both the 'mps' device and
# double precision is asking for something the hardware cannot do. Skipping is
# the honest outcome; letting it fail would bury real Metal regressions under
# noise that no amount of backend work can clear.
_DOUBLE_PRECISION_PARAMS = frozenset({"float64", "complex128"})


def _is_double_precision_on_metal(item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False

    # Only string parameters are of interest, and restricting to them also
    # keeps unhashable ones (arrays, Atoms) away from the set membership test.
    values = [value for value in callspec.params.values() if isinstance(value, str)]
    return "mps" in values and any(
        value in _DOUBLE_PRECISION_PARAMS for value in values
    )


def pytest_collection_modifyitems(config, items):
    skip_metal_double = pytest.mark.skip(
        reason="Metal (MPS) is single precision; float64 cannot run on this device"
    )
    for item in items:
        if _is_double_precision_on_metal(item):
            item.add_marker(skip_metal_double)

    if config.getoption("--runslow"):
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
