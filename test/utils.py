from typing import Iterable

import dask.array as da
import numpy as np
import pytest
from hypothesis import assume

from abtem.core.backend import cp, get_array_module
from abtem.inelastic.phonons import BaseFrozenPhonons
from abtem.potentials.iam import Potential
from abtem.waves import Waves


def assert_array_matches_device(array, device):
    assert get_array_module(array) is get_array_module(device)


def assert_array_matches_laziness(array, lazy):
    if lazy:
        assert isinstance(array, da.core.Array)
    else:
        assert not isinstance(array, da.core.Array)


def remove_dummy_dimensions(shape):
    return tuple(s for s in shape if s > 1)


def ensure_is_tuple(x, length: int = 1):
    if not isinstance(x, tuple):
        x = (x,) * length
    elif isinstance(x, Iterable):
        x = tuple(x)
    assert len(x) == length
    return x


def array_is_close(
    a1,
    a2,
    rel_tol=np.inf,
    abs_tol=np.inf,
    check_above_abs=0.0,
    check_above_rel=0.0,
    mask=None,
):
    if mask is not None:
        a1 = a1[mask]
        a2 = a2[mask]

    if rel_tol < np.inf:
        element_is_checked = (a2 > check_above_abs) * (
            a2 > (a2.max() * check_above_rel)
        )
        rel_error = (a1[element_is_checked] - a2[element_is_checked]) / a2[
            element_is_checked
        ]
        if np.any(np.abs(rel_error) > rel_tol):
            return False

    if abs_tol < np.inf:
        if np.any(np.abs(a1 - a2) > abs_tol):
            return False

    return True


def assume_valid_probe_and_detectors(probe, detectors):
    integration_limits = [detector.angular_limits(probe) for detector in detectors]
    outer_limit = max([outer for inner, outer in integration_limits])
    min_range = min([outer - inner for inner, outer in integration_limits])
    assume(min(probe.angular_sampling) < min_range)
    assume(outer_limit <= min(probe.cutoff_angles))


def assert_scanned_measurement_as_expected(
    measurements, atoms, waves, detectors, scan=None, parameter_series=None
):
    if not isinstance(measurements, list):
        measurements = [measurements]

    assert len(measurements) == len(detectors)

    for detector, measurement in zip(detectors, measurements):
        expected_shape = ()

        if isinstance(atoms, BaseFrozenPhonons):
            if (not atoms.ensemble_mean) or isinstance(measurement, Waves):
                expected_shape = (len(atoms),)

        if parameter_series is not None:
            if (
                hasattr(parameter_series, "__len__")
                and not parameter_series.ensemble_mean
            ):
                expected_shape += (len(parameter_series),)

        if detector.detect_every:
            num_detect_thicknesses = len(Potential(atoms)) // detector.detect_every
            if len(Potential(atoms)) % detector.detect_every != 0:
                num_detect_thicknesses += 1

            if num_detect_thicknesses > 1:
                expected_shape += (num_detect_thicknesses,)

        if scan is not None:
            expected_shape += scan.shape

        expected_shape = tuple(s for s in expected_shape if s > 1)
        expected_shape += detector.measurement_shape(waves)

        assert expected_shape == measurement.shape
        # assert not np.all(measurement.array == 0.)

        if detector.to_cpu:
            assert isinstance(measurement.array, np.ndarray)
        elif waves.device != "cpu":
            assert_array_matches_device(measurement.array, waves.device)


def _gpu_count() -> int:
    if cp is None:
        return 0
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover -- driver/runtime hiccup
        return 0


# Gated on a usable DEVICE, not on cupy being importable. cupy imports fine with
# no GPU present -- a hidden device (HIP_VISIBLE_DEVICES=""), a container without
# /dev/kfd, a CI image that pip-installs cupy on a CPU runner -- and the failure
# then surfaces later, at the first allocation, inside the array module. That
# turns "hide the GPU" from a way to isolate GPU-specific behaviour into a way
# to break the suite. `requires_multigpu` below already used _gpu_count(); only
# this single-GPU gate was left keyed on the import.
def _mps_is_usable() -> bool:
    """Whether the Metal (MPS) backend is loaded and usable in this process."""
    from abtem.core import backend

    if backend.tp is None:
        return False

    from abtem.core._torch import is_available

    return is_available()


def _accelerator_device():
    """The non-CPU device this machine actually has, or None.

    CUDA wins where both are present. Its test is _gpu_count() rather than the
    cupy import, for the reason given just above.
    """
    if _gpu_count() >= 1:
        return "gpu"
    if _mps_is_usable():
        return "mps"
    return None


_ACCELERATOR = _accelerator_device()

# The accelerator half of every ["cpu", gpu] device parametrization. It used to
# be the literal "gpu" (CuPy/CUDA); it now resolves to whichever accelerator the
# machine actually has, so the same tests exercise Metal on Apple silicon and
# CUDA elsewhere. A test that needs the device string must compare against
# `gpu.values[0]`, never the literal "gpu" -- or better, derive the array module
# from `device` with `get_array_module`.
gpu = pytest.param(
    _ACCELERATOR or "gpu",
    marks=pytest.mark.skipif(_ACCELERATOR is None, reason="no gpu or mps"),
)

# The same gate as a standalone marker, for tests that are GPU-only rather than
# parametrized over devices. Several files had hand-rolled `skipif(cp is None)`
# or a bare `importorskip("cupy")`, both of which ask whether cupy is installed
# rather than whether a device exists.
requires_gpu = pytest.mark.skipif(_gpu_count() < 1, reason="no gpu")


try:
    import dask_cuda as _dask_cuda  # noqa: F401

    _HAS_DASK_CUDA = True
except ImportError:
    _HAS_DASK_CUDA = False


# Skip marker for tests that genuinely need to distribute across GPUs: they
# require both >=2 GPUs and dask-cuda (one worker process per GPU). Use together
# with `pytest.mark.multigpu` so the suite can be selected with `-m multigpu`.
requires_multigpu = pytest.mark.skipif(
    _gpu_count() < 2 or not _HAS_DASK_CUDA,
    reason="requires >=2 GPUs and dask-cuda",
)


# Skip marker for the Metal backend. Note that 'enable_mps' selects the library
# load order and so has to be set before abTEM is imported -- setting it from
# inside a test is too late, which is why this tests what actually loaded rather
# than what the configuration says.
requires_mps = pytest.mark.skipif(
    not _mps_is_usable(),
    reason=(
        "requires the Metal (MPS) backend: macOS on Apple silicon with PyTorch "
        "installed, and 'enable_mps' set before abTEM is imported "
        "(e.g. DASK_ENABLE_MPS=true pytest ...)"
    ),
)


def synthetic_transition_potential(
    Z: int = 14,
    gpts: tuple[int, int] = (64, 64),
    extent: tuple[float, float] | None = (8.0, 8.0),
    energy: float | None = 100e3,
    n_transitions: int = 2,
    seed: int = 0,
):
    """A seeded ``TransitionPotentialArray`` with a synthetic payload.

    Tests that exercise machinery *around* the transition potentials --
    graph transport, caching, detector wiring -- need an object of the right
    shape, not real physics. Building one directly skips the GPAW atomic
    solvers, so these tests also run where GPAW is not installed (CI).
    """
    from abtem.core.axes import OrdinalAxis
    from abtem.inelastic.core_loss import TransitionPotentialArray

    rng = np.random.default_rng(seed)
    array = (
        rng.standard_normal((n_transitions, *gpts))
        + 1j * rng.standard_normal((n_transitions, *gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=Z,
        array=array,
        energy=energy,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": Z, "n": 1, "l": 0},
    )
