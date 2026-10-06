from typing import Iterable

import ase.build
import dask.array as da
import numpy as np
import pytest
from hypothesis import assume

from abtem.core.backend import asnumpy, cp, get_array_module
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
    rel_tol=None,
    abs_tol=None,
    check_above_abs=0.0,
    check_above_rel=0.0,
    mask=None,
):
    """Whether ``a1`` is within ``rel_tol`` (relative to ``a2``) and/or
    ``abs_tol`` of ``a2``. The caller must assert the result.

    At least one tolerance is required. Both used to default to ``inf``,
    which disables the check it controls, so a call with neither returned
    True whatever the arrays held.
    """
    if rel_tol is None and abs_tol is None:
        raise TypeError("array_is_close requires rel_tol and/or abs_tol")
    if rel_tol is None:
        rel_tol = np.inf
    if abs_tol is None:
        abs_tol = np.inf

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
    """Whether the Metal (MPS) backend is usable in this process.

    Asking loads it -- PyTorch is imported on the first request for the 'mps'
    device -- which is what any Metal test is about to do anyway.
    """
    from abtem.core import backend

    try:
        backend.check_mps_is_available()
    except RuntimeError:
        return False
    return True


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
#
# It carries the `gpu` marker, like requires_gpu, so `pytest -m "not gpu"`
# deselects the accelerator half of these tests on a machine that has one.
gpu = pytest.param(
    _ACCELERATOR or "gpu",
    marks=(
        pytest.mark.gpu,
        pytest.mark.skipif(_ACCELERATOR is None, reason="no gpu or mps"),
    ),
)


class _GpuRequirement:
    """A composite decorator: applies both the wrapped ``skipif`` and a
    dedicated, condition-free ``pytest.mark.gpu`` alongside it.

    ``conftest.py``'s GPU-worker-grouping hook prefers that marker's mere
    presence over matching the skipif's ``reason=`` text, since the marker
    survives a reword of that text where a string match would not. A
    class (rather than a plain function with attributes bolted on) so
    ``.marks``/``.mark`` below are properly typed, not dynamic attributes
    mypy can't see.

    Usable as a bare decorator (``@requires_gpu``) directly; ``.marks``
    exists for the one place that needs actual ``Mark``-compatible objects
    instead of a decorator -- ``pytest.param(..., marks=requires_gpu.marks)``
    -- and ``.mark`` exposes the skipif's own ``Mark`` (reason text
    included) for ``conftest.py``'s fallback check.
    """

    def __init__(self, skipif: "pytest.MarkDecorator"):
        self._skipif = skipif
        self.marks: tuple["pytest.MarkDecorator", "pytest.MarkDecorator"] = (
            pytest.mark.gpu,
            skipif,
        )
        self.mark = skipif.mark

    def __call__(self, func):
        func = self._skipif(func)
        return pytest.mark.gpu(func)


# The same gate as a standalone marker, for tests that are GPU-only rather than
# parametrized over devices. Several files had hand-rolled `skipif(cp is None)`
# or a bare `importorskip("cupy")`, both of which ask whether cupy is installed
# rather than whether a device exists.
requires_gpu = _GpuRequirement(pytest.mark.skipif(_gpu_count() < 1, reason="no gpu"))


# Shared `device`/`lazy` parametrize decorators. Every test file used to
# spell `@pytest.mark.parametrize("device", ["cpu", gpu])` (or the
# argument-order-flipped `[gpu, "cpu"]`) inline; that copy-paste had drifted
# into two orderings with no functional difference. Use these two decorators
# instead so the order/spelling is uniform everywhere.
devices = pytest.mark.parametrize("device", [gpu, "cpu"])
lazy_params = pytest.mark.parametrize("lazy", [True, False])


try:
    import dask_cuda as _dask_cuda  # noqa: F401

    _HAS_DASK_CUDA = True
except ImportError:
    _HAS_DASK_CUDA = False


# Skip marker for tests that genuinely need to distribute across GPUs: they
# require both >=2 GPUs and dask-cuda (one worker process per GPU). Use together
# with `pytest.mark.multigpu` so the suite can be selected with `-m multigpu`.
#
# Also applies pytest.mark.gpu via the same _GpuRequirement as requires_gpu
# above -- requires_multigpu tests already carry the separate `multigpu`
# marker too (applied explicitly alongside this one wherever it's used),
# which conftest.py's grouping hook already checks independently of
# anything here; this is for symmetry with requires_gpu, so a
# requires_multigpu-only test (if one is ever written without the
# `multigpu` marker) is still caught.
requires_multigpu = _GpuRequirement(
    pytest.mark.skipif(
        _gpu_count() < 2 or not _HAS_DASK_CUDA,
        reason="requires >=2 GPUs and dask-cuda",
    )
)


# Skip marker for the Metal backend, which -- like CUDA -- is exercised whenever
# the machine has it. Deselect it with -k "not mps".
requires_mps = pytest.mark.skipif(
    not _mps_is_usable(),
    reason=(
        "requires the Metal (MPS) backend: macOS on Apple silicon with PyTorch "
        "installed"
    ),
)


def synthetic_transition_potential(
    Z: int = 14,
    gpts: tuple[int, int] = (64, 64),
    extent: tuple[float, float] | None = (8.0, 8.0),
    energy: float | None = 100e3,
    n_transitions: int = 2,
    seed: int = 0,
    device: str = "cpu",
):
    """A seeded ``TransitionPotentialArray`` with a synthetic payload.

    Tests that exercise machinery *around* the transition potentials --
    graph transport, caching, detector wiring -- need an object of the right
    shape, not real physics. Building one directly skips the GPAW atomic
    solvers, so these tests also run where GPAW is not installed (CI).
    """
    from abtem.core.axes import OrdinalAxis
    from abtem.inelastic.core_loss import TransitionPotentialArray

    xp = get_array_module(device)
    rng = np.random.default_rng(seed)
    array = xp.asarray(
        (
            rng.standard_normal((n_transitions, *gpts))
            + 1j * rng.standard_normal((n_transitions, *gpts))
        ).astype(np.complex64)
    )
    return TransitionPotentialArray(
        Z=Z,
        array=array,
        energy=energy,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": Z, "n": 1, "l": 0},
    )


def to_host_array(measurement):
    """The array of a measurement, wave function, or bare array, on the host.

    Detectors return host arrays by default, but reductions to wave functions
    and their derived measurements may stay on the device (and can be dask-
    or cupy-backed), so cross-device/laziness comparisons should go through
    this rather than each test reimplementing the unwrap-then-convert dance.
    """
    array = measurement.array if hasattr(measurement, "array") else measurement
    if isinstance(array, da.core.Array):
        array = array.compute()
    return asnumpy(array)


def _to_host_array_object(obj):
    """A computed host-memory copy of ``obj``.

    ``to_cpu`` first, since it returns a new object: ``compute`` works in
    place and would otherwise turn the caller's lazy object eager.
    """
    obj = obj.to_cpu()
    if obj.is_lazy:
        obj.compute()
    return obj


def assert_array_objects_equal(
    a,
    b,
    rtol: float = 0.0,
    atol: float = 0.0,
    check_dtype: bool = True,
):
    """Assert two ArrayObjects (Waves, measurements, ...) are equal in value.

    ``a == b`` cannot be used for this: ``safe_equality`` skips any attribute
    whose comparison is a dask value, so for lazy objects only the metadata
    is ever compared and e.g. ``Images(da.zeros(...)) == Images(da.ones(...))``
    is True (abTEM issue #413). This computes both sides, moves them to the
    host and checks, in turn: type, shape, dtype, base and ensemble axes
    metadata, the ``metadata`` dict, every other constructor attribute
    (sampling, energy, ...) and finally the array values.

    The default tolerance is exact, for comparing two objects that went
    through the same code path (a round trip, a copy, a stack slice). For a
    comparison across genuinely different paths (PRISM vs. multislice, lazy
    vs. eager reductions) the caller must pass ``rtol``/``atol`` chosen for
    the quantity being compared.

    Lists/tuples of ArrayObjects (multi-detector output) are compared
    element-wise.
    """
    from abtem.core.utils import safe_equality

    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)), f"{type(a)} vs {type(b)}"
        assert len(a) == len(b), f"{len(a)} vs {len(b)} array objects"
        for a_i, b_i in zip(a, b):
            assert_array_objects_equal(
                a_i, b_i, rtol=rtol, atol=atol, check_dtype=check_dtype
            )
        return

    assert type(a) is type(b), f"{type(a).__name__} vs {type(b).__name__}"

    a = _to_host_array_object(a)
    b = _to_host_array_object(b)

    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    if check_dtype:
        assert a.dtype == b.dtype, f"dtype {a.dtype} vs {b.dtype}"

    assert len(a.ensemble_axes_metadata) == len(b.ensemble_axes_metadata)
    for i, (axis_a, axis_b) in enumerate(
        zip(a.ensemble_axes_metadata, b.ensemble_axes_metadata)
    ):
        assert axis_a == axis_b, f"ensemble axis {i}: {axis_a!r} != {axis_b!r}"

    assert len(a.base_axes_metadata) == len(b.base_axes_metadata)
    for i, (axis_a, axis_b) in enumerate(
        zip(a.base_axes_metadata, b.base_axes_metadata)
    ):
        assert axis_a == axis_b, f"base axis {i}: {axis_a!r} != {axis_b!r}"

    assert set(a.metadata) == set(b.metadata), (
        f"metadata keys differ: {sorted(set(a.metadata) ^ set(b.metadata))}"
    )
    np.testing.assert_equal(a.metadata, b.metadata, err_msg="metadata differs")

    # Everything else the object carries (sampling, energy, extent, ...),
    # with the array itself excluded: it is compared below, with tolerances.
    exclude = ("_array", "_metadata", "_ensemble_axes_metadata")
    exclude += tuple(getattr(a, "_eq_exclude", ()))
    differing = [
        key
        for key in a.__dict__
        if key not in exclude
        and not safe_equality(
            _AttributeHolder(a.__dict__[key]), _AttributeHolder(b.__dict__.get(key))
        )
    ]
    assert not differing, f"attributes differ: {differing}"

    np.testing.assert_allclose(
        asnumpy(a.array), asnumpy(b.array), rtol=rtol, atol=atol
    )


class _AttributeHolder:
    """Wraps one attribute so ``safe_equality`` compares just that value."""

    def __init__(self, value):
        self.value = value


def si_cubic_atoms():
    """A cubic-conventional-cell Si ``Atoms`` (diamond structure)."""
    return ase.build.bulk("Si", cubic=True)


def si_diamond_atoms():
    """The same cubic-conventional-cell diamond Si, built explicitly.

    Equivalent to :func:`si_cubic_atoms`; kept as a separate name for tests
    that spell out ``crystalstructure="diamond", a=5.43`` explicitly rather
    than relying on ASE's default lattice constant for Si.
    """
    return ase.build.bulk("Si", crystalstructure="diamond", a=5.43, cubic=True)
