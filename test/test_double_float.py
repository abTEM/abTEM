"""Tests for emulated double precision on Apple Metal (``abtem.core._double_float``).

The kernels need a real Metal device, so these are skipped everywhere else,
including torch's CPU device (``ABTEM_TORCH__DEVICE=cpu``), which cannot compile
Metal shaders. Accuracy is asserted against NumPy's float64 at tolerances far
below float32's ~1e-7, which is the point of the module.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import requires_mps

import abtem
from abtem.core import config

pytestmark = [
    *requires_mps,
    pytest.mark.skipif(
        str(config.get("torch.device")).lower() != "mps",
        reason="Metal shaders need the real MPS device",
    ),
]


@pytest.fixture
def double_float():
    from abtem.core import _double_float

    return _double_float


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _relative_error(x, reference):
    return np.abs(x - reference).max() / np.abs(reference).max()


def test_round_trip_keeps_float64_precision(double_float):
    rng = np.random.default_rng(0)
    x = _random_complex(rng, (3, 16, 16))

    y = double_float.to_numpy(double_float.from_numpy(x))

    # A float32 round trip is off by ~6e-8 relative, here ~1e-15.
    assert _relative_error(y, x) < 1e-14


@pytest.mark.parametrize(
    "shape",
    [(8, 8), (3, 8, 16), (2, 64, 128), (1, 256, 256), (4, 24, 24), (2, 45, 100)],
)
def test_fft2_matches_numpy_float64(double_float, shape):
    rng = np.random.default_rng(1)
    x = _random_complex(rng, shape)
    # The input the kernels actually see: x rounded to a pair of float32.
    seen = double_float.to_numpy(double_float.from_numpy(x))

    forward = double_float.to_numpy(double_float.fft2(double_float.from_numpy(x)))

    assert _relative_error(forward, np.fft.fft2(seen)) < 1e-13


@pytest.mark.parametrize(
    "length",
    [2, 3, 5, 6, 7, 9, 10, 12, 15, 21, 25, 35, 49, 60, 120, 245, 343, 360, 500, 1000],
)
def test_fft2_is_accurate_for_every_supported_length(double_float, length):
    rng = np.random.default_rng(length)
    # Both axes get the length in turn, so the row and the column kernels run.
    for shape in [(2, 4, length), (2, length, 4)]:
        x = _random_complex(rng, shape)
        seen = double_float.to_numpy(double_float.from_numpy(x))

        out = double_float.to_numpy(double_float.fft2(double_float.from_numpy(x)))

        assert _relative_error(out, np.fft.fft2(seen)) < 1e-13


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 1280), (2, 1280, 4), (1, 8, 2048), (1, 2048, 8), (1, 1050, 1050)],
)
def test_axes_beyond_one_threadgroup_use_a_four_step_transform(double_float, shape):
    rng = np.random.default_rng(sum(shape))
    x = _random_complex(rng, shape)
    seen = double_float.to_numpy(double_float.from_numpy(x))
    transformed = double_float.fft2(double_float.from_numpy(x))

    out = double_float.to_numpy(transformed)
    back = double_float.to_numpy(double_float.fft2(transformed, inverse=True))

    assert _relative_error(out, np.fft.fft2(seen)) < 1e-13
    assert _relative_error(back, seen) < 1e-13


@pytest.mark.parametrize("shape", [(4, 32, 32), (2, 128, 64), (2, 45, 100)])
def test_inverse_fft2_undoes_fft2(double_float, shape):
    rng = np.random.default_rng(2)
    x = double_float.from_numpy(_random_complex(rng, shape))

    back = double_float.fft2(double_float.fft2(x), inverse=True)

    assert (
        _relative_error(double_float.to_numpy(back), double_float.to_numpy(x)) < 1e-13
    )


def test_fft2_leaves_its_input_untouched(double_float):
    rng = np.random.default_rng(3)
    x = double_float.from_numpy(_random_complex(rng, (2, 16, 16)))
    before = double_float.to_numpy(x)

    double_float.fft2(x)

    np.testing.assert_array_equal(double_float.to_numpy(x), before)


def test_multiply_broadcasts_a_kernel_over_the_batch(double_float):
    rng = np.random.default_rng(4)
    x = _random_complex(rng, (5, 32, 16))
    kernel = _random_complex(rng, (32, 16))
    xd, kd = double_float.from_numpy(x), double_float.from_numpy(kernel)

    product = double_float.to_numpy(double_float.multiply(xd, kd))

    seen_x, seen_k = double_float.to_numpy(xd), double_float.to_numpy(kd)
    assert _relative_error(product, seen_x * seen_k) < 1e-13


@pytest.mark.parametrize("shape", [(11, 8), (16, 13), (2**21, 4), (1, 8), (8, 22)])
def test_unsupported_sizes_raise(double_float, shape):
    x = double_float.from_numpy(np.ones(shape, dtype=np.complex128))

    with pytest.raises(ValueError, match="no prime factor above 7"):
        double_float.fft2(x)


def _exit_wave(emulate, gpts):
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 8)
    with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
        potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=1.0)
        wave = abtem.PlaneWave(energy=200e3)
        wave.grid.match(potential)
        return np.asarray(wave.multislice(potential).compute().array)


@pytest.mark.parametrize("gpts", [64, 60, 72, 75])
def test_emulated_multislice_matches_float64_on_the_cpu(gpts):
    reference = _exit_wave(False, gpts)
    emulated = _exit_wave(True, gpts)

    assert emulated.dtype == np.complex128
    # The float32 multislice is off by ~1e-6 at this depth; emulated double
    # precision stays near float64 rounding.
    assert np.abs(emulated - reference).max() < 1e-10


def test_emulated_multislice_matches_float64_with_a_detector():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 8)

    def measure(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=64, slice_thickness=1.0)
            probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
            probe.grid.match(potential)
            scan = abtem.GridScan(
                start=(0, 0), end=(2, 2), gpts=(2, 2), potential=potential
            )
            detector = abtem.PixelatedDetector(max_angle=40)
            return np.asarray(probe.scan(potential, scan, detector).compute().array)

    reference, emulated = measure(False), measure(True)

    assert np.abs(emulated - reference).max() < 1e-10 * np.abs(reference).max()


def test_emulation_without_float64_waves_raises():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    with config.set({"precision": "float32", "torch.emulate-float64": True}):
        potential = abtem.Potential(atoms, gpts=32, slice_thickness=1.0)
        wave = abtem.PlaneWave(energy=200e3)
        wave.grid.match(potential)

        with pytest.raises(NotImplementedError, match="complex128"):
            wave.multislice(potential).compute()


def test_emulated_potential_matches_float64_on_the_host():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 4)

    def build(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=96, slice_thickness=1.0)
            return np.asarray(potential.build().compute().array)

    reference, emulated = build(False), build(True)

    assert emulated.dtype == np.float64
    assert np.abs(emulated - reference).max() < 1e-12 * np.abs(reference).max()


def test_emulated_potential_keeps_its_cache_out_of_the_pickle():
    import pickle

    from abtem.integrals import ScatteringFactorProjectionIntegrals

    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    integrator = ScatteringFactorProjectionIntegrals()
    with config.set({"precision": "float64", "torch.emulate-float64": True}):
        integrator.integrate_on_grid(atoms, 0.0, 1.0, (32, 32), (0.17, 0.17))
        assert len(integrator._emulated_kernels) == 1

        restored = pickle.loads(pickle.dumps(integrator))

    assert len(restored._emulated_kernels) == 0
