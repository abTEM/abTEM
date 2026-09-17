"""Tests for the experimental Metal (MPS) backend on Apple silicon.

Skipped unless PyTorch is installed on Apple silicon *and* 'enable_mps' was set
before abTEM was imported, e.g.::

    DASK_ENABLE_MPS=true pytest test/test_mps.py

Metal is single precision, so every comparison against the CPU reference is made
at float32 tolerances rather than exactly.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import requires_mps

import abtem
from abtem.core.backend import (
    asnumpy,
    copy_to_device,
    device_name_from_array_module,
    get_array_module,
)

pytestmark = requires_mps


@pytest.fixture
def atoms():
    return bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 3)


@pytest.fixture(autouse=True)
def eager():
    with abtem.config.set({"dask.lazy": False, "precision": "float32"}):
        yield


def test_array_module_round_trip():
    xp = get_array_module("mps")

    assert device_name_from_array_module(xp) == "mps"
    assert get_array_module("metal") is xp
    assert get_array_module("torch") is xp

    array = np.random.RandomState(0).randn(4, 5).astype(np.float32)
    on_device = copy_to_device(array, "mps")

    assert get_array_module(on_device) is xp
    assert on_device.dtype == np.float32
    assert on_device.shape == array.shape
    assert np.array_equal(asnumpy(on_device), array)
    assert np.array_equal(copy_to_device(on_device, "cpu"), array)


def test_scalar_keeps_zero_dimensions():
    # np.ascontiguousarray promotes a 0-d scalar to shape (1,); a spurious
    # dimension there propagates into every broadcast against it.
    xp = get_array_module("mps")

    assert xp.asarray(20.0).shape == ()
    assert xp.asarray(np.float32(20.0)).shape == ()
    assert xp.asarray([20.0]).shape == (1,)


def test_inferred_double_precision_is_narrowed():
    # A Python float is float64 to NumPy, but Metal has no float64; an inferred
    # dtype narrows rather than failing.
    xp = get_array_module("mps")

    assert xp.asarray((0.1, 0.2)).dtype == np.float32

    with pytest.raises(RuntimeError, match="single-precision"):
        xp.asarray(np.zeros(4), dtype=np.float64)


def test_double_precision_configuration_is_rejected():
    with abtem.config.set({"precision": "float64"}):
        with pytest.raises(RuntimeError, match="single-precision"):
            get_array_module("mps")


def test_metal_device_rejects_double_precision_when_configured():
    # The combination is refused where it is set, not later inside a
    # computation, and a refused set leaves the configuration untouched.
    before = (abtem.config.get("device"), abtem.config.get("precision"))

    with pytest.raises(ValueError, match="single-precision"):
        abtem.config.set({"device": "mps", "precision": "float64"})

    assert (abtem.config.get("device"), abtem.config.get("precision")) == before

    with abtem.config.set({"device": "mps"}):
        with pytest.raises(ValueError, match="single-precision"):
            abtem.config.set({"precision": "float64"})

        assert abtem.config.get("precision") == "float32"

    assert (abtem.config.get("device"), abtem.config.get("precision")) == before


def test_unsupported_operation_names_itself():
    xp = get_array_module("mps")

    with pytest.raises(AttributeError, match="not_a_real_ufunc"):
        xp.not_a_real_ufunc


def test_fft_round_trip():
    array = (np.random.RandomState(1).randn(2, 32, 32) * (1 + 1j)).astype(np.complex64)

    on_device = copy_to_device(array, "mps")
    restored = asnumpy(abtem.core.fft.ifft2(abtem.core.fft.fft2(on_device)))

    assert np.allclose(restored, array, atol=1e-4)
    assert np.allclose(
        asnumpy(abtem.core.fft.fft2(on_device)), np.fft.fft2(array), atol=1e-3
    )


def test_scatter_add_matches_numpy():
    xp = get_array_module("mps")
    rng = np.random.RandomState(2)

    rows = rng.randint(0, 16, 32)
    cols = rng.randint(0, 16, 32)
    values = rng.randn(32).astype(np.float32)

    expected = np.zeros((16, 16), dtype=np.float32)
    np.add.at(expected, (rows, cols), values)

    result = xp.zeros((16, 16), dtype=np.float32)
    xp.add.at(result, (xp.asarray(rows), xp.asarray(cols)), xp.asarray(values))

    assert np.allclose(asnumpy(result), expected, atol=1e-5)


def test_potential_matches_cpu(atoms):
    arrays = [
        asnumpy(abtem.Potential(atoms, gpts=128, device=device).build().array)
        for device in ("cpu", "mps")
    ]

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_plane_wave_multislice_matches_cpu(atoms):
    arrays = []
    for device in ("cpu", "mps"):
        potential = abtem.Potential(atoms, gpts=128, device=device)
        waves = abtem.PlaneWave(energy=100e3, device=device).multislice(potential)
        arrays.append(asnumpy(waves.array))

    assert arrays[1].dtype == np.complex64
    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_stem_scan_matches_cpu(atoms):
    arrays = []
    for device in ("cpu", "mps"):
        potential = abtem.Potential(atoms, gpts=128, device=device)
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(3, 3))
        detector = abtem.AnnularDetector(inner=50, outer=150)
        arrays.append(
            asnumpy(probe.scan(potential, scan=scan, detectors=detector).array)
        )

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())
