"""Tests for the experimental Metal (MPS) backend on Apple silicon.

Skipped unless PyTorch is installed on Apple silicon *and* 'enable_mps' was set
before abTEM was imported, e.g.::

    ABTEM_ENABLE_MPS=true pytest test/test_mps.py

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


def test_angle_and_round_take_their_second_argument_positionally():
    # numpy.angle(z, deg) and numpy.round(a, decimals) both accept a second
    # positional argument that torch either spells as a keyword or does not
    # take at all; dask's own angle() passes deg positionally.
    xp = get_array_module("mps")
    rng = np.random.RandomState(3)

    z = (rng.randn(4, 5) + 1j * rng.randn(4, 5)).astype(np.complex64)
    on_device = copy_to_device(z, "mps")

    assert np.allclose(asnumpy(xp.angle(on_device)), np.angle(z), atol=1e-5)
    assert np.allclose(asnumpy(xp.angle(on_device, True)), np.angle(z, True), atol=1e-3)

    x = rng.randn(6).astype(np.float32)
    assert np.allclose(asnumpy(xp.round(copy_to_device(x, "mps"), 2)), np.round(x, 2))


def test_lazy_phase_matches_cpu():
    arrays = []
    for device in ("cpu", "mps"):
        with abtem.config.set({"dask.lazy": True}):
            probe = abtem.Probe(
                energy=100e3, semiangle_cutoff=20, gpts=64, extent=10, device=device
            )
            arrays.append(asnumpy(probe.build().phase().compute().array))

    # Compared as a wrapped difference: a probe's phase sits on the branch cut
    # over much of the plane, where a float32 rounding either way flips the
    # value between +pi and -pi.
    difference = np.angle(np.exp(1j * (arrays[0] - arrays[1])))

    assert np.abs(difference).max() < 1e-3


@pytest.mark.parametrize("lazy_unit", [False, True])
def test_crystal_potential_from_built_unit_matches_cpu(atoms, lazy_unit):
    # A unit potential the caller built themselves is used as-is; built
    # lazily, its array is a dask array rather than one of the device's own.
    arrays = []
    for device in ("cpu", "mps"):
        with abtem.config.set({"dask.lazy": True}):
            unit = abtem.Potential(atoms, gpts=64, device=device).build(lazy=lazy_unit)
            crystal = abtem.CrystalPotential(unit, repetitions=(2, 2, 2))
            waves = abtem.PlaneWave(energy=100e3, device=device).multislice(crystal)
            arrays.append(asnumpy(waves.compute().array))

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_where_without_x_and_y():
    # numpy.where(condition) is numpy.nonzero(condition); torch.where's
    # one-argument form is spelled the same way but the wrapper required all
    # three. Reached from e.g. LineProfiles.width, via a sign-change search.
    xp = get_array_module("mps")

    array = copy_to_device(np.array([0.0, 1.0, 0.0, 2.0, 3.0], np.float32), "mps")
    (indices,) = xp.where(array > 0.5)

    assert np.array_equal(asnumpy(indices), np.array([1, 3, 4]))

    with pytest.raises(ValueError, match="both or neither"):
        xp.where(array > 0.5, array)


def test_line_profile_width_matches_cpu():
    widths = []
    for device in ("cpu", "mps"):
        probe = abtem.Probe(
            energy=100e3, semiangle_cutoff=20, gpts=128, extent=20, device=device
        )
        profile = (
            probe.build()
            .intensity()
            .interpolate_line_at_position(center=(10, 10), angle=0, extent=10)
        )
        widths.append(float(asnumpy(profile.width(height=0.5))))

    assert widths[1] == pytest.approx(widths[0], rel=1e-4)
