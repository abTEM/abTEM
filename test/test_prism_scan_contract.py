"""PRISM returns what multislice of the same probes returns.

At interpolation 1 the reduced S-matrix is the multislice exit wave of
``s_matrix.dummy_probes()`` to float round-off (multislice is linear in the
incident wave), so shape, axes metadata and values must agree with
``Probe.scan``. The sizes differ along every axis: 2 frozen-phonon
configurations, a 3 x 5 scan, a 64 x 64 grid and a 32 x 32 diffraction
pattern.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem.core.backend import asnumpy
from abtem.scan import validate_scan

devices = pytest.mark.parametrize("device", ["cpu", gpu])

ATOMS = bulk("Si", cubic=True) * (1, 1, 2)
POSITION = (2.0, 1.5)

# float32 round-off of two reduction orders; the largest difference over
# these cases is about 1e-6 of the maximum
TOLERANCE = 1e-5


def _potential(frozen_phonons=None, device="cpu", exit_planes=None):
    atoms = ATOMS
    if frozen_phonons is not None:
        atoms = abtem.FrozenPhonons(
            ATOMS, num_configs=2, sigmas=0.05, seed=1, ensemble_mean=frozen_phonons
        )
    return abtem.Potential(
        atoms, gpts=64, slice_thickness=2, exit_planes=exit_planes, device=device
    )


def _s_matrix(potential, **kwargs):
    kwargs = {"energy": 100e3, "interpolation": 1, "downsample": False, **kwargs}
    return abtem.SMatrix(potential=potential, semiangle_cutoff=20, **kwargs)


def _host(measurement):
    measurement = measurement.to_cpu()
    if measurement.is_lazy:
        measurement = measurement.compute()
    return measurement


def _assert_matches(measured, expected):
    """Type, shape, axes metadata and values (to float32 round-off) agree."""
    measured = measured if isinstance(measured, list) else [measured]
    expected = expected if isinstance(expected, list) else [expected]
    assert len(measured) == len(expected)
    expected = [_host(e) for e in expected]
    atol = TOLERANCE * max(float(np.abs(e.array).max()) for e in expected)

    for a, b in zip(measured, expected):
        a = _host(a)
        assert type(a) is type(b), f"{type(a).__name__} vs {type(b).__name__}"
        assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
        assert len(a.ensemble_axes_metadata) == len(b.ensemble_axes_metadata)
        for axis_a, axis_b in zip(a.ensemble_axes_metadata, b.ensemble_axes_metadata):
            assert axis_a == axis_b, f"{axis_a!r} != {axis_b!r}"
        np.testing.assert_allclose(
            asnumpy(a.array), asnumpy(b.array), rtol=TOLERANCE, atol=atol
        )


_PATHS = {
    "eager": dict(lazy=False),
    "lazy": dict(lazy=True, disable_s_matrix_chunks=False),
    "lazy-whole-s-matrix": dict(lazy=True, disable_s_matrix_chunks=True),
}


@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("frozen_phonons", [None, False])
@pytest.mark.parametrize("detector", ["annular", "pixelated", "waves"])
@devices
def test_prism_single_position_matches_multislice(
    path, frozen_phonons, detector, device
):
    # a bare (x, y) loses its position axis, as Probe.scan drops it
    potential = _potential(frozen_phonons, device=device)
    s_matrix = _s_matrix(potential, device=device)
    make_detector = {
        "annular": lambda: abtem.AnnularDetector(30, 90),
        "pixelated": lambda: abtem.PixelatedDetector(max_angle=100),
        "waves": lambda: abtem.WavesDetector(),
    }[detector]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=POSITION, detectors=make_detector(), lazy=False
    )
    measured = s_matrix.scan(
        scan=POSITION, detectors=make_detector(), **_PATHS[path]
    ).compute()

    _assert_matches(measured, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_s_matrix_array_reduce_single_position_matches_multislice(lazy):
    potential = _potential(exit_planes=2)
    s_matrix = _s_matrix(potential)
    detector = abtem.AnnularDetector(30, 90)

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=POSITION, detectors=detector, lazy=False
    )
    measured = s_matrix.build(lazy=lazy).reduce(scan=POSITION, detectors=detector)
    measured = measured.compute()

    assert measured.shape == (4,)
    _assert_matches(measured, expected)


@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("exit_planes", [None, 2])
@devices
def test_reduce_without_a_scan_has_no_position_axis(path, exit_planes, device):
    potential = _potential(exit_planes=exit_planes, device=device)
    s_matrix = _s_matrix(potential, device=device)
    centre = (potential.extent[0] / 2, potential.extent[1] / 2)

    def detectors():
        return [abtem.AnnularDetector(30, 90), abtem.PixelatedDetector(max_angle=100)]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=centre, detectors=detectors(), lazy=False
    )
    measured = s_matrix.reduce(detectors=detectors(), **_PATHS[path]).compute()

    planes = () if exit_planes is None else (4,)
    assert measured[0].shape == planes
    assert measured[1].shape == planes + expected[1].base_shape
    _assert_matches(list(measured), list(expected))


@pytest.mark.parametrize("lazy", [False, True])
@devices
def test_array_reduce_squeezes_a_validated_bare_position(lazy, device):
    s_matrix = _s_matrix(_potential(device=device), interpolation=2, device=device)
    built = s_matrix.build(lazy=lazy)
    scan = validate_scan(POSITION, s_matrix)

    measured = built.reduce(scan=scan, detectors=abtem.AnnularDetector(30, 90))
    expected = built.reduce(scan=POSITION, detectors=abtem.AnnularDetector(30, 90))

    assert measured.shape == expected.shape == ()
