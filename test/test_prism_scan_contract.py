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
from utils import assert_array_objects_equal, devices

import abtem

ATOMS = bulk("Si", cubic=True) * (1, 1, 2)
POSITION = (2.0, 1.5)

# float32 round-off of two reduction orders; the largest difference over
# these cases is about 1e-6 of the maximum
TOLERANCE = 1e-5


def _potential(frozen_phonons=None, device="cpu", num_configs=2, exit_planes=None):
    atoms = ATOMS
    if frozen_phonons is not None:
        atoms = abtem.FrozenPhonons(
            ATOMS,
            num_configs=num_configs,
            sigmas=0.05,
            seed=1,
            ensemble_mean=frozen_phonons,
        )
    return abtem.Potential(
        atoms, gpts=64, slice_thickness=2, exit_planes=exit_planes, device=device
    )


def _s_matrix(potential, **kwargs):
    kwargs = {"energy": 100e3, "interpolation": 1, "downsample": False, **kwargs}
    return abtem.SMatrix(potential=potential, semiangle_cutoff=20, **kwargs)


def _grid_scan(potential):
    return abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        gpts=(3, 5),
        fractional=True,
        endpoint=False,
        potential=potential,
    )


def _max_abs(measurements):
    measurements = measurements if isinstance(measurements, list) else [measurements]
    return max(float(np.abs(m.to_cpu().array).max()) for m in measurements)


def _assert_matches(measured, expected):
    assert_array_objects_equal(
        measured,
        expected,
        rtol=TOLERANCE,
        atol=TOLERANCE * _max_abs(expected),
    )


_PATHS = {
    "eager": dict(lazy=False),
    "lazy": dict(lazy=True, disable_s_matrix_chunks=False),
    "lazy-whole-s-matrix": dict(lazy=True, disable_s_matrix_chunks=True),
}


@pytest.mark.parametrize("path", list(_PATHS))
@devices
def test_prism_keeps_waves_of_every_frozen_phonon_configuration(path, device):
    # ensemble_mean averages the annular intensities over the configurations;
    # wave functions are never averaged, so they keep the configuration axis
    potential = _potential(frozen_phonons=True, device=device)
    s_matrix = _s_matrix(potential, device=device)
    scan = _grid_scan(potential)

    def detectors():
        return [abtem.WavesDetector(), abtem.AnnularDetector(30, 90)]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=scan, detectors=detectors(), lazy=False
    )
    measured = s_matrix.scan(scan=scan, detectors=detectors(), **_PATHS[path])
    measured = measured.compute()

    assert measured[0].shape == (2, 3, 5, 64, 64)
    assert measured[1].shape == (3, 5)
    _assert_matches(list(measured), list(expected))


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


@pytest.mark.parametrize("lazy", [False, True])
def test_multi_energy_prism_single_position_drops_the_position_axis(lazy):
    potential = _potential()
    s_matrix = _s_matrix(potential, energy=[80e3, 100e3])
    detector = abtem.AnnularDetector(30, 90)

    measured = s_matrix.scan(scan=POSITION, detectors=detector, lazy=lazy).compute()

    assert measured.shape == (2,)
    for i, energy in enumerate((80e3, 100e3)):
        expected = (
            _s_matrix(potential, energy=energy)
            .dummy_probes()
            .scan(potential=potential, scan=POSITION, detectors=detector, lazy=False)
        )
        np.testing.assert_allclose(
            measured.array[i], expected.array, rtol=TOLERANCE, atol=0
        )


@pytest.mark.parametrize(
    "entry", ["scan-eager", "scan-lazy", "build-reduce", "build-reduce-modes"]
)
@pytest.mark.parametrize("pixelated", [False, True])
def test_upsampled_prism_single_position_drops_the_position_axis(entry, pixelated):
    # oracle: the same reduction at a one-position CustomScan, which keeps
    # its position axis
    potential = _potential()
    s_matrix = _s_matrix(potential, interpolation=2, upsample=True)
    # the default method on CPU expands to an SMatrixArray and reduces that
    method = "modes" if entry.endswith("modes") else "auto"
    s_matrix_array = s_matrix.build(lazy=False)

    def reduce(scan):
        if pixelated:
            detector = abtem.PixelatedDetector(max_angle=60)
        else:
            detector = abtem.AnnularDetector(30, 90)
        if entry.startswith("build-reduce"):
            return s_matrix_array.reduce(scan=scan, detectors=detector, method=method)
        return s_matrix.scan(
            scan=scan, detectors=detector, lazy=entry == "scan-lazy"
        ).compute()

    expected = reduce(abtem.CustomScan([POSITION]))
    measured = reduce(POSITION)

    assert measured.shape == expected.shape[1:]
    assert measured.axes_metadata == expected.axes_metadata[1:]
    np.testing.assert_array_equal(measured.array, expected.array[0])


def test_composite_blend_single_position_drops_the_position_axis():
    # the blend adds the intensities of two reductions at the one position
    potential = _potential()
    s_matrix_array = _s_matrix(potential, interpolation=2, upsample=True).build(
        lazy=False
    )

    def reduce(scan):
        return s_matrix_array.reduce(
            scan=scan,
            detectors=abtem.AnnularDetector(30, 90),
            blend_angle=40.0,
            blend_window_gpts="period",
        )

    expected = reduce(abtem.CustomScan([POSITION]))
    measured = reduce(POSITION)

    assert measured.shape == ()
    np.testing.assert_array_equal(measured.array, expected.array[0])
