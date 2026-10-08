"""PRISM with a potential that has more than one exit plane.

Multislice is linear in the incident wave, so at interpolation 1 the reduced
S-matrix is the multislice exit wave of the same probe, at every exit plane, to
float round-off. The exit-plane count (4) differs from every other axis size in
these cases: 2 frozen-phonon configurations, a 3 x 5 scan, 37 plane waves, a
64 x 64 grid and a 32 x 32 diffraction pattern, so an axis put in the wrong
place changes the shape or the values.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import assert_array_objects_equal, devices

import abtem
from abtem.core.axes import ThicknessAxis

ATOMS = bulk("Si", cubic=True) * (1, 1, 2)


def _potential(frozen_phonons=None, device="cpu"):
    atoms = ATOMS
    if frozen_phonons is not None:
        atoms = abtem.FrozenPhonons(
            ATOMS, num_configs=2, sigmas=0.05, seed=1, ensemble_mean=frozen_phonons
        )
    # 6 slices, exit planes at the entrance and after every second slice
    potential = abtem.Potential(
        atoms, gpts=64, slice_thickness=2, exit_planes=2, device=device
    )
    assert len(potential.exit_planes) == 4
    return potential


def _s_matrix(potential, **kwargs):
    kwargs = {"energy": 100e3, "interpolation": 1, "downsample": False, **kwargs}
    return abtem.SMatrix(potential=potential, semiangle_cutoff=20, **kwargs)


def _scan(potential):
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


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("disable_s_matrix_chunks", [False, True])
@pytest.mark.parametrize("frozen_phonons", [None, False, True])
@pytest.mark.parametrize("pixelated", [False, True])
@devices
def test_prism_exit_planes_match_multislice(
    lazy, disable_s_matrix_chunks, frozen_phonons, pixelated, device
):
    potential = _potential(frozen_phonons, device=device)
    s_matrix = _s_matrix(potential, device=device)
    scan = _scan(potential)

    def detectors():
        annular = abtem.AnnularDetector(30, 90)
        if pixelated:
            return [annular, abtem.PixelatedDetector(max_angle=100)]
        return annular

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=scan, detectors=detectors(), lazy=lazy
    )
    measured = s_matrix.scan(
        scan=scan,
        detectors=detectors(),
        lazy=lazy,
        disable_s_matrix_chunks=disable_s_matrix_chunks,
    )
    if lazy:
        expected = expected.compute()
        measured = measured.compute()

    # float32 round-off of two different reduction orders: measured at
    # 7.4e-7 of the maximum
    tolerance = 1e-5
    assert_array_objects_equal(
        measured, expected, rtol=tolerance, atol=tolerance * _max_abs(expected)
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_prism_exit_planes_match_truncated_potentials(lazy):
    # With interpolation the reduction is not multislice, but multislice is
    # causal: the S-matrix at an exit plane is the S-matrix of the potential
    # truncated after that plane's slice, reduced the same way.
    potential = _potential()
    scan = _scan(potential)
    detector = abtem.PixelatedDetector(max_angle=100)

    measured = _s_matrix(potential, interpolation=2).scan(
        scan=scan, detectors=detector, lazy=lazy
    )
    if lazy:
        measured = measured.compute()

    whole = abtem.Potential(ATOMS, gpts=64, slice_thickness=2).build(lazy=False)

    assert measured.shape == (4, 3, 5, 18, 18)
    assert isinstance(measured.axes_metadata[0], ThicknessAxis)
    for i, plane in enumerate(potential.exit_planes):
        if plane == -1:
            reference = abtem.SMatrix(
                extent=potential.extent,
                gpts=potential.gpts,
                energy=100e3,
                semiangle_cutoff=20,
                interpolation=2,
                downsample=False,
            )
        else:
            reference = _s_matrix(whole[: plane + 1], interpolation=2)

        expected = reference.scan(scan=scan, detectors=detector, lazy=False)
        np.testing.assert_allclose(
            measured.array[i],
            expected.array,
            rtol=1e-5,
            atol=1e-5 * np.abs(expected.array).max(),
        )


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("frozen_phonons", [None, False])
@pytest.mark.parametrize("store_on_host", [False, True])
@devices
def test_s_matrix_build_has_an_exit_plane_axis(
    lazy, frozen_phonons, store_on_host, device
):
    potential = _potential(frozen_phonons, device=device)
    s_matrix_array = _s_matrix(
        potential, device=device, store_on_host=store_on_host
    ).build(lazy=lazy)

    ensemble_shape = potential.ensemble_shape + (4,)
    assert s_matrix_array.array.shape == ensemble_shape + (37, 64, 64)
    assert s_matrix_array.ensemble_shape == ensemble_shape

    thickness_axis = s_matrix_array.ensemble_axes_metadata[-1]
    assert isinstance(thickness_axis, ThicknessAxis)
    assert thickness_axis.values == tuple(potential.exit_thicknesses)

    computed = s_matrix_array.copy().compute().array
    assert computed.shape == s_matrix_array.array.shape


def test_s_matrix_build_lazy_equals_eager_with_exit_planes():
    s_matrix = _s_matrix(_potential(False))
    lazy = s_matrix.build(lazy=True).compute().array
    eager = s_matrix.build(lazy=False).array
    np.testing.assert_array_equal(lazy, eager)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("frozen_phonons", [None, False])
def test_multi_energy_build_with_exit_planes_stacks_the_single_energy_builds(
    lazy, frozen_phonons
):
    # The energies have different plane-wave sets (29 at 80 keV, 37 at
    # 100 keV); each single-energy build sits at its own wave vectors in the
    # union, after the potential-ensemble and exit-plane axes.
    potential = _potential(frozen_phonons)
    energies = (80e3, 100e3)
    built = abtem.SMatrix(
        potential=potential,
        energy=list(energies),
        semiangle_cutoff=20,
        interpolation=1,
        downsample=False,
    ).build(lazy=lazy)

    assert built.array.shape == (2,) + potential.ensemble_shape + (4, 37, 64, 64)
    array = built.copy().compute().array
    assert array.shape == built.array.shape
    union = {tuple(q): i for i, q in enumerate(np.asarray(built.wave_vectors))}

    for i, energy in enumerate(energies):
        single = _s_matrix(potential, energy=energy).build(lazy=False)
        indices = [union[tuple(q)] for q in np.asarray(single.wave_vectors)]
        np.testing.assert_array_equal(array[i][..., indices, :, :], single.array)


@pytest.mark.parametrize("lazy", [True, False])
def test_multi_energy_prism_exit_planes_match_multislice(lazy):
    potential = _potential()
    scan = _scan(potential)
    energies = (80e3, 100e3)
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=list(energies),
        semiangle_cutoff=20,
        interpolation=1,
        downsample=False,
    )
    measured = s_matrix.scan(
        scan=scan, detectors=abtem.AnnularDetector(30, 90), lazy=lazy
    )
    if lazy:
        measured = measured.compute()

    assert measured.shape == (2, 4, 3, 5)
    for i, energy in enumerate(energies):
        probe = abtem.SMatrix(
            potential=potential,
            energy=energy,
            semiangle_cutoff=20,
            interpolation=1,
            downsample=False,
        ).dummy_probes()
        expected = probe.scan(
            potential=potential,
            scan=scan,
            detectors=abtem.AnnularDetector(30, 90),
            lazy=False,
        )
        np.testing.assert_allclose(
            measured.array[i],
            expected.array,
            rtol=1e-5,
            atol=1e-5 * np.abs(expected.array).max(),
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda s, scan: s.build(lazy=False),
        lambda s, scan: s.build(lazy=True),
        lambda s, scan: s.scan(scan=scan, lazy=False),
        lambda s, scan: s.scan(scan=scan, lazy=True),
    ],
    ids=["build-eager", "build-lazy", "scan-eager", "scan-lazy"],
)
def test_upsample_refuses_exit_planes(call):
    potential = _potential()
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )
    with pytest.raises(NotImplementedError, match="exit plane"):
        call(s_matrix, _scan(potential))
