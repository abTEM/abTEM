import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem import CPRISM, CustomScan, GridScan, Potential, Probe, SMatrix


def _small_potential(gpts=96, repetitions=(2, 2, 3)):
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(atoms, gpts=gpts, slice_thickness=2)


def _relative_error(measurement, reference):
    return (
        np.sqrt(((measurement.array - reference.array) ** 2).mean())
        / reference.array.mean()
    )


@pytest.mark.parametrize("interpolation", [(1, 1), (2, 2), (2, 4)])
def test_c_prism_matches_probe(interpolation):
    c_prism = CPRISM(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=interpolation,
        tolerance=1e-6,
    )

    c_prism_array = c_prism.build(lazy=False)

    probe_diffraction_patterns = (
        c_prism.dummy_probes().build(lazy=False).diffraction_patterns(max_angle=None)
    )
    diffraction_patterns = c_prism_array.reduce().diffraction_patterns(max_angle=None)

    assert np.allclose(
        np.squeeze(diffraction_patterns.array),
        np.squeeze(probe_diffraction_patterns.array),
        atol=1e-5,
    )


def test_c_prism_rank_one_in_vacuum():
    c_prism = CPRISM(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        tolerance=1e-6,
    )
    assert c_prism.build(lazy=False).rank == 1


def test_c_prism_matches_multislice_no_interpolation():
    potential = _small_potential()

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    scan = CustomScan([(2.2, 3.3), (5.0, 5.0)])

    probe_diffraction_patterns = probe.multislice(
        potential=potential, scan=scan, lazy=False
    ).diffraction_patterns(max_angle=50)

    c_prism = CPRISM(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        tolerance=1e-6,
        max_rank=10_000,
    )

    diffraction_patterns = c_prism.reduce(scan=scan, lazy=False).diffraction_patterns(
        max_angle=50
    )

    assert np.allclose(
        diffraction_patterns.array, probe_diffraction_patterns.array, atol=1e-5
    )


def test_c_prism_beats_prism_at_same_interpolation():
    potential = _small_potential(repetitions=(2, 2, 4))

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(
        start=(0, 0), end=potential.extent, sampling=probe.aperture.nyquist_sampling
    )

    reference = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=False
    )

    prism_measurement = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    ).scan(scan=scan, detectors=detector, lazy=False)

    c_prism_measurement = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    ).scan(scan=scan, detectors=detector, lazy=False)

    prism_error = _relative_error(prism_measurement, reference)
    c_prism_error = _relative_error(c_prism_measurement, reference)

    assert c_prism_error < prism_error
    # the annular detector integrates any aliased ghost probes of the
    # interpolation as error; the band-limited interpolant must keep the
    # absolute error small, not just smaller than PRISM
    assert c_prism_error < 0.05


def test_c_prism_off_grid_positions():
    potential = _small_potential()

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    scan = CustomScan([(2.6180339, 4.1415926), (7.7182818, 3.3025851)])

    probe_diffraction_patterns = probe.multislice(
        potential=potential, scan=scan, lazy=False
    ).diffraction_patterns(max_angle=50)

    c_prism = CPRISM(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        tolerance=1e-6,
        max_rank=10_000,
    )

    diffraction_patterns = c_prism.reduce(scan=scan, lazy=False).diffraction_patterns(
        max_angle=50
    )

    assert np.allclose(
        diffraction_patterns.array, probe_diffraction_patterns.array, atol=1e-5
    )


def test_c_prism_windowed():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

    full = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    ).scan(scan=scan, detectors=detector, lazy=False)

    windowed = CPRISM(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        window_gpts=48,
    ).scan(scan=scan, detectors=detector, lazy=False)

    # the cropping window truncates the high-angle scattering tails, hence the
    # annular dark field signal is slightly reduced
    assert np.allclose(windowed.array, full.array, rtol=0.12)


def test_c_prism_window_normalization():
    c_prism = CPRISM(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        tolerance=1e-6,
        window_gpts=48,
    )

    diffraction_patterns = (
        c_prism.build(lazy=False).reduce().diffraction_patterns(max_angle=None)
    )

    assert np.allclose(diffraction_patterns.array.sum(), 1.0, atol=1e-2)


@pytest.mark.parametrize("lazy", [False, True])
def test_c_prism_frozen_phonons(lazy):
    atoms = bulk("Si", cubic=True) * (2, 2, 3)
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=13)
    potential = Potential(frozen_phonons, gpts=96, slice_thickness=2)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    c_prism = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    )

    measurement = c_prism.scan(scan=scan, detectors=detector, lazy=lazy)

    if lazy:
        measurement = measurement.compute()

    assert measurement.shape == (3, 3)
    assert np.all(measurement.array >= 0.0)


def test_c_prism_lazy_matches_eager():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    kwargs = dict(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    )

    eager = CPRISM(**kwargs).scan(scan=scan, detectors=detector, lazy=False)
    lazy = CPRISM(**kwargs).scan(scan=scan, detectors=detector, lazy=True).compute()

    assert np.allclose(eager.array, lazy.array, rtol=1e-4, atol=1e-8)


def test_c_prism_detectors():
    potential = _small_potential()

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))

    detectors = [
        abtem.AnnularDetector(inner=40, outer=100),
        abtem.FlexibleAnnularDetector(),
        abtem.PixelatedDetector(max_angle=100),
        abtem.WavesDetector(),
    ]

    c_prism = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    )

    measurements = c_prism.scan(scan=scan, detectors=detectors, lazy=False)

    assert len(measurements) == len(detectors)
    for measurement in measurements:
        assert measurement.shape[:2] == (2, 2)


def test_c_prism_ctf_ensemble():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))

    ctf = abtem.CTF(defocus=np.linspace(0, 50, 3))

    c_prism = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    )

    measurement = c_prism.scan(scan=scan, detectors=detector, ctf=ctf, lazy=False)

    assert measurement.shape == (3, 2, 2)


def test_c_prism_downsampled_gpts_independent_of_interpolation():
    potential = _small_potential()

    downsampled_gpts = set()
    for interpolation in [(1, 1), (2, 2), (2, 4), (4, 4)]:
        c_prism = CPRISM(
            potential=potential,
            energy=100e3,
            semiangle_cutoff=20,
            interpolation=interpolation,
        )
        downsampled_gpts.add(c_prism.downsampled_gpts)

    assert len(downsampled_gpts) == 1


def test_c_prism_aberrated_ctf_matches_probe():
    # the azimuthal angle convention of the reduction coefficients must match
    # polar_spatial_frequencies, ie. arctan2(ky, kx); the real-space intensity
    # of an aberrated probe is sensitive to the convention
    ctf = abtem.CTF(
        energy=100e3,
        semiangle_cutoff=20,
        defocus=50,
        astigmatism=40,
        astigmatism_angle=0.5236,
        coma=3e3,
        coma_angle=1.0,
    )

    c_prism = CPRISM(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        tolerance=1e-6,
    )
    c_prism_array = c_prism.build(lazy=False)

    probe = Probe._from_ctf(extent=20, gpts=c_prism_array.gpts, ctf=ctf, energy=100e3)
    probe_intensity = probe.build(lazy=False).intensity().array

    c_prism_intensity = c_prism_array.reduce(ctf=ctf).intensity().array

    assert np.allclose(
        np.squeeze(c_prism_intensity),
        np.squeeze(probe_intensity),
        atol=1e-3 * probe_intensity.max(),
    )


def test_c_prism_identical_to_prism_without_interpolation():
    # at an interpolation factor of (1, 1) the plane-wave expansion is complete
    # and no compression is performed, hence C-PRISM is identical to PRISM
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    prism_measurement = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1
    ).scan(scan=scan, detectors=detector, lazy=False)

    c_prism_measurement = CPRISM(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1
    ).scan(scan=scan, detectors=detector, lazy=False)

    assert np.allclose(c_prism_measurement.array, prism_measurement.array)
