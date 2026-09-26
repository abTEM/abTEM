"""A zero semiangle cutoff is a parallel beam (a plane wave).

The probe itself is well defined at a zero cutoff, and so is anything given explicit
parameters. Quantities abTEM would otherwise derive from the cutoff -- the Nyquist
and default scan sampling, the direct-beam radius, the default angular range of a
CTF, PRISM's plane-wave expansion -- are not, and must raise a ValueError that says
what to pass instead.
"""

import ase.build
import numpy as np
import pytest

import abtem
from abtem.core.energy import energy2wavelength
from abtem.prism.s_matrix import SMatrix
from abtem.transfer import CTF, Aperture, nyquist_sampling

ENERGY = 60e3


@pytest.fixture(scope="module")
def potential():
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    return abtem.Potential(atoms, sampling=0.1, slice_thickness=2)


def _haadf(probe, potential, scan):
    detector = abtem.AnnularDetector(70, 220)
    measurement = probe.scan(potential, scan=scan, detectors=detector)
    return measurement.compute().array


def test_nyquist_sampling_of_a_parallel_beam_raises():
    with pytest.raises(ValueError, match="semiangle_cutoff=0"):
        nyquist_sampling(0.0, ENERGY)

    with pytest.raises(ValueError, match="semiangle_cutoff=0"):
        Aperture(semiangle_cutoff=0.0, energy=ENERGY).nyquist_sampling


def test_nyquist_sampling_of_a_positive_cutoff_is_unchanged():
    wavelength = energy2wavelength(ENERGY)
    assert nyquist_sampling(20.0, ENERGY) == 1 / (4 * 20.0 / wavelength * 1e-3)
    np.testing.assert_array_equal(
        nyquist_sampling(np.array([10.0, 20.0]), ENERGY),
        [nyquist_sampling(10.0, ENERGY), nyquist_sampling(20.0, ENERGY)],
    )


@pytest.mark.parametrize(
    "make_scan",
    [
        pytest.param(lambda: abtem.GridScan(), id="GridScan"),
        pytest.param(lambda: abtem.LineScan(start=(0, 0), end=(3, 4)), id="LineScan"),
    ],
)
def test_default_scan_of_a_parallel_beam_raises(potential, make_scan):
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=0)

    with pytest.raises(ValueError, match="explicit `sampling` or `gpts`"):
        _haadf(probe, potential, make_scan())


def test_default_scan_of_an_all_zero_cutoff_distribution_raises(potential):
    cutoffs = abtem.distributions.from_values([0.0, 0.0])
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=cutoffs)

    with pytest.raises(ValueError, match="explicit `sampling` or `gpts`"):
        _haadf(probe, potential, abtem.GridScan())


def test_explicit_scan_of_a_parallel_beam_works(potential):
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=0)

    one_position = _haadf(probe, potential, abtem.GridScan(gpts=1))
    grid = _haadf(probe, potential, abtem.GridScan(sampling=1.0))

    # a plane wave is translation invariant
    assert one_position.shape == (1, 1)
    assert grid.size > 1
    scale = np.abs(grid).max()
    np.testing.assert_allclose(grid, grid.flat[0], rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(one_position.ravel(), grid.flat[0], atol=1e-6 * scale)


def test_default_scan_with_a_positive_cutoff_is_unchanged(potential):
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
    probe.grid.match(potential)
    scan = abtem.GridScan()
    scan.match_probe(probe)

    expected = 0.99 * nyquist_sampling(20, ENERGY)
    extent = np.array(potential.extent)
    assert scan.gpts == tuple(int(n) for n in np.ceil(extent / expected))


@pytest.mark.parametrize("semiangle_cutoff", [0, 0.0, -5.0])
def test_prism_rejects_a_non_positive_cutoff(potential, semiangle_cutoff):
    with pytest.raises(ValueError, match="positive 'semiangle_cutoff'"):
        SMatrix(potential=potential, energy=ENERGY, semiangle_cutoff=semiangle_cutoff)


@pytest.fixture(scope="module")
def parallel_beam_patterns():
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    potential = abtem.Potential(atoms, sampling=(0.1, 0.07), slice_thickness=2)
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=0)
    return probe.multislice(potential).diffraction_patterns().compute()


def test_block_direct_of_a_parallel_beam_needs_an_explicit_radius(
    parallel_beam_patterns,
):
    assert parallel_beam_patterns.metadata["semiangle_cutoff"] == 0

    with pytest.raises(ValueError, match="Pass `radius` explicitly"):
        parallel_beam_patterns.block_direct()


def test_block_direct_remedy_blocks_only_the_zero_angle_pixel(parallel_beam_patterns):
    patterns = parallel_beam_patterns

    blocked = patterns.block_direct(radius=0, margin=False)

    changed = np.argwhere(blocked.array != patterns.array)
    center = tuple(n // 2 for n in patterns.shape[-2:])
    assert [tuple(index) for index in changed] == [center]


def test_block_direct_with_a_positive_cutoff_keeps_the_margin(potential):
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20)
    patterns = probe.multislice(potential).diffraction_patterns().compute()

    expected = patterns.bandlimit(20 + max(patterns.angular_sampling), outer=np.inf)

    np.testing.assert_array_equal(patterns.block_direct().array, expected.array)


def test_ctf_default_angular_range_of_a_parallel_beam_raises():
    ctf = CTF(energy=ENERGY, semiangle_cutoff=0, defocus=50)

    with pytest.raises(ValueError, match="Pass `max_angle` explicitly"):
        ctf.profiles()

    with pytest.raises(ValueError, match="Pass `max_angle` explicitly"):
        ctf.to_diffraction_patterns()

    # the remedy works
    assert ctf.profiles(max_angle=30).shape[-1] > 1
    assert ctf.to_diffraction_patterns(max_angle=30).shape == (128, 128)


def test_ctf_to_diffraction_patterns_without_an_aperture():
    """An infinite cutoff (the CTF default) takes the 50 mrad range of CTF.profiles."""
    ctf = CTF(energy=ENERGY, defocus=50)

    patterns = ctf.to_diffraction_patterns()

    reference = ctf.to_diffraction_patterns(max_angle=50)
    np.testing.assert_allclose(patterns.sampling, reference.sampling)
    np.testing.assert_array_equal(patterns.array, reference.array)
