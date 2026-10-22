"""RealSpaceMultislice(order="exact"): the one-way root of the full operator by a
Lanczos iteration, instead of a Taylor series."""

import ase
import ase.build
import numpy as np
import pytest
from utils import devices, to_host_array

import abtem
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.finite_difference import LaplaceOperator, multislice_step
from abtem.multislice import RealSpaceMultislice

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

# Strong beams of a graphene monolayer, as the flux that the exact two-way solution
# (Numerov, 0.01 Å steps, propagating beams only) puts in the three families of the
# hexagonal lattice at 0.469, 0.813 and 0.939 1/Å, at 1 keV.
EXACT_FAMILY_FLUX_1_KEV = np.array([0.044630, 0.039530, 0.006040])


def graphene_potential(slice_thickness=0.05, device="cpu"):
    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=6.0, axis=2)
    return abtem.Potential(
        atoms,
        gpts=(48, 84),
        slice_thickness=slice_thickness,
        projection="finite",
        periodic=True,
        device=device,
    )


def family_intensities(array):
    nx, ny = array.shape
    intensity = np.abs(np.fft.fft2(array) / (nx * ny)) ** 2
    fx = np.fft.fftfreq(nx, 2.46 / nx)[:, None]
    fy = np.fft.fftfreq(ny, 4.2608 / ny)[None]
    radius = np.hypot(fx, fy)
    return np.array(
        [intensity[np.abs(radius - f) < 0.01].sum() for f in (0.469, 0.813, 0.939)]
    )


def exact_algorithm(**kwargs):
    return RealSpaceMultislice(order="exact", expansion_scope="full", **kwargs)


def test_the_exact_root_is_close_to_the_exact_solution_where_the_series_is_not():
    # Oracle: the exact two-way solution (stored above). At 1 keV the split step is 8 %
    # off and the series of order 1 is 7 % off; the exact root is within 2 %.
    potential = graphene_potential()
    waves = abtem.PlaneWave(energy=1000.0).multislice(
        potential, algorithm=exact_algorithm(), lazy=False
    )
    flux = family_intensities(to_host_array(waves))
    assert np.abs(flux / EXACT_FAMILY_FLUX_1_KEV - 1).max() < 0.02


@devices
@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_the_exact_root_conserves_intensity(energy, device):
    # Oracle: the step is unitary on the band, up to the rounding of 120 slices.
    waves = abtem.PlaneWave(energy=energy, device=device).multislice(
        graphene_potential(0.1, device), algorithm=exact_algorithm(), lazy=False
    )
    array = to_host_array(waves)
    assert float((np.abs(array) ** 2).mean()) == pytest.approx(1.0, abs=2e-4)


def test_the_backscattered_wave_of_flat_slices_is_the_difference_of_wave_numbers():
    # Oracle: for two flat slices and a plane wave the band is not touched, and
    # B = (k_next - k_current) / (2 k_next) with k = K0 sqrt(1 + sigma U / (pi K0)).
    energy, thickness = 1000.0, 0.5
    wavelength = energy2wavelength(energy)
    k0, sigma = 1 / wavelength, energy2sigma(energy)
    potentials = (0.0, 2.0)  # volts
    array = np.stack([np.full((32, 32), u * thickness) for u in potentials]).astype(
        np.float32
    )
    slices = abtem.PotentialArray(array, slice_thickness=thickness, extent=(8.0, 8.0))
    waves = abtem.PlaneWave(energy=energy).match_grid(slices).build(lazy=False)

    forward, backscattered = multislice_step(
        waves.copy(),
        slices[0],
        slices[1],
        LaplaceOperator(6),
        order="exact",
        fully_corrected=True,
        correct_forward=False,
    )

    def wave_number(u):
        return k0 * np.sqrt(1 + sigma * u / (np.pi * k0))

    expected = (wave_number(potentials[1]) - wave_number(potentials[0])) / (
        2 * wave_number(potentials[1])
    )
    ratio = to_host_array(backscattered).mean() / to_host_array(forward).mean()
    assert ratio.real == pytest.approx(expected, rel=1e-4)
    assert abs(ratio.imag) < 1e-5


def test_the_correction_is_subtracted_from_the_forward_wave_as_for_the_series():
    # Oracle: with the correction the forward wave is the plain one minus the
    # backscattered wave, which does not depend on the switch.
    potential = graphene_potential(0.1)
    slices = potential.build(lazy=False)
    waves = abtem.PlaneWave(energy=1000.0).match_grid(slices).build(lazy=False)
    # the slice before the atomic plane, whose potential differs from the next one
    first = int(np.argmax(np.abs(np.asarray(slices.array)).max(axis=(1, 2)))) - 1

    def step(correction):
        return tuple(
            to_host_array(w)
            for w in multislice_step(
                waves.copy(),
                slices[first],
                slices[first + 1],
                LaplaceOperator(6),
                order="exact",
                fully_corrected=True,
                correct_forward=correction,
            )
        )

    forward_off, backscattered_off = step(False)
    forward_on, backscattered_on = step(True)
    assert np.abs(backscattered_off).max() > 1e-6
    np.testing.assert_allclose(backscattered_on, backscattered_off, atol=1e-7)
    np.testing.assert_allclose(forward_off - forward_on, backscattered_off, atol=1e-6)


def test_the_exact_root_agrees_with_the_converged_series_at_high_energy():
    # Oracle: where 2 sigma V / k is far below 1 the Taylor series converges to the
    # root: at 100 keV the series of order 3 and the exact root give the same wave
    # to 0.3 % (the series is truncated, and the stencil of the exact root is the same).
    potential = graphene_potential(0.1)
    wave = {}
    for name, algorithm in (
        ("exact", exact_algorithm()),
        ("series", RealSpaceMultislice(order=3, expansion_scope="full")),
    ):
        wave[name] = to_host_array(
            abtem.PlaneWave(energy=100e3).multislice(
                potential, algorithm=algorithm, lazy=False
            )
        )
    assert np.abs(wave["exact"] - wave["series"]).max() < 6e-3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": "exact"},  # the propagator scope
        {"order": 0},
        {"order": "second", "expansion_scope": "full"},
        {"order": "exact", "expansion_scope": "full", "propagating_cutoff": 1.0},
        {"order": "exact", "expansion_scope": "full", "lanczos_tolerance": 0.0},
        {"order": "exact", "expansion_scope": "full", "max_iterations": 1},
    ],
)
def test_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        RealSpaceMultislice(**kwargs)


# The strong beams of a graphene monolayer for a wave tilted by (80, 30) mrad, at 1 keV:
# the flux that the exact two-way solution (Numerov, 0.01 Å steps, propagating beams
# only; the beam g of a wave stored without the carrier of its direction is the plane
# wave g + k_t) puts in the beam at the grid index (i, j) of the 48 x 84 grid.
TILTED_EXACT_FLUX = {
    (0, 0): 0.8911565,
    (0, 82): 0.0075034,
    (47, 1): 0.0074885,
    (0, 2): 0.0074791,
    (1, 83): 0.0074624,
    (1, 1): 0.0074409,
    (47, 83): 0.0074407,
    (47, 81): 0.0066886,
}


def test_the_exact_root_follows_the_exact_solution_for_a_tilted_wave():
    # Oracle: the exact two-way solution (stored above). The series of order 1 with
    # the paraxial translation of the tilt is 1-8 % off on these beams; the exact root
    # in the frame of the tilted wave is within 1.5 %.
    wave = abtem.PlaneWave(energy=1000.0, tilt=(80.0, 30.0))

    def errors(algorithm):
        array = to_host_array(
            wave.multislice(graphene_potential(), algorithm=algorithm, lazy=False)
        )
        intensity = np.abs(np.fft.fft2(array) / array.size) ** 2
        return np.array(
            [intensity[index] / flux - 1 for index, flux in TILTED_EXACT_FLUX.items()]
        )

    assert np.abs(errors(exact_algorithm())).max() < 0.015
    assert (
        np.abs(errors(RealSpaceMultislice(order=1, expansion_scope="full"))).max()
        > 0.05
    )


def test_an_ensemble_of_tilts_is_the_tilts_one_by_one_for_the_exact_root():
    # Oracle: each member is the exit wave of its tilt alone.
    potential = graphene_potential(0.1)
    tilts_x, tilts_y = [0.0, 80.0], [0.0, 30.0]
    ensemble = to_host_array(
        abtem.PlaneWave(energy=1000.0, tilt=(tilts_x, tilts_y)).multislice(
            potential, algorithm=exact_algorithm(), lazy=False
        )
    )
    assert ensemble.shape == (2, 2, 48, 84)
    for i, tx in enumerate(tilts_x):
        for j, ty in enumerate(tilts_y):
            single = abtem.PlaneWave(energy=1000.0, tilt=(tx, ty)).multislice(
                potential, algorithm=exact_algorithm(), lazy=False
            )
            np.testing.assert_allclose(ensemble[i, j], to_host_array(single), atol=2e-5)
