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
# hexagonal lattice at 0.469, 0.813 and 0.939 1/Å, at 1 keV. The exit wave of a
# multislice algorithm is the wave function, whose beams have the intensity |psi_g|^2;
# the flux of a beam is that intensity times kz / k, so the exact intensity is the flux
# over kz / k.
EXACT_FAMILY_FLUX_1_KEV = np.array([0.044630, 0.039530, 0.006040])
EXACT_FAMILY_INTENSITY_1_KEV = EXACT_FAMILY_FLUX_1_KEV / np.sqrt(
    1 - (energy2wavelength(1000.0) * np.array([0.469, 0.813, 0.939])) ** 2
)


def graphene_potential(
    slice_thickness=0.05, device="cpu", vacuum=6.0, gpts=(48, 84)
):
    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=vacuum, axis=2)
    return abtem.Potential(
        atoms,
        gpts=gpts,
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


def plain_exit_wave(potential, energy, tilt=(0.0, 0.0)):
    """The exit wave of the exact root without the backscatter correction: every slice
    stepped without a next slice, which has no backscattered wave to subtract."""
    slices = potential.build(lazy=False)
    waves = (
        abtem.PlaneWave(energy=energy, tilt=tilt).match_grid(slices).build(lazy=False)
    )
    for index in range(len(slices)):
        waves = multislice_step(
            waves,
            slices[index],
            None,
            LaplaceOperator(6),
            order="exact",
            fully_corrected=True,
        )[0]
    return to_host_array(waves)


@pytest.mark.slow
def test_the_exact_root_is_close_to_the_exact_solution_where_the_series_is_not():
    # Oracle: the exact two-way solution (stored above). At 1 keV the series of order 1
    # is 7 % off; the exact root with the backscatter correction is within 1.4 %.
    waves = abtem.PlaneWave(energy=1000.0).multislice(
        graphene_potential(), algorithm=exact_algorithm(), lazy=False
    )
    corrected = family_intensities(to_host_array(waves))
    assert np.abs(corrected / EXACT_FAMILY_INTENSITY_1_KEV - 1).max() < 0.02


@devices
@pytest.mark.parametrize("energy", [200.0, 1000.0])
def test_the_exact_root_conserves_intensity(energy, device):
    # Oracle: the step is unitary on the band, up to the rounding of 120 slices (the
    # correction is not unitary: it adds what the exact wave has more intensity than
    # flux, 1 % at 1 keV).
    array = plain_exit_wave(graphene_potential(0.2, device, vacuum=3.0), energy)
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
    )

    def wave_number(u):
        return k0 * np.sqrt(1 + sigma * u / (np.pi * k0))

    expected = (wave_number(potentials[1]) - wave_number(potentials[0])) / (
        2 * wave_number(potentials[1])
    )
    # the forward wave has the backscattered wave subtracted: the plain one is their sum
    plain = to_host_array(forward).mean() + to_host_array(backscattered).mean()
    ratio = to_host_array(backscattered).mean() / plain
    assert ratio.real == pytest.approx(expected, rel=1e-4)
    assert abs(ratio.imag) < 1e-5


def test_the_correction_is_subtracted_from_the_forward_wave_as_for_the_series():
    # Oracle: the plain forward wave is the step without a next slice, which has no
    # backscattered wave; with a next slice the forward wave is the plain one minus the
    # backscattered wave.
    potential = graphene_potential(0.1)
    slices = potential.build(lazy=False)
    waves = abtem.PlaneWave(energy=1000.0).match_grid(slices).build(lazy=False)
    # the slice before the atomic plane, whose potential differs from the next one
    first = int(np.argmax(np.abs(np.asarray(slices.array)).max(axis=(1, 2)))) - 1

    def step(next_slice):
        return tuple(
            to_host_array(w)
            for w in multislice_step(
                waves.copy(),
                slices[first],
                next_slice,
                LaplaceOperator(6),
                order="exact",
                fully_corrected=True,
            )
        )

    plain = step(None)[0]
    forward, backscattered = step(slices[first + 1])
    assert np.abs(backscattered).max() > 1e-6
    np.testing.assert_allclose(plain - forward, backscattered, atol=1e-6)


def test_the_exact_root_agrees_with_the_converged_series_at_high_energy():
    # Oracle: where 2 sigma V / k is far below 1 the Taylor series converges to the
    # root: at 100 keV the series of order 3 and the exact root give the same wave
    # to 0.3 % (the series is truncated, and the stencil of the exact root is the same).
    potential = graphene_potential(0.1, vacuum=3.0)
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
# the flux (the intensity times kz / the kz of the incident beam) that the exact two-way
# solution (Numerov, 0.01 Å steps, propagating beams only; the beam g of a wave stored
# without the carrier of its direction is the plane wave g + k_t) puts in the beam at
# the grid index (i, j) of the 48 x 84 grid.
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


@pytest.mark.slow
def test_the_exact_root_follows_the_exact_solution_for_a_tilted_wave():
    # Oracle: the exact two-way solution (stored above, as intensities). The series of
    # order 1 with the paraxial translation of the tilt is 1-10 % off on these beams;
    # the exact root in the frame of the tilted wave is within 1 %.
    tilt = (80.0, 30.0)
    wave = abtem.PlaneWave(energy=1000.0, tilt=tilt)
    wavelength = energy2wavelength(1000.0)
    tangent = np.tan(np.asarray(tilt) / 1e3)
    norm = np.sqrt(1 + np.sum(tangent**2))
    k_t = tangent / (wavelength * norm)

    def exact_intensity(index, flux):
        qx = np.fft.fftfreq(48, 2.46 / 48)[index[0]] + k_t[0]
        qy = np.fft.fftfreq(84, 4.2608 / 84)[index[1]] + k_t[1]
        return flux * (1 / norm) / np.sqrt(1 - wavelength**2 * (qx**2 + qy**2))

    def errors(algorithm):
        array = to_host_array(
            wave.multislice(graphene_potential(), algorithm=algorithm, lazy=False)
        )
        intensity = np.abs(np.fft.fft2(array) / array.size) ** 2
        return np.array(
            [
                intensity[index] / exact_intensity(index, flux) - 1
                for index, flux in TILTED_EXACT_FLUX.items()
            ]
        )

    assert np.abs(errors(exact_algorithm())).max() < 0.012
    assert (
        np.abs(errors(RealSpaceMultislice(order=1, expansion_scope="full"))).max()
        > 0.05
    )


def test_an_ensemble_of_tilts_is_the_tilts_one_by_one_for_the_exact_root():
    # Oracle: each member is the exit wave of its tilt alone.
    potential = graphene_potential(0.2, vacuum=3.0, gpts=(24, 42))
    tilts_x = [0.0, 80.0]
    ensemble = to_host_array(
        abtem.PlaneWave(energy=1000.0, tilt=(tilts_x, 30.0)).multislice(
            potential, algorithm=exact_algorithm(), lazy=False
        )
    )
    assert ensemble.shape == (2, 24, 42)
    for i, tx in enumerate(tilts_x):
        single = abtem.PlaneWave(energy=1000.0, tilt=(tx, 30.0)).multislice(
            potential, algorithm=exact_algorithm(), lazy=False
        )
        np.testing.assert_allclose(ensemble[i], to_host_array(single), atol=2e-5)
