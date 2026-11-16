import warnings

import ase
import ase.build
import numpy as np
import pytest
from scipy.linalg import sqrtm
from scipy.special import binom
from utils import float64_devices, to_host_array

import abtem
from abtem.antialias import antialias_aperture
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.finite_difference import LaplaceOperator, _exact_root_step, multislice_step
from abtem.multislice import LanczosMultislice, RealSpaceMultislice

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning",
    # the series it is compared with is deprecated
    "ignore:RealSpaceMultislice\\(expansion_scope='full'\\):FutureWarning",
)


def _sto_potential(**kwargs):
    atoms = ase.Atoms(
        symbols="SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        cell=[3.9127, 3.9127, 3.9127],
        pbc=True,
    ) * (1, 1, 2)
    return abtem.Potential(
        atoms, gpts=(40, 40), slice_thickness=0.75, projection="finite", **kwargs
    )


def test_the_forward_wave_of_the_full_scope_is_corrected_by_eq_10_of_the_micron_paper():
    # Oracle: for one slice with a next slice, the forward wave is the plain series
    # (the step without a next slice, which has no backscattered wave) minus the
    # backscattered wave (Eq. 10), which is computed from the same stepped wave.
    slices = _sto_potential().build(lazy=False)
    waves = abtem.PlaneWave(energy=30e3).match_grid(slices).build(lazy=False)

    def step(next_slice):
        result = multislice_step(
            waves.copy(),
            slices[0],
            next_slice,
            LaplaceOperator(6),
            order=3,
            fully_corrected=True,
        )
        return tuple(to_host_array(w) for w in result)

    plain = step(None)[0]
    forward, backscattered = step(slices[1])

    assert np.abs(backscattered).max() > 1e-3
    # the step returns the backscattered wave band-limited but subtracts it unfiltered
    mask = antialias_aperture(waves._valid_gpts, waves._valid_sampling, np)
    subtracted = np.fft.ifft2(np.fft.fft2(plain - forward) * mask)
    np.testing.assert_allclose(subtracted, backscattered, atol=1e-6)


def test_requesting_the_backscattered_wave_leaves_the_forward_wave_unchanged():
    # Oracle: the correction is always applied, so the forward wave is the same
    # whether or not the backscattered wave is returned too.
    potential = _sto_potential(exit_planes=1)
    algorithm = LanczosMultislice()
    forward_only = abtem.PlaneWave(energy=30e3).multislice(
        potential, algorithm=algorithm, lazy=False
    )
    transmitted, reflected = abtem.PlaneWave(energy=30e3).multislice(
        potential,
        algorithm=algorithm,
        detectors=[abtem.WavesDetector(), abtem.WavesDetector(reflected=True)],
        lazy=False,
    )
    assert np.abs(to_host_array(reflected)).max() > 0
    np.testing.assert_array_equal(
        to_host_array(transmitted), to_host_array(forward_only)
    )


# The exact two-way solution (TwoWaySolver: Numerov in steps of 0.25 / k, 4.5 1/Å of
# beams, evanescent ones included) for graphene (48 x 84 points, 6 Å cell, finite
# projection, slices of 0.05 Å) at 2 keV: the intensity |psi_g|^2 of the Bragg beams in
# three bands of |g| (0.01-1.5, 1.5-3.0 and 3.0-4.4 1/Å) and in total.
EXACT_BAND_INTENSITY_2_KEV = np.array([0.05449581, 0.00692510, 0.00045739])
EXACT_TOTAL_INTENSITY_2_KEV = 1.0028191


def _graphene_band_intensities(array):
    nx, ny = array.shape
    intensity = np.abs(np.fft.fft2(array) / (nx * ny)) ** 2
    radius = np.hypot(
        np.fft.fftfreq(nx, 2.46 / nx)[:, None], np.fft.fftfreq(ny, 4.2608 / ny)[None]
    )
    edges = (0.01, 1.5, 3.0, 4.4)
    return np.array(
        [
            intensity[(radius >= low) & (radius < high)].sum()
            for low, high in zip(edges[:-1], edges[1:])
        ]
    )


@pytest.mark.slow
def test_the_correction_brings_the_wide_angle_beams_to_the_exact_solution():
    # Oracle: the exact two-way solution above. At 2 keV the one-way root without the
    # correction (the step without a next slice) puts 19 % too little intensity in the
    # beams at 1.5-3 1/Å and 65 % too little beyond; with the correction of Eq. 10 they
    # are 1 % and 10 % off, and the total intensity is that of the exact wave (the
    # exact wave has a total above 1, since what it conserves is the flux).
    atoms = abtem.orthogonalize_cell(ase.build.graphene(a=2.46, vacuum=1.0))
    atoms.center(vacuum=3.0, axis=2)
    potential = abtem.Potential(
        atoms, gpts=(48, 84), slice_thickness=0.05, projection="finite"
    ).build(lazy=False)
    wave = abtem.PlaneWave(energy=2000.0)

    corrected = to_host_array(
        wave.multislice(
            potential,
            algorithm=LanczosMultislice(),
            lazy=False,
        )
    )
    plain = wave.match_grid(potential).build(lazy=False)
    for index in range(len(potential)):
        plain = _exact_root_step(plain, potential[index], None, 0.95, 1e-8, 60)[0]
    plain = to_host_array(plain)

    error = {
        name: np.abs(_graphene_band_intensities(array) / EXACT_BAND_INTENSITY_2_KEV - 1)
        for name, array in (("plain", plain), ("corrected", corrected))
    }
    assert error["plain"][1] > 0.15 and error["plain"][2] > 0.5
    assert error["corrected"][0] < 0.01
    assert error["corrected"][1] < 0.03
    assert error["corrected"][2] < 0.15
    total = {
        name: float((np.abs(array) ** 2).mean())
        for name, array in (("plain", plain), ("corrected", corrected))
    }
    assert total["corrected"] == pytest.approx(EXACT_TOTAL_INTENSITY_2_KEV, abs=5e-4)
    assert abs(total["plain"] - EXACT_TOTAL_INTENSITY_2_KEV) > 2e-3


def test_the_full_scope_warns_about_the_infinite_projection():
    # Oracle: the infinite projection (the default of Potential) puts the whole
    # projected potential of an atom in one slice, so the potential per unit
    # thickness that the full scope needs is not defined; the propagator scope uses
    # the potential to first order only and does not mind.
    atoms = ase.Atoms("Si", cell=[3.0, 3.0, 3.0], pbc=True)
    potential = abtem.Potential(atoms, gpts=(16, 16), slice_thickness=1.0)
    plane_wave = abtem.PlaneWave(energy=30e3)

    with pytest.warns(UserWarning, match="projection='finite'"):
        plane_wave.multislice(
            potential,
            algorithm=LanczosMultislice(),
            lazy=False,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plane_wave.multislice(potential, algorithm=RealSpaceMultislice(), lazy=False)


def _series_step(order):
    def step(waves, potential_slice, next_slice):
        return multislice_step(
            waves,
            potential_slice,
            next_slice,
            LaplaceOperator(6),
            order=order,
            fully_corrected=True,
        )

    return step


def _algorithm(name):
    if name == "lanczos":
        return LanczosMultislice()
    return RealSpaceMultislice(order=1, expansion_scope="full")


def _reflected_and_transmitted(potential, algorithm, amplitude=1.0):
    """The wave reflected at the entrance and the transmitted waves at every exit
    plane, of a plane wave of the given amplitude at 1 keV."""
    waves = abtem.PlaneWave(energy=1000.0).match_grid(potential).build(lazy=False)
    waves._array = waves._array * amplitude
    transmitted, reflected = waves.multislice(
        potential,
        algorithm=algorithm,
        detectors=[abtem.WavesDetector(), abtem.WavesDetector(reflected=True)],
    )
    return to_host_array(reflected), to_host_array(transmitted)


@float64_devices
@pytest.mark.parametrize("order", [1, 2])
def test_the_full_expansion_is_linear_in_the_incident_wave(order, device):
    # Oracle: the Schrödinger equation is linear, so doubling the incident wave
    # doubles the reflected wave and the transmitted waves at every exit plane. The
    # potential is laterally structured, so the backscatter is not a constant.
    array = np.random.default_rng(0).random((4, 16, 12)).astype(np.float32)
    potential = abtem.PotentialArray(
        array, slice_thickness=0.1, extent=(4.0, 3.0), exit_planes=1
    )
    algorithm = RealSpaceMultislice(order=order, expansion_scope="full")

    once = _reflected_and_transmitted(potential, algorithm, 1.0)
    twice = _reflected_and_transmitted(potential, algorithm, 2.0)

    for name, single, double in zip(("reflected", "transmitted"), once, twice):
        assert np.abs(single).max() > 1e-4, name
        deviation = np.linalg.norm(double - 2 * single) / np.linalg.norm(2 * single)
        assert deviation < 1e-12, name


@float64_devices
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("thickness", [0.01, 0.05])
@pytest.mark.parametrize("amplitude", [1.0, 1e-3])
def test_the_backscatter_of_a_uniform_step_is_the_series_of_the_wave_numbers(
    order, thickness, amplitude, device
):
    # Oracle: for a constant wave across a step from 0 to U, k_next - k_current acts
    # as K0 sum_{i=1..order} binom(1/2, i) x^i, and 1 / k_next as the series
    # sum_{i=0..order} binom(-1/2, i) x^i / K0 (Eq. 13 of the Micron paper), with
    # x = sigma U / (pi K0): the backscatter over the wave is
    # (k_next - k_current) / (2 k_next) of the same truncated series.
    energy, volts = 1000.0, 10.0
    k0, sigma = 1 / energy2wavelength(energy), energy2sigma(energy)
    x = sigma * volts / (np.pi * k0)
    difference = sum(binom(0.5, i) * x**i for i in range(1, order + 1))
    inverse = sum(binom(-0.5, i) * x**i for i in range(order + 1))
    expected = difference * inverse / 2

    array = np.zeros((2, 8, 8), dtype=np.float32)
    array[1] = volts * thickness
    slices = abtem.PotentialArray(array, slice_thickness=thickness, extent=(4.0, 4.0))
    waves = abtem.PlaneWave(energy=energy).match_grid(slices).build(lazy=False)
    waves._array = waves._array * amplitude

    _, backscattered = _series_step(order)(waves, slices[0], slices[1])

    ratio = to_host_array(backscattered) / amplitude
    np.testing.assert_allclose(ratio, expected, rtol=1e-6, atol=0)


@float64_devices
@pytest.mark.parametrize("order", [4, 6])
def test_the_backscatter_of_a_structured_wave_is_the_exact_operator_of_eq_7(
    order, device
):
    # Oracle: B = (2 k_next)^-1 (k_next - k_current), with the wave-number operators
    # k = K0 sqrt(1 + (laplacian / (4 pi K0) + sigma V) / (pi K0)) as dense matrices
    # of the stencil of the step on a 12 x 12 grid, and their matrix square roots.
    # 1 / k_next acts to the left of the difference, which differs from acting to
    # its right for a structured wave and structured potentials. The first slice is
    # so thin that the wave is the same after it. The comparison is at 30 keV, where
    # the series of order 4 is converged to 2e-8 of the backscatter.
    n, sampling, energy = 12, 0.2, 30e3
    rng = np.random.default_rng(1)
    q2 = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None] ** 2

    def structured_potential():  # smooth, 0 to 20 V
        v = np.fft.ifft2(np.fft.fft2(rng.random((n, n))) * np.exp(-60 * q2)).real
        return 20 * (v - v.min()) / np.ptp(v)

    volts = [structured_potential(), structured_potential()]
    thicknesses = [1e-9, 0.1]
    slices = abtem.PotentialArray(
        np.stack([v * t for v, t in zip(volts, thicknesses)]),
        slice_thickness=thicknesses,
        sampling=sampling,
    )
    waves = abtem.PlaneWave(energy=energy).match_grid(slices).build(lazy=False)
    xp = get_array_module(waves.array)
    wave = rng.random((n, n)) * np.exp(2j * np.pi * rng.random((n, n)))
    waves._array = xp.asarray(wave.reshape(waves.array.shape))

    stencil = LaplaceOperator(6).get_stencil(waves, device=waves.device)
    basis = np.eye(n * n, dtype=complex).reshape(n * n, n, n)
    laplacian = asnumpy(stencil(xp.asarray(basis))).reshape(n * n, n * n).T
    k0, sigma = 1 / energy2wavelength(energy), energy2sigma(energy)

    def wave_number(v):
        a = laplacian / (4 * np.pi * k0) + np.diag(sigma * v.ravel())
        return k0 * sqrtm(np.eye(n * n) + a / (np.pi * k0))

    k_current, k_next = (wave_number(v) for v in volts)
    mask = antialias_aperture(waves._valid_gpts, waves._valid_sampling, np)

    def bandlimit(a):
        return np.fft.ifft2(np.fft.fft2(a.reshape(n, n)) * mask).ravel()

    expected = bandlimit(
        np.linalg.solve(2 * k_next, (k_next - k_current) @ bandlimit(wave))
    )

    _, backscattered = _series_step(order)(waves, slices[0], slices[1])

    np.testing.assert_allclose(
        to_host_array(backscattered).ravel(),
        expected,
        rtol=0,
        atol=1e-6 * np.abs(expected).max(),
    )


@float64_devices
@pytest.mark.parametrize("algorithm", ["series", "lanczos"])
def test_a_uniform_medium_sliced_unevenly_reflects_nothing(algorithm, device):
    # Oracle: a laterally uniform medium of 10 V with no vacuum has no interface,
    # however it is sliced, so it reflects nothing (|r|^2 < 1e-20).
    thicknesses = np.array([0.01, 0.02, 0.03, 0.01, 0.02] * 6)
    potential = abtem.PotentialArray(
        np.ones((len(thicknesses), 8, 8)) * (10.0 * thicknesses)[:, None, None],
        slice_thickness=list(thicknesses),
        sampling=0.5,
    )
    waves = abtem.PlaneWave(energy=1000.0).match_grid(potential).build(lazy=False)
    reflected = waves.multislice(
        potential,
        algorithm=_algorithm(algorithm),
        detectors=abtem.WavesDetector(reflected=True),
    )

    assert np.abs(to_host_array(reflected.array)).max() ** 2 < 1e-20


@float64_devices
@pytest.mark.parametrize("algorithm", ["series", "lanczos"])
def test_the_reflection_of_a_slab_does_not_depend_on_how_unevenly_it_is_sliced(
    algorithm, device
):
    # Oracle: the reflection of a slab of potential V0 and thickness d, in vacuum, is
    # r12 (1 - exp(2 i k2 d)) / (1 - r12^2 exp(2 i k2 d)), with k2 the wave number in
    # the slab (Airy), whatever the slicing. The slab is sliced in 0.01-0.03 Å.
    energy, v0 = 1000.0, 10.0
    thickness = np.array([0.01, 0.02, 0.03, 0.01, 0.02] * 6)
    k1 = 2 * np.pi / energy2wavelength(energy)
    k2 = np.sqrt(k1**2 + 2 * k1 * energy2sigma(energy) * v0)
    r12 = (k1 - k2) / (k1 + k2)
    phase = np.exp(2j * k2 * thickness.sum())
    expected = abs(r12 * (1 - phase) / (1 - r12**2 * phase)) ** 2

    thicknesses = np.r_[np.full(5, 0.02), thickness, np.full(5, 0.02)]
    volts = np.r_[np.zeros(5), np.full(len(thickness), v0), np.zeros(5)]
    potential = abtem.PotentialArray(
        np.ones((len(volts), 8, 8), np.float32)
        * (volts * thicknesses).astype(np.float32)[:, None, None],
        slice_thickness=list(thicknesses),
        sampling=0.5,
    )
    waves = abtem.PlaneWave(energy=energy).match_grid(potential).build(lazy=False)
    reflected = waves.multislice(
        potential,
        algorithm=_algorithm(algorithm),
        detectors=abtem.WavesDetector(reflected=True),
    )

    assert abs(to_host_array(reflected.array).mean()) ** 2 == pytest.approx(
        expected, rel=0.01
    )
