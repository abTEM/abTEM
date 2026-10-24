import warnings

import ase
import ase.build
import numpy as np
import pytest
from utils import to_host_array

import abtem
from abtem.antialias import antialias_aperture
from abtem.finite_difference import LaplaceOperator, multislice_step
from abtem.multislice import RealSpaceMultislice

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


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
    algorithm = RealSpaceMultislice(order=3, expansion_scope="full")
    forward_only = abtem.PlaneWave(energy=30e3).multislice(
        potential, algorithm=algorithm, lazy=False
    )
    transmitted, backscattered = abtem.PlaneWave(energy=30e3).multislice(
        potential, algorithm=algorithm, return_backscattered=True, lazy=False
    )
    assert np.abs(to_host_array(backscattered)[0]).max() > 0
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
            algorithm=RealSpaceMultislice(order="exact", expansion_scope="full"),
            lazy=False,
        )
    )
    plain = wave.match_grid(potential).build(lazy=False)
    for index in range(len(potential)):
        plain = multislice_step(
            plain,
            potential[index],
            None,
            LaplaceOperator(6),
            order="exact",
            fully_corrected=True,
        )[0]
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
            algorithm=RealSpaceMultislice(expansion_scope="full"),
            lazy=False,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plane_wave.multislice(potential, algorithm=RealSpaceMultislice(), lazy=False)
