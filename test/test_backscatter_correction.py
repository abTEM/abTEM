import ase
import numpy as np
import pytest
from utils import to_host_array

import abtem
from abtem.antialias import antialias_aperture
from abtem.finite_difference import LaplaceOperator, multislice_step
from abtem.multislice import RealSpaceMultislice

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_backscatter_correction_is_off_by_default():
    assert RealSpaceMultislice().backscatter_correction is False
    assert RealSpaceMultislice(expansion_scope="full").backscatter_correction is False


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


def test_backscatter_correction_switch_is_eq_10_of_the_micron_paper():
    # Oracle: for one slice with a next slice, the corrected forward wave is the plain
    # series minus the backscattered wave (Eq. 10), and the backscattered wave itself
    # does not depend on the switch, since it is computed from the same stepped forward
    # wave.
    slices = _sto_potential().build(lazy=False)
    waves = abtem.PlaneWave(energy=30e3).match_grid(slices).build(lazy=False)

    def step(correction):
        return tuple(
            to_host_array(w)
            for w in multislice_step(
                waves.copy(),
                slices[0],
                slices[1],
                LaplaceOperator(6),
                order=3,
                fully_corrected=True,
                correct_forward=correction,
            )
        )

    forward_off, backscattered_off = step(False)
    forward_on, backscattered_on = step(True)

    assert np.abs(backscattered_off).max() > 1e-3
    np.testing.assert_allclose(backscattered_on, backscattered_off, atol=1e-7)
    # the step returns the backscattered wave band-limited but subtracts it unfiltered
    mask = antialias_aperture(waves._valid_gpts, waves._valid_sampling, np)
    subtracted = np.fft.ifft2(np.fft.fft2(forward_off - forward_on) * mask)
    np.testing.assert_allclose(subtracted, backscattered_off, atol=1e-6)


def test_forward_wave_is_independent_of_requesting_backscatter_when_correction_is_off():
    # Oracle: with the correction off the next slice is only used for the backscattered
    # wave, so asking for it must leave the forward wave bit-identical.
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
