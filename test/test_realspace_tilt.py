"""RealSpaceMultislice applies the tilt of the wave function as the Fourier
propagator does at orders 1 and 2: the beams are translated by tan(tilt) per unit
depth (issue #627)."""

import ase.build
import numpy as np
import pytest
from utils import synthetic_transition_potential, to_host_array

import abtem
from abtem.multislice import (
    FourierMultislice,
    RealSpaceMultislice,
    _translate_tilted_waves,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def silicon_potential(slice_thickness):
    return abtem.Potential(
        ase.build.bulk("Si", cubic=True),
        gpts=(32, 32),
        slice_thickness=slice_thickness,
        projection="finite",
    )


def exit_wave(algorithm, potential, **kwargs):
    waves = abtem.PlaneWave(energy=100e3, **kwargs).multislice(
        potential, algorithm=algorithm, lazy=False
    )
    return to_host_array(waves)


def test_translate_tilted_waves_shifts_a_bump_by_tan_tilt_per_depth():
    # Oracle: a Gaussian moves by tan(50 mrad) * 20 Å = 1.0008 Å along x, and
    # does not move along y.
    n, extent = 128, 20.0
    x = np.arange(n) * extent / n
    gaussian = np.exp(-((x - 8.0) ** 2) / (2 * 0.8**2))
    array = (gaussian[:, None] * gaussian[None]).astype(np.complex64)
    waves = abtem.Waves(
        array,
        extent=extent,
        energy=100e3,
        metadata={"base_tilt_x": 50.0, "base_tilt_y": 0.0},
    )

    def centre(a):
        return (np.abs(a) ** 2 * x[:, None]).sum() / (np.abs(a) ** 2).sum()

    shifted = to_host_array(_translate_tilted_waves(waves.copy(), 20.0))
    assert centre(shifted) - centre(array) == pytest.approx(
        np.tan(0.05) * 20.0, abs=5e-3
    )
    # along y the centre stays

    def centre_y(a):
        return (np.abs(a) ** 2 * x[None]).sum() / (np.abs(a) ** 2).sum()

    assert centre_y(shifted) == pytest.approx(centre_y(array), abs=1e-3)


def test_translate_untilted_waves_is_the_identity():
    waves = abtem.PlaneWave(energy=100e3, extent=10.0, gpts=32).build(lazy=False)
    before = to_host_array(waves).copy()
    np.testing.assert_array_equal(
        to_host_array(_translate_tilted_waves(waves, 5.0)), before
    )


@pytest.mark.parametrize("scope", ["propagator", "full"])
@pytest.mark.parametrize("tilt", [(5.0, 0.0), (20.0, 10.0)])
def test_the_tilt_changes_the_exit_wave_as_it_does_for_the_fourier_propagator(
    scope, tilt
):
    # Oracle: the Fourier propagator of the same order. The two algorithms differ
    # without tilt (stencil, and the order in which the slice is split); the change
    # that the tilt makes to that difference is within 10 % of the effect of the
    # tilt, and the tilt has an effect that is far above that.
    potential = silicon_potential(0.25)
    real_space = RealSpaceMultislice(order=1, expansion_scope=scope)
    fourier = FourierMultislice(order=1)

    effect = exit_wave(fourier, potential, tilt=tilt) - exit_wave(fourier, potential)
    real_space_effect = exit_wave(real_space, potential, tilt=tilt) - exit_wave(
        real_space, potential
    )
    assert np.abs(effect).max() > 1e-2
    assert np.abs(real_space_effect).max() > 1e-2
    assert np.abs(real_space_effect - effect).max() < 0.1 * np.abs(effect).max()


def test_a_tilt_ensemble_is_the_tilts_one_by_one():
    # Oracle: each member of a tilt ensemble is the exit wave of that tilt alone.
    potential = silicon_potential(0.5)
    algorithm = RealSpaceMultislice(order=1)
    tilts = [0.0, 10.0, 20.0]

    ensemble = exit_wave(algorithm, potential, tilt=(tilts, 0.0))
    assert ensemble.shape[0] == len(tilts)
    for member, tilt in zip(ensemble, tilts):
        np.testing.assert_allclose(
            member,
            exit_wave(algorithm, potential, tilt=(tilt, 0.0)),
            atol=1e-5,
        )


def test_the_core_loss_multislice_applies_the_tilt():
    # Oracle: the relative change of the core-loss signal with the tilt of the probe
    # is that of the Fourier propagator.
    atoms = list(
        abtem.FrozenPhonons(ase.build.bulk("Si", cubic=True), 1, sigmas=0.1, seed=1)
    )[0]
    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=atoms.cell[2, 2] / 3
    )

    def signal(algorithm, tilt):
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, tilt=tilt)
        probe.grid.match(potential)
        measurement = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=synthetic_transition_potential(
                gpts=potential.gpts, extent=potential.extent, n_transitions=2
            ),
            scan=np.array([[0.0, 0.0]]),
            detectors=abtem.AnnularDetector(inner=0.0, outer=None),
            lazy=False,
            sites=atoms,
            algorithm=algorithm,
        )
        return float(np.asarray(measurement.array).ravel()[0])

    change = {}
    for name, algorithm in (
        ("fourier", FourierMultislice(order=1)),
        ("real space", RealSpaceMultislice(order=1)),
    ):
        change[name] = (
            signal(algorithm, (30.0, 0.0)) / signal(algorithm, (0.0, 0.0)) - 1
        )

    assert abs(change["fourier"]) > 5e-3
    assert change["real space"] == pytest.approx(change["fourier"], rel=0.2)
