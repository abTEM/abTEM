"""Core-loss PRISM (SMatrix.transition_potential_scan) against core-loss multislice
(Probe.transition_potential_scan): output shapes, frames and metadata."""

import ase
import numpy as np
import pytest
from utils import devices, synthetic_transition_potential, to_host_array

import abtem

GPTS = (48, 56)
ENERGY = 100e3
ATOMS = ase.Atoms("B2", positions=[(2, 2, 1), (4, 5, 3)], cell=(8, 6, 8), pbc=True)

# float32 round-off of the PRISM reduction and of the multislice oracle
TOLERANCE = 1e-4


def _potential(device, num_configs=None, ensemble_mean=True):
    atoms = ATOMS
    if num_configs:
        atoms = abtem.FrozenPhonons(
            ATOMS,
            num_configs=num_configs,
            sigmas=0.1,
            seed=3,
            ensemble_mean=ensemble_mean,
        )
    return abtem.Potential(atoms, gpts=GPTS, slice_thickness=2, device=device)


def _scan(potential):
    return abtem.GridScan(
        (0, 0),
        (1, 1),
        gpts=(3, 5),
        fractional=True,
        endpoint=False,
        potential=potential,
    )


def _detector(name):
    return {
        "pixelated": abtem.PixelatedDetector,
        "flexible": abtem.FlexibleAnnularDetector,
        "annular": lambda: abtem.AnnularDetector(5, 40),
        "waves": abtem.WavesDetector,
        "real-space": lambda: abtem.PixelatedDetector(reciprocal_space=False),
    }[name]()


def _tp(potential):
    return synthetic_transition_potential(
        Z=5,
        gpts=GPTS,
        extent=potential.extent,
        energy=ENERGY,
        device=potential.device,
    )


def _assert_close(measured, expected):
    a, b = to_host_array(measured), to_host_array(expected)
    assert np.abs(a - b).max() <= TOLERANCE * np.abs(b).max()


def _prism(potential, detector, lazy, **kwargs):
    s_matrix = abtem.SMatrix(
        potential=potential, energy=ENERGY, semiangle_cutoff=20, **kwargs
    )
    result = s_matrix.transition_potential_scan(
        _tp(potential),
        scan=_scan(potential),
        detectors=detector,
        sites=ATOMS,
        lazy=lazy,
        double_channel=False,
    )
    return result.copy().compute(progress_bar=False) if lazy else result


def _multislice(potential, detector):
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=20, device=potential.device)
    return probe.transition_potential_scan(
        potential,
        _tp(potential),
        scan=_scan(potential),
        detectors=detector,
        sites=ATOMS,
        lazy=False,
        double_channel=False,
    )


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "num_configs, ensemble_mean", [(None, True), (3, True), (3, False)]
)
@pytest.mark.parametrize("detector", ["waves", "real-space"])
def test_real_space_outputs_match_multislice(
    device, detector, num_configs, ensemble_mean, lazy
):
    potential = _potential(device, num_configs, ensemble_mean)
    measured = _prism(
        potential, _detector(detector), lazy, interpolation=1, downsample=False
    )
    expected = _multislice(potential, _detector(detector))
    assert measured.shape == expected.shape
    _assert_close(measured, expected)


@devices
@pytest.mark.parametrize("interpolation, upsample", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("num_configs", [None, 3])
@pytest.mark.parametrize("detector", ["pixelated", "flexible", "annular", "waves"])
def test_lazy_declared_shape_matches_computed_and_eager(
    device, detector, num_configs, interpolation, upsample
):
    potential = _potential(device, num_configs)
    kwargs = dict(interpolation=interpolation, downsample="cutoff", upsample=upsample)
    s_matrix = abtem.SMatrix(
        potential=potential, energy=ENERGY, semiangle_cutoff=20, **kwargs
    )
    declared = s_matrix.transition_potential_scan(
        _tp(potential),
        scan=_scan(potential),
        detectors=_detector(detector),
        sites=ATOMS,
        lazy=True,
        double_channel=False,
    )
    computed = declared.copy().compute(progress_bar=False)
    eager = _prism(potential, _detector(detector), lazy=False, **kwargs)
    assert declared.shape == computed.shape == eager.shape
    _assert_close(computed, eager)


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("num_configs", [None, 3])
def test_results_carry_the_probe_metadata_of_multislice(device, num_configs, lazy):
    # the semiangle cutoff and the base tilt of the probe are what
    # DiffractionPatterns.block_direct reads
    potential = _potential(device, num_configs)
    kwargs = dict(interpolation=1, downsample=False)
    measured = _prism(potential, _detector("pixelated"), lazy, **kwargs)
    expected = _multislice(potential, _detector("pixelated"))

    for key in ("semiangle_cutoff", "base_tilt_x", "base_tilt_y"):
        assert measured.metadata[key] == expected.metadata[key]
    _assert_close(measured.block_direct(), expected.block_direct())
