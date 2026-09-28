"""A probe with several energies, through multislice and a detector.

Each energy of a multi-energy result must equal a separate run at that energy alone.
The sizes all differ (3 energies, 2 frozen-phonon configurations, a 4 x 7 scan, 5 exit
planes), so a result whose axes are mislabelled or stacked in the wrong place shows up
as a wrong shape or wrong values rather than passing by coincidence.
"""

import ase.build
import dask
import numpy as np
import pytest

import abtem
from abtem.core.axes import EnergyAxis

ENERGIES = [50e3, 60e3, 70e3]

DETECTORS = {
    "annular": lambda: abtem.AnnularDetector(30, 100),
    "flexible": lambda: abtem.FlexibleAnnularDetector(step_size=10, inner=10, outer=80),
    "segmented": lambda: abtem.SegmentedDetector(
        nbins_radial=2, nbins_azimuthal=3, inner=20, outer=80
    ),
    "waves": lambda: abtem.WavesDetector(),
}

_references = {}


def _potential(frozen_phonons, exit_planes=None):
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    if frozen_phonons:
        atoms = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.08, seed=1)
    return abtem.Potential(
        atoms, sampling=0.1, slice_thickness=2, exit_planes=exit_planes, device="cpu"
    )


def _scan(kind, potential):
    if kind == "gridscan":
        return abtem.GridScan(
            (0, 0), (1, 1), gpts=(4, 7), fractional=True, potential=potential
        )
    if kind == "linescan":
        return abtem.LineScan(start=(0, 0), end=(3, 5), gpts=4)
    if kind == "customscan":
        return abtem.CustomScan(
            np.array([[0.5, 0.5], [1.0, 2.0], [2.0, 1.5], [3.0, 3.0]])
        )
    return None


def _run(energy, detector, frozen_phonons, scan, lazy, exit_planes=None):
    potential = _potential(frozen_phonons, exit_planes)
    probe = abtem.Probe(energy=energy, semiangle_cutoff=20, device="cpu")
    probe.grid.match(potential)
    scan = _scan(scan, potential)
    detectors = DETECTORS[detector]()
    with dask.config.set(scheduler="synchronous"):
        if scan is None:
            result = probe.multislice(potential, detectors=detectors, lazy=lazy)
        else:
            result = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
        if lazy:
            result = result.compute(progress_bar=False)
    return result


def _assert_each_energy_matches_a_single_energy_run(
    detector, frozen_phonons, scan, lazy, exit_planes=None
):
    result = _run(ENERGIES, detector, frozen_phonons, scan, lazy, exit_planes)
    kinds = [type(axis) for axis in result.axes_metadata]
    assert kinds.count(EnergyAxis) == 1
    energy_axis = kinds.index(EnergyAxis)
    assert list(result.axes_metadata[energy_axis].values) == ENERGIES

    for i, energy in enumerate(ENERGIES):
        key = (energy, detector, frozen_phonons, scan, exit_planes)
        if key not in _references:
            _references[key] = _run(
                energy, detector, frozen_phonons, scan, False, exit_planes
            ).array
        reference = _references[key]
        member = np.take(result.array, i, axis=energy_axis)
        assert member.shape == reference.shape
        np.testing.assert_allclose(
            member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
        )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("scan", ["gridscan", None], ids=["gridscan", "no-scan"])
@pytest.mark.parametrize(
    "frozen_phonons", [False, True], ids=["static", "frozen-phonons"]
)
@pytest.mark.parametrize("detector", ["annular", "waves"])
def test_each_energy_matches_a_single_energy_run(detector, frozen_phonons, scan, lazy):
    """Covers the energy axis being stacked in front of the potential's ensemble axes
    (eager, frozen phonons), a detector's reordering of the scan axes not being shifted
    past them (frozen phonons with a GridScan), and a lazy block returning its declared
    rather than its natural axis order (lazy GridScan)."""
    _assert_each_energy_matches_a_single_energy_run(
        detector, frozen_phonons, scan, lazy
    )


@pytest.mark.parametrize(
    "frozen_phonons", [False, True], ids=["static", "frozen-phonons"]
)
@pytest.mark.parametrize("detector", ["flexible", "segmented"])
def test_binned_detectors_with_a_multi_energy_scan(detector, frozen_phonons):
    _assert_each_energy_matches_a_single_energy_run(
        detector, frozen_phonons, "gridscan", lazy=False
    )


@pytest.mark.parametrize(
    "scan, frozen_phonons, lazy",
    [
        ("linescan", False, True),
        ("linescan", True, False),
        ("customscan", False, False),
        ("customscan", True, False),
    ],
)
def test_other_scan_types_with_a_multi_energy_probe(scan, frozen_phonons, lazy):
    _assert_each_energy_matches_a_single_energy_run(
        "annular", frozen_phonons, scan, lazy
    )


@pytest.mark.parametrize("detector, scan", [("annular", "gridscan"), ("waves", None)])
def test_exit_planes_with_a_multi_energy_probe(detector, scan):
    """The exit-plane axis is one of the leading axes the energy axis must be stacked
    behind."""
    _assert_each_energy_matches_a_single_energy_run(
        detector, True, scan, lazy=False, exit_planes=1
    )


def test_annular_detector_on_multi_energy_exit_waves():
    """AnnularDetector integrates through integrate_radial, which already puts the scan
    axes last; its result must not be reordered a second time."""
    waves = _run(ENERGIES, "waves", False, "gridscan", lazy=False)
    detected = abtem.AnnularDetector(30, 100).detect(waves)
    integrated = waves.diffraction_patterns(max_angle="full").integrate_radial(30, 100)

    assert detected.shape == integrated.shape == (3, 4, 7)
    assert [type(axis) for axis in detected.axes_metadata] == [
        type(axis) for axis in integrated.axes_metadata
    ]
    np.testing.assert_allclose(
        detected.array,
        integrated.array,
        rtol=0,
        atol=1e-5 * np.abs(integrated.array).max(),
    )
