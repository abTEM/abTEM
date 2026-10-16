"""Lazy frozen-phonon multislice with several exit planes and a detector that
drops the wave-function axes (``AnnularDetector`` and its relatives).

``MultisliceTransform`` partitions a frozen-phonon potential with several exit
planes as one two-dimensional argument (configurations x exit planes).
``ArrayObject._apply_transform`` packs each block's outputs into an object
array that must have one dimension per dimension of the blockwise output. When
the detector drops the wave-function axes, dask concatenates those packed
blocks along the dropped axes, which needs every dimension to be present.

Sizes are chosen so that every axis has a different length (2 configurations,
3 exit planes, a 4 x 5 scan): a result with mislabelled or transposed axes
cannot match the reference by coincidence.
"""

from __future__ import annotations

import ase.build
import numpy as np
import pytest
from utils import devices, to_host_array

import abtem

NUM_CONFIGS = 2
SEEDS = (3, 11)
NUM_EXIT_PLANES = 3
SCAN_GPTS = (4, 5)
GPTS = (48, 48)


def _atoms():
    return ase.build.bulk("Si", cubic=True)


def _frozen_phonons(ensemble_mean: bool = True):
    return abtem.FrozenPhonons(
        _atoms(),
        num_configs=NUM_CONFIGS,
        sigmas=0.1,
        seed=SEEDS,
        ensemble_mean=ensemble_mean,
    )


def _potential(atoms, device):
    # Si (5.431 A) in 1.36 A slices is four slices; exit_planes=2 gives the
    # entrance plane and two exit planes.
    potential = abtem.Potential(
        atoms, gpts=GPTS, slice_thickness=1.36, exit_planes=2, device=device
    )
    assert len(potential.exit_planes) == NUM_EXIT_PLANES
    return potential


def _scan(potential, kind: str):
    if kind == "grid":
        return abtem.GridScan(
            start=(0, 0),
            end=(0.5, 0.75),
            fractional=True,
            potential=potential,
            gpts=SCAN_GPTS,
        )
    if kind == "line":
        return abtem.LineScan(start=(0, 0), end=(3, 4), gpts=7)
    raise ValueError(kind)


def _scan_shape(kind: str) -> tuple[int, ...]:
    return SCAN_GPTS if kind == "grid" else (7,)


def _run(atoms, kind, lazy, device, detectors=None):
    potential = _potential(atoms, device)
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    if detectors is None:
        detectors = abtem.AnnularDetector(inner=30, outer=90)
    return probe.scan(
        potential, scan=_scan(potential, kind), detectors=detectors, lazy=lazy
    )


def _per_configuration(kind, device):
    """One static-potential run per displaced configuration."""
    return [
        to_host_array(_run(atoms, kind, lazy=False, device=device))
        for atoms in _frozen_phonons()
    ]


def _assert_close(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual, expected, rtol=1e-5, atol=1e-6 * np.abs(expected).max()
    )


@devices
@pytest.mark.parametrize("kind", ["grid", "line"])
def test_ensemble_mean_matches_eager_and_configuration_mean(device, kind):
    lazy = _run(_frozen_phonons(), kind, lazy=True, device=device)
    assert lazy.is_lazy
    assert lazy.shape == (NUM_EXIT_PLANES, *_scan_shape(kind))

    lazy_values = to_host_array(lazy.compute())
    eager_values = to_host_array(
        _run(_frozen_phonons(), kind, lazy=False, device=device)
    )
    _assert_close(lazy_values, eager_values)

    reference = np.mean(_per_configuration(kind, device), axis=0)
    _assert_close(lazy_values, reference)

    # The planes differ, so a permuted exit-plane axis cannot pass.
    assert not np.allclose(lazy_values[1], lazy_values[2])


@devices
def test_without_ensemble_mean_each_configuration_matches_its_static_run(device):
    lazy = _run(_frozen_phonons(ensemble_mean=False), "grid", lazy=True, device=device)
    assert lazy.shape == (NUM_CONFIGS, NUM_EXIT_PLANES, *SCAN_GPTS)

    lazy_values = to_host_array(lazy.compute())
    eager_values = to_host_array(
        _run(_frozen_phonons(ensemble_mean=False), "grid", lazy=False, device=device)
    )
    _assert_close(lazy_values, eager_values)

    for config, reference in enumerate(_per_configuration("grid", device)):
        _assert_close(lazy_values[config], reference)


@devices
def test_crystal_potential_frozen_phonons_lazy_matches_eager(device):
    def scan(lazy):
        unit = abtem.Potential(_atoms(), gpts=GPTS, slice_thickness=1.36, device=device)
        potential = abtem.CrystalPotential(
            unit,
            repetitions=(1, 1, 2),
            num_frozen_phonons=NUM_CONFIGS,
            seeds=SEEDS,
            exit_planes=4,
        )
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
        probe.grid.match(potential)
        return probe.scan(
            potential,
            scan=_scan(potential, "grid"),
            detectors=abtem.AnnularDetector(inner=30, outer=90),
            lazy=lazy,
        )

    lazy = scan(lazy=True)
    num_planes = lazy.shape[0]
    assert num_planes not in (NUM_CONFIGS, *SCAN_GPTS)
    assert lazy.shape == (num_planes, *SCAN_GPTS)

    _assert_close(to_host_array(lazy.compute()), to_host_array(scan(lazy=False)))


@devices
def test_annular_detector_alongside_a_detector_that_keeps_its_axes(device):
    def detectors():
        return [
            abtem.AnnularDetector(inner=30, outer=90),
            abtem.FlexibleAnnularDetector(),
        ]

    lazy = _run(
        _frozen_phonons(), "grid", lazy=True, device=device, detectors=detectors()
    )
    eager = _run(
        _frozen_phonons(), "grid", lazy=False, device=device, detectors=detectors()
    )
    assert lazy[0].shape == (NUM_EXIT_PLANES, *SCAN_GPTS)

    # In place: both detectors come from one graph, computed once here, so
    # to_host_array below only copies to the host.
    lazy.compute()
    for lazy_measurement, eager_measurement in zip(lazy, eager):
        _assert_close(to_host_array(lazy_measurement), to_host_array(eager_measurement))
