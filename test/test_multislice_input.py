"""The multislice loop must never mutate the wave functions passed to it.

The input batch may be a dask task input shared with other tasks (e.g. when
frozen-phonon configurations are partitioned across tasks), so
multislice_and_detect works on per-configuration copies. These tests pin that
invariant so the copies can be kept to the minimum (one per configuration,
no additional pristine duplicate of the batch).
"""

import numpy as np
from ase.build import bulk

import abtem


def _probe_waves(potential):
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    return probe.build(lazy=False)


def test_multislice_does_not_mutate_input_waves():
    atoms = bulk("Si", cubic=True) * (1, 1, 2)
    potential = abtem.Potential(atoms, gpts=(32, 32), slice_thickness=2)
    waves = _probe_waves(potential)
    before = waves.array.copy()
    waves.multislice(potential)
    np.testing.assert_array_equal(waves.array, before)


def test_multislice_does_not_mutate_input_waves_frozen_phonons():
    atoms = bulk("Si", cubic=True) * (1, 1, 2)
    phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=13)
    potential = abtem.Potential(phonons, gpts=(32, 32), slice_thickness=2)
    waves = _probe_waves(potential)
    before = waves.array.copy()
    waves.multislice(potential)
    np.testing.assert_array_equal(waves.array, before)
