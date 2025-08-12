#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:15:32 2025

@author: arg6
"""

import ase
from abtem.prism.s_matrix import SMatrix
import abtem
from abtem import PixelatedDetector, FrozenPhonons, PlaneWave
import pytest
import pdb
from abtem.prism.s_matrix import SMatrix
from abtem import AnnularDetector, PixelatedDetector

# if you want to directly explore one of the failing tests in debug mode, this
# is the snippet I used.
# pytest.main(
#     ["test_simulate.py::test_s_matrix_multislice_detect_with_frozen_phonons"]
# )

# This next bit attempts to recreate the components of the failing test for
# triaging.

# make a simple atom and phonon objects, as well as an SMatrix object
# roughly identical to those in the above test
atoms = ase.Atoms(
    "N2",
    positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)],
    cell=[2, 2, 2],
    pbc=True,
)
frozen_ph = FrozenPhonons(atoms, num_configs=10, sigmas=0.1, seed=100)
prism_wave = SMatrix(5, 80000, gpts=32, extent=1)

# the following command succeeeds using Atoms to define the potential both
# before and after my changes.
atom_exit_waves = prism_wave.multislice(atoms)
# This line used to fail, but now succeeds with the changes from this PR
phonon_exit_wave = prism_wave.multislice(frozen_ph)


# However, both fail when the reduce function is applied. I believe this is
# because there is metadata missing.
detector = PixelatedDetector(max_angle=None)
atom_exit_waves.reduce(detectors=detector)
