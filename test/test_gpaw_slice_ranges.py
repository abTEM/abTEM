"""A range of slices of a GPAWPotential must equal the same slices of the whole.

generate_slices(first_slice, last_slice), build(first_slice, last_slice) and the
chunked multislice (potential_chunk_size, the potential.slice-chunk-size
configuration) all ask for part of the slices.
"""

import sys

import numpy as np
import pytest
from ase import Atoms

import abtem
from abtem.inelastic.phonons import FrozenPhonons
from abtem.potentials.iam import PotentialArray

try:
    from gpaw import GPAW, PW

    from abtem.potentials.gpaw import GPAWPotential
except ImportError:
    pass

pytestmark = pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")

GPTS = (32, 28)


@pytest.fixture(scope="module")
def calculator():
    atoms = Atoms(
        "CO",
        positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
        cell=(3.2, 2.8, 3.6),
        pbc=True,
    )
    atoms.calc = GPAW(mode=PW(250), h=0.2, txt=None, symmetry="off")
    atoms.get_potential_energy()
    return atoms.calc


@pytest.fixture(autouse=True)
def double_precision_numpy_fft():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


def _potential(calculator, reps=(1, 1, 1), slice_thickness=0.9, **kwargs):
    return GPAWPotential(
        calculator,
        gpts=(GPTS[0] * reps[0], GPTS[1] * reps[1]),
        slice_thickness=slice_thickness,
        repetitions=reps,
        **kwargs,
    )


def _relative_difference(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("reps", [(1, 1, 1), (2, 3, 2), (1, 1, 4)])
@pytest.mark.parametrize("first, last", [(1, 3), (2, None), (0, 2), (-1, None)])
def test_generate_slices_of_a_range_equal_the_slices_of_the_whole(
    calculator, reps, first, last
):
    potential = _potential(calculator, reps)
    whole = potential.build(lazy=False).array
    n = len(potential)
    first = first % n
    stop = n if last is None else last

    part = np.stack(
        [
            s.array[0]
            for s in potential.generate_slices(first_slice=first, last_slice=last)
        ]
    )

    assert part.shape == whole[first:stop].shape
    np.testing.assert_allclose(
        part, whole[first:stop], rtol=0, atol=1e-10 * np.abs(whole).max()
    )


def test_a_range_of_a_single_slice(calculator):
    potential = _potential(calculator, (1, 1, 2))
    whole = potential.build(lazy=False).array

    for i in range(len(potential)):
        (slic,) = list(potential.generate_slices(first_slice=i, last_slice=i + 1))
        np.testing.assert_allclose(
            slic.array[0], whole[i], rtol=0, atol=1e-10 * np.abs(whole).max()
        )


def test_build_of_a_range_equals_the_slices_of_the_whole(calculator):
    potential = _potential(calculator, (2, 1, 2))
    whole = potential.build(lazy=False).array

    part = potential.build(first_slice=2, last_slice=6, lazy=False)

    assert part.array.shape == whole[2:6].shape
    np.testing.assert_allclose(
        part.array, whole[2:6], rtol=0, atol=1e-10 * np.abs(whole).max()
    )


def test_generate_slices_of_a_range_with_frozen_phonons(calculator):
    frozen_phonons = FrozenPhonons(calculator.atoms, num_configs=1, sigmas=0.05, seed=2)
    potential = _potential(calculator, (2, 1, 2), frozen_phonons=frozen_phonons)
    whole = potential.build(lazy=False).array[0]

    part = np.stack([s.array[0] for s in potential.generate_slices(2, 5)])

    np.testing.assert_allclose(
        part, whole[2:5], rtol=0, atol=1e-10 * np.abs(whole).max()
    )


def _chunked_potential(calculator, reps):
    # One frozen-phonon configuration without displacement: the multislice needs a
    # configuration count, which a GPAWPotential of a single calculator does not give.
    frozen_phonons = FrozenPhonons(calculator.atoms, num_configs=1, sigmas=0.0)
    return _potential(calculator, reps, frozen_phonons=frozen_phonons)


def _exit_wave(potential, lazy, **kwargs):
    waves = abtem.PlaneWave(energy=100e3).multislice(potential, lazy=lazy, **kwargs)
    return (waves.compute() if lazy else waves).array


def _built_array(potential):
    built = potential.build(lazy=False)
    return PotentialArray(
        built.array[0], sampling=built.sampling, slice_thickness=built.slice_thickness
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "reps, chunk_size", [((1, 1, 1), 2), ((1, 1, 4), 2), ((1, 1, 4), 1), ((1, 1, 4), 3)]
)
def test_chunked_multislice_equals_the_multislice_of_the_built_potential(
    calculator, lazy, reps, chunk_size
):
    potential = _chunked_potential(calculator, reps)
    array = _built_array(potential)

    exit_wave = _exit_wave(potential, lazy, potential_chunk_size=chunk_size)
    reference = _exit_wave(array, lazy)

    assert _relative_difference(exit_wave, reference) < 1e-10


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_multislice_with_the_chunk_size_configuration(calculator, chunk_size):
    potential = _chunked_potential(calculator, (1, 1, 4))
    array = _built_array(potential)

    with abtem.config.set({"potential.slice-chunk-size": chunk_size}):
        exit_wave = _exit_wave(potential, lazy=False)
    reference = _exit_wave(array, lazy=False)

    assert _relative_difference(exit_wave, reference) < 1e-10
