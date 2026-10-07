"""Multislice through a TransmissionFunction equals multislice through the
PotentialArray it was made from, once it is band-limited the way multislice
band-limits a potential (a TransmissionFunction is used as given)."""

import numpy as np
import pytest
from ase import Atoms
from utils import assert_array_objects_equal

import abtem
from abtem.antialias import AntialiasAperture

ENERGY = 100e3


def _potential(kind):
    atoms = Atoms(
        "CO",
        positions=[(1.0, 1.0, 0.5), (2.5, 1.7, 2.5)],
        cell=(4.0, 3.0, 4.0),
        pbc=True,
    )
    if kind == "frozen_phonons":
        atoms = abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.1, seed=1)
    exit_planes = 1 if kind == "exit_planes" else None
    potential = abtem.Potential(
        atoms, gpts=(40, 30), slice_thickness=1.0, exit_planes=exit_planes
    )
    return potential.build(lazy=kind == "lazy")


@pytest.mark.parametrize("kind", ["plain", "lazy", "exit_planes", "frozen_phonons"])
@pytest.mark.parametrize("waves", ["builder", "eager", "lazy"])
def test_multislice_through_a_transmission_function(kind, waves):
    potential = _potential(kind)
    transmission_function = AntialiasAperture().bandlimit(
        potential.transmission_function(ENERGY), in_place=False
    )
    plane_wave = abtem.PlaneWave(energy=ENERGY, extent=(4.0, 3.0), gpts=(40, 30))
    if waves != "builder":
        plane_wave = plane_wave.build(lazy=waves == "lazy")

    expected = plane_wave.multislice(potential).compute()
    result = plane_wave.multislice(transmission_function).compute()

    # The same transmission function goes through the same steps either way.
    assert_array_objects_equal(result, expected)


@pytest.mark.parametrize("kind", ["plain", "lazy", "exit_planes", "frozen_phonons"])
def test_transmission_function_keeps_what_the_potential_carries(kind):
    potential = _potential(kind)
    potential.metadata["label"] = "kept"
    transmission_function = potential.transmission_function(ENERGY)

    assert transmission_function.metadata["label"] == "kept"
    assert transmission_function.exit_planes == potential.exit_planes
    assert (
        transmission_function.ensemble_axes_metadata == potential.ensemble_axes_metadata
    )
    assert transmission_function.shape == potential.shape


@pytest.mark.parametrize("kind", ["plain", "exit_planes", "frozen_phonons"])
def test_a_chunk_of_a_transmission_function_holds_its_slices(kind):
    potential = _potential(kind)
    potential.metadata["label"] = "kept"
    transmission_function = potential.transmission_function(ENERGY)
    chunk = transmission_function.get_chunk(1, 3)

    np.testing.assert_array_equal(
        chunk.array, transmission_function.array[..., 1:3, :, :]
    )
    assert chunk.slice_thickness == transmission_function.slice_thickness[1:3]
    assert chunk.ensemble_axes_metadata == transmission_function.ensemble_axes_metadata
    assert chunk.energy == transmission_function.energy
    assert chunk.metadata == transmission_function.metadata
