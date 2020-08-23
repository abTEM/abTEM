import numpy as np
from ase.io import read

from abtem.potentials import Potential, PotentialArray
from abtem.waves import Probe, Waves


def test_export_import_potential(tmp_path):
    atoms = read('data/orthogonal_graphene.cif')

    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'srtio3_110_potential.hdf5'

    potential = Potential(atoms, sampling=.05)
    precalculated_potential = potential.build(pbar=False)
    precalculated_potential.write(path)
    imported_potential = PotentialArray.read(path)
    assert np.allclose(imported_potential.array, precalculated_potential.array)
    assert np.allclose(imported_potential.extent, precalculated_potential.extent)
    assert np.allclose(imported_potential._slice_thicknesses, precalculated_potential._slice_thicknesses)


def test_export_import_waves(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'waves.hdf5'

    waves = Probe(semiangle_cutoff=30, sampling=.05, extent=10, energy=80e3).build()
    waves.write(path)
    imported_waves = Waves.read(path)
    assert np.allclose(waves.array, imported_waves.array)
    assert np.allclose(waves.extent, imported_waves.extent)
    assert np.allclose(waves.energy, imported_waves.energy)
