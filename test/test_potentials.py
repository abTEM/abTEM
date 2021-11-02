import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from abtem.potentials import Potential  # , CrystalPotential


@pytest.fixture
def potential():
    atoms = Atoms('C', positions=[(0, 0, 0)], cell=(4, 6, 4))
    potential = Potential(atoms=atoms, sampling=.1)
    return potential


def test_potential(potential):
    assert (potential.extent[0] == 4) & (potential.extent[1] == 6)
    assert potential.thickness == 4
    assert potential.num_slices == 8
    assert np.all(potential.slice_thickness == potential.thickness / potential.num_slices)
    assert potential[0].array.shape == (1, 40, 60)
    assert potential.build().array.shape == (8, 40, 60)
    assert potential.build().compute(pbar=False).array.shape == (8, 40, 60)


def test_potential_projection():
    atoms = Atoms('C', positions=[(3, 3, 10)], cell=(6, 6, 20))

    potential = Potential(atoms=atoms, sampling=.05, projection='finite', cutoff_tolerance=1e-3)
    finite_array = potential.build().compute(pbar=False).array.sum(0)

    potential = Potential(atoms=atoms, sampling=.05, projection='infinite')
    infinite_array = potential.build().compute(pbar=False).array.sum(0)

    abs_error = finite_array - infinite_array
    rel_error = np.zeros_like(abs_error)
    tol = 2e-3
    rel_error[abs_error > tol] = abs_error[abs_error > tol] / infinite_array[abs_error > tol]

    assert np.all(rel_error < .03)


def test_potential_raises():
    with pytest.raises(RuntimeError) as e:
        Potential(Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 0)))

    assert str(e.value) == 'cell has no thickness'


def test_potential_centered():
    Lx = 5
    Ly = 5
    gpts_x = 40
    gpts_y = 60

    atoms1 = Atoms('C', positions=[(0, Ly / 2, 2)], cell=[Lx, Ly, 4])
    atoms2 = Atoms('C', positions=[(Lx / 2, Ly / 2, 2)], cell=[Lx, Ly, 4])
    potential1 = Potential(atoms1, gpts=(gpts_x, gpts_y), slice_thickness=4, cutoff_tolerance=1e-2)
    potential2 = Potential(atoms2, gpts=(gpts_x, gpts_y), slice_thickness=4, cutoff_tolerance=1e-2)

    assert np.allclose(potential1[0].array[0, :, gpts_y // 2],
                       np.roll(potential2[0].array[0, :, gpts_y // 2], gpts_x // 2))
    assert np.allclose(potential2[0].array[0, :, gpts_y // 2][1:], (potential2[0].array[0, :, gpts_y // 2])[::-1][:-1])
    assert np.allclose(potential2[0].array[0, gpts_x // 2][1:], potential2[0].array[0, gpts_x // 2][::-1][:-1])


def test_crystal_potential(graphene_atoms):
    potential_unit = Potential(graphene_atoms, parametrization='kirkland', gpts=64, projection='infinite')
    crystal_potential = CrystalPotential(potential_unit, (1, 2, 2))
    potential = Potential(graphene_atoms * (1, 2, 2), parametrization='kirkland', gpts=(64, 128), projection='infinite')

    for (start, __, a), (_, __, b) in zip(crystal_potential.generate_slices(), potential.generate_slices()):
        assert np.allclose(a.array, b.array, atol=1e-5, rtol=1e-5)


def test_z_periodic():
    atoms = bulk('C', 'sc', a=2)

    potential = Potential(atoms, slice_thickness=1, z_periodic=False, sampling=.05).build().project()

    potential_z_periodic = Potential(atoms, slice_thickness=1, z_periodic=True, sampling=.05).build().project()

    assert np.any((potential - potential_z_periodic).array)

    atoms.center(axis=2, vacuum=4)

    potential = Potential(atoms, slice_thickness=1, z_periodic=False, sampling=.05).build().project()
    potential_z_periodic = Potential(atoms, slice_thickness=1, z_periodic=True, sampling=.05).build().project()

    assert np.all((potential - potential_z_periodic).array == 0)


@pytest.mark.gpu
def test_potential_build_gpu():
    atoms = Atoms('CO', positions=[(2, 3, 1), (3, 2, 3)], cell=(4, 6, 4.3))
    potential = Potential(atoms=atoms, sampling=.1, device='gpu')

    array_potential = potential.build()
    assert np.all(asnumpy(array_potential[2].array) == asnumpy(potential[2].array))

    potential = Potential(atoms=atoms, sampling=.1, device='cpu')
    assert np.allclose(asnumpy(array_potential[2].array), potential[2].array)


@pytest.mark.gpu
def test_potential_storage():
    atoms = Atoms('CO', positions=[(2, 3, 1), (3, 2, 3)], cell=(4, 6, 4.3))

    potential = Potential(atoms=atoms, sampling=.1, device='gpu')
    assert type(potential.build().array) is cp.ndarray

    potential = Potential(atoms=atoms, sampling=.1, device='gpu', storage='cpu')
    assert type(potential.build().array) is np.ndarray


def test_potential_infinite(graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.1, projection='infinite', parametrization='kirkland')
    potential.build(pbar=False)


@pytest.mark.gpu
def test_potential_infinite_gpu(graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.1, projection='infinite', parametrization='kirkland',
                          device='gpu')

    potential.build(pbar=False)
