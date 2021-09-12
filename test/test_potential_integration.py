import numpy as np
from ase import Atoms

from abtem.cpu_kernels import interpolate_radial_functions
from abtem.parametrizations import kirkland_projected, kirkland, load_kirkland_parameters
from abtem.potentials import PotentialIntegrator, Potential, kappa, _disc_meshgrid


def test_gaussian_integral():
    sigma = 3
    f = lambda r: np.exp(-r ** 2 / (2 * sigma ** 2))
    r = np.array([0, 30])
    integrator = PotentialIntegrator(f, r, 30)
    value = integrator.integrate(np.array([0.]), -50, 50)
    assert np.isclose(value[0][0][0], sigma * np.sqrt(2 * np.pi))


def test_projected_kirkland():
    f = lambda r: kirkland(r, parameters[6])
    parameters = load_kirkland_parameters()
    r = np.geomspace(.01, 10, 100)
    integrator = PotentialIntegrator(f, r, 20)
    assert np.allclose(integrator.integrate(np.array([0.]), -10, 10)[0], kirkland_projected(r, parameters[6]))


def test_cutoff():
    tolerance = 1e-5
    atoms = Atoms([6], [(0, 0, 0)], cell=(1, 1, 1))
    potential = Potential(atoms, cutoff_tolerance=tolerance)
    assert np.isclose(potential.function(potential.get_cutoff(6), potential.parameters[6]), tolerance)


def test_interpolation():  # just a sanity check
    from abtem.potentials import kappa

    sampling = .005
    L = 20
    atoms = Atoms('C', positions=[(0, 0, 1.5)], cell=(L, L, 3))

    potential = Potential(atoms, sampling=sampling, cutoff_tolerance=1e-3, slice_thickness=3, z_periodic=False)

    interpolated = potential[0].array[0, 0]
    integrator = potential.get_integrator(6)

    integrated = integrator.integrate(np.array([0.]), -1.5, 1.5)[0][0]

    r = np.linspace(0, L, len(interpolated), endpoint=False)

    x = np.linspace(.005, 1.5, 10)
    relative_errors = (np.interp(x, integrator.r, integrated) / kappa -
                       np.interp(x, r, interpolated)) / np.interp(x, r, interpolated)

    absolute_errors = (np.interp(x, integrator.r, integrated) / kappa - np.interp(x, r, interpolated))

    print(np.abs(relative_errors).max())
    print(np.abs(absolute_errors).max())

    assert np.all((np.abs(relative_errors) < 1e-4) + (np.abs(absolute_errors) < 1e-4))


def test_geomspace():
    sampling = .1
    rc = 10
    n = 100

    for rf in np.linspace(.01, rc + 1, 1000):
        r = np.geomspace(sampling, rc, n)
        i = max(np.searchsorted(r, rf) - 1, 0)
        dt = np.log(rc / sampling) / (n - 1)
        j = min(max(np.floor(np.log(rf / sampling) / dt), 0), len(r) - 1)
        assert i == j

# def interpolate_radial_functions_launcher(func, positions, shape, cutoff, inner_cutoff):
#     n = np.int(np.ceil(cutoff - inner_cutoff))
#     r = np.geomspace(inner_cutoff, cutoff, 100 * n)[:-1]
#
#     v = func(r)
#     v = np.tile(v.reshape(1, -1), (len(positions), 1))
#     dvdr = np.zeros_like(v)
#     dvdr[:, :-1] = np.diff(v) / np.diff(r)
#
#     # padded_shape = (shape[0] + 2 * margin, shape[1] + 2 * margin)
#     array = np.zeros(shape, dtype=np.float32)
#
#     position_indices = np.rint(positions).astype(np.int)
#
#     rows, cols = disc_meshgrid(np.int(np.ceil(r[-1])))
#     disc_indices = np.hstack((rows[:, None], cols[:, None]))
#
#     rows, cols = np.indices(shape)
#     x = rows.astype(np.float32)
#     y = cols.astype(np.float32)
#
#     interpolate_radial_functions(array, x, y, position_indices, disc_indices, positions, v, r, dvdr)
#     return array
#
#
# def test_interpolate():
#     sigma = 128
#     func = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))
#
#     shape = (2048, 2048)
#     positions = np.array([[0, 0]])
#     cutoff = 10 * sigma
#     inner_cutoff = 0.001
#
#     array = interpolate_radial_functions_launcher(func, positions, shape, cutoff, inner_cutoff=inner_cutoff)
#     r = np.linspace(inner_cutoff, shape[0], shape[0], endpoint=False)
#
#     assert np.allclose(func(r), array[0], atol=1e-6)
