import numpy as np
from ase import Atoms

from abtem.cpu_kernels import interpolate_radial_functions
from abtem.parametrizations import kirkland_projected, kirkland, load_kirkland_parameters
from abtem.potentials import PotentialIntegrator, Potential, kappa, disc_meshgrid


def test_gaussian_integral():
    sigma = 3
    f = lambda r: np.exp(-r ** 2 / (2 * sigma ** 2))
    r = np.array([0, 30])
    integrator = PotentialIntegrator(f, r)
    value = integrator.integrate(-30, 30)
    assert np.isclose(value[0], sigma * np.sqrt(2 * np.pi))


def test_projected_kirkland():
    f = lambda r: kirkland(r, parameters[6])
    parameters = load_kirkland_parameters()
    r = np.geomspace(.01, 10, 100)
    integrator = PotentialIntegrator(f, r)
    assert np.allclose(integrator.integrate(-10, 10), kirkland_projected(r, parameters[6]))


def test_cutoff():
    tolerance = 1e-5
    atoms = Atoms([6], [(0, 0, 0)], cell=(1, 1, 1))
    potential = Potential(atoms, cutoff_tolerance=tolerance)
    assert np.isclose(potential.function(potential._get_cutoff(6), potential.parameters[6]), tolerance)


def test_interpolation():  # just a sanity check, most of the error comes from the test itself
    from abtem.potentials import kappa

    sampling = .005
    L = 20
    atoms = Atoms('C', positions=[(0, 0, 1.5)], cell=(L, L, 10))

    potential = Potential(atoms, sampling=sampling, cutoff_tolerance=1e-3, slice_thickness=10)

    interpolated = potential.calculate_slice(0)[0]
    integrator = potential._integrators[6][0]

    integrated = np.sum([value[0] for value in integrator.cache._cache.values()], axis=0)

    r = np.linspace(0, L, len(interpolated), endpoint=False)

    x = np.linspace(.005, 1.5, 10)
    relative_errors = (np.interp(x, integrator.r, integrated) / kappa -
                       np.interp(x, r, interpolated)) / np.interp(x, r, interpolated)

    absolute_errors = (np.interp(x, integrator.r, integrated) / kappa - np.interp(x, r, interpolated))

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


def interpolate_radial_functions_launcher(func, positions, shape, cutoff, inner_cutoff=0.):
    n = np.int(np.ceil(cutoff - inner_cutoff))
    r = np.linspace(inner_cutoff, cutoff, 2 * n)

    values = func(r)
    values = np.tile(values.reshape(1, -1), (len(positions), 1))
    margin = np.int(np.ceil(r[-1]))

    padded_shape = (shape[0] + 2 * margin, shape[1] + 2 * margin)
    array = np.zeros((padded_shape[0], padded_shape[1]), dtype=np.float32)

    positions = positions + margin
    positions_indices = np.rint(positions).astype(np.int)[:, 0] * padded_shape[1] + \
                        np.rint(positions).astype(np.int)[:, 1]

    disc_rows, disc_cols = disc_meshgrid(margin)
    disc_indices = disc_rows * padded_shape[1] + disc_cols

    rows, cols = np.indices(padded_shape)
    array_rows = rows.ravel()
    array_cols = cols.ravel()

    interpolate_radial_functions(array, array_rows, array_cols, positions_indices, disc_indices, positions, values, r)
    return array[margin:-margin, margin:-margin]


def test_interpolate():
    sigma = 128
    func = lambda x: np.exp(-x ** 2 / (2 * sigma ** 2))

    shape = (1024, 1024)
    positions = np.array([[0, 0]])
    cutoff = 8 * sigma

    array = interpolate_radial_functions_launcher(func, positions, shape, cutoff)
    r = np.linspace(0, shape[0], shape[0], endpoint=False)
    assert np.allclose(func(r), array[0], atol=1e-6)
