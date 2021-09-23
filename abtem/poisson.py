import numpy as np
from ase import units

from abtem.utils import fft_crop

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)


def squared_wavenumbers(shape, box):
    Lx, Ly, Lz = box
    k, l, m = (np.fft.fftfreq(n, d=1 / n) for n in shape)
    k2 = k[:, None, None] ** 2 / Lx ** 2 + l[None, :, None] ** 2 / Ly ** 2 + m[None, None, :] ** 2 / Lz ** 2
    return 2 ** 2 * np.pi ** 2 * k2


def _solve_fourier_space(charge, box, out_shape):
    k2 = squared_wavenumbers(charge.shape, box)
    V = np.zeros(charge.shape, dtype=np.complex)

    nonzero = np.ones_like(V, dtype=bool)
    nonzero[0, 0, 0] = False

    V[nonzero] = charge[nonzero] / k2[nonzero]
    V[0, 0, 0] = 0

    V = fft_crop(V, out_shape)
    V = - np.fft.ifftn(V).real / eps0
    V -= V.min()
    return V


def solve_potential(charge_density, cell, out_shape):
    box = np.diag(cell)
    return _solve_fourier_space(np.fft.fftn(charge_density), box, out_shape)


def solve_system(atoms, charge_density=None, shape=None):
    density_shape = charge_density.shape

    if shape is None:
        shape = density_shape

    Lx, Ly, Lz = np.diag(atoms.cell)
    k, l, m = (np.fft.fftfreq(n, d=1 / n) for n in density_shape)
    pixel_volume = np.prod(density_shape) / np.prod(np.diag(atoms.cell))

    if solve_system is not None:
        fourier_density = np.fft.fftn(charge_density)
    else:
        fourier_density = np.zeros(density_shape, dtype=np.complex64)

    for atom in atoms:
        scale = -atom.number * pixel_volume
        x, y, z = atom.position
        fourier_density += scale * np.exp(
            -2 * np.pi * 1j * (k[:, None, None] / Lx * x + l[None, :, None] / Ly * y + m[None, None, :] / Lz * z))

    return _solve_fourier_space(fourier_density, (Lx, Ly, Lz), shape)
