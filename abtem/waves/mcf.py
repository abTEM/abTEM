from typing import Union, Tuple, TYPE_CHECKING

from scipy.sparse.linalg import eigsh


from abtem.core.axes import OrdinalAxis
from abtem.core.backend import validate_device, get_array_module
from abtem.core.energy import HasAcceleratorMixin, Accelerator
from abtem.core.fft import fft_crop, ifft2
from abtem.core.grid import spatial_frequencies, Grid
import numpy as np
from abtem.waves.transfer import WaveTransform

import math
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from abtem.waves.waves import Waves


class DiagonalizedMCF(WaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 energy: float,
                 focal_spread: float,
                 source_diameter: float,
                 semiangle_cutoff: float,
                 num_eigenvectors: int):
        self._focal_spread = focal_spread
        self._source_diameter = source_diameter
        self._semiangle_cutoff = semiangle_cutoff
        self._num_eigenvectors = num_eigenvectors

        self._accelerator = Accelerator(energy=energy)
        super().__init__(from_alpha_and_phi=False)

    @property
    def semiangle_cutoff(self):
        return self._semiangle_cutoff

    @property
    def focal_spread(self):
        return self._focal_spread

    @property
    def source_diameter(self):
        return self._source_diameter

    @property
    def num_eigenvectors(self):
        return self._num_eigenvectors

    def _cropped_shape(self, extent):
        fourier_space_sampling = 1 / extent[0], 1 / extent[1]
        return (int(np.ceil(2 * self.semiangle_cutoff / (fourier_space_sampling[0] * self.wavelength * 1e3))),
                int(np.ceil(2 * self.semiangle_cutoff / (fourier_space_sampling[1] * self.wavelength * 1e3))))

    def _evaluate_flat_cropped_mcf(self, waves) -> np.ndarray:
        xp = get_array_module(waves.device)

        waves.grid.check_is_defined()

        grid = Grid(extent=waves.extent, gpts=self._cropped_shape(waves.extent))

        kx, ky = spatial_frequencies(gpts=grid.gpts, sampling=grid.sampling, xp=xp)
        # kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        A = k2 < (self.semiangle_cutoff / self.wavelength / 1e3) ** 2

        A, kx, ky, k2 = A.ravel(), kx.ravel(), ky.ravel(), k2.ravel()

        A = xp.multiply.outer(A, A)
        kx = xp.subtract.outer(kx, kx)
        ky = xp.subtract.outer(ky, ky)
        k2 = xp.subtract.outer(k2, k2)

        Ec = xp.exp(-(0.5 * np.pi * self.wavelength * self._focal_spread) ** 2 * k2 ** 2)
        Es = np.exp(-2 * (np.pi * self.source_diameter) ** 2 * (kx ** 2 + ky ** 2))
        E = Es * Ec * A
        return E

    def evaluate_for_waves(self, waves, apply_weights=True, return_correlation: bool = False):
        E = self._evaluate_flat_cropped_mcf(waves)

        values, vectors = eigsh(E, k=self.num_eigenvectors)
        order = np.argsort(-values)

        order = order[:self.num_eigenvectors]
        vectors = vectors[:, order].T.reshape((self.num_eigenvectors,) + self._cropped_shape(waves.extent))
        values = values[order]

        vectors = fft_crop(vectors, waves.gpts)

        if return_correlation:
            raise NotImplementedError

        return np.abs(values[:, None, None]) ** .5 * vectors

    @property
    def ensemble_axes_metadata(self):
        return [OrdinalAxis()]

    def apply(self, waves: 'Waves', out_space: 'str' = 'in_space'):

        if out_space == 'in_space':
            fourier_space_out = waves.fourier_space
        elif out_space in ('fourier_space', 'real_space'):
            fourier_space_out = out_space == 'fourier_space'
        else:
            raise ValueError

        xp = get_array_module(waves.device)
        self.energy = waves.energy

        kernel = self.evaluate_for_waves(waves)

        waves = waves.ensure_fourier_space()

        waves_dims = tuple(range(len(kernel.shape) - 2))
        kernel_dims = tuple(range(len(kernel.shape) - 2, len(waves.array.shape) - 2 + len(kernel.shape) - 2))

        array = xp.expand_dims(waves.array, axis=waves_dims) * xp.expand_dims(kernel, axis=kernel_dims)

        if not fourier_space_out:
            array = ifft2(array, overwrite_x=False)

        d = waves._copy_as_dict(copy_array=False)
        d['fourier_space'] = fourier_space_out
        d['array'] = array
        d['ensemble_axes_metadata'] = self.ensemble_axes_metadata + d['ensemble_axes_metadata']
        return waves.__class__(**d)

    def ensemble_partial(self):
        pass

        # W = np.zeros((num_eigenvectors,) + self._cropped_shape(extent), dtype=float)

        # print(W.shape)

        # S = 0
        # for i in range(num_eigenvectors):
        #    vector = vectors[:, order[i]]
        #    value = values[order[i]]
        #    W[i] = vector.reshape(W.shape[1:])

        # return W
        # S = S + value * np.multiply.outer(vector, vector)

        # for n in range(num_eigenvectors):
        #     V = vecs[:, vals_idx[n]]
        #     S = S + vals[vals_idx[n]] * np.multiply.outer(V, V)  # Reconstruct MCF with eigenvectors
        #     W0 = np.reshape(V, (ny_c, nx_c))  # Reshape an 1D eigenvector back to 2D
        #     W[n, :, :] = np.pad(W0, [(pad_top, pad_bottom), (pad_left, pad_right)],
        #                         mode='constant')  # Pad the eigenvectors to original size
        # R = np.corrcoef(E.ravel(), S.ravel())
        # vals = vals[vals_idx]
        # return W, vals, R


class params(object):
    def __init__(self, HT, sigmac, ds, theta0):
        self.HT = float(HT)  # Accelerating voltage. Unit: kV
        self.sigmac = float(sigmac)  # Focal spread. Unit: Angstrom
        self.ds = float(ds)  # Source diameter. Unit: Angstrom
        self.theta0 = float(theta0)  # Half aperture angle. Unit: mrad


def diag_mcf(params, ny, nx, dq, num_eig):
    HT = params.HT
    sigmac = params.sigmac
    sigmas = params.ds / np.sqrt(2 * math.pi)
    theta0 = params.theta0 * 1e-3

    # Create q-grid
    wav = 0.3877 / np.sqrt(HT * (1 + 0.9788 * 1.0e-3 * HT))
    q0 = theta0 / wav
    nyct = np.floor(ny / 2)
    nxct = np.floor(nx / 2)
    qx = np.arange(-nxct, (nx - nxct))
    qy = np.arange(-nyct, (ny - nyct))

    qx = dq * qx
    qy = dq * qy

    qx, qy = np.meshgrid(qx, qy)
    q2 = np.square(qx) + np.square(qy)
    q = np.sqrt(q2)

    # Aperture function
    A = np.zeros((ny, nx), dtype=float)
    A[q <= q0] = 1

    # Remove redundant elements in order to reduce the size of flattened matrix
    A_id = np.nonzero(A)
    crop_y1 = np.min(A_id[0])
    crop_y2 = np.max(A_id[0]) + 1
    crop_x1 = np.min(A_id[1])
    crop_x2 = np.max(A_id[1]) + 1
    A = A[crop_y1:crop_y2, crop_x1:crop_x2]
    q2 = q2[crop_y1:crop_y2, crop_x1:crop_x2]
    qx = qx[crop_y1:crop_y2, crop_x1:crop_x2]
    qy = qy[crop_y1:crop_y2, crop_x1:crop_x2]
    ny_c, nx_c = np.shape(A)

    print(A.shape)

    # Reshape 2D matrices to 1D arrays
    A = A.ravel()
    q2 = q2.ravel()
    qx = qx.ravel()
    qy = qy.ravel()

    # Construct flattened MCF
    A = np.multiply.outer(A, A)
    qx = np.subtract.outer(qx, qx)
    qy = np.subtract.outer(qy, qy)
    q2 = np.subtract.outer(q2, q2)
    Ec = np.exp(-0.5 * (math.pi * wav * sigmac) ** 2 * np.power(q2, 2))  # Temporal MCF
    Es = np.exp(-2 * (math.pi * sigmas) ** 2 * (np.power(qx, 2) + np.power(qy, 2)))  # Spatial MCF
    E = Ec * Es * A  # Total MCF including aperture function

    print(E.shape)

    import matplotlib.pyplot as plt
    plt.imshow(E)
    plt.show()

    del Ec
    del Es
    del A
    del q2
    del qx
    del qy

    # Diagonalize the MCF and sort the largest k eigenvalues
    vals, vecs = eigsh(E, k=num_eig)
    vals_idx = np.argsort(-vals)

    # Calculate amount of padding in each dimension of eigenvectors
    pad_top = int(np.floor((ny - ny_c) / 2))
    pad_bottom = ny - ny_c - pad_top
    pad_left = int(np.floor((nx - nx_c) / 2))
    pad_right = nx - nx_c - pad_left

    # Reshape and pad the eigenvectors
    S = 0
    W = np.zeros((num_eig, ny, nx), dtype=float)
    for n in range(num_eig):
        V = vecs[:, vals_idx[n]]
        S = S + vals[vals_idx[n]] * np.multiply.outer(V, V)  # Reconstruct MCF with eigenvectors
        W0 = np.reshape(V, (ny_c, nx_c))  # Reshape an 1D eigenvector back to 2D
        W[n, :, :] = np.pad(W0, [(pad_top, pad_bottom), (pad_left, pad_right)],
                            mode='constant')  # Pad the eigenvectors to original size
    R = np.corrcoef(E.ravel(), S.ravel())
    vals = vals[vals_idx]
    return W, vals, R
