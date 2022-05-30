from functools import partial
from typing import Union, Tuple, TYPE_CHECKING

import dask.array as da
import numpy as np
from scipy.sparse.linalg import eigsh

from abtem.core.axes import OrdinalAxis
from abtem.core.backend import get_array_module
from abtem.core.energy import HasAcceleratorMixin, Accelerator
from abtem.core.fft import fft_crop
from abtem.core.grid import spatial_frequencies, Grid
from abtem.waves.transfer import ArrayWaveTransform

if TYPE_CHECKING:
    pass


class DiagonalMCF(ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 focal_spread: float,
                 source_diameter: float,
                 eigenvectors: Union[int, Tuple[int]],
                 energy: float = None,
                 semiangle_cutoff: float = None, ):
        """
        Diagonal mixed coherence function


        Parameters
        ----------
        focal_spread : float
            The standard deviation of the gaussian focal spread assuming [Å].
        source_size : float
            The standard deviation of the size of the 2d gaussian shaped electron source [Å].
        eigenvectors : int, or tuple of int
            The subset of eigenvectors used to represent
        energy : float, optional
            Electron energy [eV]. If not given, this will be matched to a wavefunction.
        semiangle_cutoff : float, optional
            Half aperture angle [mrad]. If not given, this will be matched to a wavefunction.
        """

        self._focal_spread = focal_spread
        self._source_diameter = source_diameter
        self._semiangle_cutoff = semiangle_cutoff

        if np.isscalar(eigenvectors):
            eigenvectors = range(eigenvectors)

        self._eigenvectors = tuple(eigenvectors)

        self._accelerator = Accelerator(energy=energy)
        super().__init__()

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
    def eigenvectors(self):
        return self._eigenvectors

    def _cropped_shape(self, extent, semiangle_cutoff, wavelength):
        fourier_space_sampling = 1 / extent[0], 1 / extent[1]
        return (int(np.ceil(2 * semiangle_cutoff / (fourier_space_sampling[0] * wavelength * 1e3))),
                int(np.ceil(2 * semiangle_cutoff / (fourier_space_sampling[1] * wavelength * 1e3))))

    def _safe_semiangle_cutoff(self, waves):
        if self.semiangle_cutoff is None:
            try:
                semiangle_cutoff = waves.metadata['semiangle_cutoff']
            except KeyError:
                raise RuntimeError('"semiangle_cutoff" could not be inferred from Waves, please provide as an argument')
        else:
            semiangle_cutoff = self.semiangle_cutoff

        return semiangle_cutoff

    def _evaluate_flat_cropped_mcf(self, waves) -> np.ndarray:
        waves.grid.check_is_defined()

        semiangle_cutoff = self._safe_semiangle_cutoff(waves)

        grid = Grid(extent=waves.extent, gpts=self._cropped_shape(waves.extent, semiangle_cutoff, waves.wavelength))

        kx, ky = spatial_frequencies(gpts=grid.gpts, sampling=grid.sampling, xp=np)
        # kx, ky = np.fft.fftshift(kx), np.fft.fftshift(ky)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        A = k2 < (semiangle_cutoff / waves.wavelength / 1e3) ** 2

        A, kx, ky, k2 = A.ravel(), kx.ravel(), ky.ravel(), k2.ravel()

        A = np.multiply.outer(A, A)
        kx = np.subtract.outer(kx, kx)
        ky = np.subtract.outer(ky, ky)
        k2 = np.subtract.outer(k2, k2)

        Ec = np.exp(-(0.5 * np.pi * waves.wavelength * self.focal_spread) ** 2 * k2 ** 2)
        Es = np.exp(-1 * (np.pi * self.source_diameter) ** 2 * (kx ** 2 + ky ** 2))
        E = Es * Ec * A
        return E

    def evaluate(self, waves, apply_weights=True, return_correlation: bool = False):

        semiangle_cutoff = self._safe_semiangle_cutoff(waves)

        E = self._evaluate_flat_cropped_mcf(waves)

        values, vectors = eigsh(E, k=max(self.eigenvectors) + 1)
        order = np.argsort(-values)

        selected = order[np.array(self.eigenvectors)]
        vectors = vectors[:, selected].T.reshape(
            (len(selected),) + self._cropped_shape(waves.extent, semiangle_cutoff, waves.wavelength))
        values = values[selected]

        vectors = fft_crop(vectors, waves.gpts)

        # R = np.corrcoef(E.ravel(), S.ravel())

        if return_correlation:
            raise NotImplementedError

        xp = get_array_module(waves.device)

        vectors = xp.array(vectors)
        values = xp.array(values)

        return xp.abs(values[:, None, None]) ** .5 * vectors

    @property
    def ensemble_axes_metadata(self):
        return [OrdinalAxis()]

    def ensemble_partial(self):
        def diagonal_mcf(*args, kwargs):
            kwargs['eigenvectors'] = tuple(args[0])
            arr = np.zeros((1,), dtype=object)
            arr.itemset(DiagonalMCF(**kwargs))
            return arr

        kwargs = self._copy_as_dict()
        del kwargs['eigenvectors']
        return partial(diagonal_mcf, kwargs=kwargs)

    @property
    def default_ensemble_chunks(self):
        return 'auto',

    def ensemble_blocks(self, chunks):
        return da.from_array(self.eigenvectors, chunks=chunks),

    @property
    def ensemble_shape(self):
        return len(self.eigenvectors),

    def _copy_as_dict(self):
        return {'focal_spread': self.focal_spread,
                'source_diameter': self.source_diameter,
                'eigenvectors': self.eigenvectors,
                'energy': self.energy,
                'semiangle_cutoff': self.semiangle_cutoff}

    def copy(self):
        return self.__class__(**self._copy_as_dict())
