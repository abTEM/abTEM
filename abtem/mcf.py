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
from abtem.transfer import ArrayWaveTransform

if TYPE_CHECKING:
    pass


class DiagonalMCF(ArrayWaveTransform, HasAcceleratorMixin):

    def __init__(self,
                 eigenvectors: Union[int, Tuple[int]],
                 focal_spread: float = 0.,
                 source_size: float = 0.,
                 rectangular_offset: Tuple[float, float] = (0., 0.),
                 energy: float = None,
                 semiangle_cutoff: float = None, ):
        """
        The diagonal mixed coherence may be used to efficient calculate partial coherence for electron probes.

        Parameters
        ----------
        focal_spread : float, optional
            The standard deviation of the Gaussian focal spread assuming [Å].
        source_size : float, optional
            The standard deviation of the 2D gaussian shaped electron source [Å].
        rectangular_offset : two float, optional
            The standard deviation of the 2D gaussian shaped electron source [Å].
        eigenvectors : int, or tuple of int
            The subset of eigenvectors of the decomposed mixed coherence used to represent the electron probe. It is
            possible to parallelize over eigenvectors.
        energy : float, optional
            Electron energy [eV]. If not given, this will be matched to a wave function.
        semiangle_cutoff : float, optional
            Aperture half-angle [mrad]. If not given, this will be matched to a wave function.
        """

        self._focal_spread = focal_spread
        self._source_size = source_size
        self._rectangular_offset = rectangular_offset
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
    def source_size(self):
        return self._source_size

    @property
    def rectangular_offset(self):
        return self._rectangular_offset

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
                raise RuntimeError('"Semiangle_cutoff" could not be inferred from Waves, please provide as an argument.')
        else:
            semiangle_cutoff = self.semiangle_cutoff

        return semiangle_cutoff

    def _evaluate_flat_cropped_mcf(self, waves) -> np.ndarray:
        waves.grid.check_is_defined()

        semiangle_cutoff = self._safe_semiangle_cutoff(waves)

        grid = Grid(extent=waves.extent, gpts=self._cropped_shape(waves.extent, semiangle_cutoff, waves.wavelength))

        kx, ky = spatial_frequencies(gpts=grid.gpts, sampling=grid.sampling, xp=np)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        kx, ky = np.meshgrid(kx, ky, indexing='ij')

        A = k2 < (semiangle_cutoff / waves.wavelength / 1e3) ** 2

        A, kx, ky, k2 = (arr.ravel().astype(np.float32) for arr in (A, kx, ky, k2))

        A = np.multiply.outer(A, A)
        kx = np.subtract.outer(kx, kx)
        ky = np.subtract.outer(ky, ky)
        k2 = np.subtract.outer(k2, k2)

        E = A
        if self.focal_spread > 0.:
            E *= np.exp(-(0.5 * np.pi * waves.wavelength * self.focal_spread) ** 2 * k2 ** 2)

        if self.source_size > 0.:
            E *= np.exp(-(np.pi * self.source_size) ** 2 * (kx ** 2 + ky ** 2))

        if self.rectangular_offset != (0., 0.):
            E *= np.sinc(kx * self.rectangular_offset[0]) * np.sinc(ky * self.rectangular_offset[1])

        return E

    def evaluate(self, waves, return_correlation: bool = False):
        """
        Evaluate the diagonal mixed coherence function for given wave functions.

        Parameters
        ----------
        waves : Waves
            Wave functions to which the diagonal mixed coherence function is applied.
        return_correlation : bool
            Return correlation coefficients (default is False).

        Returns
        -------
        mcf : np.ndarray
            Array representing the diagonal mixed coherence function.
        """
        semiangle_cutoff = self._safe_semiangle_cutoff(waves)

        E = self._evaluate_flat_cropped_mcf(waves)

        if max(self.eigenvectors) + 1 >= E.shape[0]:
            raise RuntimeError()
        
        values, vectors = eigsh(E, k=max(self.eigenvectors) + 1)
        order = np.argsort(-values)

        selected = order[np.array(self.eigenvectors)]
        vectors = vectors[:, selected].T.reshape(
            (len(selected),) + self._cropped_shape(waves.extent, semiangle_cutoff, waves.wavelength))
        values = values[selected]

        vectors = fft_crop(vectors, waves.gpts)

        # TODO: implement returing correlation coefficients
        # R = np.corrcoef(E.ravel(), S.ravel())

        if return_correlation:
            raise NotImplementedError

        xp = get_array_module(waves.device)

        vectors = xp.array(vectors)
        values = xp.array(values)

        return xp.sqrt(xp.abs(values[:, None, None])) * vectors

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
                'source_size': self.source_size,
                'rectangular_offset': self.rectangular_offset,
                'eigenvectors': self.eigenvectors,
                'energy': self.energy,
                'semiangle_cutoff': self.semiangle_cutoff}

    def copy(self):
        return self.__class__(**self._copy_as_dict())
