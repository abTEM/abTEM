import copy
from abc import ABCMeta, abstractmethod
from numbers import Number

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
from abtem.basic.axes import HasAxesMetadata
from abtem.basic.backend import cp, asnumpy, get_array_module, get_scipy_module
from abtem.measure.utils import polar_detector_bins, sum_run_length_encoded

if cp is not None:
    from abtem.basic.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
else:
    sum_run_length_encoded_cuda = None


class HasReductionMixin:
    _illegal_reduction: tuple
    _axes_metadata: dict

    def _validate_axis(self, axis):
        if isinstance(axis, str):
            pass

        pass


class AbstractMeasurement(HasAxesMetadata, metaclass=ABCMeta):

    def __init__(self, array, axes_metadata, metadata, base_axes):
        self._array = array
        self._axes_metadata = axes_metadata
        self._metadata = metadata
        self._base_axes = base_axes

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def base_shape(self):
        return self.shape[-self.base_dimensions:]

    @property
    def base_dimensions(self):
        return len(self._base_axes)

    @property
    def dimensions(self):
        return len(self._array.shape)

    @property
    def collection_dimensions(self):
        return self.dimensions - self.base_dimensions

    @property
    def base_axes(self):
        base_axes = ()
        for base_axis in self._base_axes:
            if base_axis < 0:
                base_axis = self.dimensions + base_axis

            base_axes += (base_axis,)
        return base_axes

    @property
    def collection_axes(self):
        return tuple(range(self.collection_dimensions))

    def _validate_axes(self, axes):
        if isinstance(axes, Number):
            return (axes,)
        return axes

    def _check_is_base_axes(self, axes):
        axes = self._validate_axes(axes)
        return len(set(axes).intersection(self.base_axes)) > 0

    def mean(self, axes=None):
        return self._reduction(da.mean, axes=axes)

    def sum(self, axes=None):
        return self._reduction(da.mean, axes=axes)

    def std(self, axes=None):
        return self._reduction(da.std, axes=axes)

    def _reduction(self, reduction_func, axes=None):
        if axes is None:
            axes = self.collection_axes

        if self._check_is_base_axes(axes):
            raise RuntimeError('base axes cannot be reduced')

        axes_metadata = self._remove_axes_metadata(axes)
        new_copy = self.copy(copy_array=False)
        new_copy._array = reduction_func(new_copy._array, axes)
        new_copy._axes_metadata = axes_metadata
        return new_copy

    def __getitem__(self, items):
        if isinstance(items, Number):
            items = (items,)

        array = self._array[items]

        removed_axes = []
        for i, item in enumerate(items):
            if isinstance(item, Number):
                removed_axes.append(i)

        axes_metadata = self._remove_axes_metadata(removed_axes)

        if self._check_is_base_axes(removed_axes):
            raise RuntimeError('base axes cannot be indexed')

        return

        print(axes_metadata)

    def asnumpy(self):
        new_copy = self.copy(copy_array=False)
        new_copy._array = asnumpy(self._array)
        return new_copy

    @abstractmethod
    def copy(self, copy_array=True):
        pass


class Images(AbstractMeasurement):

    def __init__(self, array, axes_metadata=None, metadata=None):
        super().__init__(array, axes_metadata, metadata, (-2, -1))

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, axes_metadata=copy.deepcopy(self.axes_metadata))

    def compute(self, **kwargs):
        array = self._array.compute(**kwargs)
        return self.__class__(array, axes_metadata=self._axes_metadata)

    @property
    def sampling(self):
        return tuple(axes_metadata['sampling'] for axes_metadata in self.axes_metadata[-2:])

    @property
    def extent(self):
        return tuple(d * n for d, n in zip(self.sampling, self.base_shape))

    def gaussian_filter(self, sigma, padding_mode: str = 'wrap'):

        scipy = get_scipy_module(self._array)

        sigma = [s / sampling for s, calibration in zip(sigma, self.calibrations)]

        array = scipy.gaussian_filter(self.array, sigma, mode=padding_mode)
        return self.__class__

    def show(self, ax=None):
        if ax is None:
            ax = plt.subplot()

        slic = (0,) * self.collection_dimensions

        ax.imshow(asnumpy(self._array)[slic].T, extent=[0, self.extent[0], 0, self.extent[1]], origin='lower')
        ax.set_xlabel(f'{self.axes_metadata[-2]["label"]} [Å]')
        ax.set_ylabel(f'{self.axes_metadata[-1]["label"]} [Å]')


class DiffractionPatterns(AbstractMeasurement):

    def __init__(self,
                 array,
                 fftshift: bool = False,
                 axes_metadata=None,
                 metadata=None):
        self._fftshift = fftshift
        super().__init__(array, axes_metadata, metadata, (-2, -1))

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, axes_metadata=copy.deepcopy(self.axes_metadata),
                              metadata=copy.deepcopy(self.axes_metadata), fftshift=self.fftshift)

    def compute(self):
        array = self._array.compute()
        return self.__class__(array, fftshift=self.fftshift, axes_metadata=self.axes_metadata)

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def angular_sampling(self):
        return tuple(axes_metadata['sampling'] for axes_metadata in self.axes_metadata[-2:])

    @property
    def max_angles(self):
        return (self.shape[0] // 2 * self.angular_sampling[0], self.shape[1] // 2 * self.angular_sampling[1])

    def _check_max_angle(self, angle):
        if (angle > self.max_angles[0]) or (angle > self.max_angles[1]):
            raise RuntimeError('integration angle exceeds the maximum scattering angle')

    def polar_binning(self, nbins_radial, nbins_azimuthal, inner, outer):
        self._check_max_angle(outer)
        xp = get_array_module(self.array)

        indices = dask.delayed(polar_detector_bins, pure=True)(gpts=self.array.shape[-2:],
                                                               sampling=self.angular_sampling,
                                                               inner=inner,
                                                               outer=outer,
                                                               nbins_radial=nbins_radial,
                                                               nbins_azimuthal=nbins_azimuthal,
                                                               fftshift=self.fftshift,
                                                               return_indices=True)

        def radial_binning(array, indices, nbins_radial, nbins_azimuthal):
            xp = get_array_module(array)

            separators = xp.concatenate((xp.array([0]), xp.cumsum(xp.array([len(i) for i in indices]))))

            new_shape = array.shape[:-2] + (nbins_radial, nbins_azimuthal)

            array = array.reshape((-1, array.shape[-2] * array.shape[-1],))[..., np.concatenate(indices)]

            result = xp.zeros((array.shape[0], len(indices),), dtype=xp.float32)

            if xp is cp:
                sum_run_length_encoded_cuda(array, result, separators)

            else:
                sum_run_length_encoded(array, result, separators)

            return result.reshape(new_shape)

        array = self.array.map_blocks(radial_binning, indices=indices, nbins_radial=nbins_radial,
                                      nbins_azimuthal=nbins_azimuthal,
                                      drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                      chunks=self.array.chunks[:-2] + ((nbins_radial,), (nbins_azimuthal,),),
                                      new_axis=(len(self.shape) - 2, len(self.shape) - 1,),
                                      meta=xp.array((), dtype=xp.float32))

        return array

    def radial_binning(self, step_size=1., inner=0., outer=None):
        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)

        return self.polar_binning(nbins_radial, 1, inner, outer)

    def integrate_annular_disc(self, inner, outer):
        self._check_max_angle(outer)

        bins = dask.delayed(polar_detector_bins, pure=True)(gpts=self.array.shape[-2:],
                                                            sampling=self.angular_sampling,
                                                            inner=inner,
                                                            outer=outer,
                                                            nbins_radial=1,
                                                            nbins_azimuthal=1,
                                                            fftshift=self.fftshift)

        xp = get_array_module(self.array)
        bins = da.from_delayed(bins, shape=self.array.shape[-2:], dtype=xp.float32)
        bins = bins.map_blocks(xp.array)

        def integrate_fourier_space(array, bins):
            xp = get_array_module(array)
            return xp.sum(array * (bins == 0), axis=(-2, -1))

        integrated_intensity = self.array.map_blocks(integrate_fourier_space, bins=bins,
                                                     drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                                     dtype=xp.array((), dtype=xp.float32))

        return Images(integrated_intensity, axes_metadata=self.axes_metadata[:-2])
