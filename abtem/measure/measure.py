import copy
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, Tuple

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import zarr

from abtem.basic.axes import HasAxesMetadata
from abtem.basic.backend import cp, asnumpy, get_array_module, get_scipy_module
from abtem.basic.dask import computable, HasDaskArray, requires_dask_array
from abtem.basic.fft import fft2_interpolate
from abtem.measure.utils import polar_detector_bins, sum_run_length_encoded
from abtem.visualize.utils import domain_coloring

if cp is not None:
    from abtem.basic.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
else:
    sum_run_length_encoded_cuda = None


class AbstractMeasurement(HasDaskArray, HasAxesMetadata, metaclass=ABCMeta):

    def __init__(self, array, axes_metadata, metadata, base_axes):
        self._array = array
        self._axes_metadata = axes_metadata
        self._metadata = metadata
        self._base_axes = base_axes

        super().__init__(array)

    @property
    def metadata(self):
        return self._metadata

    @property
    def axes_metadata(self):
        return self._axes_metadata

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

    def _get_measurements(self, items, **kwargs):
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

        return self.__class__(array=array, axes_metadata=axes_metadata, metadata=self.metadata, **kwargs)

    # def __getitem__(self, items):
    #     if isinstance(items, Number):
    #         items = (items,)
    #
    #     array = self._array[items]
    #
    #     removed_axes = []
    #     for i, item in enumerate(items):
    #         if isinstance(item, Number):
    #             removed_axes.append(i)
    #
    #     axes_metadata = self._remove_axes_metadata(removed_axes)
    #
    #     if self._check_is_base_axes(removed_axes):
    #         raise RuntimeError('base axes cannot be indexed')
    #
    #     return self.__class__

    @requires_dask_array
    def to_zarr(self, url, overwrite=False):

        with zarr.open(url, mode='w') as root:
            self.array.to_zarr(url, component='array', overwrite=overwrite)
            root.attrs['axes_metadata'] = self.axes_metadata
            root.attrs['metadata'] = self.metadata
            root.attrs['cls'] = self.__class__.__name__

    @staticmethod
    def from_zarr(url, chunks=None):

        with zarr.open(url, mode='r') as f:
            axes_metadata = f.attrs['axes_metadata']
            metadata = f.attrs['metadata']
            cls = globals()[f.attrs['cls']]

        array = da.from_zarr(url, component='array')
        return cls(array, axes_metadata=axes_metadata, metadata=metadata)

    def asnumpy(self):
        new_copy = self.copy(copy_array=False)
        new_copy._array = asnumpy(self._array)
        return new_copy

    @abstractmethod
    def copy(self, copy_array=True):
        pass


class Images(AbstractMeasurement):

    def __init__(self, array, sampling, axes_metadata=None, metadata=None):
        self._sampling = sampling
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-2, -1))

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, sampling=self.sampling, axes_metadata=copy.deepcopy(self.axes_metadata))

    @property
    def sampling(self):
        return self._sampling

    @property
    def extent(self):
        return tuple(d * n for d, n in zip(self.sampling, self.base_shape))

    @computable
    @requires_dask_array
    def interpolate(self,
                    sampling: Union[float, Tuple[float, float]] = None,
                    gpts: Union[int, Tuple[int, int]] = None,
                    method: str = 'fft',
                    boundary: str = 'periodic'):

        if method == 'fft' and boundary != 'periodic':
            raise ValueError()

        if sampling is None and gpts is None:
            raise ValueError()

        if gpts is None and sampling is not None:
            if isinstance(sampling, Number):
                sampling = (sampling,) * 2
            gpts = tuple(int(np.ceil(l / d)) for d, l in zip(sampling, self.extent))
        elif gpts is not None:
            if isinstance(gpts, Number):
                gpts = (gpts,) * 2
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        array = dask.delayed(fft2_interpolate)(self.array, gpts)

        array = da.from_delayed(array, shape=self.shape[:-2] + gpts, meta=xp.array((), dtype=xp.float32))

        return self.__class__(array, sampling=sampling, axes_metadata=self.axes_metadata, metadata=self.metadata)

    def is_compatible(self, other):

        if self.shape != other.shape:
            return False

        if self.sampling != self.sampling:
            return False

        if self.axes_metadata != other.axes_metadata:
            return False

        return True

    def subtract(self, other):
        self.is_compatible(other)
        return self.__class__(self.array - other.array,
                              sampling=self.sampling,
                              axes_metadata=copy.copy(self.axes_metadata),
                              metadata=copy.copy(self.metadata))

    def tile(self, reps):
        new_array = np.tile(self.array, reps)
        return self.__class__(new_array, sampling=self.sampling, axes_metadata=copy.copy(self.axes_metadata),
                              metadata=copy.copy(self.metadata))

    def gaussian_filter(self, sigma: Union[float, Tuple[float, float]], boundary: str = 'periodic'):
        xp = get_array_module(self.array)
        scipy = get_scipy_module(self._array)

        if isinstance(sigma, Number):
            sigma = (sigma,) * 2

        sigma = (0,) * (len(self.shape) - 2) + tuple(s / d for s, d in zip(sigma, self.sampling))

        array = self.array.map_overlap(scipy.ndimage.gaussian_filter,
                                       sigma=sigma,
                                       boundary=boundary,
                                       depth=(0,) * (len(self.shape) - 2) + (int(np.ceil(4.0 * max(sigma))),) * 2,
                                       meta=xp.array((), dtype=xp.float32))

        return self.__class__(array, axes_metadata=self.axes_metadata, metadata=self.metadata)

    def show(self, ax=None, cbar=False, **kwargs):
        if ax is None:
            ax = plt.subplot()

        slic = (0,) * self.collection_dimensions

        array = asnumpy(self._array)[slic].T

        if np.iscomplexobj(array):
            array = domain_coloring(array)

        im = ax.imshow(array, extent=[0, self.extent[0], 0, self.extent[1]], origin='lower', **kwargs)
        ax.set_xlabel('x [Å]')
        ax.set_ylabel('y [Å]')

        if cbar:
            plt.colorbar(im, ax=ax)

        return ax, im


class LineProfiles(AbstractMeasurement):

    def __init__(self, array, axes_metadata, metadata=None):
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-1,))

    def show(self, ax=None, title=''):
        if ax is None:
            ax = plt.subplot()

        ax.plot(self.array)
        ax.set_title(title)

        return ax

    def copy(self):
        pass


class DiffractionPatterns(AbstractMeasurement):

    def __init__(self,
                 array,
                 angular_sampling,
                 fftshift: bool = False,
                 axes_metadata=None,
                 metadata=None):

        self._fftshift = fftshift
        self._angular_sampling = angular_sampling
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-2, -1))

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, axes_metadata=copy.deepcopy(self.axes_metadata),
                              metadata=copy.deepcopy(self.axes_metadata), fftshift=self.fftshift)

    def __getitem__(self, items):
        return self._get_measurements(items, fftshift=self.fftshift)

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def angular_sampling(self):
        return self._angular_sampling

    @property
    def max_angles(self):
        return (self.shape[-2] // 2 * self.angular_sampling[0], self.shape[-1] // 2 * self.angular_sampling[1])

    @property
    def fourier_space_extent(self):
        limits = []
        for i in (-2, -1):
            if self.shape[i] % 2:
                limits += [(-(self.shape[i] - 1) // 2 * self.angular_sampling[i],
                            (self.shape[i] - 1) // 2 * self.angular_sampling[i])]
            else:
                limits += [(-self.shape[i] // 2 * self.angular_sampling[i],
                            (self.shape[i] // 2 - 1) * self.angular_sampling[i])]
        return limits

    def _check_max_angle(self, angle):
        if (angle > self.max_angles[0]) or (angle > self.max_angles[1]):
            raise RuntimeError('integration angle exceeds the maximum simulated angle')

    @computable
    @requires_dask_array
    def polar_binning(self, nbins_radial, nbins_azimuthal, inner, outer, rotation=0.):
        self._check_max_angle(outer)
        xp = get_array_module(self.array)

        indices = dask.delayed(polar_detector_bins, pure=True)(gpts=self.array.shape[-2:],
                                                               sampling=self.angular_sampling,
                                                               inner=inner,
                                                               outer=outer,
                                                               nbins_radial=nbins_radial,
                                                               nbins_azimuthal=nbins_azimuthal,
                                                               fftshift=self.fftshift,
                                                               rotation=rotation,
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

        radial_sampling = (outer - inner) / nbins_radial
        azimuthal_sampling = 2 * np.pi / nbins_azimuthal

        axes_metadata = self.axes_metadata

        return PolarMeasurements(array,
                                 radial_sampling=radial_sampling,
                                 azimuthal_sampling=azimuthal_sampling,
                                 radial_offset=inner,
                                 azimuthal_offset=rotation,
                                 axes_metadata=axes_metadata,
                                 metadata=self.metadata)

    @computable
    @requires_dask_array
    def radial_binning(self, step_size=1., inner=0., outer=None):
        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)
        return self.polar_binning(nbins_radial, 1, inner, outer)

    @computable
    @requires_dask_array
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

        if len(self.scan_axes) == 1:
            return LineProfiles(integrated_intensity, axes_metadata=self.axes_metadata[:-2])
        else:
            sampling = [self.axes_metadata[axis]['sampling'] for axis in self.scan_axes]
            return Images(integrated_intensity, sampling=sampling, axes_metadata=self.axes_metadata[:-2])

    def center_of_mass(self):
        x, y = self.angular_coordinates()

        com_x = (self.array * x[:, None]).sum(axis=(-2, -1))
        com_y = (self.array * y[None]).sum(axis=(-2, -1))

        sampling = [self.axes_metadata[axis]['sampling'] for axis in self.scan_axes]

        com = com_x + 1.j * com_y
        com = Images(array=com, sampling=sampling, axes_metadata=self.axes_metadata, metadata=self.metadata)
        return com

        # com_x =

        # grids = np.ogrid[[slice(0, i) for i in input.shape]]

        # results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
        #           for dir in range(input.ndim)]

        # for i in range(self.shape[-2]):
        #     for j in range(self.shape[-1]):
        #         com[i, j] = scipy.ndimage.measurements.center_of_mass(self.array[i, j])
        # com = com - center[None, None]
        #

    #
    #     if measurement.dimensions == 3:
    #         for i in range(measurement.array.shape[0]):
    #             com[i] = scipy.ndimage.measurements.center_of_mass(measurement.array[i])
    #         com = com - center[None]
    #     else:

    # def center_of_mass(self, return_magnitude=False):
    #     """
    #     Calculate the center of mass of a measurement.
    #
    #     Parameters
    #     ----------
    #     measurement : Measurement
    #         A collection of diffraction patterns.
    #     return_icom : bool
    #         If true, return the integrated center of mass.
    #
    #     Returns
    #     -------
    #     Measurement
    #     """
    #     if (measurement.dimensions != 3) and (measurement.dimensions != 4):
    #         raise RuntimeError()
    #
    #     if not (measurement.calibrations[-1].units == measurement.calibrations[-2].units):
    #         raise RuntimeError()
    #
    #     shape = measurement.array.shape[-2:]
    #     center = np.array(shape) / 2 - np.array([.5 * (shape[-2] % 2), .5 * (shape[-1] % 2)])
    #     com = np.zeros(measurement.array.shape[:-2] + (2,))
    #
    #     if measurement.dimensions == 3:
    #         for i in range(measurement.array.shape[0]):
    #             com[i] = scipy.ndimage.measurements.center_of_mass(measurement.array[i])
    #         com = com - center[None]
    #     else:
    #         for i in range(measurement.array.shape[0]):
    #             for j in range(measurement.array.shape[1]):
    #                 com[i, j] = scipy.ndimage.measurements.center_of_mass(measurement.array[i, j])
    #         com = com - center[None, None]
    #
    #     com[..., 0] = com[..., 0] * measurement.calibrations[-2].sampling
    #     com[..., 1] = com[..., 1] * measurement.calibrations[-1].sampling
    #
    #     if return_icom:
    #         if measurement.dimensions != 4:
    #             raise RuntimeError('the integrated center of mass is only defined for 4d measurements')
    #
    #         sampling = (measurement.calibrations[0].sampling, measurement.calibrations[1].sampling)
    #
    #         icom = intgrad2d((com[..., 0], com[..., 1]), sampling)
    #         return Measurement(icom, measurement.calibrations[:-2])
    #     elif return_magnitude:
    #         magnitude = np.sqrt(com[..., 0] ** 2 + com[..., 1] ** 2)
    #         return Measurement(magnitude, measurement.calibrations[:-2], units='mrad', name='com')
    #
    #     else:
    #         return (Measurement(com[..., 0], measurement.calibrations[:-2], units='mrad', name='com_x'),
    #                 Measurement(com[..., 1], measurement.calibrations[:-2], units='mrad', name='com_y'))

    def angular_coordinates(self):
        alpha_x = np.linspace(self.fourier_space_extent[0][0], self.fourier_space_extent[0][1], self.shape[-2])
        alpha_y = np.linspace(self.fourier_space_extent[1][0], self.fourier_space_extent[1][1], self.shape[-1])
        return alpha_x, alpha_y

    @computable
    @requires_dask_array
    def block_direct(self, radius=None):

        if radius is None:
            radius = max(self.angular_sampling) * 1.1

        def block_direct(array):
            alpha_x, alpha_y = self.angular_coordinates()
            alpha = alpha_x[:, None] ** 2 + alpha_y[None] ** 2
            block = alpha > radius ** 2
            return array * block

        xp = get_array_module(self.array)
        array = da.from_delayed(dask.delayed(block_direct)(self.array), shape=self.shape,
                                meta=xp.array((), dtype=xp.float32))
        return self.__class__(array, angular_sampling=self.angular_sampling, axes_metadata=self.axes_metadata,
                              metadata=self.metadata, fftshift=self.fftshift)

    def show(self, ax=None, power=1., **kwargs):
        if ax is None:
            ax = plt.subplot()

        slic = (0,) * self.collection_dimensions
        extent = self.fourier_space_extent[0] + self.fourier_space_extent[1]

        array = asnumpy(self._array)[slic].T ** power

        im = ax.imshow(array, extent=extent, origin='lower', **kwargs)
        ax.set_xlabel(f'{self.axes_metadata[-2]["label"]} [mrad]')
        ax.set_ylabel(f'{self.axes_metadata[-1]["label"]} [mrad]')
        return ax, im


class PolarMeasurements(AbstractMeasurement):

    def __init__(self, array, radial_sampling, azimuthal_sampling, radial_offset=0., azimuthal_offset=0.,
                 axes_metadata=None, metadata=None):

        self._radial_sampling = radial_sampling
        self._azimuthal_sampling = azimuthal_sampling
        self._radial_offset = radial_offset
        self._azimuthal_offset = azimuthal_offset

        super().__init__(array, axes_metadata, metadata, base_axes=(-2, -1))

    @property
    def inner_angle(self):
        return self._radial_offset

    @property
    def outer_angle(self):
        return self._radial_offset + self.radial_sampling * self.shape[-2]

    @property
    def radial_sampling(self):
        return self._radial_sampling

    @property
    def azimuthal_sampling(self):
        return self._azimuthal_sampling

    @property
    def azimuthal_offset(self):
        return self._azimuthal_offset

    def _check_radial_angle(self, angle):
        if angle < self.inner_angle or angle > self.outer_angle:
            raise RuntimeError()

    def integrate_radial(self, inner, outer):
        return self.integrate(radial_limits=(inner, outer))

    def integrate(self, radial_limits=None, azimutal_limits=None, detector_regions=None):

        sampling = [self.axes_metadata[axis]['sampling'] for axis in self.scan_axes]

        if detector_regions is not None:
            array = self.array.reshape(self.shape[:-2] + (-1,))[..., list(detector_regions)].sum(axis=-1)
            return Images(array=array, sampling=sampling, axes_metadata=self.axes_metadata[:-2], metadata=self.metadata)

        if radial_limits is None:
            radial_slice = slice(None)
        else:
            inner_index = int((radial_limits[0] - self.inner_angle) / self.radial_sampling)
            outer_index = int((radial_limits[1] - self.inner_angle) / self.radial_sampling)
            radial_slice = slice(inner_index, outer_index)

        if azimutal_limits is None:
            azimuthal_slice = slice(None)
        else:
            left_index = int(azimutal_limits[0] / self.radial_sampling)
            right_index = int(azimutal_limits[1] / self.radial_sampling)
            azimuthal_slice = slice(left_index, right_index)

        array = self.array[..., radial_slice, azimuthal_slice].sum(axis=(-2, -1))

        return Images(array=array, sampling=sampling, axes_metadata=self.axes_metadata[:-2], metadata=self.metadata)

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, axes_metadata=copy.deepcopy(self.axes_metadata),
                              metadata=copy.deepcopy(self.axes_metadata))

    def compute(self, **kwargs):
        array = self._array.compute(**kwargs)
        return self.__class__(array,
                              radial_sampling=self.radial_sampling,
                              azimuthal_sampling=self.azimuthal_sampling,
                              radial_offset=self._radial_offset,
                              azimuthal_offset=self._azimuthal_offset,
                              axes_metadata=self.axes_metadata,
                              metadata=self.metadata)

    def show(self, ax=None, min_azimuthal_division=np.pi / 20, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np

        array = self.array[(0,) * (len(self.shape) - 2)]

        repeat = int(self.azimuthal_sampling / min_azimuthal_division)
        r = np.pi / (4 * repeat) + self.azimuthal_offset
        azimuthal_grid = np.linspace(r, 2 * np.pi + r, self.shape[-1] * repeat, endpoint=False)

        d = (self.outer_angle - self.inner_angle) / 2 / self.shape[-2]
        radial_grid = np.linspace(self.inner_angle + d, self.outer_angle - d, self.shape[-2])

        z = np.repeat(array, repeat, axis=-1)
        r, th = np.meshgrid(radial_grid, azimuthal_grid)

        if ax is None:
            ax = plt.subplot(projection="polar")

        im = ax.pcolormesh(th, r, z.T, shading='auto', **kwargs)
        ax.set_rlim([0, self.outer_angle * 1.1])

        ax.grid()
        return ax, im
