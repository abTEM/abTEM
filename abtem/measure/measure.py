import copy
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, Tuple, TypeVar, Dict, List, Sequence

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import zarr
from ase import Atom

from abtem.basic.axes import HasAxesMetadata
from abtem.basic.backend import cp, asnumpy, get_array_module, get_ndimage_module
from abtem.basic.dask import HasDaskArray, requires_dask_array
from abtem.basic.fft import fft2_interpolate
from abtem.basic.interpolate import interpolate_bilinear
from abtem.measure.utils import polar_detector_bins, sum_run_length_encoded
from abtem.visualize.utils import domain_coloring, add_domain_coloring_cbar

if cp is not None:
    from abtem.basic.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
else:
    sum_run_length_encoded_cuda = None
    interpolate_bilinear_cuda = None

T = TypeVar('T', bound='AbstractMeasurement')


def _to_hyperspy_axes_metadata(axes_metadata, shape):
    hyperspy_axes = []
    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axes.append({'size': n})

        if 'sampling' in metadata:
            hyperspy_axes[-1]['scale'] = metadata['sampling']

        if 'units' in metadata:
            hyperspy_axes[-1]['units'] = metadata['units']

        if 'label' in metadata:
            hyperspy_axes[-1]['name'] = metadata['label']

        if 'offset' in metadata:
            hyperspy_axes[-1]['offset'] = metadata['offset']
        else:
            hyperspy_axes[-1]['offset'] = 0.

    return hyperspy_axes


def from_zarr(url):
    with zarr.open(url, mode='r') as f:
        axes_metadata = f.attrs['axes_metadata']
        metadata = f.attrs['metadata']
        kwargs = f.attrs['kwargs']
        cls = globals()[f.attrs['cls']]

    array = da.from_zarr(url, component='array')
    return cls(array, axes_metadata=axes_metadata, metadata=metadata, **kwargs)


class AbstractMeasurement(HasDaskArray, HasAxesMetadata, metaclass=ABCMeta):

    def __init__(self, array, axes_metadata, metadata, base_axes):
        self._array = array

        if axes_metadata is None:
            axes_metadata = []

        if metadata is None:
            metadata = {}

        self._axes_metadata = axes_metadata
        self._metadata = metadata
        self._base_axes = base_axes
        super().__init__(array)

        self._check_axes_metadata()

    # def squeeze(self):
    #
    #     shape = self.shape[:self.base_shape]
    #
    #     print(shape)
    #
    #
    #     #self._array =
    #

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        pass

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def axes_metadata(self) -> List[Dict]:
        return self._axes_metadata + self.base_axes_metadata

    @property
    def extra_axes_metadata(self) -> List[Dict]:
        return self._axes_metadata

    @property
    def base_shape(self) -> Tuple[float, ...]:
        return self.shape[-self.base_dimensions:]

    @property
    def base_dimensions(self) -> int:
        return len(self._base_axes)

    @property
    def dimensions(self) -> int:
        return len(self._array.shape)

    @property
    def collection_dimensions(self) -> int:
        return self.dimensions - self.base_dimensions

    @property
    def base_axes(self) -> Tuple[float, ...]:
        base_axes = ()
        for base_axis in self._base_axes:
            if base_axis < 0:
                base_axis = self.dimensions + base_axis

            base_axes += (base_axis,)
        return base_axes

    @property
    def collection_axes(self) -> Tuple[float, ...]:
        return tuple(range(self.collection_dimensions))

    def _validate_axes(self, axes):
        if isinstance(axes, Number):
            return (axes,)
        return axes

    def _check_is_base_axes(self, axes):
        axes = self._validate_axes(axes)
        return len(set(axes).intersection(self.base_axes)) > 0

    def _check_axes_metadata(self):
        if len(self.axes_metadata) != len(self.array.shape):
            raise RuntimeError()

    def mean(self, axes=None, split_every=2):
        return self._reduction(da.mean, axes=axes, split_every=split_every)

    def sum(self, axes=None, split_every=2):
        return self._reduction(da.sum, axes=axes, split_every=split_every)

    def std(self, axes=None, split_every=2):
        return self._reduction(da.std, axes=axes, split_every=split_every)

    def _reduction(self, reduction_func, axes=None, split_every=2):
        if axes is None:
            axes = self.collection_axes

        if self._check_is_base_axes(axes):
            raise RuntimeError('base axes cannot be reduced')

        axes_metadata = self._remove_axes_metadata(axes)[:-len(self.base_shape)]

        new_copy = self.copy(copy_array=False)
        new_copy._array = reduction_func(new_copy._array, axes, split_every=split_every)
        new_copy._axes_metadata = axes_metadata

        return new_copy

    def _get_measurements(self, items, **kwargs):
        if isinstance(items, Number):
            items = (items,)

        new_copy = self.copy(copy_array=False)

        new_copy._array = self._array[items]

        removed_axes = []
        for i, item in enumerate(items):
            if isinstance(item, Number):
                removed_axes.append(i)

        new_copy._axes_metadata = self._remove_axes_metadata(removed_axes)

        if self._check_is_base_axes(removed_axes):
            raise RuntimeError('base axes cannot be indexed')

        return new_copy

        # return self.__class__(array=array, axes_metadata=axes_metadata, metadata=self.metadata, **kwargs)

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

    def _to_zarr(self, url, compute=True, overwrite=False, **kwargs):
        with zarr.open(url, mode='w') as root:
            array = self.array.to_zarr(url, compute=compute, component='array', overwrite=overwrite)
            root.attrs['axes_metadata'] = self._axes_metadata
            root.attrs['metadata'] = self.metadata
            root.attrs['kwargs'] = kwargs
            root.attrs['cls'] = self.__class__.__name__
        return array

    @staticmethod
    def from_zarr(url):
        return from_zarr(url)

    @abstractmethod
    def to_hyperspy(self):
        pass

    @abstractmethod
    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False):
        pass

    def to_cpu(self):
        new_copy = self.copy(copy_array=False)
        new_copy._array = asnumpy(self._array)
        return new_copy

    @abstractmethod
    def copy(self, copy_array=True) -> T:
        pass


class Images(AbstractMeasurement):

    def __init__(self,
                 array: Union[da.core.Array, np.array],
                 sampling: Union[float, Tuple[float, float]],
                 axes_metadata: List[Dict] = None,
                 metadata: Dict = None):

        if isinstance(sampling, Number):
            sampling = (sampling,) * 2

        self._sampling = sampling
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-2, -1))

    @property
    def base_axes_metadata(self) -> List[Dict]:
        return [{'label': 'x', 'type': 'real-space', 'sampling': self.sampling[0]},
                {'label': 'y', 'type': 'real-space', 'sampling': self.sampling[1]}]

    @requires_dask_array
    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False, **kwargs):
        return self._to_zarr(url=url, overwrite=overwrite, sampling=self.sampling, compute=compute, **kwargs)

    def to_hyperspy(self):
        from hyperspy._signals.signal2d import Signal2D

        base_axes = [
            {'scale': self.sampling[1], 'units': 'Å', 'name': 'y', 'offset': 0., 'size': self.array.shape[1]},
            {'scale': self.sampling[0], 'units': 'Å', 'name': 'x', 'offset': 0., 'size': self.array.shape[0]}
            ]

        extra_axes = [{'size': n} for n in self.array.shape[:-2]]

        return Signal2D(self.array.T, axes=extra_axes + base_axes)

    def copy(self, copy_array=True) -> 'Images':
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, sampling=self.sampling, axes_metadata=copy.deepcopy(self._axes_metadata))

    @property
    def sampling(self) -> Tuple[float, float]:
        return self._sampling

    @property
    def coordinates(self):
        x = np.linspace(0., self.shape[-2] * self.sampling[0], self.shape[-2])
        y = np.linspace(0., self.shape[-1] * self.sampling[1], self.shape[-1])
        return x, y

    @property
    def extent(self) -> Tuple[float, float]:
        return (self.sampling[0] * self.base_shape[0], self.sampling[1] * self.base_shape[1])

    def interpolate(self,
                    sampling: Union[float, Tuple[float, float]] = None,
                    gpts: Union[int, Tuple[int, int]] = None,
                    method: str = 'fft',
                    boundary: str = 'periodic') -> 'Images':

        if method == 'fft' and boundary != 'periodic':
            raise ValueError()

        if sampling is None and gpts is None:
            raise ValueError()

        if gpts is None and sampling is not None:
            if np.isscalar(sampling):
                sampling = (sampling,) * 2
            gpts = tuple(int(np.ceil(l / d)) for d, l in zip(sampling, self.extent))
        elif gpts is not None:
            if np.isscalar(gpts):
                gpts = (gpts,) * 2
        else:
            raise ValueError()

        xp = get_array_module(self.array)

        sampling = (self.extent[0] / gpts[0], self.extent[1] / gpts[1])

        if self.is_lazy:
            array = dask.delayed(fft2_interpolate)(self.array, gpts)
            array = da.from_delayed(array, shape=self.shape[:-2] + gpts, meta=xp.array((), dtype=self.array.dtype))
        else:
            array = fft2_interpolate(self.array, gpts)

        return self.__class__(array, sampling=sampling, axes_metadata=self.axes_metadata[:-2], metadata=self.metadata)

    def interpolate_line(self,
                         start: Union[Tuple[float, float], Atom],
                         end: Union[Tuple[float, float], Atom] = None,
                         angle: float = 0.,
                         gpts: int = None,
                         sampling: float = None,
                         width: float = None,
                         margin: float = 0.,
                         interpolation_method: str = 'splinef2d') -> 'LineProfiles':
        """
        Interpolate 2d measurement along a line.

        Parameters
        ----------
        start : two float, Atom
            Start point on line [Å].
        end : two float, Atom, optional
            End point on line [Å].
        angle : float, optional
            The angle of the line. This is only used when an "end" is not give.
        gpts : int
            Number of grid points along line.
        sampling : float
            Sampling rate of grid points along line [1 / Å].
        width : float, optional
            The interpolation will be averaged across line of this width.
        margin : float, optional
            The line will be extended by this amount at both ends.
        interpolation_method : str, optional
            The interpolation method.

        Returns
        -------
        Measurement
            Line profile measurement.
        """

        # if (gpts is None) & (sampling is None):
        #    sampling = (measurement.calibrations[0].sampling + measurement.calibrations[1].sampling) / 2.

        from abtem.waves.scan import LineScan

        if (sampling is None) and (gpts is None):
            sampling = min(self.sampling)

        scan = LineScan(start=start, end=end, angle=angle, gpts=gpts, sampling=sampling, margin=margin)

        start = scan.margin_start
        end = scan.margin_end

        positions = scan.get_positions() / self.sampling

        from scipy.ndimage import map_coordinates

        def interpolate_1d_from_2d(array, positions):
            old_shape = array.shape
            array = array.reshape((-1,) + array.shape[-2:])
            output = np.zeros((array.shape[0], positions.shape[0]))

            for i in range(array.shape[0]):
                map_coordinates(array[i], positions.T, output=output[i])

            output = output.reshape(old_shape[:-2] + (output.shape[-1],))

            return output

        array = self.array.map_blocks(interpolate_1d_from_2d,
                                      positions=positions,
                                      drop_axis=(self.num_ensemble_axes, self.num_ensemble_axes + 1),
                                      chunks=self.array.chunks[:-2] + ((positions.shape[0],),),
                                      new_axis=(self.num_ensemble_axes,),
                                      meta=np.array((), dtype=np.float32))

        return LineProfiles(array=array, start=scan.start, end=scan.end, axes_metadata=self._axes_metadata)

    def is_compatible(self, other) -> bool:

        if self.shape != other.shape:
            return False

        if self.sampling != self.sampling:
            return False

        if self.axes_metadata != other.axes_metadata:
            return False

        return True

    def subtract(self, other) -> 'Images':
        self.is_compatible(other)
        return self.__class__(self.array - other.array,
                              sampling=self.sampling,
                              axes_metadata=copy.copy(self._axes_metadata),
                              metadata=copy.copy(self.metadata))

    def tile(self, reps) -> 'Images':
        new_array = np.tile(self.array, reps)
        return self.__class__(new_array, sampling=self.sampling, axes_metadata=copy.copy(self._axes_metadata),
                              metadata=copy.copy(self.metadata))

    def gaussian_filter(self, sigma: Union[float, Tuple[float, float]], boundary: str = 'periodic'):
        xp = get_array_module(self.array)
        ndimage = get_ndimage_module(self._array)

        gaussian_filter = ndimage.gaussian_filter

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sigma = (0,) * (len(self.shape) - 2) + tuple(s / d for s, d in zip(sigma, self.sampling))

        array = self.array.map_overlap(gaussian_filter,
                                       sigma=sigma,
                                       boundary=boundary,
                                       depth=(0,) * (len(self.shape) - 2) + (int(np.ceil(4.0 * max(sigma))),) * 2,
                                       meta=xp.array((), dtype=xp.float32))

        return self.__class__(array, sampling=self.sampling, axes_metadata=self.axes_metadata, metadata=self.metadata)

    def show(self, ax=None, cbar=False, power=1., **kwargs):
        self.compute(pbar=False)

        if ax is None:
            ax = plt.subplot()

        slic = (0,) * self.collection_dimensions

        array = asnumpy(self._array)[slic].T ** power

        if np.iscomplexobj(array):
            colored_array = domain_coloring(array)
        else:
            colored_array = array

        im = ax.imshow(colored_array, extent=[0, self.extent[0], 0, self.extent[1]], origin='lower', **kwargs)
        ax.set_xlabel('x [Å]')
        ax.set_ylabel('y [Å]')

        if cbar:
            if np.iscomplexobj(array):
                abs_array = np.abs(array)
                add_domain_coloring_cbar(ax, (abs_array.min(), abs_array.max()))
            else:
                plt.colorbar(im, ax=ax)

        return ax, im


class LineProfiles(AbstractMeasurement):

    def __init__(self,
                 array,
                 start: Tuple[float, float] = None,
                 end: Tuple[float, float] = None,
                 sampling: float = None,
                 axes_metadata=None,
                 metadata=None):
        from abtem.waves.scan import LineScan

        if start is None:
            start = (0., 0.)

        if end is None:
            end = (start[0] + len(array) * sampling, start[1])

        self._linescan = LineScan(start=start, end=end, sampling=sampling, gpts=array.shape[-1])
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-1,))

    @property
    def start(self) -> Tuple[float, float]:
        return self._linescan.start

    @property
    def end(self) -> Tuple[float, float]:
        return self._linescan.end

    @property
    def extent(self) -> float:
        return self._linescan.extent[0]

    @property
    def sampling(self) -> float:
        return self._linescan.sampling[0]

    @property
    def base_axes_metadata(self) -> list:
        return [{'sampling': self.sampling}]

    def tile(self):
        new_copy = self.copy(copy_array=False)
        # new_copy._array

    def to_hyperspy(self):
        from hyperspy._signals.signal1d import Signal1D
        return Signal1D(self.array, axes=_to_hyperspy_axes_metadata(self.axes_metadata, self.shape)).as_lazy()

    @requires_dask_array
    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False):
        self._to_zarr(url=url, compute=compute, overwrite=overwrite, sampling=self.sampling)

    def show(self, ax=None, title='', label=None):
        if ax is None:
            ax = plt.subplot()

        ax.plot(self.array.reshape((-1, self.array.shape[-1])).T, label=label)
        ax.set_title(title)

        return ax

    def copy(self, copy_array=True) -> 'LineProfiles':
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array, start=self.start, end=self.end,
                              axes_metadata=copy.deepcopy(self.extra_axes_metadata))


class RadialFourierSpaceLineProfiles(LineProfiles):

    def __init__(self, array, sampling, axes_metadata=None, metadata=None):
        super().__init__(array=array, start=(0., 0.), end=(0., array.shape[-1] * sampling), sampling=sampling,
                         axes_metadata=axes_metadata, metadata=metadata)

    def show(self, ax=None, title='', **kwargs):
        if ax is None:
            ax = plt.subplot()

        x = np.linspace(0., len(self.array) * self.sampling * 1000, len(self.array))

        p = ax.plot(x, self.array, **kwargs)
        ax.set_xlabel('Scattering angle [mrad]')
        ax.set_title(title)
        return ax, p


class DiffractionPatterns(AbstractMeasurement):

    def __init__(self,
                 array,
                 angular_sampling: Tuple[float, float],
                 fftshift: bool = False,
                 axes_metadata=None,
                 metadata=None):

        self._fftshift = fftshift
        self._angular_sampling = (float(angular_sampling[0]), float(angular_sampling[1]))
        super().__init__(array=array, axes_metadata=axes_metadata, metadata=metadata, base_axes=(-2, -1))

    @property
    def base_axes_metadata(self):
        return [{'label': 'scattering_angle_x', 'type': 'fourier_space', 'sampling': self.angular_sampling[0],
                 'offset': self.fourier_space_extent[0][0], 'units': 'mrad'},
                {'label': 'scattering_angle_y', 'type': 'fourier_space', 'sampling': self.angular_sampling[1],
                 'offset': self.fourier_space_extent[1][0], 'units': 'mrad'}]

    @requires_dask_array
    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False, **kwargs):
        return self._to_zarr(url=url,
                             compute=compute,
                             overwrite=overwrite,
                             angular_sampling=self.angular_sampling,
                             fftshift=self.fftshift,
                             **kwargs)

    def to_hyperspy(self):
        from hyperspy._signals.signal2d import Signal2D
        return Signal2D(self.array, axes=_to_hyperspy_axes_metadata(self.axes_metadata, self.shape)).as_lazy()

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array,
                              angular_sampling=self.angular_sampling,
                              axes_metadata=copy.deepcopy(self.extra_axes_metadata),
                              metadata=copy.deepcopy(self.metadata),
                              fftshift=self.fftshift)

    def __getitem__(self, items):
        return self._get_measurements(items, fftshift=self.fftshift)

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def angular_sampling(self) -> Tuple[float, float]:
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

    def interpolate(self, new_sampling):

        def bilinear_nodes_and_weight(old_shape, new_shape, old_angular_sampling, new_angular_sampling, xp):
            nodes = []
            weights = []

            old_sampling = (1 / old_angular_sampling[0] / old_shape[0],
                            1 / old_angular_sampling[1] / old_shape[1])

            new_sampling = (1 / new_angular_sampling[0] / new_shape[0],
                            1 / new_angular_sampling[1] / new_shape[1])

            for n, m, r, d in zip(old_shape, new_shape, old_sampling, new_sampling):
                k = xp.fft.fftshift(xp.fft.fftfreq(n, r).astype(xp.float32))
                k_new = xp.fft.fftshift(xp.fft.fftfreq(m, d).astype(xp.float32))
                distances = k_new[None] - k[:, None]
                distances[distances < 0.] = np.inf
                w = distances.min(0) / (k[1] - k[0])
                w[w == np.inf] = 0.
                nodes.append(distances.argmin(0))
                weights.append(w)

            v, u = nodes
            vw, uw = weights
            v, u, vw, uw = xp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])
            return v, u, vw, uw

        def resampled_gpts(new_sampling, gpts, angular_sampling):
            if new_sampling == 'uniform':
                scale_factor = (angular_sampling[0] / max(angular_sampling),
                                angular_sampling[1] / max(angular_sampling))

                new_gpts = (int(np.ceil(gpts[0] * scale_factor[0])),
                            int(np.ceil(gpts[1] * scale_factor[1])))

                if np.abs(new_gpts[0] - new_gpts[1]) <= 2:
                    new_gpts = (min(new_gpts),) * 2

                new_angular_sampling = (angular_sampling[0] / scale_factor[0],
                                        angular_sampling[1] / scale_factor[1])

            else:
                raise RuntimeError('')

            return new_gpts, new_angular_sampling

        xp = get_array_module(self.array)

        new_gpts, new_angular_sampling = resampled_gpts(new_sampling, self.array.shape[-2:], self.angular_sampling)

        v, u, vw, uw = bilinear_nodes_and_weight(self.array.shape[-2:],
                                                 new_gpts,
                                                 self.angular_sampling,
                                                 new_angular_sampling,
                                                 xp)

        return interpolate_bilinear(self.array, v, u, vw, uw)

    def _check_max_angle(self, angle):
        if (angle > self.max_angles[0]) or (angle > self.max_angles[1]):
            raise RuntimeError(
                f'integration angle exceeds the maximum simulated angle ({angle} > {min(self.max_angles)})')

    def gaussian_filter(self, sigma: Union[float, Tuple[float, float]], boundary: str = 'periodic'):
        xp = get_array_module(self.array)
        ndimage = get_ndimage_module(self._array)

        gaussian_filter = ndimage.gaussian_filter

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sampling = [self.axes_metadata[axis]['sampling'] for axis in self.scan_axes]

        sigma = self.num_ensemble_axes * (0,) + tuple(s / d for s, d in zip(sigma, sampling)) + (0,) * 2

        array = self.array.map_overlap(gaussian_filter,
                                       sigma=sigma,
                                       boundary=boundary,
                                       depth=self.num_ensemble_axes * (0,) + (int(np.ceil(4.0 * max(sigma))),) * 2 +
                                             (0,) * 2,
                                       meta=xp.array((), dtype=xp.float32))

        return self.__class__(array, angular_sampling=self.angular_sampling, axes_metadata=self.axes_metadata,
                              metadata=self.metadata, fftshift=self.fftshift)

    def polar_binning(self, nbins_radial, nbins_azimuthal, inner, outer, rotation=0.):
        self._check_max_angle(outer)
        xp = get_array_module(self.array)

        def radial_binning(array, nbins_radial, nbins_azimuthal):
            xp = get_array_module(array)

            indices = polar_detector_bins(gpts=array.shape[-2:],
                                          sampling=self.angular_sampling,
                                          inner=inner,
                                          outer=outer,
                                          nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          fftshift=self.fftshift,
                                          rotation=rotation,
                                          return_indices=True)

            separators = xp.concatenate((xp.array([0]), xp.cumsum(xp.array([len(i) for i in indices]))))

            new_shape = array.shape[:-2] + (nbins_radial, nbins_azimuthal)

            array = array.reshape((-1, array.shape[-2] * array.shape[-1],))[..., np.concatenate(indices)]

            result = xp.zeros((array.shape[0], len(indices),), dtype=xp.float32)

            if xp is cp:
                sum_run_length_encoded_cuda(array, result, separators)

            else:
                sum_run_length_encoded(array, result, separators)

            return result.reshape(new_shape)

        if self.is_lazy:
            array = self.array.map_blocks(radial_binning, nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                          chunks=self.array.chunks[:-2] + ((nbins_radial,), (nbins_azimuthal,),),
                                          new_axis=(len(self.shape) - 2, len(self.shape) - 1,),
                                          meta=xp.array((), dtype=xp.float32))
        else:

            array = radial_binning(self.array, nbins_radial=nbins_radial, nbins_azimuthal=nbins_azimuthal)

        radial_sampling = (outer - inner) / nbins_radial
        azimuthal_sampling = 2 * np.pi / nbins_azimuthal

        axes_metadata = self.axes_metadata[:-2]

        return PolarMeasurements(array,
                                 radial_sampling=radial_sampling,
                                 azimuthal_sampling=azimuthal_sampling,
                                 radial_offset=inner,
                                 azimuthal_offset=rotation,
                                 axes_metadata=axes_metadata,
                                 metadata=self.metadata)

    def radial_binning(self, step_size=1., inner=0., outer=None):
        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)
        return self.polar_binning(nbins_radial, 1, inner, outer)

    def integrate_radial(self, inner, outer):
        self._check_max_angle(outer)

        xp = get_array_module(self.array)

        def integrate_fourier_space(array, sampling):

            bins = polar_detector_bins(gpts=array.shape[-2:],
                                       sampling=sampling,
                                       inner=inner,
                                       outer=outer,
                                       nbins_radial=1,
                                       nbins_azimuthal=1,
                                       fftshift=self.fftshift)

            xp = get_array_module(array)
            bins = xp.asarray(bins, dtype=xp.float32)

            return xp.sum(array * (bins == 0), axis=(-2, -1))

        if self.is_lazy:
            integrated_intensity = self.array.map_blocks(integrate_fourier_space,
                                                         sampling=self.angular_sampling,
                                                         drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                                         meta=xp.array((), dtype=xp.float32))
        else:
            integrated_intensity = integrate_fourier_space(self.array, sampling=self.angular_sampling)

        if len(self.scan_axes) > 0:
            sampling = [self.axes_metadata[axis]['sampling'] for axis in self.scan_axes]
        else:
            sampling = (1., 1.)

        if len(self.real_space_axes) == 1:
            return LineProfiles(integrated_intensity, sampling=sampling[0], axes_metadata=self.axes_metadata[:-3])
        else:
            return Images(integrated_intensity, sampling=sampling, axes_metadata=self.axes_metadata[:-4])

    def integrated_center_of_mass(self) -> Images:
        def intgrad2d(gradient, sampling):
            gx, gy = gradient
            (nx, ny) = gx.shape
            ikx = np.fft.fftfreq(nx, d=sampling[0])
            iky = np.fft.fftfreq(ny, d=sampling[1])
            grid_ikx, grid_iky = np.meshgrid(ikx, iky, indexing='ij')
            k = grid_ikx ** 2 + grid_iky ** 2
            k[k == 0] = 1e-12
            That = (np.fft.fft2(gx) * grid_ikx + np.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
            T = np.real(np.fft.ifft2(That))
            T -= T.min()
            return T

        com = self.center_of_mass()

        sampling = (self.axes_metadata[self.scan_axes[0]]['sampling'],
                    self.axes_metadata[self.scan_axes[1]]['sampling'])

        icom = intgrad2d((com.array.real, com.array.imag), sampling)

        return Images(array=icom, sampling=sampling, axes_metadata=self._axes_metadata[:-2], metadata=self.metadata)

    def center_of_mass(self) -> Images:
        x, y = self.angular_coordinates()

        com_x = (self.array * x[:, None]).sum(axis=(-2, -1))
        com_y = (self.array * y[None]).sum(axis=(-2, -1))

        sampling = (self.axes_metadata[self.scan_axes[0]]['sampling'],
                    self.axes_metadata[self.scan_axes[1]]['sampling'])

        com = com_x + 1.j * com_y

        return Images(array=com, sampling=sampling, axes_metadata=self._axes_metadata[:-2], metadata=self.metadata)

    def angular_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        xp = get_array_module(self.array)
        alpha_x = xp.linspace(self.fourier_space_extent[0][0], self.fourier_space_extent[0][1], self.shape[-2])
        alpha_y = xp.linspace(self.fourier_space_extent[1][0], self.fourier_space_extent[1][1], self.shape[-1])
        return alpha_x, alpha_y

    def block_direct(self, radius: float = None) -> 'DiffractionPatterns':

        if radius is None:
            radius = max(self.angular_sampling) * 1.1

        def block_direct(array):
            alpha_x, alpha_y = self.angular_coordinates()
            alpha = alpha_x[:, None] ** 2 + alpha_y[None] ** 2
            block = alpha > radius ** 2
            return array * block

        xp = get_array_module(self.array)

        if self.is_lazy:
            array = da.from_delayed(dask.delayed(block_direct)(self.array), shape=self.shape,
                                    meta=xp.array((), dtype=xp.float32))
        else:
            array = block_direct(self.array)

        return self.__class__(array, angular_sampling=self.angular_sampling, axes_metadata=self._axes_metadata,
                              metadata=self.metadata, fftshift=self.fftshift)

    def show(self, ax=None, cbar=False, power=1., **kwargs):
        self.compute()

        if ax is None:
            ax = plt.subplot()

        slic = (0,) * self.collection_dimensions
        extent = self.fourier_space_extent[0] + self.fourier_space_extent[1]

        array = asnumpy(self._array)[slic].T ** power

        im = ax.imshow(array, extent=extent, origin='lower', **kwargs)
        ax.set_xlabel('Scattering angle x [mrad]')
        ax.set_ylabel('Scattering angle y [mrad]')

        if cbar:
            plt.colorbar(im, ax=ax)

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
    def base_axes_metadata(self) -> list:
        return [{'label': 'scattering_angle_x', 'type': 'fourier_space', 'sampling': self.radial_sampling,
                 'offset': self.radial_offset, 'units': 'mrad'},
                {'label': 'scattering_angle_y', 'type': 'fourier_space', 'sampling': self.azimuthal_sampling,
                 'offset': self.azimuthal_offset, 'units': 'rad'}]

    def to_hyperspy(self):
        pass

    @requires_dask_array
    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False):
        return self._to_zarr(url=url,
                             compute=compute,
                             overwrite=overwrite,
                             radial_sampling=self.radial_sampling,
                             azimuthal_sampling=self.azimuthal_sampling,
                             radial_offset=self.radial_offset,
                             azimuthal_offset=self.azimuthal_offset)

    @property
    def radial_offset(self) -> float:
        return self._radial_offset

    @property
    def outer_angle(self) -> float:
        return self._radial_offset + self.radial_sampling * self.shape[-2]

    @property
    def radial_sampling(self) -> float:
        return self._radial_sampling

    @property
    def azimuthal_sampling(self) -> float:
        return self._azimuthal_sampling

    @property
    def azimuthal_offset(self) -> float:
        return self._azimuthal_offset

    def _check_radial_angle(self, angle):
        if angle < self.radial_offset or angle > self.outer_angle:
            raise RuntimeError()

    def integrate_radial(self, inner, outer) -> Union[Images, LineProfiles]:
        return self.integrate(radial_limits=(inner, outer))

    def integrate(self,
                  radial_limits: Tuple[float, float] = None,
                  azimutal_limits: Tuple[float, float] = None,
                  detector_regions: Sequence[int] = None) -> Union[Images, LineProfiles]:

        sampling = (self.axes_metadata[self.scan_axes[0]]['sampling'],
                    self.axes_metadata[self.scan_axes[1]]['sampling'])

        if detector_regions is not None:
            array = self.array.reshape(self.shape[:-2] + (-1,))[..., list(detector_regions)].sum(axis=-1)
            return Images(array=array, sampling=sampling, axes_metadata=self.axes_metadata[:-4], metadata=self.metadata)

        if radial_limits is None:
            radial_slice = slice(None)
        else:
            inner_index = int((radial_limits[0] - self.radial_offset) / self.radial_sampling)
            outer_index = int((radial_limits[1] - self.radial_offset) / self.radial_sampling)
            radial_slice = slice(inner_index, outer_index)

        if azimutal_limits is None:
            azimuthal_slice = slice(None)
        else:
            left_index = int(azimutal_limits[0] / self.radial_sampling)
            right_index = int(azimutal_limits[1] / self.radial_sampling)
            azimuthal_slice = slice(left_index, right_index)

        array = self.array[..., radial_slice, azimuthal_slice].sum(axis=(-2, -1))

        return Images(array=array, sampling=sampling, axes_metadata=self.axes_metadata[:-4], metadata=self.metadata)

    def copy(self, copy_array=True):
        if copy_array:
            array = self._array.copy()
        else:
            array = self._array
        return self.__class__(array,
                              radial_offset=self.radial_offset,
                              radial_sampling=self.radial_sampling,
                              azimuthal_offset=self.azimuthal_offset,
                              azimuthal_sampling=self.azimuthal_sampling,
                              axes_metadata=copy.deepcopy(self._axes_metadata),
                              metadata=copy.deepcopy(self._axes_metadata))

    def compute(self, **kwargs):
        array = self._array.compute(**kwargs)
        return self.__class__(array,
                              radial_sampling=self.radial_sampling,
                              azimuthal_sampling=self.azimuthal_sampling,
                              radial_offset=self._radial_offset,
                              azimuthal_offset=self._azimuthal_offset,
                              axes_metadata=self._axes_metadata,
                              metadata=self.metadata)

    def show(self, ax=None, min_azimuthal_division=np.pi / 20, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np

        array = self.array[(0,) * (len(self.shape) - 2)]

        repeat = int(self.azimuthal_sampling / min_azimuthal_division)
        r = np.pi / (4 * repeat) + self.azimuthal_offset
        azimuthal_grid = np.linspace(r, 2 * np.pi + r, self.shape[-1] * repeat, endpoint=False)

        d = (self.outer_angle - self.radial_offset) / 2 / self.shape[-2]
        radial_grid = np.linspace(self.radial_offset + d, self.outer_angle - d, self.shape[-2])

        z = np.repeat(array, repeat, axis=-1)
        r, th = np.meshgrid(radial_grid, azimuthal_grid)

        if ax is None:
            ax = plt.subplot(projection="polar")

        im = ax.pcolormesh(th, r, z.T, shading='auto', **kwargs)
        ax.set_rlim([0, self.outer_angle * 1.1])

        ax.grid()
        return ax, im
