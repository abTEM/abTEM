import warnings
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, TypeVar, Dict, List, Sequence, Type, TYPE_CHECKING

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from ase import Atom
from matplotlib.axes import Axes

from abtem.core.array import HasArray
from abtem.core.axes import HasAxes, RealSpaceAxis, AxisMetadata, FourierSpaceAxis, LinearAxis, NonLinearAxis, \
    SampleAxis, OrdinalAxis
from abtem.core.backend import cp, get_array_module, get_ndimage_module, copy_to_device
from abtem.core.complex import abs2
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft2, fft_interpolate
from abtem.core.grid import adjusted_gpts
from abtem.core.interpolate import interpolate_bilinear
from abtem.core.utils import CopyMixin, EqualityMixin
from abtem.measure.utils import polar_detector_bins, sum_run_length_encoded
from abtem.potentials.temperature import validate_seeds
from abtem.visualize.mpl import show_measurement_2d_exploded

if cp is not None:
    from abtem.core.cuda import sum_run_length_encoded as sum_run_length_encoded_cuda
    from abtem.core.cuda import interpolate_bilinear as interpolate_bilinear_cuda
else:
    sum_run_length_encoded_cuda = None
    interpolate_bilinear_cuda = None

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import hyperspy.api as hs

        Signal2D = hs.signals.Signal2D
        Signal1D = hs.signals.Signal1D
except ImportError:
    Signal2D = None
    Signal1D = None

if TYPE_CHECKING:
    from abtem.waves.base import WavesLikeMixin

T = TypeVar('T', bound='AbstractMeasurement')

missing_hyperspy_message = 'This functionality of abTEM requires hyperspy, see https://hyperspy.org/.'


def _to_hyperspy_axes_metadata(axes_metadata, shape):
    hyperspy_axes = []

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    for metadata, n in zip(axes_metadata, shape):
        hyperspy_axes.append({'size': n})

        axes_mapping = {'sampling': 'scale',
                        'units': 'units',
                        'label': 'name',
                        'offset': 'offset'
                        }

        if isinstance(metadata, NonLinearAxis):
            # TODO : when hyperspy supports arbitrary (non-uniform) DataAxis this should be updated

            if len(metadata.values) > 1:
                sampling = metadata.values[1] - metadata.values[0]
            else:
                sampling = 1.

            metadata = LinearAxis(label=metadata.label,
                                  units=metadata.units,
                                  sampling=sampling,
                                  offset=metadata.values[0])

        for attr, mapped_attr in axes_mapping.items():
            if hasattr(metadata, attr):
                hyperspy_axes[-1][mapped_attr] = getattr(metadata, attr)

    return hyperspy_axes


def scanned_measurement_type(measurement: Union['AbstractMeasurement', 'WavesLikeMixin']) \
        -> Type['AbstractMeasurement']:
    if len(measurement.scan_axes) == 0:
        return SinglePointMeasurement

    elif len(measurement.scan_axes) == 1:
        return LineProfiles

    elif len(measurement.scan_axes) == 2:
        return Images

    else:
        raise RuntimeError(f'no measurement type for {measurement.__class__} with {len(measurement.scan_axes)} scan '
                           f'axes')


class AbstractMeasurement(HasArray, HasAxes, EqualityMixin, CopyMixin, metaclass=ABCMeta):

    def __init__(self, array, ensemble_axes_metadata, metadata, allow_complex=False,
                 allow_base_axis_chunks=False):
        #

        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []

        if metadata is None:
            metadata = {}

        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._metadata = metadata

        self._array = array

        self.check_axes_metadata()

        # if len(array.shape) < len(self.base_shape):
        #     raise RuntimeError(f'array dim smaller than base dim of measurement type {self.__class__}')
        #
        # if not allow_complex:
        #     if np.iscomplexobj(array):
        #         raise RuntimeError(f'complex dtype not implemented for {self.__class__}')
        #
        # if not allow_base_axis_chunks:
        #     if self.is_lazy and (not all(len(chunks) == 1 for chunks in array.chunks[-2:])):
        #         raise RuntimeError(f'chunks not allowed in base axes of {self.__class__}')

    def iterate_ensemble(self, keep_dims: bool = False):
        for i in np.ndindex(self.ensemble_shape):
            yield i, self.get_items(i, keep_dims=keep_dims)

    @property
    def ensemble_axes_metadata(self):
        return self._ensemble_axes_metadata

    @property
    def energy(self):
        if not 'energy' in self.metadata.keys():
            raise RuntimeError('energy not in measurement metadata')
        return self.metadata['energy']

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    # def scan_from_metadata(self):
    #     start = ()
    #     end = ()
    #     gpts = ()
    #     endpoint = ()
    #     for n, metadata in zip(self.scan_shape, self.scan_axes_metadata):
    #         start += (metadata.offset,)
    #         end += (metadata.offset + metadata.sampling * n,)
    #         gpts += (n,)
    #         endpoint += (metadata.endpoint,)
    #
    #     if len(start) == 2:
    #         return GridScan(start=start, end=end, gpts=gpts, endpoint=endpoint)  # noqa

    def scan_positions(self):
        positions = ()
        for n, metadata in zip(self.scan_shape, self.scan_axes_metadata):
            positions += (
                np.linspace(metadata.offset, metadata.offset + metadata.sampling * n, n, endpoint=metadata.endpoint),)
        return positions

    def scan_extent(self):
        extent = ()
        for n, metadata in zip(self.scan_shape, self.scan_axes_metadata):
            extent += (metadata.sampling * n,)
        return extent

    @property
    @abstractmethod
    def base_axes_metadata(self) -> list:
        pass

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def dimensions(self) -> int:
        return len(self._array.shape)

    # def check_is_compatible(self, other: 'AbstractMeasurement'):
    #     if not isinstance(other, self.__class__):
    #         raise RuntimeError(f'Incompatible measurement types ({self.__class__} is not {other.__class__})')
    #
    #     if self.shape != other.shape:
    #         raise RuntimeError()
    #
    #     for (key, value), (other_key, other_value) in zip(self.copy_kwargs(exclude=('array',)).items(),
    #                                                       other.copy_kwargs(exclude=('array',)).items()):
    #         if np.any(value != other_value):
    #             raise RuntimeError(f'{key}, {other_key} {value} {other_value}')

    def relative_difference(self, other, min_relative_tol=0.):
        difference = self - other

        xp = get_array_module(self.array)

        # if min_relative_tol > 0.:
        valid = xp.abs(self.array) >= min_relative_tol * self.array.max()
        difference._array[valid] /= self.array[valid]
        difference._array[valid == 0] = difference.array.min()
        # else:
        #    difference._array[:] /= self.array

        return difference

    def power(self, number):
        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = self.array ** number
        return self.__class__(**kwargs)

    @abstractmethod
    def from_array_and_metadata(self, array, axes_metadata, metadata) -> 'T':
        pass

    def reduce_ensemble(self) -> 'T':
        return self.mean(axes=self._ensemble_axes_to_reduce())

    def _apply_element_wise_func(self, func: callable) -> 'T':
        d = self.copy_kwargs(exclude=('array',))
        d['array'] = func(self.array)
        return self.__class__(**d)

    @property
    @abstractmethod
    def _area_per_pixel(self):
        pass

    @staticmethod
    def _add_poisson_noise(array, seed, total_dose, block_info=None):
        xp = get_array_module(array)

        if block_info is not None:
            chunk_index = np.ravel_multi_index(block_info[0]['chunk-location'], block_info[0]['num-chunks'])
        else:
            chunk_index = 0

        rng = xp.random.default_rng(seed + chunk_index)
        randomized_seed = int(rng.integers(np.iinfo(np.int32).max))  # fixes strange cupy bug

        rng = xp.random.RandomState(seed=randomized_seed)

        return rng.poisson(xp.clip(array, a_min=0., a_max=None) * total_dose).astype(xp.float32)

    def poisson_noise(self,
                      dose_per_area: float = None,
                      total_dose: float = None,
                      samples: int = 1,
                      seed: int = None):
        """
        Add Poisson noise to the DiffractionPattern.

        Parameters
        ----------
        dose_per_area : float, optional
            The irradiation dose in electrons per Å^2. Provide either "total_dose" or "dose_per_area".
        total_dose : float, optional
            The irradiation dose per diffraction pattern. Provide either "total_dose" or "dose_per_area".
        samples : int, optional
            The number of samples to draw from the poisson distribution. If this is greater than 1, an additional
            ensemble axis will be added to the measurement.
        seed : int, optional
            Seed the random number generator.

        Returns
        -------
        noisy_diffraction_patterns: DiffractionPatterns
            The noisy diffraction patterns.
        """

        wrong_dose_error = RuntimeError('provide one of "dose_per_area" or "total_dose"')

        if dose_per_area is not None:
            if total_dose is not None:
                raise wrong_dose_error

            total_dose = self._area_per_pixel * dose_per_area

        elif total_dose is not None:
            if dose_per_area is not None:
                raise wrong_dose_error

        else:
            raise wrong_dose_error

        xp = get_array_module(self.array)

        seeds = validate_seeds(seed, samples)

        arrays = []
        for seed in seeds:
            if self.is_lazy:

                arrays.append(
                    self.array.map_blocks(self._add_poisson_noise, total_dose=total_dose, seed=seed,
                                          meta=xp.array((), dtype=xp.float32)))
            else:
                arrays.append(self._add_poisson_noise(self.array, total_dose=total_dose, seed=seed))

        if len(seeds) > 1:
            if self.is_lazy:
                arrays = da.stack(arrays)
            else:
                arrays = xp.stack(arrays)
            axes_metadata = [SampleAxis(label='sample')]
        else:
            arrays = arrays[0]
            axes_metadata = []

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = arrays
        kwargs['ensemble_axes_metadata'] = axes_metadata + kwargs['ensemble_axes_metadata']
        return self.__class__(**kwargs)

    @abstractmethod
    def to_hyperspy(self):
        pass


def interpolate_stack(array, positions, mode, order, **kwargs):
    map_coordinates = get_ndimage_module(array).map_coordinates
    xp = get_array_module(array)

    positions_shape = positions.shape
    positions = positions.reshape((-1, 2))

    old_shape = array.shape
    array = array.reshape((-1,) + array.shape[-2:])
    array = xp.pad(array, ((0, 0), (2 * order,) * 2, (2 * order,) * 2), mode=mode)

    positions = positions + 2 * order
    output = xp.zeros((array.shape[0], positions.shape[0]), dtype=np.float32)

    for i in range(array.shape[0]):
        map_coordinates(array[i], positions.T, output=output[i], order=order, **kwargs)

    output = output.reshape(old_shape[:-2] + positions_shape[:-1])
    return output


class Images(AbstractMeasurement):
    _base_dims = 2

    def __init__(self,
                 array: Union[da.core.Array, np.array],
                 sampling: Union[float, Tuple[float, float]],
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: Dict = None):

        """
        A collection of 2d images such as images from HRTEM or STEM-ADF. The complex valued images may be used to
        represent reconstructed phase.

        Parameters
        ----------
        array : ndarray
            2D or greater array containing data with `float` type or ´complex type´. The second-to-last and last
            dimensions are the image y and x axis, respectively.
        sampling : two float
            Lateral sampling of images in x and y [Å].
        ensemble_axes_metadata : list of AxisMetadata, optional
            Metadata associated with an ensemble axis.
        metadata : dict, optional
            A dictionary defining simulation metadata.
        """

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = float(sampling[0]), float(sampling[1])

        self._sampling = sampling

        super().__init__(array=array,
                         ensemble_axes_metadata=ensemble_axes_metadata,
                         metadata=metadata,
                         allow_complex=True,
                         allow_base_axis_chunks=True)

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None) -> 'Images':

        real_space_axes = tuple(i for i, axis in enumerate(axes_metadata) if isinstance(axis, RealSpaceAxis))

        if len(real_space_axes) < 2:
            raise RuntimeError()

        scan_axes_metadata = [axes_metadata[i] for i in real_space_axes[-2:]]

        other_axes_metadata = [axes_metadata[i] for i, metadata in enumerate(axes_metadata)
                               if i not in real_space_axes[-2:]]

        sampling = (scan_axes_metadata[-2].sampling, scan_axes_metadata[-1].sampling)

        return cls(array, sampling=sampling, ensemble_axes_metadata=other_axes_metadata, metadata=metadata)

    @property
    def _area_per_pixel(self):
        return np.prod(self.sampling)

    @property
    def sampling(self) -> Tuple[float, float]:
        """ Sampling of images in x and y [Å]. """
        return self._sampling

    @property
    def extent(self) -> Tuple[float, float]:
        """ Extent of images in x and y [Å]. """
        return self.sampling[0] * self.base_shape[0], self.sampling[1] * self.base_shape[1]

    @property
    def coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Coordinates of pixels in x and y [Å]. """
        x = np.linspace(0., self.shape[-2] * self.sampling[0], self.shape[-2])
        y = np.linspace(0., self.shape[-1] * self.sampling[1], self.shape[-1])
        return x, y

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [RealSpaceAxis(label='x', sampling=self.sampling[0], units='Å'),
                RealSpaceAxis(label='y', sampling=self.sampling[1], units='Å')]

    def _check_is_complex(self):
        if not np.iscomplexobj(self.array):
            raise RuntimeError('function not implemented for non-complex image')

    def real(self):
        """ Calculate the phase from complex images. """
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).real)

    def imag(self):
        """ Calculate the phase from complex images. """
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).real)

    def angle(self):
        """ Calculate the phase from complex images. """
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).angle)

    def abs(self):
        """ Calculate the absolute value from complex images. """
        self._check_is_complex()
        return self._apply_element_wise_func(get_array_module(self.array).abs)

    def intensity(self):
        """ Calculate the absolute square from complex images. """
        self._check_is_complex()
        return self._apply_element_wise_func(abs2)

    def integrate_gradient(self):
        """
        Calculate integrated gradients. Requires complex images where the real and imaginary represents the x and y
        component of a gradient.

        Returns
        -------
        intgrad_images : Images
        """
        self._check_is_complex()
        if self.is_lazy:
            xp = get_array_module(self.array)
            array = self.array.rechunk(self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],)))
            array = array.map_blocks(integrate_gradient_2d, sampling=self.sampling, meta=xp.array((), dtype=np.float32))
        else:
            array = integrate_gradient_2d(self.array, sampling=self.sampling)

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def to_hyperspy(self) -> 'Signal2D':
        """
        Convert Images to a hyperspy Signal2D.
        """

        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.to_cpu().array, self.ensemble_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def crop(self, extent: Tuple[float, float], offset: Tuple[float, float] = (0., 0.)):
        """
        Crop images to a smaller extent.

        Parameters
        ----------
        extent : two scalar
            Extent of rectangular cropping region in x and y, respectively.
        offset : two scalar, optional
            Lower corner of cropping region in x and y, respectively. Default is (0,0).

        Returns
        -------
        cropped_images : Images
        """

        offset = (int(np.round(self.base_shape[0] * offset[0] / self.extent[0])),
                  int(np.round(self.base_shape[1] * offset[1] / self.extent[1])))
        new_shape = (int(np.round(self.base_shape[0] * extent[0] / self.extent[0])),
                     int(np.round(self.base_shape[1] * extent[1] / self.extent[1])))

        array = self.array[..., offset[0]:offset[0] + new_shape[0], offset[1]:offset[1] + new_shape[1]]

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def interpolate(self,
                    sampling: Union[float, Tuple[float, float]] = None,
                    gpts: Union[int, Tuple[int, int]] = None,
                    method: str = 'fft',
                    boundary: str = 'periodic',
                    order: int = 3,
                    normalization: str = 'values',
                    cval: float = 0.) -> 'Images':
        """
        Interpolate images producing equivalent images with a different sampling.

        Parameters
        ----------
        sampling : float, two float
            Sampling of images after interpolation in x and y [Å].
        gpts : two int
            Number of grid points of images after interpolation in x and y. Do not use if `sampling` is used.
        method : {'fft', 'spline'}
            The interpolation method.

                ``fft`` :
                    Interpolate by cropping or zero-padding in Fourier space. This method should be preferred for
                    periodic images.

                ``spline`` :
                    Interpolate using spline interpolation. This method should be preferred for non-periodic images.

        boundary : {'periodic', 'reflect', 'constant'}
            The boundary parameter determines how the input array is extended beyond its boundaries for spline
            interpolation.

                ``periodic`` :
                    The images are extended by wrapping around to the opposite edge. Use this mode for periodic
                    (Default)

                ``reflect`` :
                    The images are extended by reflecting about the edge of the last pixel.

                ``constant`` :
                    The images are extended by filling all values beyond the edge with the same constant value, defined
                    by the cval parameter.

        order : int
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        normalization : {'values', 'amplitude'}
            The normalization parameter determines the preserved quantity after normalization.

                ``values`` :
                    The pixelwise values of the wave function are preserved.

                ``amplitude`` :
                    The total amplitude of the wave function is preserved.

        cval : scalar, optional
            Value to fill past edges in spline interpolation input if boundary is ‘constant’. Default is 0.0.

        Returns
        -------
        interpolated_images : Images
        """
        if method == 'fft' and boundary != 'periodic':
            raise ValueError('only periodic boundaries available for FFT interpolation')

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

        def interpolate_spline(array, old_gpts, new_gpts, pad_mode, order, cval):
            xp = get_array_module(array)
            x = xp.linspace(0., old_gpts[0], new_gpts[0], endpoint=False)
            y = xp.linspace(0., old_gpts[1], new_gpts[1], endpoint=False)
            positions = xp.meshgrid(x, y, indexing='ij')
            positions = xp.stack(positions, axis=-1)
            return interpolate_stack(array, positions, pad_mode, order=order, cval=cval)

        if boundary == 'periodic':
            boundary = 'wrap'

        array = None
        if self.is_lazy:
            array = self.array.rechunk(chunks=self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],)))
            if method == 'fft':
                array = array.map_blocks(fft_interpolate,
                                         new_shape=gpts,
                                         normalization=normalization,
                                         chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                                         meta=xp.array((), dtype=self.array.dtype))

            elif method == 'spline':
                array = array.map_blocks(interpolate_spline,
                                         old_gpts=self.shape[-2:],
                                         new_gpts=gpts,
                                         order=order,
                                         cval=cval,
                                         pad_mode=boundary,
                                         chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                                         meta=xp.array((), dtype=self.array.dtype))

        else:
            if method == 'fft':
                array = fft_interpolate(self.array, gpts, normalization=normalization)
            elif method == 'spline':
                array = interpolate_spline(self.array,
                                           old_gpts=self.shape[-2:],
                                           new_gpts=gpts,
                                           pad_mode=boundary,
                                           order=order,
                                           cval=cval)

        if array is None:
            raise RuntimeError()

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['sampling'] = sampling
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def interpolate_line_at_position(self,
                                     center: Union[Tuple[float, float], Atom],
                                     angle: float,
                                     extent: float,
                                     gpts: int = None,
                                     sampling: float = None,
                                     width: float = 0.,
                                     order: int = 3,
                                     endpoint: bool = True):

        from abtem.waves.scan import LineScan

        scan = LineScan.at_position(position=center, extent=extent, angle=angle)

        return self.interpolate_line(scan.start, scan.end, gpts=gpts, sampling=sampling, width=width, order=order,
                                     endpoint=endpoint)

    def interpolate_line(self,
                         start: Union[Tuple[float, float], Atom] = None,
                         end: Union[Tuple[float, float], Atom] = None,
                         gpts: int = None,
                         sampling: float = None,
                         width: float = 0.,
                         order: int = 3,
                         endpoint: bool = False) -> 'LineProfiles':
        """
        Interpolate image along a line.

        Parameters
        ----------
        start : two float, Atom, optional
            Start point of line [Å]. Also provide end, do not provide center, angle and extent.
        end : two float, Atom, optional
            End point of line [Å].
        gpts : int
            Number of grid points along line.
        sampling : float
            Sampling rate of grid points along line [1 / Å].
        width : float, optional
            The interpolation will be averaged across line of this width.
        order : str, optional
            The interpolation method.

        Returns
        -------
        line_profiles : LineProfiles
        """

        from abtem.waves.scan import LineScan

        if self.is_complex:
            raise NotImplementedError

        if (sampling is None) and (gpts is None):
            sampling = min(self.sampling)

        xp = get_array_module(self.array)

        if start is None:
            start = (0., 0.)

        if end is None:
            end = (0., self.extent[0])

        scan = LineScan(start=start, end=end, gpts=gpts, sampling=sampling, endpoint=endpoint)

        positions = xp.asarray(scan.get_positions(lazy=False) / self.sampling)

        if width:
            direction = xp.array(scan.end) - xp.array(scan.start)
            direction = direction / xp.linalg.norm(direction)
            perpendicular_direction = xp.array([-direction[1], direction[0]])
            n = xp.floor(width / min(self.sampling) / 2) * 2 + 1
            perpendicular_positions = xp.linspace(-n / 2, n / 2, int(n))[:, None] * perpendicular_direction[None]
            positions = perpendicular_positions[None, :] + positions[:, None]

        if self.is_lazy:
            array = self.array.map_blocks(interpolate_stack,
                                          positions=positions,
                                          mode='wrap',
                                          order=order,
                                          drop_axis=self.base_axes,
                                          new_axis=self.base_axes[0],
                                          chunks=self.array.chunks[:-2] + (positions.shape[0],),
                                          meta=xp.array((), dtype=np.float32))
        else:
            array = interpolate_stack(self.array, positions, mode='wrap', order=order)

        if width:
            array = array.mean(-1)

        return LineProfiles(array=array, sampling=scan.sampling,
                            ensemble_axes_metadata=self.ensemble_axes_metadata, metadata=self.metadata)

    def tile(self, repetitions: Tuple[int, int]) -> 'Images':
        """
        Tile images.

        Parameters
        ----------
        repetitions : two int
            The number of repetitions of the images along the x and y axis, respectively.

        Returns
        -------
        tiled_images : Images
        """
        if len(repetitions) != 2:
            raise RuntimeError()
        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = np.tile(self.array, (1,) * (len(self.array.shape) - 2) + repetitions)
        return self.__class__(**kwargs)

    def gaussian_filter(self, sigma: Union[float, Tuple[float, float]], boundary: str = 'periodic', cval: float = 0.):
        """
        Apply 2d gaussian filter to images.

        Parameters
        ----------
        sigma : float, two float
            Standard deviation for Gaussian kernel in the x and y-direction. If given as a single number, the standard
            deviation is equal for both axes.

        boundary : {'periodic', 'reflect', 'constant'}
            The boundary parameter determines how the images are extended beyond their boundaries when the filter
            overlaps with a border.

                ``periodic`` :
                    The images are extended by wrapping around to the opposite edge. Use this mode for periodic
                    (Default)

                ``reflect`` :
                    The images are extended by reflecting about the edge of the last pixel.

                ``constant`` :
                    The images are extended by filling all values beyond the edge with the same constant value, defined
                    by the cval parameter.
        Returns
        -------
        filtered_images : Images
        """
        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self.array).gaussian_filter

        if boundary == 'periodic':
            mode = 'wrap'
        elif boundary in ('reflect', 'constant'):
            mode = boundary
        else:
            raise ValueError()

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        sigma = (0,) * (len(self.shape) - 2) + tuple(s / d for s, d in zip(sigma, self.sampling))

        if self.is_lazy:
            depth = tuple(min(int(np.ceil(4.0 * s)), n) for s, n in zip(sigma, self.base_shape))
            array = self.array.map_overlap(gaussian_filter,
                                           sigma=sigma,
                                           boundary=boundary,
                                           mode=mode,
                                           cval=cval,
                                           depth=(0,) * (len(self.shape) - 2) + depth,
                                           meta=xp.array((), dtype=xp.float32))
        else:
            array = gaussian_filter(self.array, sigma=sigma, mode=mode, cval=cval)

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def diffractograms(self) -> 'DiffractionPatterns':
        """
        Calculate the diffractograms (i.e. power spectra) from the images.

        Returns
        -------
        diffractograms : DiffractionPatterns
        """
        xp = get_array_module(self.array)

        def diffractograms(array):
            array = fft2(array)
            return xp.fft.fftshift(xp.abs(array), axes=(-2, -1))

        if self.is_lazy:
            array = self.array.rechunk(chunks=self.array.chunks[:-2] + ((self.shape[-2],), (self.shape[-1],)))
            array = array.map_blocks(diffractograms, meta=xp.array((), dtype=xp.float32))
        else:
            array = diffractograms(self.array)

        sampling = 1 / self.extent[0], 1 / self.extent[1]
        return DiffractionPatterns(array=array,
                                   sampling=sampling,
                                   ensemble_axes_metadata=self.ensemble_axes_metadata,
                                   metadata=self.metadata)

    def show(self,
             cmap: str = 'viridis',
             explode: bool = False,
             ax: Axes = None,
             figsize: Tuple[int, int] = None,
             title: Union[bool, str] = True,
             panel_titles: Union[bool, List[str]] = True,
             x_ticks: bool = True,
             y_ticks: bool = True,
             x_label: Union[bool, str] = True,
             y_label: Union[bool, str] = True,
             row_super_label: Union[bool, str] = False,
             col_super_label: Union[bool, str] = False,
             power: float = 1.,
             vmin: float = None,
             vmax: float = None,
             common_color_scale=False,
             cbar: bool = False,
             cbar_labels: str = None,
             sizebar: bool = False,
             float_formatting: str = '.2f',
             panel_labels: dict = None,
             image_grid_kwargs: dict = None,
             imshow_kwargs: dict = None,
             anchored_text_kwargs: dict = None,
             ) -> Axes:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        cmap : str, optional
            TODO
        explode : bool, optional
            If True, a grid of images are created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown.
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        panel_titles : bool or list of str, optional
            Add titles to each panel. If True a title will be created from the axis metadata. If given as a list of
            strings an item must exist for each panel.
        x_ticks : bool or list, optional
            If False, the ticks on the x-axis will be removed.
        y_ticks : bool or list, optional
            If False, the ticks on the y-axis will be removed.
        x_label : bool or str, optional
            Add label to the x-axis of every plot. If True (default) the label will created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the x-axis of every plot. If True (default) the label will created from the corresponding axis
            metadata. A string may be given to override this.
        row_super_label : bool or str, optional
            Add super label to the rows of an image grid. If True the label will be created from the corresponding axis
            metadata. A string may be given to override this. The default is no super label.
        col_super_label : bool or str, optional
            Add super label to the columns of an image grid. If True the label will be created from the corresponding
            axis metadata. A string may be given to override this. The default is no super label.
        power : float
            Show image on a power scale.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
            keyword arguments to mpl_toolkits.axes_grid1.axes_grid.ImageGrid through `image_grid_kwargs`.
        cbar_labels : str or list of str
        sizebar : bool, optional,
            Add a size bar to the image(s).
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.
        panel_labels : list of str
            A list of labels for each panel of a grid of images.
        image_grid_kwargs : dict
            Additional keyword arguments passed to mpl_toolkits.axes_grid1.axes_grid.ImageGrid.
        imshow_kwargs : dict
            Additional keyword arguments passed to matplotlib.axes.Axes.imshow.
        anchored_text_kwargs : dict
            Additional keyword arguments passed to matplotlib.offsetbox.AnchoredText. This is used for creating panel
            labels.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if not explode:
            measurements = self[(0,) * len(self.ensemble_shape)]
        else:
            if ax is not None:
                raise NotImplementedError('`ax` not implemented for with `explode = True`')
            measurements = self

        return show_measurement_2d_exploded(measurements=measurements,
                                            cmap=cmap,
                                            figsize=figsize,
                                            super_title=title,
                                            sub_title=panel_titles,
                                            x_ticks=x_ticks,
                                            y_ticks=y_ticks,
                                            x_label=x_label,
                                            y_label=y_label,
                                            row_super_label=row_super_label,
                                            col_super_label=col_super_label,
                                            power=power,
                                            vmin=vmin,
                                            vmax=vmax,
                                            common_color_scale=common_color_scale,
                                            cbar=cbar,
                                            cbar_labels=cbar_labels,
                                            sizebar=sizebar,
                                            float_formatting=float_formatting,
                                            panel_labels=panel_labels,
                                            image_grid_kwargs=image_grid_kwargs,
                                            imshow_kwargs=imshow_kwargs,
                                            anchored_text_kwargs=anchored_text_kwargs,
                                            axes=ax
                                            )


class SinglePointMeasurement(AbstractMeasurement):
    _base_dims = 0

    def __init__(self, array, ensemble_axes_metadata, metadata):
        super().__init__(array, ensemble_axes_metadata, metadata)


class AbstractMeasurement1d(AbstractMeasurement):
    _base_dims = 1

    def __init__(self,
                 array: np.ndarray,
                 sampling: float = None,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        self._sampling = sampling

        super().__init__(array=array,
                         ensemble_axes_metadata=ensemble_axes_metadata,
                         metadata=metadata,
                         allow_complex=True,
                         allow_base_axis_chunks=True)

    @property
    def _area_per_pixel(self):
        raise RuntimeError('cannot infer pixel area from metadata')

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None) -> 'T':
        sampling = axes_metadata[-1].sampling
        axes_metadata = axes_metadata[:-1]
        return cls(array, sampling=sampling, ensemble_axes_metadata=axes_metadata, metadata=metadata)

    @property
    def extent(self) -> float:
        return self.sampling * self.shape[-1]

    @property
    def sampling(self) -> float:
        return self._sampling

    @property
    @abstractmethod
    def base_axes_metadata(self) -> List[Union[RealSpaceAxis, FourierSpaceAxis]]:
        pass

    def interpolate(self,
                    sampling: float = None,
                    gpts: int = None,
                    order: int = 3,
                    endpoint: bool = False) -> 'T':

        map_coordinates = get_ndimage_module(self.array).map_coordinates
        xp = get_array_module(self.array)

        if (gpts is not None) and (sampling is not None):
            raise RuntimeError()

        if sampling is None and gpts is None:
            sampling = self.sampling

        if gpts is None:
            gpts = int(np.ceil(self.extent / sampling))

        if sampling is None:
            sampling = self.extent / gpts

        def interpolate(array, gpts, endpoint, order):
            old_shape = array.shape
            array = array.reshape((-1, array.shape[-1]))

            array = xp.pad(array, ((0,) * 2, (3,) * 2), mode='wrap')
            new_points = xp.linspace(3., array.shape[-1] - 3., gpts, endpoint=endpoint)[None]

            new_array = xp.zeros(array.shape[:-1] + (gpts,), dtype=xp.float32)
            for i in range(len(array)):
                map_coordinates(array[i], new_points, new_array[i], order=order)

            return new_array.reshape(old_shape[:-1] + (gpts,))

        if self.is_lazy:
            array = self.array.rechunk(self.array.chunks[:-1] + ((self.shape[-1],),))
            array = array.map_blocks(interpolate, gpts=gpts, endpoint=endpoint, order=order,
                                     chunks=self.array.chunks[:-1] + ((gpts,)), meta=xp.array((), dtype=xp.float32))
        else:
            array = interpolate(self.array, gpts, endpoint, order)

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        kwargs['sampling'] = sampling
        return self.__class__(**kwargs)

    def to_hyperspy(self):
        if Signal1D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.to_cpu().array, self.ensemble_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal1D(array, axes=axes_extra[::-1] + axes_base[::-1])

        if self.is_lazy:
            s = s.as_lazy()

        return s

    def _plot(self, x, y, ax, label, **kwargs):
        y = copy_to_device(y, np)
        x, y = np.squeeze(x), np.squeeze(y)

        if np.iscomplexobj(self.array):
            if label is None:
                label = ''

            line1 = ax.plot(x, y.real, label=f'Real {label}', **kwargs)
            line2 = ax.plot(x, y.imag, label=f'Imag. {label}', **kwargs)
            line = (line1, line2)
        else:
            line = ax.plot(x, y, label=label, **kwargs)

        return line

    def show(self,
             ax: Axes = None,
             title: str = None,
             figsize=None,
             angular_units: bool = False):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if title is None and 'name' in self.metadata:
            title = self.metadata['name']

        if title is not None:
            ax.set_title(title)

        x = np.linspace(0, self.extent, self.shape[-1], endpoint=False)

        if angular_units:
            x *= energy2wavelength(self.energy) * 1e3

        for index, line_profile in self.iterate_ensemble(keep_dims=True):

            labels = []
            for axis in line_profile.ensemble_axes_metadata:

                if isinstance(axis, OrdinalAxis):
                    labels += [f'{axis.values[0]}']

            label = ''.join(labels)

            self._plot(x, line_profile.array, ax, label)

        ax.set_xlabel(f'{self.axes_metadata[-1].label} [{self.axes_metadata[-1].units}]')

        if len(self.ensemble_shape) > 0:
            ax.legend()

        return ax


class LineProfiles(AbstractMeasurement1d):

    def __init__(self,
                 array: np.ndarray,
                 sampling: float = None,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        super().__init__(array=array,
                         sampling=sampling,
                         ensemble_axes_metadata=ensemble_axes_metadata,
                         metadata=metadata)

    @property
    def base_axes_metadata(self) -> List[RealSpaceAxis]:
        return [RealSpaceAxis(label='r', sampling=self.sampling, units='Å')]

    def tile(self, reps: int) -> 'LineProfiles':
        kwargs = self.copy_kwargs(exclude=('array',))
        xp = get_array_module(self.array)
        reps = (1,) * (len(self.array.shape) - 1) + (reps,)

        if self.is_lazy:
            kwargs['array'] = da.tile(self.array, reps)
        else:
            kwargs['array'] = xp.tile(self.array, reps)

        return self.__class__(**kwargs)


class FourierSpaceLineProfiles(AbstractMeasurement1d):

    def __init__(self,
                 array: np.ndarray,
                 sampling: float = None,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):
        super().__init__(array=array,
                         sampling=sampling,
                         ensemble_axes_metadata=ensemble_axes_metadata,
                         metadata=metadata)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [FourierSpaceAxis(label='k', sampling=self.sampling, units='1 / Å')]


def integrate_gradient_2d(gradient, sampling):
    xp = get_array_module(gradient)
    gx, gy = gradient.real, gradient.imag
    (nx, ny) = gx.shape[-2:]
    ikx = xp.fft.fftfreq(nx, d=sampling[0])
    iky = xp.fft.fftfreq(ny, d=sampling[1])
    grid_ikx, grid_iky = xp.meshgrid(ikx, iky, indexing='ij')
    k = grid_ikx ** 2 + grid_iky ** 2
    k[k == 0] = 1e-12
    That = (xp.fft.fft2(gx) * grid_ikx + xp.fft.fft2(gy) * grid_iky) / (2j * np.pi * k)
    T = xp.real(xp.fft.ifft2(That))
    T -= xp.min(T)
    return T


def _fourier_space_bilinear_nodes_and_weight(old_shape: Tuple[int, int],
                                             new_shape: Tuple[int, int],
                                             old_angular_sampling: Tuple[float, float],
                                             new_angular_sampling: Tuple[float, float],
                                             xp) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = []
    weights = []

    old_sampling = 1 / old_angular_sampling[0] / old_shape[0], 1 / old_angular_sampling[1] / old_shape[1]
    new_sampling = 1 / new_angular_sampling[0] / new_shape[0], 1 / new_angular_sampling[1] / new_shape[1]

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


def _reduced_scanned_images_or_line_profiles(new_array, old_measurement):
    if scanned_measurement_type(old_measurement) is LineProfiles:
        sampling = old_measurement.ensemble_axes_metadata[-1].sampling

        ensemble_axes_metadata = old_measurement.ensemble_axes_metadata[:-1]

        return LineProfiles(new_array,
                            sampling=sampling,
                            ensemble_axes_metadata=ensemble_axes_metadata,
                            metadata=old_measurement.metadata)

    elif scanned_measurement_type(old_measurement) is Images:

        ensemble_axes_metadata = old_measurement.ensemble_axes_metadata[:-2]

        sampling = old_measurement.ensemble_axes_metadata[-2].sampling, \
                   old_measurement.ensemble_axes_metadata[-1].sampling

        images = Images(new_array,
                        sampling=sampling,
                        ensemble_axes_metadata=ensemble_axes_metadata,
                        metadata=old_measurement.metadata)

        return images
    else:
        return new_array


class DiffractionPatterns(AbstractMeasurement):
    _base_dims = 2

    def __init__(self,
                 array: Union[np.ndarray, da.core.Array],
                 sampling: Union[float, Tuple[float, float]],
                 fftshift: bool = False,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        """
        One or more diffraction patterns.

        Parameters
        ----------
        array : ndarray
            2D or greater array containing data with `float` type. The second-to-last and last dimensions are the
            reciprocal space y and x axis of the diffraction pattern.
        sampling : float or two float
            The Fourier space sampling of the diffraction patterns [1 / Å].
        fftshift : bool, optional
            If True, the diffraction patterns are assumed to have the zero-frequency component to the center of the
            spectrum, otherwise the centers are assumed to be at (0,0).
        ensemble_axes_metadata : list of AxisMetadata, optional
            Metadata associated with an ensemble axis.
        metadata : dict, optional
            A dictionary defining simulation metadata.
        """

        if np.isscalar(sampling):
            sampling = (float(sampling),) * 2
        else:
            sampling = float(sampling[0]), float(sampling[1])

        self._fftshift = fftshift
        self._sampling = sampling

        self._base_axes = (-2, -1)

        super().__init__(array=array, ensemble_axes_metadata=ensemble_axes_metadata, metadata=metadata)

    @property
    def _area_per_pixel(self):
        if len(self.scan_sampling) == 2:
            return np.prod(self.scan_sampling)
        else:
            raise RuntimeError('cannot infer pixel area from metadata')

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata=None):
        sampling = (axes_metadata[-2].sampling, axes_metadata[-1].sampling)
        fftshift = axes_metadata[-1].fftshift
        axes_metadata = axes_metadata[:-2]
        return cls(array, sampling=sampling, ensemble_axes_metadata=axes_metadata, fftshift=fftshift, metadata=metadata)

    @property
    def base_axes_metadata(self):
        return [FourierSpaceAxis(sampling=self.sampling[0], label='kx', units='1 / Å', fftshift=self.fftshift),
                FourierSpaceAxis(sampling=self.sampling[1], label='ky', units='1 / Å', fftshift=self.fftshift)]

    def to_hyperspy(self):
        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.to_cpu().array, self.ensemble_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1])

        # s.set_signal_type('electron_diffraction')
        for axis in s.axes_manager.signal_axes:
            axis.offset = -int(axis.size / 2) * axis.scale
        if self.is_lazy:
            s = s.as_lazy()

        return s

    @property
    def fftshift(self):
        return self._fftshift

    @property
    def sampling(self) -> Tuple[float, float]:
        return self._sampling

    @property
    def angular_sampling(self) -> Tuple[float, float]:
        return self.sampling[0] * self.wavelength * 1e3, self.sampling[1] * self.wavelength * 1e3

    @property
    def max_angles(self):
        return self.shape[-2] // 2 * self.angular_sampling[0], self.shape[-1] // 2 * self.angular_sampling[1]

    @property
    def equivalent_real_space_extent(self):
        return 1 / self.sampling[0], 1 / self.sampling[1]

    @property
    def equivalent_real_space_sampling(self):
        return 1 / self.sampling[0] / self.base_shape[0], 1 / self.sampling[1] / self.base_shape[1]

    @property
    def scan_extent(self):
        extent = ()
        for d, n in zip(self.scan_sampling, self.scan_shape):
            extent += (d * n,)
        return extent

    @property
    def limits(self) -> List[Tuple[float, float]]:
        limits = []
        for i in (-2, -1):
            if self.shape[i] % 2:
                limits += [(-(self.shape[i] - 1) // 2 * self.sampling[i], (self.shape[i] - 1) // 2 * self.sampling[i])]
            else:
                limits += [(-self.shape[i] // 2 * self.sampling[i], (self.shape[i] // 2 - 1) * self.sampling[i])]
        return limits

    @property
    def angular_limits(self) -> List[Tuple[float, float]]:
        limits = self.limits
        limits[0] = limits[0][0] * self.wavelength * 1e3, limits[0][1] * self.wavelength * 1e3
        limits[1] = limits[1][0] * self.wavelength * 1e3, limits[1][1] * self.wavelength * 1e3
        return limits

    @property
    def angular_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Angular coordinates. """
        xp = get_array_module(self.array)
        limits = self.angular_limits
        alpha_x = xp.linspace(limits[0][0], limits[0][1], self.shape[-2], dtype=xp.float32)
        alpha_y = xp.linspace(limits[1][0], limits[1][1], self.shape[-1], dtype=xp.float32)
        return alpha_x, alpha_y

    @staticmethod
    def _batch_interpolate_bilinear(array, new_sampling, sampling, new_gpts):

        xp = get_array_module(array)
        v, u, vw, uw = _fourier_space_bilinear_nodes_and_weight(array.shape[-2:],
                                                                new_gpts,
                                                                sampling,
                                                                new_sampling,
                                                                xp)

        old_shape = array.shape
        array = array.reshape((-1,) + array.shape[-2:])

        old_sums = array.sum((-2, -1), keepdims=True)

        if xp is cp:
            array = interpolate_bilinear_cuda(array, v, u, vw, uw)
        else:
            array = interpolate_bilinear(array, v, u, vw, uw)

        array = array / array.sum((-2, -1), keepdims=True) * old_sums

        return array.reshape(old_shape[:-2] + array.shape[-2:])

    # def interpolate_scan(self, sampling: Union[str, float, Tuple[float, float]] = None):
    #     """
    #     Interpolate between scan positions producing additional diffraction patterns.
    #
    #     Parameters
    #     ----------
    #     sampling : two float or 'uniform'
    #         Target scan sampling after interpolation in x and y [1 / Å].
    #
    #     Returns
    #     -------
    #     interpolated_diffraction_patterns : DiffractionPatterns
    #     """
    #
    #     if np.isscalar(sampling):
    #         sampling = (sampling,) * 2
    #
    #     scan_sampling = self.scan_sampling
    #
    #     if len(scan_sampling) != 2:
    #         raise NotImplementedError()
    #
    #     sampling, gpts = adjusted_gpts(sampling, self.scan_sampling, self.base_shape)
    #
    #     xp = get_array_module(self.array)
    #
    #     if self.is_lazy:
    #         array = da.moveaxis(self.array, self.scan_axes, (-2, -1))
    #         array = array.rechunk(('auto',) * (len(array.shape) - 2) + (-1, -1))
    #         array = array.map_blocks(self._batch_interpolate_bilinear,
    #                                  sampling=scan_sampling,
    #                                  new_sampling=sampling,
    #                                  new_gpts=gpts,
    #                                  chunks=array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
    #                                  dtype=np.float32)
    #         array = da.moveaxis(self.array, (-2, -1), self.scan_axes)
    #     else:
    #         array = xp.moveaxis(self.array, self.scan_axes, (-2, -1))
    #         array = self._batch_interpolate_bilinear(array, sampling=self.sampling, new_sampling=sampling,
    #                                                  new_gpts=gpts)
    #         array = xp.moveaxis(self.array, (-2, -1), self.scan_axes)
    #
    #     return array
    #     # print(array.chunks[-2:], gpts[0] / self.scan_shape[0] * sum(array.chunks[-2]), gpts)

    def interpolate(self, sampling: Union[str, float, Tuple[float, float]] = None):
        """
        Interpolate diffraction patterns producing equivalent patterns with a different sampling.

        Parameters
        ----------
        sampling : two float or 'uniform'
            Sampling of diffraction patterns after interpolation in x and y [1 / Å]. If given as 'uniform' the
            diffraction patterns are downsampled along the axis with the smallest pixel size such that the sampling
            is uniform.

        Returns
        -------
        interpolated_diffraction_patterns : DiffractionPatterns
        """

        if sampling == 'uniform':
            sampling = (max(self.sampling),) * 2

        elif not isinstance(sampling, str) and np.isscalar(sampling):
            sampling = (sampling,) * 2

        sampling, gpts = adjusted_gpts(sampling, self.sampling, self.base_shape)

        if self.is_lazy:
            array = self.array.map_blocks(self._batch_interpolate_bilinear,
                                          sampling=self.sampling,
                                          new_sampling=sampling,
                                          new_gpts=gpts,
                                          chunks=self.array.chunks[:-2] + ((gpts[0],), (gpts[1],)),
                                          dtype=np.float32)
        else:
            array = self._batch_interpolate_bilinear(self.array, sampling=self.sampling, new_sampling=sampling,
                                                     new_gpts=gpts)

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['sampling'] = sampling
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def _check_integration_limits(self, inner: float, outer: float):
        if inner >= outer:
            raise RuntimeError(f'inner detection ({inner} mrad) angle exceeds outer detection angle'
                               f'({outer} mrad)')

        if (outer > self.max_angles[0]) or (outer > self.max_angles[1]):
            raise RuntimeError(
                f'outer integration limit exceeds the maximum simulated angle ({outer} mrad > '
                f'{min(self.max_angles)} mrad), increase the number of grid points')

        # integration_range = outer - inner
        # if integration_range < min(self.angular_sampling):
        #     raise RuntimeError(
        #         f'integration range ({integration_range} mrad) smaller than angular sampling of simulation'
        #         f' ({min(self.angular_sampling)} mrad)')

    def gaussian_source_size(self, sigma: Union[float, Tuple[float, float]]) -> 'DiffractionPatterns':
        """
        Simulate Gaussian source size by applying a gaussian filter.

        The filter is not applied to diffraction pattern individually, the intensity of diffraction patterns are mixed
        across scan axes. Applying this filter requires two linear scan axes.

        Applying this filter before integrating the diffraction patterns will produce the same image as integrating
        the diffraction patterns first then applying a gaussian filter.

        Parameters
        ----------
        sigma : float, two float
            Standard deviation of Gaussian kernel in the x and y-direction. If given as a single number, the standard
            deviation is equal for both axes.

        Returns
        -------
        filtered_diffraction_patterns : DiffractionPatterns
        """
        if self.num_scan_axes < 2:
            raise RuntimeError(
                'gaussian_source_size not implemented for DiffractionPatterns with less than 2 scan axes')

        if np.isscalar(sigma):
            sigma = (sigma,) * 2

        xp = get_array_module(self.array)
        gaussian_filter = get_ndimage_module(self._array).gaussian_filter

        scan_axes = self.scan_axes

        padded_sigma = ()
        depth = ()
        i = 0
        for axis, n in zip(self.ensemble_axes, self.ensemble_shape):
            if axis in scan_axes:
                scan_sampling = self.scan_sampling[i]
                padded_sigma += (sigma[i] / scan_sampling,)
                depth += (min(int(np.ceil(4.0 * sigma[i] / scan_sampling)), n),)
                i += 1
            else:
                padded_sigma += (0.,)
                depth += (0,)

        padded_sigma += (0.,) * 2
        depth += (0,) * 2

        if self.is_lazy:
            array = self.array.map_overlap(gaussian_filter,
                                           sigma=padded_sigma,
                                           mode='wrap',
                                           depth=depth,
                                           meta=xp.array((), dtype=xp.float32))
        else:
            array = gaussian_filter(self.array, sigma=padded_sigma, mode='wrap')

        return self.__class__(array,
                              sampling=self.sampling,
                              ensemble_axes_metadata=self.ensemble_axes_metadata,
                              metadata=self.metadata,
                              fftshift=self.fftshift)

    def polar_binning(self,
                      nbins_radial: int,
                      nbins_azimuthal: int,
                      inner: float = 0.,
                      outer: float = None,
                      rotation: float = 0.,
                      offset: Tuple[float, float] = (0., 0.)):
        """
        Create polar measurements from the diffraction patterns by binning the measurements on a polar grid. This
        method may be used to simulate a segmented detector with a specified .

        Each bin is a segment of an annulus and the bins are spaced equally in the radial and azimuthal directions.
        The bins fit between a given inner and outer integration limit, they may be rotated around the origin and their
        center may be shifted from the origin.

        Parameters
        ----------
        nbins_radial : int
            Number of radial bins.
        nbins_azimuthal : int
            Number of angular bins.
        inner : float
            Inner integration limit of the bins [mrad].
        outer : float
            Outer integration limit of the bins [mrad].
        rotation : float
            Rotation of the bins around the origin [mrad].
        offset : two float
            Offset of the bins from the origin in x and y [mrad].

        Returns
        -------
        polar_measurements : PolarMeasurements
        """

        if outer is None:
            outer = min(self.max_angles)

        self._check_integration_limits(inner, outer)
        xp = get_array_module(self.array)

        def radial_binning(array, nbins_radial, nbins_azimuthal, sampling):
            xp = get_array_module(array)

            indices = polar_detector_bins(gpts=array.shape[-2:],
                                          sampling=sampling,
                                          inner=inner,
                                          outer=outer,
                                          nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          fftshift=self.fftshift,
                                          rotation=rotation / 1e3,
                                          offset=offset,
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
            array = self.array.map_blocks(radial_binning,
                                          nbins_radial=nbins_radial,
                                          nbins_azimuthal=nbins_azimuthal,
                                          sampling=self.angular_sampling,
                                          drop_axis=(len(self.shape) - 2, len(self.shape) - 1),
                                          chunks=self.array.chunks[:-2] + ((nbins_radial,), (nbins_azimuthal,),),
                                          new_axis=(len(self.shape) - 2, len(self.shape) - 1,),
                                          meta=xp.array((), dtype=xp.float32))
        else:
            array = radial_binning(self.array,
                                   nbins_radial=nbins_radial,
                                   nbins_azimuthal=nbins_azimuthal,
                                   sampling=self.angular_sampling)

        radial_sampling = (outer - inner) / nbins_radial
        azimuthal_sampling = 2 * np.pi / nbins_azimuthal

        return PolarMeasurements(array,
                                 radial_sampling=radial_sampling,
                                 azimuthal_sampling=azimuthal_sampling,
                                 radial_offset=inner,
                                 azimuthal_offset=rotation,
                                 ensemble_axes_metadata=self.ensemble_axes_metadata,
                                 metadata=self.metadata)

    def radial_binning(self,
                       step_size: float = 1.,
                       inner: float = 0.,
                       outer: float = None) -> 'PolarMeasurements':
        """
        Create polar measurements from the diffraction patterns by binning the measurements in annular regions.

        This is equivalent to detecting a wave function using the FlexibleAnnularDetector.

        Parameters
        ----------
        step_size : float, optional
            Radial extent of the bins [mrad]. Default is 1.
        inner : float, optional
            Inner integration limit of the bins [mrad].
        outer : float, optional
            Outer integration limit of the bins [mrad].

        Returns
        -------
        radially_binned_measurement : PolarMeasurements
        """

        if outer is None:
            outer = min(self.max_angles)

        nbins_radial = int((outer - inner) / step_size)
        return self.polar_binning(nbins_radial, 1, inner, outer)

    def integrate_radial(self, inner: float, outer: float) -> Images:
        """
        Create images by integrating the diffraction patterns over an annulus defined by an inner and outer integration
        angle.

        Parameters
        ----------
        inner : float
            Inner integration limit [mrad].
        outer : float
            Outer integration limit [mrad].

        Returns
        -------
        integrated_images : Images
        """

        self._check_integration_limits(inner, outer)

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

        return _reduced_scanned_images_or_line_profiles(integrated_intensity, self)

    def integrated_center_of_mass(self) -> Images:
        """
        Calculate integrated center of mass (iCOM) images from diffraction patterns. This method is only implemented
        for diffraction patterns with exactly two scan axes.

        Returns
        -------
        icom_images : Images
        """
        com = self.center_of_mass()

        if isinstance(com, Images):
            return com.integrate_gradient()
        else:
            raise RuntimeError(f'integrated center of mass not implemented for DiffractionPatterns with '
                               f'{self.num_scan_axes} scan axes')

    @staticmethod
    def _com(array, x, y):
        com_x = (array * x[:, None]).sum(axis=(-2, -1))
        com_y = (array * y[None]).sum(axis=(-2, -1))
        com = com_x + 1.j * com_y
        return com

    def center_of_mass(self) -> Union[Images, LineProfiles]:
        """
        Calculate center of mass images or line profiles from diffraction patterns. The results are complex type where
        the real and imaginary part represents the x and y component.

        Returns
        -------
        com_images : Images

        com_line_profiles : LineProfiles
        """

        x, y = self.angular_coordinates

        xp = get_array_module(self.array)

        x, y = xp.asarray(x), xp.asarray(y)

        if self.is_lazy:
            array = self.array.map_blocks(self._com, x=x, y=y, drop_axis=self.base_axes, dtype=np.complex64)
        else:
            array = self._com(self.array, x=x, y=y)

        return _reduced_scanned_images_or_line_profiles(array, self)

    def epie(self,
             probe_guess,
             max_batch: int = 8,
             max_iter: int = 4,
             alpha: float = 1.,
             beta: float = 1.,
             fix_probe: bool = False,
             fix_com: bool = True,
             crop_to_scan: bool = True) -> Images:

        """
        Ptychographic reconstruction with the extended ptychographical engine (ePIE) algorithm.
        
        probe_guess : abtem.waves.Probe
            The initial guess for the probe.
        max_batch : int
            Maximum number of probe positions to update at every step.
        max_iter : int
            Maximum number of iterations to run the ePIE algorithm.
        alpha : float
            Step size of iterative updates for the object.
        beta : float
            Step size of iterative updates for the probe.
        fix_probe : bool
            If True, the probe will not be updated by the algorithm. Default is False.
        fix_com : bool
            If True, the center of mass of the probe will be centered. Default is True.
        crop_to_scan : bool
            If true, the output is cropped to the scan area.
        """

        from abtem.reconstruct.epie import epie

        reconstruction = epie(self,
                              probe_guess,
                              max_batch=max_batch,
                              max_iter=max_iter,
                              alpha=alpha,
                              beta=beta,
                              fix_com=fix_com,
                              fix_probe=fix_probe)

        if crop_to_scan:
            reconstruction = reconstruction.crop(self.scan_extent)

        return reconstruction

    def bandlimit(self, inner: float, outer: float) -> 'DiffractionPatterns':
        """
        Bandlimit the diffraction patterns by setting everything outside an annulus defined by two radial angles to
        zero.

        Parameters
        ----------
        inner : float
            Inner limit of zero region [mrad].
        outer : float
            Outer limit of zero region [mrad].

        Returns
        -------
        bandlimited_diffraction_patterns : DiffractionPatterns
        """

        def bandlimit(array, inner, outer):
            alpha_x, alpha_y = self.angular_coordinates
            alpha = alpha_x[:, None] ** 2 + alpha_y[None] ** 2
            block = (alpha >= inner ** 2) * (alpha < outer ** 2)
            return array * block

        xp = get_array_module(self.array)

        if self.is_lazy:
            array = self.array.map_blocks(bandlimit, inner=inner, outer=outer, meta=xp.array((), dtype=xp.float32))
        else:
            array = bandlimit(self.array, inner, outer)

        kwargs = self.copy_kwargs(exclude=('array',))
        kwargs['array'] = array
        return self.__class__(**kwargs)

    def block_direct(self, radius: float = None) -> 'DiffractionPatterns':
        """
        Block the direct beam by setting the pixels of the zeroth order disk to zero.

        Parameters
        ----------
        radius : float, optional
            The radius of the zeroth order disk to block [mrad]. If not given this will be inferred from the metadata.

        Returns
        -------
        diffraction_patterns : DiffractionPatterns
        """

        if radius is None:
            if 'semiangle_cutoff' in self.metadata.keys():
                radius = self.metadata['semiangle_cutoff']
            else:
                radius = max(self.angular_sampling) * 1.0001

        return self.bandlimit(radius, outer=np.inf)

    def select_frequency_bin(self, bins):
        bins = np.array(bins)
        center = np.array([self.base_shape[0] // 2, self.base_shape[1] // 2])
        indices = bins + center
        if len(bins.shape) == 2:
            return self.array[..., indices[:, 0], indices[:, 1]]
        else:
            return self.array[..., indices[0], indices[1]]

    def show(self,
             units: str = 'reciprocal',
             cmap: str = 'viridis',
             explode: bool = False,
             ax: Axes = None,
             figsize: Tuple[int, int] = None,
             title: Union[bool, str] = True,
             panel_titles: Union[bool, List[str]] = True,
             x_ticks: bool = True,
             y_ticks: bool = True,
             x_label: Union[bool, str] = True,
             y_label: Union[bool, str] = True,
             row_super_label: Union[bool, str] = False,
             col_super_label: Union[bool, str] = False,
             power: float = 1.,
             vmin: float = None,
             vmax: float = None,
             common_color_scale=False,
             cbar: bool = False,
             cbar_labels: str = None,
             sizebar: bool = False,
             float_formatting: str = '.2f',
             panel_labels: dict = None,
             image_grid_kwargs: dict = None,
             imshow_kwargs: dict = None,
             anchored_text_kwargs: dict = None,
             ) -> Axes:
        """
        Show the image(s) using matplotlib.

        Parameters
        ----------
        units : bool, optional
        cmap : str, optional
        explode : bool, optional
            If True, a grid of images are created for all the items of the last two ensemble axes. If False, the first
            ensemble item is shown.
        ax : matplotlib.axes.Axes, optional
            If given the plots are added to the axis. This is not available for image grids.
        figsize : two int, optional
            The figure size given as width and height in inches, passed to matplotlib.pyplot.figure.
        title : bool or str, optional
            Add a title to the figure. If True is given instead of a string the title will be given by the value
            corresponding to the "name" key of the metadata dictionary, if this item exists.
        panel_titles : bool or list of str, optional
            Add titles to each panel. If True a title will be created from the axis metadata. If given as a list of
            strings an item must exist for each panel.
        x_ticks : bool or list, optional
            If False, the ticks on the x-axis will be removed.
        y_ticks : bool or list, optional
            If False, the ticks on the y-axis will be removed.
        x_label : bool or str, optional
            Add label to the x-axis of every plot. If True (default) the label will created from the corresponding axis
            metadata. A string may be given to override this.
        y_label : bool or str, optional
            Add label to the x-axis of every plot. If True (default) the label will created from the corresponding axis
            metadata. A string may be given to override this.
        row_super_label : bool or str, optional
            Add super label to the rows of an image grid. If True the label will be created from the corresponding axis
            metadata. A string may be given to override this. The default is no super label.
        col_super_label : bool or str, optional
            Add super label to the columns of an image grid. If True the label will be created from the corresponding
            axis metadata. A string may be given to override this. The default is no super label.
        power : float
            Show image on a power scale.
        vmin : float, optional
            Minimum of the intensity color scale. Default is the minimum of the array values.
        vmax : float, optional
            Maximum of the intensity color scale. Default is the maximum of the array values.
        common_color_scale : bool, optional
            If True all images in an image grid are shown on the same colorscale, and a single colorbar is created (if
            it is requested). Default is False.
        cbar : bool, optional
            Add colorbar(s) to the image(s). The position and size of the colorbar(s) may be controlled by passing
            keyword arguments to mpl_toolkits.axes_grid1.axes_grid.ImageGrid through `image_grid_kwargs`.
        sizebar : bool, optional,
            Add a size bar to the image(s).
        float_formatting : str, optional
            A formatting string used for formatting the floats of the panel titles.
        panel_labels : list of str
            A list of labels for each panel of a grid of images.
        image_grid_kwargs : dict
            Additional keyword arguments passed to mpl_toolkits.axes_grid1.axes_grid.ImageGrid.
        imshow_kwargs : dict
            Additional keyword arguments passed to matplotlib.axes.Axes.imshow.
        anchored_text_kwargs : dict
            Additional keyword arguments passed to matplotlib.offsetbox.AnchoredText. This is used for creating panel
            labels.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if not explode:
            measurements = self[(0,) * len(self.ensemble_shape)]
        else:
            if ax is not None:
                raise NotImplementedError('`ax` not implemented for with `explode = True`')
            measurements = self

        angular_units = ('angular', 'mrad')
        bins = ('bins',)
        reciprocal_units = ('reciprocal', '1/Å')

        if units.lower() in angular_units:
            x_label = 'scattering angle x [mrad]'
            y_label = 'scattering angle y [mrad]'
            extent = list(self.angular_limits[0] + self.angular_limits[1])
        elif units in bins:
            def bin_extent(n):
                if n % 2 == 0:
                    return -n // 2 - .5, n // 2 - .5
                else:
                    return -n // 2 + .5, n // 2 + .5

            x_label = 'frequency bin n'
            y_label = 'frequency bin m'
            extent = bin_extent(self.base_shape[0]) + bin_extent(self.base_shape[1])
        elif units.lower().strip() in reciprocal_units:
            extent = list(self.limits[0] + self.limits[1])
        else:
            raise ValueError()

        return show_measurement_2d_exploded(measurements=measurements,
                                            figsize=figsize,
                                            super_title=title,
                                            sub_title=panel_titles,
                                            x_ticks=x_ticks,
                                            y_ticks=y_ticks,
                                            x_label=x_label,
                                            y_label=y_label,
                                            extent=extent,
                                            cmap=cmap,
                                            row_super_label=row_super_label,
                                            col_super_label=col_super_label,
                                            power=power,
                                            vmin=vmin,
                                            vmax=vmax,
                                            common_color_scale=common_color_scale,
                                            cbar=cbar,
                                            cbar_labels=cbar_labels,
                                            sizebar=sizebar,
                                            float_formatting=float_formatting,
                                            panel_labels=panel_labels,
                                            image_grid_kwargs=image_grid_kwargs,
                                            imshow_kwargs=imshow_kwargs,
                                            anchored_text_kwargs=anchored_text_kwargs,
                                            axes=ax
                                            )

    # def show(self,
    #          ax: Axes = None,
    #          cbar: bool = False,
    #          power: float = 1.,
    #          title: str = None,
    #          figsize: Tuple[float, float] = None,
    #          angular_units: bool = False,
    #          max_angle: float = None,
    #          vmin=None,
    #          vmax=None,
    #          **kwargs):
    #
    #     if ax is None:
    #         fig, ax = plt.subplots(figsize=figsize)
    #
    #     if title is not None:
    #         ax.set_title(title)
    #
    #     if angular_units:
    #         ax.set_xlabel('Scattering angle x [mrad]')
    #         ax.set_ylabel('Scattering angle y [mrad]')
    #         extent = self.angular_limits[0] + self.angular_limits[1]
    #     else:
    #         ax.set_xlabel('Spatial frequency x [1 / Å]')
    #         ax.set_ylabel('Spatial frequency y [1 / Å]')
    #         extent = self.limits[0] + self.limits[1]
    #
    #     slic = (0,) * self.num_ensemble_axes
    #
    #     array = asnumpy(self.array)[slic].T
    #
    #     if np.iscomplexobj(array):
    #         if power != 1.:
    #             raise ValueError
    #
    #         colored_array = domain_coloring(array, vmin=vmin, vmax=vmax)
    #     else:
    #         colored_array = array ** power
    #
    #     im = ax.imshow(colored_array, origin='lower', vmin=vmin, extent=extent, vmax=vmax, **kwargs)
    #
    #     if cbar:
    #         if np.iscomplexobj(array):
    #             vmin = np.abs(array).min() if vmin is None else vmin
    #             vmax = np.abs(array).max() if vmax is None else vmax
    #             add_domain_coloring_cbar(ax, vmin, vmax)
    #         else:
    #             plt.colorbar(im, ax=ax)
    #
    #     # im = ax.imshow(array, extent=extent, origin='lower', **kwargs)
    #
    #     if max_angle:
    #         ax.set_xlim([-max_angle, max_angle])
    #         ax.set_ylim([-max_angle, max_angle])
    #
    #     if cbar:
    #         plt.colorbar(im, ax=ax)
    #
    #     return ax, im


class PolarMeasurements(AbstractMeasurement):
    _base_dims = 2

    def __init__(self,
                 array: np.ndarray,
                 radial_sampling: float,
                 azimuthal_sampling: float,
                 radial_offset: float = 0.,
                 azimuthal_offset: float = 0.,
                 ensemble_axes_metadata: List[AxisMetadata] = None,
                 metadata: dict = None):

        self._radial_sampling = radial_sampling
        self._azimuthal_sampling = azimuthal_sampling
        self._radial_offset = radial_offset
        self._azimuthal_offset = azimuthal_offset

        super().__init__(array=array, ensemble_axes_metadata=ensemble_axes_metadata, metadata=metadata)

    @property
    def _area_per_pixel(self):
        if len(self.scan_sampling) == 2:
            return np.prod(self.scan_sampling)
        else:
            raise RuntimeError('cannot infer pixel area from metadata')

    @classmethod
    def from_array_and_metadata(cls, array, axes_metadata, metadata) -> 'PolarMeasurements':
        radial_sampling = axes_metadata[-2].sampling
        radial_offset = axes_metadata[-2].offset
        azimuthal_sampling = axes_metadata[-1].sampling
        azimuthal_offset = axes_metadata[-1].offset
        return cls(array, radial_sampling=radial_sampling, radial_offset=radial_offset,
                   azimuthal_sampling=azimuthal_sampling, azimuthal_offset=azimuthal_offset,
                   ensemble_axes_metadata=axes_metadata[:-2], metadata=metadata)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [LinearAxis(label='Radial scattering angle', offset=self.radial_offset,
                           sampling=self.radial_sampling, units='mrad'),
                LinearAxis(label='Azimuthal scattering angle', offset=self.azimuthal_offset,
                           sampling=self.azimuthal_sampling, units='rad')]

    def to_hyperspy(self):
        if Signal2D is None:
            raise RuntimeError(missing_hyperspy_message)

        axes_base = _to_hyperspy_axes_metadata(
            self.base_axes_metadata,
            self.base_shape,
        )
        axes_extra = _to_hyperspy_axes_metadata(
            self.ensemble_axes_metadata,
            self.ensemble_shape,
        )

        # We need to transpose the navigation axes to match hyperspy convention
        array = np.transpose(self.to_cpu().array, self.ensemble_axes[::-1] + self.base_axes[::-1])
        # The index in the array corresponding to each axis is determine from
        # the index in the axis list
        s = Signal2D(array, axes=axes_extra[::-1] + axes_base[::-1]).squeeze()

        if self.is_lazy:
            s = s.as_lazy()

        return s

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

    def integrate_radial(self, inner: float, outer: float) -> Union[Images, LineProfiles]:
        return self.integrate(radial_limits=(inner, outer))

    def integrate(self,
                  radial_limits: Tuple[float, float] = None,
                  azimuthal_limits: Tuple[float, float] = None,
                  detector_regions: Sequence[int] = None) -> Union[Images, LineProfiles]:

        if detector_regions is not None:
            if (radial_limits is not None) or (azimuthal_limits is not None):
                raise RuntimeError()

            array = self.array.reshape(self.shape[:-2] + (-1,))[..., list(detector_regions)].sum(axis=-1)
        else:
            if radial_limits is None:
                radial_slice = slice(None)
            else:
                inner_index = int((radial_limits[0] - self.radial_offset) / self.radial_sampling)
                outer_index = int((radial_limits[1] - self.radial_offset) / self.radial_sampling)
                radial_slice = slice(inner_index, outer_index)

            if azimuthal_limits is None:
                azimuthal_slice = slice(None)
            else:
                left_index = int(azimuthal_limits[0] / self.radial_sampling)
                right_index = int(azimuthal_limits[1] / self.radial_sampling)
                azimuthal_slice = slice(left_index, right_index)

            array = self.array[..., radial_slice, azimuthal_slice].sum(axis=(-2, -1))

        return _reduced_scanned_images_or_line_profiles(array, self)

    def show(self, ax: Axes = None, title: str = None, min_azimuthal_division: float = np.pi / 20, grid: bool = False,
             **kwargs):

        if ax is None:
            ax = plt.subplot(projection="polar")

        if title is not None:
            ax.set_title(title)

        array = self.array[(0,) * (len(self.shape) - 2)]

        repeat = int(self.azimuthal_sampling / min_azimuthal_division)
        r = np.pi / (4 * repeat) + self.azimuthal_offset
        azimuthal_grid = np.linspace(r, 2 * np.pi + r, self.shape[-1] * repeat, endpoint=False)

        d = (self.outer_angle - self.radial_offset) / 2 / self.shape[-2]
        radial_grid = np.linspace(self.radial_offset + d, self.outer_angle - d, self.shape[-2])

        z = np.repeat(array, repeat, axis=-1)
        r, th = np.meshgrid(radial_grid, azimuthal_grid)

        im = ax.pcolormesh(th, r, z.T, shading='auto', **kwargs)
        ax.set_rlim([0, self.outer_angle * 1.1])

        if grid:
            ax.grid()

        return ax, im
