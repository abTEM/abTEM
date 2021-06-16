"""Module for often-used base classes."""
from collections import OrderedDict
from copy import copy
from typing import Optional, Union, Sequence, Any, Callable, Tuple

import numpy as np

from abtem.device import copy_to_device, get_array_module, get_device_function
from abtem.utils import energy2wavelength, energy2sigma, spatial_frequencies


class Event(object):
    """
    Event class for registering callbacks.
    """

    def __init__(self):
        self.callbacks = []
        self._notify_count = 0

    @property
    def notify_count(self):
        """
        Number of times this event has been notified.
        """
        return self._notify_count

    def notify(self, change):
        """
        Notify this event. All registered callbacks are called.
        """

        self._notify_count += 1
        for callback in self.callbacks:
            callback(change)

    def observe(self, callbacks: Union[Callable, Sequence[Callable]]):
        """
        Register new callbacks.

        Parameters
        ----------
        callbacks : callable
            The callbacks to register.
        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks += callbacks


class HasEventMixin:
    _event: Event

    @property
    def event(self):
        return self._event

    def observe(self, callback):
        self.event.observe(callback)


def watched_method(event: 'str'):
    """
    Decorator for class methods that have to notify.

    Parameters
    ----------
    event : str
        Name class property with target event.
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args, **kwargs):
            instance = args[0]
            result = func(*args, **kwargs)
            getattr(instance, event).notify({'owner': instance, 'name': property_name, 'change': True})
            return result

        return new_func

    return wrapper


def watched_property(event: 'str'):
    """
    Decorator for class properties that have to notify an event.

    Parameters
    ----------
    event : str
        Name class property with target event
    """

    def wrapper(func):
        property_name = func.__name__

        def new_func(*args):
            instance, value = args
            old = getattr(instance, property_name)
            result = func(*args)
            change = np.any(old != value)
            getattr(instance, event).notify({'notifier': instance, 'name': property_name, 'change': change,
                                             'old': old, 'new': value})
            return result

        return new_func

    return wrapper


def cache_clear_callback(target_cache: 'Cache'):
    """
    Helper function for creating a callback that clears a target cache object.

    Parameters
    ----------
    target_cache : Cache object
        The target cache object.
    """

    # noinspection PyUnusedLocal
    def callback(change):
        if change['change']:
            target_cache.clear()

    return callback


def cached_method(target_cache_property: str):
    """
    Decorator for cached methods. The method will store the output in the cache held by the target property.

    Parameters
    ----------
    target_cache_property : str
        The property holding the target cache.
    """

    def wrapper(func):

        def new_func(*args):
            cache = getattr(args[0], target_cache_property)
            key = (func,) + args[1:]

            if key in cache.cached:
                # The decorated method has been called once with the given args.
                # The calculation will be retrieved from cache.

                result = cache.retrieve(key)
                cache._hits += 1
            else:
                # The method will be called and its output will be cached.

                result = func(*args)
                cache.insert(key, result)
                cache._misses += 1

            return result

        return new_func

    return wrapper


def copy_docstring_from(source):
    def wrapper(func):
        func.__doc__ = source.__doc__
        return func

    return wrapper


class Cache:
    """
    Cache object.

    Class for handling a dictionary-based cache. When the cache is full, the first inserted item is deleted.

    Parameters
    ----------
    max_size : int
        The maximum number of values stored by this cache.
    """

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cached = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def cached(self) -> dict:
        """
        Dictionary of cached data.
        """
        return self._cached

    @property
    def hits(self) -> int:
        """
        Number of times a previously calculated object was retrieved.
        """
        return self._hits

    @property
    def misses(self) -> int:
        """
        Number of times a new object had to be calculated.
        """
        return self._hits

    def __len__(self) -> int:
        """
        Number of objects cached.
        """
        return len(self._cached)

    def insert(self, key: Any, value: Any):
        """
        Insert new value into the cache.

        Parameters
        ----------
        key : Any
            The dictionary key of the cached object.
        value : Any
            The object to cache.
        """
        self._cached[key] = value
        self._check_size()

    def retrieve(self, key: Any) -> Any:
        """
        Retrieve object from cache.

        Parameters
        ----------
        key: Any
            The key of the cached item.

        Returns
        -------
        Any
            The cached object.
        """
        return self._cached[key]

    def _check_size(self):
        """
        Delete item from cache, if it is too large.
        """
        if self._max_size is not None:
            while len(self) > self._max_size:
                self._cached.popitem(last=False)

    def clear(self):
        """
        Clear the cache.
        """
        self._cached = OrderedDict()
        self._hits = 0
        self._misses = 0


class Grid(HasEventMixin):
    """
    Grid object.

    The grid object represent the simulation grid on which the wave functions and potential are discretized.

    Parameters
    ----------
    extent : two float
        Grid extent in each dimension [Å].
    gpts : two int
        Number of grid points in each dimension.
    sampling : two float
        Grid sampling in each dimension [1 / Å].
    dimensions : int
        Number of dimensions represented by the grid.
    endpoint : bool
        If true include the grid endpoint. Default is False. For periodic grids the endpoint should not be included.
    """

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 dimensions: int = 2,
                 endpoint: Union[int, Sequence[bool]] = False,
                 lock_extent: bool = False,
                 lock_gpts: bool = False,
                 lock_sampling: bool = False):

        self._event = Event()
        self._dimensions = dimensions

        if isinstance(endpoint, bool):
            endpoint = (endpoint,) * dimensions

        self._endpoint = tuple(endpoint)

        if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
            raise RuntimeError('At most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = self._validate(extent, dtype=float)
        self._gpts = self._validate(gpts, dtype=int)
        self._sampling = self._validate(sampling, dtype=float)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        self._adjust_sampling(self.extent, self.gpts)

    def _validate(self, value, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError('Grid value length of {} != {}'.format(len(value), self._dimensions))
            return tuple((map(dtype, value)))

        if isinstance(value, (int, float, complex)):
            return (dtype(value),) * self.dimensions

        if value is None:
            return value

        raise RuntimeError('Invalid grid property ({})'.format(value))

    def __len__(self) -> int:
        return self.dimensions

    @property
    def endpoint(self) -> tuple:
        """Include the grid endpoint."""
        return self._endpoint

    @property
    def dimensions(self) -> int:
        """Number of dimensions represented by the grid."""
        return self._dimensions

    @property
    def extent(self) -> tuple:
        """Grid extent in each dimension [Å]."""
        return self._extent

    @extent.setter
    @watched_method('_event')
    def extent(self, extent: Union[float, Sequence[float]]):
        if self._lock_extent:
            raise RuntimeError('Extent cannot be modified')

        extent = self._validate(extent, dtype=np.float32)

        if self._lock_sampling or (self.gpts is None):
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(extent, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> tuple:
        """Number of grid points in each dimension."""
        return self._gpts

    @gpts.setter
    @watched_method('_event')
    def gpts(self, gpts: Union[int, Sequence[int]]):
        if self._lock_gpts:
            raise RuntimeError('Grid gpts cannot be modified')

        gpts = self._validate(gpts, dtype=int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, gpts)
        else:
            self._adjust_extent(gpts, self.sampling)

        self._gpts = gpts

    @property
    def sampling(self) -> tuple:
        """Grid sampling in each dimension [1 / Å]."""
        return self._sampling

    @sampling.setter
    @watched_method('_event')
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError('Sampling cannot be modified')

        sampling = self._validate(sampling, dtype=np.float32)
        if self._lock_gpts:
            self._adjust_extent(self.gpts, sampling)
        elif self.extent is not None:
            self._adjust_gpts(self.extent, sampling)
        else:
            self._adjust_extent(self.gpts, sampling)

        self._adjust_sampling(self.extent, self.gpts)

    def _adjust_extent(self, gpts: tuple, sampling: tuple):
        if (gpts is not None) & (sampling is not None):
            self._extent = tuple((n - 1) * d if e else n * d for n, d, e in zip(gpts, sampling, self._endpoint))
            self._extent = self._validate(self._extent, float)

    def _adjust_gpts(self, extent: tuple, sampling: tuple):
        if (extent is not None) & (sampling is not None):
            self._gpts = tuple(int(np.ceil(r / d)) + 1 if e else int(np.ceil(r / d))
                               for r, d, e in zip(extent, sampling, self._endpoint))

    def _adjust_sampling(self, extent: tuple, gpts: tuple):
        if (extent is not None) & (gpts is not None):
            self._sampling = tuple(r / (n - 1) if e else r / n for r, n, e in zip(extent, gpts, self._endpoint))
            self._sampling = self._validate(self._sampling, float)

    def check_is_defined(self):
        """
        Raise error if the grid is not defined.
        """

        if self.extent is None:
            raise RuntimeError('Grid extent is not defined')

        elif self.gpts is None:
            raise RuntimeError('Grid gpts are not defined')

    def match(self, other: Union['Grid', 'HasGridMixin'], check_match: bool = False):
        """
        Set the parameters of this grid to match another grid.

        Parameters
        ----------
        other : Grid object
            The grid that should be matched.
        check_match : bool
            If true check whether grids can match without overriding already defined grid parameters.
        """

        if check_match:
            self.check_match(other)

        if (self.extent is None) & (other.extent is None):
            raise RuntimeError('Grid extent cannot be inferred')
        elif other.extent is None:
            other.extent = self.extent
        elif np.any(np.array(self.extent, np.float32) != np.array(other.extent, np.float32)):
            self.extent = other.extent

        if (self.gpts is None) & (other.gpts is None):
            raise RuntimeError('Grid gpts cannot be inferred')
        elif other.gpts is None:
            other.gpts = self.gpts
        elif np.any(self.gpts != other.gpts):
            self.gpts = other.gpts

    def check_match(self, other):
        """
        Raise error if the grid of another object is different from this object.

        Parameters
        ----------
        other : Grid object
            The grid that should be checked.
        """

        if (self.extent is not None) & (other.extent is not None):
            if not np.all(np.isclose(self.extent, other.extent)):
                raise RuntimeError('Inconsistent grid extent ({} != {})'.format(self.extent, other.extent))

        if (self.gpts is not None) & (other.gpts is not None):
            if not np.all(self.gpts == other.gpts):
                raise RuntimeError('Inconsistent grid gpts ({} != {})'.format(self.gpts, other.gpts))

    def round_to_power(self, power: int = 2):
        """
        Round the grid gpts up to the nearest value that is a power of n. Fourier transforms are faster for arrays of
        whose size can be factored into small primes (2, 3, 5 and 7).

        Parameters
        ----------
        power : int
            The gpts will be a power of this number.
        """

        self.gpts = tuple(power ** np.ceil(np.log(n) / np.log(power)) for n in self.gpts)

    def __copy__(self):
        return self.__class__(extent=self.extent,
                              gpts=self.gpts,
                              sampling=self.sampling,
                              dimensions=self.dimensions,
                              endpoint=self.endpoint,
                              lock_extent=self._lock_extent,
                              lock_gpts=self._lock_gpts,
                              lock_sampling=self._lock_sampling)

    def copy(self):
        """
        Make a copy.
        """
        return copy(self)


class HasGridMixin:
    _grid: Grid

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    @copy_docstring_from(Grid.extent)
    def extent(self):
        return self.grid.extent

    @extent.setter
    def extent(self, extent):
        self.grid.extent = extent

    @property
    @copy_docstring_from(Grid.gpts)
    def gpts(self):
        return self.grid.gpts

    @gpts.setter
    def gpts(self, gpts):
        self.grid.gpts = gpts

    @property
    @copy_docstring_from(Grid.sampling)
    def sampling(self):
        return self.grid.sampling

    @sampling.setter
    def sampling(self, sampling):
        self.grid.sampling = sampling

    def match_grid(self, other, check_match=False):
        self.grid.match(other, check_match=check_match)


class Accelerator(HasEventMixin):
    """
    Accelerator object describes the energy of wave functions and transfer functions.

    Parameters
    ----------
    energy: float
        Acceleration energy [eV].
    """

    def __init__(self, energy: Optional[float] = None, lock_energy=False):
        if energy is not None:
            energy = float(energy)

        self._event = Event()
        self._energy = energy
        self._lock_energy = lock_energy

    @property
    def energy(self) -> float:
        """
        Acceleration energy [eV].
        """
        return self._energy

    @energy.setter
    @watched_method('_event')
    def energy(self, value: float):
        if self._lock_energy:
            raise RuntimeError('Energy cannot be modified')

        if value is not None:
            value = float(value)
        self._energy = value

    @property
    def wavelength(self) -> float:
        """ Relativistic wavelength [Å]. """
        self.check_is_defined()
        return energy2wavelength(self.energy)

    @property
    def sigma(self) -> float:
        """ Interaction parameter. """
        self.check_is_defined()
        return energy2sigma(self.energy)

    def check_is_defined(self):
        """
        Raise error if the energy is not defined.
        """
        if self.energy is None:
            raise RuntimeError('Energy is not defined')

    def check_match(self, other: 'Accelerator'):
        """
        Raise error if the accelerator of another object is different from this object.

        Parameters
        ----------
        other: Accelerator object
            The accelerator that should be checked.
        """
        if (self.energy is not None) & (other.energy is not None) & (self.energy != other.energy):
            raise RuntimeError('Inconsistent energies')

    def match(self, other, check_match=False):
        """
        Set the parameters of this accelerator to match another accelerator.

        Parameters
        ----------
        other: Accelerator object
            The accelerator that should be matched.
        check_match: bool
            If true check whether accelerators can match without overriding an already defined energy.
        """

        if check_match:
            self.check_match(other)

        if (self.energy is None) & (other.energy is None):
            raise RuntimeError('Energy cannot be inferred')

        if other.energy is None:
            other.energy = self.energy

        else:
            self.energy = other.energy

    def __copy__(self):
        return self.__class__(self.energy)

    def copy(self):
        """Make a copy."""
        return copy(self)


class HasAcceleratorMixin:
    _accelerator: Accelerator

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, new: Accelerator):
        self._accelerator = new
        self._accelerator._event = new._event

    @property
    @copy_docstring_from(Accelerator.energy)
    def energy(self):
        return self.accelerator.energy

    @energy.setter
    def energy(self, energy):
        self.accelerator.energy = energy

    @property
    @copy_docstring_from(Accelerator.wavelength)
    def wavelength(self):
        return self.accelerator.wavelength


class BeamTilt(HasEventMixin):

    def __init__(self, tilt: Tuple[float, float] = (0., 0.)):
        self._tilt = tilt
        self._event = Event()

    @property
    def tilt(self) -> Tuple[float, float]:
        """Beam tilt [mrad]."""
        return self._tilt

    @tilt.setter
    @watched_method('_event')
    def tilt(self, value: Tuple[float, float]):
        self._tilt = value


class HasBeamTiltMixin:
    _beam_tilt: BeamTilt

    @property
    @copy_docstring_from(BeamTilt.tilt)
    def tilt(self) -> Tuple[float, float]:
        return self._beam_tilt.tilt

    @tilt.setter
    def tilt(self, value: Tuple[float, float]):
        self.tilt = value


class AntialiasFilter(HasEventMixin):
    """
    Antialias filter object.
    """
    cutoff = 2 / 3.
    rolloff = .1

    def __init__(self):
        self._mask_cache = Cache(1)
        self._event = Event()
        self._event.observe(cache_clear_callback(self._mask_cache))

    @cached_method('_mask_cache')
    def get_mask(self, gpts, sampling, xp):
        if sampling is None:
            sampling = (1., 1.)

        kx, ky = spatial_frequencies(gpts, sampling)
        kx = copy_to_device(kx, xp)
        ky = copy_to_device(ky, xp)
        k = xp.sqrt(kx[:, None] ** 2 + ky[None] ** 2)

        kcut = 1 / max(sampling) / 2 * self.cutoff

        if self.rolloff > 0.:
            array = .5 * (1 + xp.cos(np.pi * (k - kcut + self.rolloff) / self.rolloff))
            array[k > kcut] = 0.
            array = xp.where(k > kcut - self.rolloff, array, xp.ones_like(k, dtype=xp.float32))
        else:
            array = xp.array(k < kcut).astype(xp.float32)
        return array

    def _bandlimit(self, array):
        xp = get_array_module(array)
        fft2_convolve = get_device_function(xp, 'fft2_convolve')
        array = fft2_convolve(array, self.get_mask(array.shape[-2:], (1, 1), xp), overwrite_x=True)
        return array

    def bandlimit(self, waves):
        """

        Parameters
        ----------
        waves

        Returns
        -------

        """
        xp = get_array_module(waves.array)
        fft2_convolve = get_device_function(xp, 'fft2_convolve')
        waves._array = fft2_convolve(waves.array, self.get_mask(waves.gpts, waves.sampling, xp), overwrite_x=True)
        return waves


class AntialiasAperture:

    def __init__(self, antialias_aperture=(2 / 3., 2 / 3.)):
        self._antialias_aperture = antialias_aperture

    @property
    def antialias_aperture(self) -> Tuple[float, float]:
        """Anti-aliasing aperture as a fraction of the Nyquist frequency."""
        return self._antialias_aperture

    @antialias_aperture.setter
    def antialias_aperture(self, value: Tuple[float, float]):
        self._antialias_aperture = value


class HasAntialiasAperture(HasEventMixin):
    _antialias_aperture: AntialiasAperture

    @property
    @copy_docstring_from(AntialiasAperture.antialias_aperture)
    def antialias_aperture(self) -> Tuple[float, float]:
        return self._antialias_aperture.antialias_aperture

    @antialias_aperture.setter
    def antialias_aperture(self, value: Tuple[float, float]):
        self._antialias_aperture.antialias_aperture = value
