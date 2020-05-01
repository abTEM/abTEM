import os
from typing import Optional, Union, Any, Sequence

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np

from abtem.config import DTYPE, CUPY_DTYPE
from abtem.utils import energy2wavelength, energy2sigma, abs2
from collections.abc import Iterable
from skimage.io import imsave
from sys import getsizeof


def notify(func):
    """
    Decorator for class methods that have to notify.

    Parameters
    ----------
    func : function
        notifying function

    Returns
    -------

    """
    name = func.__name__

    def wrapper(*args):
        obj, value = args
        old = getattr(obj, name)
        func(*args)
        change = np.any(old != value)
        obj.notify_observers({'notifier': name, 'change': change})

    return wrapper


class Observable:

    def __init__(self, **kwargs):
        """
        Observable base class.

        Base class for creating an observable class in the classic observer design pattern.

        :param kwargs: dummy
        """
        self._observers = []
        super().__init__(**kwargs)

    @property
    def observers(self) -> list:
        return self._observers

    def register_observer(self, observer: 'Observer'):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.notify(self, message)


class Observer:

    def __init__(self, **kwargs):
        """
        Observer base class.

        Base class for creating an observer class in the classic observer design pattern.

        :param kwargs: dummy
        """
        super().__init__(**kwargs)

    def observe(self, observable: Observable):
        observable.register_observer(self)

    def notify(self, observable: Observable, message: dict):
        raise NotImplementedError()


class Cache(Observer):

    def __init__(self, **kwargs):
        """
        Observer with a cache.

        This object has a dictionary for saving results that might be
        used for further calculations.

        :param kwargs: dummy
        """
        self._cache = {}
        self._clear_conditions = {}

        super().__init__(**kwargs)

    @property
    def cache(self) -> dict:
        return self._cache

    def retrieve_from_cache(self, key: str):
        return self._cache[key]

    def update_cache(self, key: str, data: Any, clear_condition: Optional[tuple] = None):
        self._cache[key] = data
        self._clear_conditions[key] = clear_condition

    def notify(self, observable: Observable, message: dict):
        pop = []
        if message['change']:
            for key, conditions in self._clear_conditions.items():
                if conditions is None:
                    pop.append(key)
                elif message['notifier'] in conditions:
                    pop.append(key)

        for key in pop:
            self._cache.pop(key, None)
            self._clear_conditions.pop(key, None)

    def clear_cache(self):
        self._cache = {}


def cached_method(clear_conditions=None):
    def wrapper(func):
        def new_func(*args):
            self = args[0]
            name = func.__name__
            try:
                return self.retrieve_from_cache(name)
            except KeyError:
                data = func(*args)
                self.update_cache(name, data, clear_conditions)
                return data

        return new_func

    return wrapper


def cached_method_with_args(clear_conditions=None):
    def wrapper(func):
        def new_func(*args):
            self = args[0]
            name = func.__name__
            try:
                return self.retrieve_from_cache(name)[args[1:]]
            except KeyError:
                data = func(*args)
                try:
                    self._cache[name][args[1:]] = data
                except KeyError:
                    self.update_cache(name, {args[1:]: data}, clear_conditions)
                return data

        return new_func

    return wrapper


class Grid(Observable):

    def __init__(self,
                 extent: Union[float, Sequence[float]] = None,
                 gpts: Union[int, Sequence[int]] = None,
                 sampling: Union[float, Sequence[float]] = None,
                 dimensions: int = 2,
                 endpoint: bool = False,
                 lock_extent=False,
                 lock_gpts=False,
                 lock_sampling=False,
                 **kwargs):

        """
        Grid object.

        The grid object represent the simulation grid on which the wave function and potential is discretized.

        Parameters
        ----------
        extent : sequence of float, float, optional
            Grid extent in each dimension [Å]
        gpts : sequence of int, int, optional
            Number of grid points in each dimension
        sampling : sequence of float, float, optional
            Grid sampling in each dimension [1 / Å]
        dimensions : int
            Number of dimensions represented by the grid.
        endpoint : bool, optional
            If true include the grid endpoint (the dafault is False). For periodic grids the endpoint should not be
            included.
        kwargs :
        """

        self._dimensions = dimensions
        self._endpoint = endpoint

        if sum([lock_extent, lock_gpts, lock_sampling]) > 1:
            raise RuntimeError('at most one of extent, gpts, and sampling may be locked')

        self._lock_extent = lock_extent
        self._lock_gpts = lock_gpts
        self._lock_sampling = lock_sampling

        self._extent = self._validate(extent, dtype=DTYPE)
        self._gpts = self._validate(gpts, dtype=np.int)
        self._sampling = self._validate(sampling, dtype=DTYPE)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)

        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)

        self._adjust_sampling(self.extent, self.gpts)

        super().__init__(**kwargs)

    def _validate(self, value, dtype):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self._dimensions:
                raise RuntimeError('grid value length of {} != {}'.format(len(value), self._dimensions))
            return np.array(value).astype(dtype)

        if isinstance(value, (int, float, complex)):
            return np.full(self._dimensions, value, dtype=dtype)

        if value is None:
            return value

        raise RuntimeError('invalid grid property ({})'.format(value))

    @property
    def endpoint(self) -> bool:
        return self._endpoint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def extent(self) -> np.ndarray:
        return self._extent

    @extent.setter
    @notify
    def extent(self, extent: Union[float, Sequence[float]]):
        if self._lock_extent:
            raise RuntimeError('extent locked')

        extent = self._validate(extent, dtype=DTYPE)

        if self._lock_sampling:
            self._adjust_gpts(extent, self.sampling)
            self._adjust_sampling(self.extent, self.gpts)
        else:
            self._adjust_sampling(extent, self.gpts)

        self._extent = extent

    @property
    def gpts(self) -> np.ndarray:
        return self._gpts

    @gpts.setter
    @notify
    def gpts(self, gpts: Union[int, Sequence[int]]):
        if self._lock_gpts:
            raise RuntimeError('gpts locked')

        gpts = self._validate(gpts, dtype=np.int)

        if self._lock_sampling:
            self._adjust_extent(gpts, self.sampling)
        else:
            self._adjust_sampling(self.extent, gpts)

        self._gpts = gpts

    @property
    def sampling(self) -> np.ndarray:
        return self._sampling

    @sampling.setter
    @notify
    def sampling(self, sampling):
        if self._lock_sampling:
            raise RuntimeError('sampling locked')

        sampling = self._validate(sampling, dtype=DTYPE)

        if self._lock_gpts:
            self._adjust_extent(self.gpts, sampling)

        else:
            self._adjust_gpts(self.extent, sampling)

        self._adjust_sampling(self.extent, self.gpts)

    def _adjust_extent(self, gpts, sampling):
        if (gpts is not None) & (sampling is not None):
            if self._endpoint:
                self._extent = (gpts - 1) * sampling
            else:
                self._extent = gpts * sampling

    def _adjust_gpts(self, extent, sampling):
        if (extent is not None) & (sampling is not None):
            if self._endpoint:
                self._gpts = np.ceil(extent / sampling).astype(np.int) + 1
            else:
                self._gpts = np.ceil(extent / sampling).astype(np.int)

    def _adjust_sampling(self, extent, gpts):
        if (extent is not None) & (gpts is not None):
            if self._endpoint:
                self._sampling = extent / (gpts - 1)
            else:
                self._sampling = extent / gpts

    def check_is_grid_defined(self):
        """ Throw error if the grid is not defined. """
        if self.extent is None:
            raise RuntimeError('grid extent is not defined')

        elif self.gpts is None:
            raise RuntimeError('grid gpts is not defined')

    @property
    def spatial_frequency_limits(self):
        return np.array([(-1 / (2 * d), 1 / (2 * d) - 1 / (d * p)) if (p % 2 == 0) else
                         (-1 / (2 * d) + 1 / (2 * d * p), 1 / (2 * d) - 1 / (2 * d * p)) for d, p in
                         zip(self.sampling, self.gpts)])

    @property
    def spatial_frequency_extent(self):
        fourier_limits = self.spatial_frequency_limits
        return fourier_limits[:, 1] - fourier_limits[:, 0]

    def match_grid(self, other):
        self.check_grids_can_match(other)

        if (self.extent is None) & (other.extent is None):
            raise RuntimeError('grid extent cannot be inferred')

        elif self.extent is None:
            self.extent = other.extent

        elif other.extent is None:
            other.extent = self.extent

        if (self.gpts is None) & (other.gpts is None):
            raise RuntimeError('grid gpts cannot be inferred')

        elif self.gpts is None:
            self.gpts = other.gpts

        elif other.gpts is None:
            other.gpts = self.gpts

    def check_grids_can_match(self, other):
        """ Throw error if the grid of another object is different from this object. """

        if (self.extent is not None) & (other.extent is not None) & np.any(self.extent != other.extent):
            raise RuntimeError('inconsistent grid extent ({} != {})'.format(self.extent, other.extent))

        elif (self.gpts is not None) & (other.gpts is not None) & np.any(self.gpts != other.gpts):
            raise RuntimeError('inconsistent grid gpts ({} != {})'.format(self.gpts, other.gpts))

    def linspace(self):
        self.check_is_grid_defined()
        return [np.linspace(0, e, g, endpoint=self.endpoint, dtype=DTYPE) for g, e in zip(self.gpts, self.extent)]

    def fftfreq(self):
        self.check_is_grid_defined()
        return [DTYPE(np.fft.fftfreq(g, s)) for g, s in zip(self.gpts, self.sampling)]

    def copy(self):
        return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
                              dimensions=self._dimensions)


class Energy(Observable):
    """
    Energy base class

    Base class for describing the energy of wavefunctions and transfer functions.

    :param energy: energy
    :type energy: optional, float
    """

    def __init__(self, energy: Optional[float] = None, **kwargs):
        """
        Energy base class.

        The Energy object is used to represent the acceleration energy of an inheriting waves object.

        Parameters
        ----------
        energy : float
            Acceleration energy [eV]
        kwargs :
        """
        if energy is not None:
            energy = DTYPE(energy)

        self._energy = energy

        super().__init__(**kwargs)

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    @notify
    def energy(self, value: float):
        if value is not None:
            value = DTYPE(value)

        self._energy = value

    @property
    def wavelength(self) -> float:
        """
        Relativistic wavelength from energy.
        :return: wavelength
        :rtype: float
        """
        self.check_is_energy_defined()
        return DTYPE(energy2wavelength(self.energy))

    @property
    def sigma(self) -> float:
        """
        Interaction parameter from energy.
        """
        self.check_is_energy_defined()
        return DTYPE(energy2sigma(self.energy))

    def check_is_energy_defined(self):
        """ Throw error if the energy is not defined. """

        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def check_energies_can_match(self, other: 'Energy'):
        if (self.energy is not None) & (other.energy is not None) & (self.energy != other.energy):
            raise RuntimeError('inconsistent energies')

    def match_energy(self, other):
        self.check_energies_can_match(other)

        if (self.energy is None) & (other.energy is None):
            raise RuntimeError('energy cannot be inferred')

        elif self.energy is None:
            self.energy = other.energy

        elif other.energy is None:
            other.energy = self.energy

    def copy(self) -> 'Energy':
        """
        :return: A copy of itself
        :rtype: Energy
        """
        return self.__class__(self.energy)


class Buildable(object):

    def __init__(self, build_on_gpu=False, **kwargs):
        self._build_on_gpu = build_on_gpu
        super().__init__(**kwargs)

    def _dtype(self):
        if self._build_on_gpu:
            return CUPY_DTYPE
        else:
            return DTYPE

    @property
    def build_on_gpu(self):
        return self._build_on_gpu

    def _array_module(self):
        if self._build_on_gpu:
            return cp
        else:
            return np


class ArrayWithGrid(Grid):

    def __init__(self, array, spatial_dimensions, extent=None, sampling=None, fourier_space=False, endpoint=False,
                 **kwargs):
        array_dimensions = len(array.shape)

        if array_dimensions < spatial_dimensions:
            raise RuntimeError('array dimensions exceeds spatial dimensions')

        self._array = array
        self._spatial_dimensions = spatial_dimensions
        self._fourier_space = fourier_space

        super().__init__(extent=extent, sampling=sampling, lock_gpts=True, dimensions=spatial_dimensions,
                         endpoint=endpoint, **kwargs)

    @property
    def shape(self):
        return self._array.shape

    @property
    def memory_usage(self):
        return getsizeof(self._array)

    @property
    def gpts(self):
        return self._array.shape[-self._spatial_dimensions:]

    @property
    def spatial_dimensions(self):
        return self._spatial_dimensions

    @property
    def fourier_space(self):
        return self._fourier_space

    @property
    def array(self):
        return self._array

    def tile(self, repetitions):
        if not isinstance(repetitions, Iterable):
            repetitions = (repetitions) * self.spatial_dimensions

        if len(repetitions) != self.spatial_dimensions:
            raise RuntimeError()

        self._extent = repetitions * self._extent
        repetitions = (1,) * (len(repetitions) - self.spatial_dimensions) + repetitions
        self._array = np.tile(self._array, repetitions)
        return self

    def save_as_image(self, fname, dtype='uint16', **kwargs):
        array = (self._array - self._array.min()) / self._array.ptp() * np.iinfo(dtype).max
        array = array.astype(dtype)
        imsave(fname, array, **kwargs)

    def to_gpu(self):
        return self.copy(to_gpu=True)

    def __getitem__(self, item):
        if len(self.array.shape) <= self.spatial_dimensions:
            raise RuntimeError()
        return self.__class__(array=self._array[item], spatial_dimensions=self.spatial_dimensions - 1,
                              extent=self.extent.copy(), fourier_space=self.fourier_space)

    def write(self, path):
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('class', (100,), dtype='S100')
            dset[:] = 'abtem.bases.ArrayWithGrid'
            f.create_dataset('array', data=self.array)
            f.create_dataset('spatial_dimensions', data=self.spatial_dimensions)
            f.create_dataset('extent', data=self.extent)
            f.create_dataset('fourier_space', data=self.fourier_space)
        return path

    @classmethod
    def read(cls, path):
        # TODO: implement chunks
        with h5py.File(path, 'r') as f:
            datasets = {}
            for key in f.keys():
                datasets[key] = f.get(key)[()]
        return cls(**datasets)

    def copy(self, to_gpu=False):
        if to_gpu:
            new_array = cp.asarray(self.array)
        else:
            new_array = self.array.copy()
        return self.__class__(array=new_array, spatial_dimensions=self.spatial_dimensions,
                              extent=self.extent.copy(), fourier_space=self.fourier_space)


class ArrayWithGrid1D(ArrayWithGrid):

    def __init__(self, array, extent=None, sampling=None, fourier_space=False, endpoint=False, **kwargs):
        super().__init__(array=array, spatial_dimensions=1, extent=extent, sampling=sampling,
                         fourier_space=fourier_space, endpoint=endpoint, **kwargs)

    def plot(self, ax=None, complex_reduction=None, fourier_space=False, title=None, figsize=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        array = np.squeeze(self.array)

        if self.fourier_space & fourier_space:
            array = np.fft.fftshift(array)

        elif (not self.fourier_space) & fourier_space:
            array = np.fft.fftshift(np.fft.fftn(array))

        elif (self.fourier_space) & (not fourier_space):
            array = np.fft.ifftn(array)

        if complex_reduction is not None:
            if isinstance(complex_reduction, str):
                if complex_reduction == 'phase':
                    complex_reduction = np.angle
                elif complex_reduction == 'abs2':
                    complex_reduction = abs2

            array = complex_reduction(array)

        if fourier_space:
            x_label = 'kx [1 / Å]'
            x = self.fftfreq()[0]

        else:
            x_label = 'x [Å]'
            x = self.linspace()[0]

        if np.iscomplexobj(array):
            ax.plot(x, array.real)
            ax.plot(x, array.imag)
        else:
            ax.plot(x, array)

        ax.set_xlabel(x_label)

        if title is not None:
            ax.set_title(title)

    def write(self, path):
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('class', (1,), dtype='S100')
            dset[:] = np.string_('abtem.bases.ArrayWithGrid1D')
            f.create_dataset('array', data=self.array)
            f.create_dataset('extent', data=self.extent)
            f.create_dataset('fourier_space', data=self.fourier_space)
        return path


class ArrayWithGrid2D(ArrayWithGrid):

    def __init__(self, array, extent=None, sampling=None, fourier_space=False, endpoint=False, **kwargs):
        super().__init__(array=array, spatial_dimensions=2, extent=extent, sampling=sampling,
                         fourier_space=fourier_space, endpoint=endpoint, **kwargs)

    def plot(self, ax=None, logscale=False, logscale_constant=1., complex_representation=None,
             fourier_space=None, title=None, cmap='gray', figsize=None, scans=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        array = np.squeeze(self.array)

        array = cp.asnumpy(array)

        if len(array.shape) > 2:
            raise RuntimeError('array dimension greater 2, set reduction operation')

        if fourier_space is None:
            fourier_space = self.fourier_space

        if self.fourier_space & fourier_space:
            array = np.fft.fftshift(array)

        elif (not self.fourier_space) & fourier_space:
            array = np.fft.fftshift(np.fft.fftn(array))

        elif (self.fourier_space) & (not fourier_space):
            array = np.fft.ifftn(array)

        if (complex_representation is None) & np.iscomplexobj(array):
            array = abs2(array)

        elif complex_representation is not None:
            if isinstance(complex_representation, str):
                if complex_representation == 'phase':
                    complex_representation = np.angle
                elif complex_representation == 'abs2':
                    complex_representation = abs2

            array = complex_representation(array)

        if logscale:
            array = np.log(1 + logscale_constant * array)

        if fourier_space:
            x_label = 'kx [1 / Å]'
            y_label = 'ky [1 / Å]'
            extent = self.spatial_frequency_limits.ravel()
        else:
            x_label = 'x [Å]'
            y_label = 'y [Å]'
            extent = [0, self.extent[0], 0, self.extent[1]]

        im = ax.imshow(array.T, extent=extent, cmap=cmap, origin='lower', **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if title is not None:
            ax.set_title(title)

        if scans is not None:
            if not isinstance(scans, Iterable):
                scans = [scans]

            for scan in scans:
                scan.add_to_mpl_plot(ax)

        return ax, im

    def write(self, path):
        with h5py.File(path, 'w') as f:
            dset = f.create_dataset('class', (1,), dtype='S100')
            dset[:] = np.string_('abtem.bases.ArrayWithGrid2D')
            f.create_dataset('array', data=self.array)
            f.create_dataset('extent', data=self.extent)
            f.create_dataset('fourier_space', data=self.fourier_space)
        return path


class ArrayWithGridAndEnergy2D(ArrayWithGrid2D, Energy):

    def __init__(self, array, extent=None, sampling=None, energy=None, fourier_space=False,
                 **kwargs):
        super().__init__(array=array, extent=extent, sampling=sampling, energy=energy, fourier_space=fourier_space,
                         **kwargs)

    def copy(self, to_gpu=False):
        if to_gpu:
            new_array = cp.asarray(self.array)
        else:
            new_array = self.array.copy()
        return self.__class__(array=new_array, spatial_dimensions=self.spatial_dimensions,
                              extent=self.extent.copy(), energy=self.energy, fourier_space=self.fourier_space)
