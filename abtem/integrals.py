"""Module to describe projection integrals of radial potential parametrizations."""

from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from numba import jit  # type: ignore
from scipy import integrate  # type: ignore
from scipy.optimize import brentq  # type: ignore
from scipy.special import erf  # type: ignore

from abtem.core.backend import (
    cp,
    cupyx,
    device_name_from_array_module,
    get_array_module,
    get_ndimage_module,
)
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import (
    disk_meshgrid,
    polar_spatial_frequencies,
    spatial_frequencies,
)
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype
from abtem.parametrizations import validate_parametrization

if cp is not None:
    from abtem.core._cuda import (
        interpolate_radial_functions as interpolate_radial_functions_cuda,
    )
else:
    interpolate_radial_functions_cuda = None

if TYPE_CHECKING:
    from abtem.parametrizations import Parametrization


_MAX_CACHE_ENTRIES = 32

# Kept as the name the scattering-factor tests import.
_MAX_SCATTERING_FACTOR_ENTRIES = _MAX_CACHE_ENTRIES


class _CacheStateMixin:
    """Normalise cache attributes restored from an older pickle.

    ``__setstate__`` gets ``__dict__`` verbatim, so an integrator pickled
    before these caches existed -- ``_sinc_cache`` only arrived in PR #269 --
    comes back missing them, and one pickled while they were plain dicts comes
    back with objects that have ``.get`` but not ``.put``. The miss path, not
    the hit path, would then raise. Restore both to the current container.
    """

    _cache_attributes: tuple[str, ...] = ()

    def __getstate__(self):
        """Drop the caches on the way out.

        abTEM pickles integrators into every task graph, so a populated cache
        rides along to every worker. Measured on a warm four-configuration
        frozen-phonon graph at 512^2, the caches this file keeps inflate it
        126x (scattering factor), 532x (quadrature) and 39x (gaussian); a
        sampling sweep takes a pickled quadrature integrator to 69 MB. That is
        the same mechanism this commit cites when deleting the two caches that
        never stored -- keeping the others un-dropped while deleting those for
        graph inflation would be incoherent.

        They are pure derived state that any worker can rebuild, and
        __setstate__ below recreates them empty. Same discipline as
        _local_potential_on_device.__getstate__ in core_loss.py (PR #375).

        This also settles the __eq__/tokenize disagreement: two integrators
        that compare equal now tokenize equal, because tokenize pickles.
        """
        state = self.__dict__.copy()
        for name in self._cache_attributes:
            state.pop(name, None)
        return state

    def __setstate__(self, state):
        """Restore, with every cache empty.

        Contents are deliberately not carried over, even when the state dict
        has them. Only a pickle written before __getstate__ existed can, and
        its entries are keyed in the *old* shape -- _sorted_disks without a
        precision component, _tables without sampling -- so reusing them would
        serve a value found under an incomplete key, which is the whole defect
        this commit exists to fix. They are derived state; recomputing costs a
        miss and gives the same answer.

        The attribute still has to be replaced rather than trusted: an
        integrator pickled before these caches existed (_sinc_cache post-dates
        PR #269) comes back missing them entirely, and one pickled while they
        were plain dicts comes back with an object that has .get but not .put,
        so the miss path -- not the hit path -- raised AttributeError.
        """
        self.__dict__.update(state)
        for name in self._cache_attributes:
            self.__dict__[name] = _DeviceArrayCache()


class _DeviceArrayCache(Mapping):
    """A bounded, least-recently-used cache of computed arrays.

    Deliberately a plain ``OrderedDict`` rather than ``functools.lru_cache``: a
    per-instance ``lru_cache`` wrapping a bound method is not picklable -- it
    resolves by ``__qualname__`` and no longer matches the class attribute of
    that name -- which breaks dask's processes scheduler and ``distributed``,
    i.e. the multi-GPU layout these device keys exist to serve. It also breaks
    ``EqualityMixin`` comparison and ``deepcopy``. See PR #388.

    Eviction cannot race: ``popitem`` and ``move_to_end`` are single C-level
    dict operations, and the ``KeyError`` a losing thread sees is caught rather
    than escaping to the caller.
    """

    def __init__(self, maxsize: int = _MAX_CACHE_ENTRIES):
        self._maxsize = maxsize
        self._entries: OrderedDict = OrderedDict()

    def get(self, key, default=None):
        """The cached value for ``key``, or ``default``, refreshing its recency.

        The two-argument form matters: ``tables`` and ``scattering_factors``
        are public properties that returned a plain dict before this container
        existed, and keeping them dict-like is the whole reason this is a
        Mapping. A one-argument override shadows ``Mapping.get`` and makes
        ``scattering_factors.get(key, "not cached")`` -- ordinary dict usage --
        raise TypeError instead of returning the default.
        """
        try:
            value = self._entries[key]
        except KeyError:
            return default
        try:
            self._entries.move_to_end(key)
        except KeyError:  # evicted by another thread; the value is still ours
            pass
        return value

    def put(self, key, value):
        """Store ``value`` under ``key`` and return it."""
        while len(self._entries) >= self._maxsize:
            try:
                self._entries.popitem(last=False)
            except KeyError:  # another thread emptied it
                break
        self._entries[key] = value
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # ``QuadratureProjectionIntegrals.tables`` and
    # ``ScatteringFactorProjectionIntegrals.scattering_factors`` are public
    # properties that returned a plain dict before this container existed, so
    # it has to keep behaving like one -- Mapping supplies keys/values/items/
    # __contains__/get on top of __getitem__/__iter__/__len__. Note Mapping
    # also supplies __eq__, which is why ours is defined *after* this in the
    # class body and wins.
    def __getitem__(self, key):
        return self._entries[key]

    def __eq__(self, other) -> bool:
        # A cache is incidental state, never identity: two integrators with the
        # same parametrization are the same integrator whether or not either
        # has been used. Comparing contents would compare numpy arrays, which
        # safe_equality turns into an unequal verdict via its ValueError guard,
        # so a used integrator would stop comparing equal to an identical fresh
        # one.
        #
        # This fixes the comparison at the *integrator* level only. A built
        # Potential still compares unequal to an identical unbuilt one, because
        # Potential.build() populates _sliced_atoms lazily -- the same defect
        # shape one level up, and out of scope here.
        return isinstance(other, _DeviceArrayCache)

    __hash__ = None


def _precision_key() -> str:
    """The current precision, as a cache-key component.

    Every cached value here is built through ``get_dtype``, directly or via
    ``spatial_frequencies``, so a key without it serves a float32 entry to a
    float64 request. Silent, and small enough that ``np.allclose`` with default
    tolerances calls the wrong answer correct.
    """
    return str(np.dtype(get_dtype(complex=False)))


def _device_cache_key(device, like=None) -> str | tuple[str, int]:
    """Name the concrete device a cached array belongs to.

    The ``device`` threaded through the integrators is the plain "cpu"/"gpu"
    string, which does not distinguish one GPU from another. Read the device
    off ``like`` -- an array the cached value will be combined with -- the way
    ``_local_potential_on_device`` (core_loss.py) and
    ``_radial_binning_device_arrays`` (measurements.py) read it off theirs.
    That is what makes the key right when one process drives several GPUs: the
    ambient CUDA context can differ from the device an array actually lives on.

    Without ``like`` there is nothing to anchor to and the current device is
    the best available answer. Callers that have an array should pass it.
    """
    xp = get_array_module(device)
    if xp is np:
        return "cpu"
    if like is not None and get_array_module(like) is not np:
        return ("gpu", int(like.device.id))
    return ("gpu", int(xp.cuda.Device().id))


class FieldIntegrator(EqualityMixin, CopyMixin, metaclass=ABCMeta):
    """Base class for projection integrator object used for calculating projection
    integrals of radial potentials.

    Parameters
    ----------
    periodic : bool
        True indicates that the projection integrals are periodic perpendicular to the
        projection direction.
    finite : bool
        True indicates that the projection integrals are finite along the projection
        direction.
    retain_data : bool, optional
        If True, intermediate calculations are kept.
    """

    def __init__(self, periodic: bool, finite: bool, retain_data: bool = False):
        self._periodic = periodic
        self._finite = finite
        self._retain_data = retain_data

    @abstractmethod
    def integrate_on_grid(
        self,
        positions: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Integrate radial potential between two limits at the given 2D positions on a
        grid. The integration limits are only used when the integration method is
        finite.

        Parameters
        ----------
        positions : numpy.ndarray
            2D array of xy-positions of the centers of each radial function [Å].
        a : numpy.ndarray
            Lower integration limit of the pr
            ojection integrals along z for each position [Å]. The limit is given
            relative to the center of the radial function.
        b : numpy.ndarray
            Upper integration limit of the projection integrals along z for each
            position [Å]. The limit is given relative to the center of the radial
            function.
        gpts : two int
            Number of grid points in `x` and `y` describing each slice of the potential.
        sampling : two float
            Sampling of the potential in `x` and `y` [1 / Å].
        device : str, optional
            The device used for calculating the potential, 'cpu' or 'gpu'. The default
            is determined by the user configuration file.
        """

    @property
    def periodic(self) -> bool:
        """True indicates that the created projection integrators are implemented only
        for periodic potentials."""
        return self._periodic

    @property
    def finite(self) -> bool:
        """True indicates that the created projection integrators are implemented only
        for infinite potential projections."""
        return self._finite

    @abstractmethod
    def cutoff(self, symbol: str) -> float:
        """Radial cutoff of the potential for the given chemical symbol."""


def correction_projected_scattering_factors(
    symbol, gpts, sampling, short_range="lobato", long_range="peng"
):
    short_range = validate_parametrization(short_range)
    long_range = validate_parametrization(long_range)

    k, _ = polar_spatial_frequencies(gpts, sampling)

    short_range = short_range.projected_scattering_factor(symbol)
    long_range = long_range.projected_scattering_factor(symbol)

    correction = short_range(k**2) - long_range(k**2)
    # correction /= sinc()
    return correction


def gaussian_projected_scattering_factors(
    symbol, gpts, sampling, parametrization="peng"
):
    parametrization = validate_parametrization(parametrization)

    parameters = parametrization.scaled_parameters(
        symbol, "projected_scattering_factor"
    )

    k, _ = polar_spatial_frequencies(gpts, sampling)

    a = parameters[0, :, None, None]
    b = parameters[1, :, None, None]

    projected_gaussians = a * np.exp(-b * k[None] ** 2.0)
    return projected_gaussians


def gaussian_projection_weights(symbol, a, b, parametrization="peng"):
    parametrization = validate_parametrization(parametrization)

    parameters = parametrization.scaled_parameters(
        symbol, "projected_scattering_factor"
    )

    scales = np.pi / np.sqrt(parameters[1])[:, None]

    weights = np.abs(erf(scales * b[None]) - erf(scales * a[None])) / 2
    return weights


def _sinc_on_device(gpts, sampling, device, device_key):
    """``sinc`` built inside the context of the device its key names."""
    if device_key == "cpu":
        return sinc(gpts, sampling, "cpu")

    import cupy as cp  # noqa: PLC0415 -- optional dependency

    with cp.cuda.Device(device_key[1]):
        return sinc(gpts, sampling, "gpu")


class GaussianProjectionIntegrals(_CacheStateMixin, FieldIntegrator):
    """
    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The correction radial potential parametrization to integrate. Used for
        correcting the dependence of the potential close to the nuclear core.
        Default is the Lobato parametrization.
    gaussian_parametrization : str or Parametrization, optional
        The Gaussian radial potential parametrization to integrate. Must be
        parametrization described by a superposition of Gaussians. Default is the Peng
        parametrization.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]. Default is 1e-3.
    """

    _cache_attributes = ("_sinc_cache",)

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        gaussian_parametrization: str | Parametrization = "peng",
        cutoff_tolerance: float = 1e-3,
    ):
        self._gaussian_parametrization = validate_parametrization(
            gaussian_parametrization
        )

        self._correction_parametrization = validate_parametrization(parametrization)

        self._cutoff_tolerance = cutoff_tolerance

        super().__init__(periodic=True, finite=True)

        # There were two further caches here, for the projected gaussians and
        # the corrections. Both computed a key, checked the dict, missed,
        # recomputed and never wrote back, so neither ever stored anything and
        # nobody noticed. They are removed rather than repaired: the values are
        # (5, *gpts) float64 arrays that abTEM pickles into every task graph,
        # so enabling them cost ~300 MB retained and inflated a warm
        # frozen-phonon graph 12.6x, to buy a speed-up nothing has ever had.
        # A correctly keyed cache can be added by whoever needs one.
        self._sinc_cache = _DeviceArrayCache()

    @property
    def cutoff_tolerance(self):
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def gaussian_parametrization(self):
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._gaussian_parametrization

    @property
    def correction_parametrization(self):
        return self._correction_parametrization

    def cutoff(self, symbol: str) -> float:
        return optimize_cutoff(
            self.gaussian_parametrization.potential(symbol),
            self.cutoff_tolerance,
            a=1e-3,
            b=1e3,
        )  # noqa

    def get_gaussians(self, symbol, gpts, sampling):
        return gaussian_projected_scattering_factors(symbol, gpts, sampling)

    def get_corrections(self, symbol, gpts, sampling):
        return correction_projected_scattering_factors(symbol, gpts, sampling)

    def _integrate_gaussians(self, positions, symbol, a, b, gpts, sampling, device):
        gaussians = self.get_gaussians(symbol, gpts, sampling)

        shifted_a = a - positions[:, 2]
        shifted_b = b - positions[:, 2]

        weights = gaussian_projection_weights(symbol, shifted_a, shifted_b)

        xp = get_array_module(device)
        fp_dtype = get_dtype(complex=False)
        cx_dtype = get_dtype(complex=True)
        positions = (positions[:, :2] / sampling).astype(fp_dtype)

        array = xp.zeros(gpts, dtype=cx_dtype)
        for i in range(5):
            temp = xp.zeros_like(array, dtype=cx_dtype)
            superpose_deltas(positions, temp, weights=weights[i])
            array += fft2(temp, overwrite_x=True) * gaussians[i].astype(cx_dtype)

        return array

    def _integrate_corrections(self, positions, symbol, a, b, gpts, sampling, device):
        corrections = self.get_corrections(symbol, gpts, sampling)

        xp = get_array_module(device)
        fp_dtype = get_dtype(complex=False)
        cx_dtype = get_dtype(complex=True)

        positions = positions[(positions[:, 2] >= a) * (positions[:, 2] < b)]
        positions = (positions[:, :2] / sampling).astype(fp_dtype)

        array = xp.zeros(gpts, dtype=cx_dtype)

        superpose_deltas(positions, array)

        corrections = fft2(array, overwrite_x=False) * corrections

        return corrections

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
        fourier_space: bool = False,
    ) -> np.ndarray:
        xp = get_array_module(device)

        array = xp.zeros(gpts, dtype=get_dtype(complex=True))
        for number in np.unique(atoms.numbers):
            positions = atoms.positions[atoms.numbers == number]
            symbol = chemical_symbols[number]

            array += self._integrate_gaussians(
                positions, symbol, a, b, gpts, sampling, device
            )
            array += self._integrate_corrections(
                positions, symbol, a, b, gpts, sampling, device
            )

        # `array` is already on the device this result must live on, so use it
        # as the anchor rather than the ambient CUDA context, and build the
        # sinc there too -- keying on one device while allocating on another is
        # the half-fix PR #388 had to correct.
        device_key = _device_cache_key(device, like=array)
        sinc_key = (tuple(gpts), tuple(sampling), device_key, _precision_key())
        sinc_array = self._sinc_cache.get(sinc_key)
        if sinc_array is None:
            sinc_array = self._sinc_cache.put(
                sinc_key, _sinc_on_device(gpts, sampling, device, device_key)
            )

        return ifft2(array / sinc_array).real


def sinc(
    gpts: tuple[int, int], sampling: tuple[float, float], device: str = "cpu"
) -> np.ndarray:
    """
    Returns an array representing a 2D sinc function centered at [0, 0]. The result is
    used to compensate for the finite size of single pixels used for representing delta
    functions.

    Parameters
    ----------
    gpts : two int
        Number of grid points in the first and second dimension to evaluate the sinc
        over.
    sampling : two float
        Size of the pixels of the grid determining the scale of the sinc.
    device : str
        The array is created on this device ('cpu' or 'gpu').

    Returns
    -------
    sinc : numpy.ndarray
        2D sinc function.
    """
    xp = get_array_module(device)
    kx, ky = spatial_frequencies(gpts, sampling, return_grid=False, xp=xp)
    k = xp.sqrt((kx[:, None] * sampling[0]) ** 2 + (ky[None] * sampling[1]) ** 2)
    dk2 = sampling[0] * sampling[1]
    k[0, 0] = 1
    sinc = xp.sin(k) / k * dk2
    sinc[0, 0] = dk2
    return sinc


def superpose_deltas(
    positions: np.ndarray,
    array: np.ndarray,
    weights: Optional[np.ndarray] = None,
    round_positions: bool = False,
) -> np.ndarray:
    """
    Add superposition of delta functions at given positions to a 2D array.

    Parameters
    ----------
    positions : numpy.ndarray
        Array of 2D positions as an nx2 array. The positions are given in units of
        pixels.
    array : numpy.ndarray
        The delta functions are added to this 2D array.
    weights : numpy.ndarray, optional
        If given each delta function is weighted by the given factor. Must match the
        length of `positions`.
    round_positions : bool, optional
        If True, the delta function positions are rounded to the center of the nearest
        pixel, otherwise subpixel precision is used.

    Returns
    -------
    array : numpy.ndarray
        The array with the delta functions added.
    """

    xp = get_array_module(array)
    shape = array.shape

    positions = xp.array(positions)

    if round_positions:
        rounded = xp.round(positions).astype(xp.int32)
        i, j = rounded[:, 0][None] % shape[0], rounded[:, 1][None] % shape[1]
        v = xp.array([1.0], dtype=get_dtype(complex=False))[:, None]
    else:
        rounded = xp.floor(positions).astype(xp.int32)
        rows, cols = rounded[:, 0], rounded[:, 1]
        x = positions[:, 0] - rows
        y = positions[:, 1] - cols
        xy = x * y
        i = xp.array([rows % shape[0], (rows + 1) % shape[0]] * 2)
        j = xp.array([cols % shape[1]] * 2 + [(cols + 1) % shape[1]] * 2)
        v = xp.array(
            [1 + xy - y - x, x - xy, y - xy, xy], dtype=get_dtype(complex=False)
        )

    if weights is not None:
        v = v * weights[None]
    
    try:
        xp.add.at(array, (i, j), v)
    except AttributeError:
        
        if device_name_from_array_module(xp) == "gpu":
            cupyx.scatter_add(array, (i, j), v)
        else:
            raise RuntimeError()

    return array


class ScatteringFactorProjectionIntegrals(_CacheStateMixin, FieldIntegrator):
    """
    A FieldIntegrator calculating infinite projections of radial potential
    parametrizations. The hybrid real and reciprocal space method by
    Wouter Van den Broek et al. is used.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The radial potential parametrization to integrate. Default is the Lobato
        parametrization.

    References
    ----------
    W. Van den Broek et al. Ultramicroscopy, 158:89-97, 2015.
    doi:10.1016/j.ultramic.2015.07.005.
    """

    _cache_attributes = ("_scattering_factors",)

    def __init__(self, parametrization: str | Parametrization = "lobato"):
        self._parametrization = validate_parametrization(parametrization)
        # See _DeviceArrayCache for why this is not a functools.lru_cache.
        self._scattering_factors = _DeviceArrayCache()
        super().__init__(periodic=True, finite=False)

    @property
    def parametrization(self) -> Parametrization:
        return self._parametrization

    def cutoff(self, symbol: str) -> float:
        return np.inf

    def _calculate_scattering_factor(
        self,
        symbol: str,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ):
        xp = get_array_module(device)
        kx, ky = spatial_frequencies(gpts, sampling, xp=np)

        k2 = kx[:, None] ** 2 + ky[None] ** 2
        f = self.parametrization.projected_scattering_factor(symbol)(k2)
        f = xp.asarray(f, dtype=get_dtype(complex=False))

        if symbol in self.parametrization.sigmas.keys():
            sigma = self.parametrization.sigmas[symbol]
            f = f * xp.exp(
                -xp.asarray(k2, dtype=get_dtype(complex=False))
                * (xp.pi * sigma / xp.sqrt(1 / 2)) ** 2
            )

        return f

    def get_scattering_factor(self, symbol, gpts, sampling, device, like=None):
        # The cached array depends on the grid and on the device it was
        # allocated on, not on the element alone: keying on ``symbol`` served
        # the first grid's array to every later grid (a broadcast error one
        # frame away, in integrate_on_grid) and the first device's array to
        # every later device (a numpy array handed to a cupy kernel).
        #
        # ``device`` itself is deliberately not part of the key: it may be an
        # array or a module (get_array_module accepts both), which is not
        # always hashable, and "gpu" and the cupy module would otherwise take
        # two entries for one physical device. ``device_key`` is canonical.
        device_key = _device_cache_key(device, like)
        key = (
            symbol,
            tuple(gpts),
            tuple(sampling),
            device_key,
            _precision_key(),
        )

        scattering_factor = self._scattering_factors.get(key)
        if scattering_factor is None:
            scattering_factor = self._scattering_factors.put(
                key,
                self._calculate_scattering_factor_on_device(
                    symbol, gpts, sampling, device_key
                ),
            )

        return scattering_factor

    def _calculate_scattering_factor_on_device(
        self, symbol, gpts, sampling, device_key
    ):
        """Build the scattering factor *on the device named by the key*.

        Anchoring only the key on ``like`` would have been half a fix: the key
        would say one device while the allocation followed the ambient CUDA
        context, so under multi-GPU use the array could be cached under a
        device it does not live on. Allocate inside that device's context, as
        ``_local_potential_on_device`` (core_loss.py) does with
        ``with like.device:``.
        """
        if device_key == "cpu":
            return self._calculate_scattering_factor(symbol, gpts, sampling, "cpu")

        import cupy as cp  # noqa: PLC0415 -- optional dependency

        with cp.cuda.Device(device_key[1]):
            return self._calculate_scattering_factor(symbol, gpts, sampling, "gpu")

    @property
    def scattering_factors(self) -> Mapping[tuple, np.ndarray]:
        """Cached projected scattering factors, keyed by element, grid and
        device."""
        return self._scattering_factors

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: np.ndarray,
        b: np.ndarray,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ):
        xp = get_array_module(device)
        if len(atoms) == 0:
            return xp.zeros(gpts, dtype=get_dtype(complex=False))

        array = xp.zeros(gpts, dtype=get_dtype(complex=False))
        for number in np.unique(atoms.numbers):
            # Anchor the cache's device key on an array the scattering factor
            # will actually be combined with, rather than on the ambient CUDA
            # context, which can name a different device under multi-GPU use.
            scattering_factor = self.get_scattering_factor(
                chemical_symbols[number], gpts, sampling, device, like=array
            )

            positions = atoms.positions[atoms.numbers == number]

            positions = (positions[:, :2] / sampling).astype(get_dtype(complex=False))

            temp_array = xp.zeros(gpts, dtype=get_dtype(complex=False))

            temp_array = superpose_deltas(positions, temp_array).astype(
                get_dtype(complex=True)
            )

            temp_array = fft2(temp_array, overwrite_x=True)

            temp_array *= scattering_factor / sinc(gpts, sampling, device)

            # if not fourier_space:
            temp_array = ifft2(temp_array, overwrite_x=True).real

            array += temp_array

        return array


# Deliberately not numba parallel=True: the workqueue threading layer (the
# fallback when neither TBB nor OpenMP is available) hard-aborts the process
# when parallel kernels are launched concurrently from multiple Python
# threads, which is exactly what happens when dask tasks build potential
# slices in a threaded scheduler. Since the kernel is nogil, thread-level
# parallelism is instead applied by the caller (see
# _threaded_interpolate_radial_functions), which is safe under any layer.
@jit(nopython=True, nogil=True)
def interpolate_radial_functions(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    disk_counts: np.ndarray,
    sampling: tuple[float, float],
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    n = radial_gpts.shape[0]
    dt = np.log(radial_gpts[-1] / radial_gpts[0]) / (n - 1)

    for i in range(positions.shape[0]):
        px = int(round(positions[i, 0] / sampling[0]))
        py = int(round(positions[i, 1] / sampling[1]))

        # The disk indices are sorted by radial distance, so the loop may stop
        # after the first disk_counts[i] pixels (those within the lateral
        # cutoff of atom i for the current slice).
        for j in range(disk_counts[i]):
            k = px + disk_indices[j, 0]
            m = py + disk_indices[j, 1]

            if (k < array.shape[0]) & (m < array.shape[1]) & (k >= 0) & (m >= 0):
                r_interp = np.sqrt(
                    (k * sampling[0] - positions[i, 0]) ** 2
                    + (m * sampling[1] - positions[i, 1]) ** 2
                )

                idx = int(np.floor(np.log(r_interp / radial_gpts[0] + 1e-12) / dt))

                if idx < 0:
                    array[k, m] += radial_functions[i, 0]
                elif idx < n - 1:
                    slope = radial_derivative[i, idx]
                    array[k, m] += (
                        radial_functions[i, idx] + (r_interp - radial_gpts[idx]) * slope
                    )


_interpolation_pool: Optional[ThreadPoolExecutor] = None

# Upper bound on the temporary per-thread accumulation buffers used by
# _threaded_interpolate_radial_functions. For very large grids the thread
# count is reduced so the buffers stay below this size, degrading gracefully
# to the serial kernel.
_INTERPOLATION_BUFFER_BUDGET = 256 * 1024**2


def _get_interpolation_pool() -> ThreadPoolExecutor:
    global _interpolation_pool
    if _interpolation_pool is None:
        _interpolation_pool = ThreadPoolExecutor(max_workers=os.cpu_count())
    return _interpolation_pool


def _threaded_interpolate_radial_functions(
    array: np.ndarray,
    positions: np.ndarray,
    disk_indices: np.ndarray,
    disk_counts: np.ndarray,
    sampling: tuple[float, float],
    radial_gpts: np.ndarray,
    radial_functions: np.ndarray,
    radial_derivative: np.ndarray,
):
    """Run the (nogil) interpolation kernel across a thread pool.

    The atoms are dealt round-robin to per-thread accumulation buffers so no
    two threads ever write to the same array; the buffers are summed into
    ``array`` afterwards. Since the kernel releases the GIL, plain Python
    threads give full parallelism without involving numba's threading layer
    (whose workqueue backend aborts on concurrent launches, e.g. from dask).
    """
    num_chunks = min(
        os.cpu_count() or 1,
        len(positions),
        max(int(_INTERPOLATION_BUFFER_BUDGET // max(array.nbytes, 1)), 1),
    )

    if num_chunks <= 1:
        interpolate_radial_functions(
            array,
            positions,
            disk_indices,
            disk_counts,
            sampling,
            radial_gpts,
            radial_functions,
            radial_derivative,
        )
        return

    buffers = np.zeros((num_chunks,) + array.shape, dtype=array.dtype)

    def run_chunk(chunk: int):
        # Round-robin selection balances the load when disk sizes vary along
        # the atom order (e.g. sorted by z relative to the slice).
        selection = slice(chunk, None, num_chunks)
        interpolate_radial_functions(
            buffers[chunk],
            np.ascontiguousarray(positions[selection]),
            disk_indices,
            np.ascontiguousarray(disk_counts[selection]),
            sampling,
            radial_gpts,
            np.ascontiguousarray(radial_functions[selection]),
            np.ascontiguousarray(radial_derivative[selection]),
        )

    pool = _get_interpolation_pool()
    futures = [pool.submit(run_chunk, chunk) for chunk in range(num_chunks)]
    for future in futures:
        future.result()

    array += buffers.sum(axis=0)


class ProjectionIntegralTable:
    """
    A ProjectionIntegrator calculating finite projections of radial potential
    parametrizations. An integral table for each used to evaluate the projection
    integrals for each atom in a slice given p integral limits. The projected potential
    evaluated along the

    Parameters
    ----------
    radial_gpts : array
        The points along a radial in the `xy`-plane where the projection integrals of
        the integral table are evaluated.
    limits : array
        The points along the projection direction where the projection integrals are
        evaluated.
    """

    def __init__(self, radial_gpts: np.ndarray, limits: np.ndarray, values: np.ndarray):
        assert values.shape[0] == len(limits)
        assert values.shape[1] == len(radial_gpts)

        self._radial_gpts = radial_gpts
        self._limits = limits
        self._values = values

    @property
    def radial_gpts(self) -> np.ndarray:
        return self._radial_gpts

    @property
    def limits(self) -> np.ndarray:
        return self._limits

    @property
    def values(self) -> np.ndarray:
        return self._values

    def _interpolate(self, x: np.ndarray) -> np.ndarray:
        # Piecewise-linear interpolation along the limits axis with linear
        # extrapolation from the end segments; equivalent to
        # scipy.interpolate.interp1d(limits, values, axis=0, kind="linear",
        # fill_value="extrapolate"), but without rebuilding an interpolator
        # for every slice.
        idx = np.searchsorted(self._limits, x, side="right") - 1
        idx = np.clip(idx, 0, len(self._limits) - 2)
        x0 = self._limits[idx]
        x1 = self._limits[idx + 1]
        weights = (x - x0) / (x1 - x0)
        return self._values[idx] + weights[..., None] * (
            self._values[idx + 1] - self._values[idx]
        )

    def integrate(self, a: float | np.ndarray, b: float | np.ndarray) -> np.ndarray:
        a = np.atleast_1d(np.asarray(a, dtype=self._limits.dtype))
        b = np.atleast_1d(np.asarray(b, dtype=self._limits.dtype))
        return self._interpolate(b) - self._interpolate(a)


def optimize_cutoff(func: Callable, tolerance: float, a: float, b: float) -> float:
    """
    Calculate the point where a function becomes lower than a given tolerance within a
    given bracketing interval.

    Parameters
    ----------
    func : callable
        The function to calculate the cutoff for.
    tolerance : float
        The tolerance to calculate the cutoff for.
    a : float
        One end of the bracketing interval.
    b : float
        The other end of the bracketing interval.

    Returns
    -------
    cutoff : float
    """
    f = brentq(f=lambda r: np.abs(func(r)) - tolerance, a=a, b=b)
    return f


def cutoff_taper(radial_gpts, cutoff, taper):
    taper_start = taper * cutoff
    taper_mask = radial_gpts > taper_start
    taper_values = np.ones_like(radial_gpts)
    taper_values[taper_mask] = (
        np.cos(np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start))
        + 1.0
    ) / 2
    return taper_values


class QuadratureProjectionIntegrals(_CacheStateMixin, FieldIntegrator):
    """
    Projection integration plan for calculating finite projection integrals based on
    Gaussian quadrature rule.

    Parameters
    ----------
    parametrization : str or Parametrization, optional
        The potential parametrization describing the radial dependence of the potential.
        Default is 'lobato'.
    cutoff_tolerance : float, optional
        The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]. Default is 1e-3.
    taper : float, optional
        The fraction from the cutoff of the radial distance from the core where the
        atomic potential starts tapering
        to zero. Default is 0.85.
    integration_step : float, optional
        The step size between integration limits used for calculating the integral
        table. Default is 0.02.
    quad_order : int, optional
        Order of quadrature integration passed to scipy.integrate.fixed_quad.
        Default is 8.
    """

    _cache_attributes = ("_tables", "_sorted_disks", "_device_arrays")

    def __init__(
        self,
        parametrization: str | Parametrization = "lobato",
        cutoff_tolerance: float = 1e-4,
        inner_cutoff_factor: float = 2.0,
        taper: float = 0.85,
        integration_step: float = 0.02,
        quad_order: int = 8,
    ):
        self._parametrization = validate_parametrization(parametrization)
        self._taper = taper
        self._quad_order = quad_order
        self._cutoff_tolerance = cutoff_tolerance
        self._inner_cutoff_factor = inner_cutoff_factor
        self._integration_step = integration_step
        # Keyed on the element ALONE until 2026-09: _calculate_integral_table
        # sets inner_limit from min(sampling), which sets the radial grid, so
        # reusing one integrator across a sampling sweep served the first
        # sampling's table and built a potential ~54 % wrong.
        self._tables = _DeviceArrayCache()
        # Host-side sorted disk, keyed by (symbol, sampling) -- correct, but
        # previously unbounded, so a sampling sweep accumulated it without
        # limit while its device-resident twin below was capped.
        self._sorted_disks = _DeviceArrayCache()
        # Device-resident copy of the sorted disk, keyed by
        # (symbol, sampling, device). The disk is invariant per (symbol,
        # sampling) across all slices, but integrate_on_grid is called once
        # per slice per species; without this cache each call re-uploads
        # it to the GPU from scratch, which measurably dominated GPU
        # build time (see PR #309 discussion) despite the array never
        # changing between calls. Only disks small enough to fit within
        # the chunked-transfer bound are cached; larger disks are streamed
        # in memory-bounded chunks instead (see integrate_on_grid).
        self._device_arrays = _DeviceArrayCache()

        super().__init__(periodic=False, finite=True)

    @property
    def parametrization(self):
        """The potential parametrization describing the radial dependence of the
        potential."""
        return self._parametrization

    @property
    def quad_order(self):
        """Order of quadrature integration."""
        return self._quad_order

    @property
    def tables(self):
        return self._tables

    @property
    def cutoff_tolerance(self) -> float:
        """The error tolerance used for deciding the radial cutoff distance of the
        potential [eV / e]."""
        return self._cutoff_tolerance

    @property
    def integration_step(self) -> float:
        """The step size between integration limits used for calculating the integral
        table."""
        return self._integration_step

    def cutoff(self, symbol: str) -> float:
        return optimize_cutoff(
            self.parametrization.potential(symbol), self.cutoff_tolerance, a=1e-3, b=1e3
        )

    @staticmethod
    def _radial_gpts(inner_cutoff: float, cutoff: float) -> np.ndarray:
        num_points = int(np.ceil(cutoff / inner_cutoff))
        return np.geomspace(inner_cutoff, cutoff, num_points)

    @staticmethod
    def _taper_values(radial_gpts: np.ndarray, cutoff: float, taper: float):
        taper_start = taper * cutoff
        taper_mask = radial_gpts > taper_start
        taper_values = np.ones_like(radial_gpts)
        taper_values[taper_mask] = (
            np.cos(
                np.pi * (radial_gpts[taper_mask] - taper_start) / (cutoff - taper_start)
            )
            + 1.0
        ) / 2
        return taper_values

    def _integral_limits(self, cutoff: float):
        limits = np.linspace(-cutoff, 0, int(np.ceil(cutoff / self._integration_step)))
        return np.concatenate((limits, -limits[::-1][1:]))

    def _calculate_integral_table(
        self, symbol: str, sampling: tuple[float, float]
    ) -> ProjectionIntegralTable:
        potential = self.parametrization.potential(symbol)
        cutoff = self.cutoff(symbol)

        inner_limit = min(sampling) / self._inner_cutoff_factor
        radial_gpts = self._radial_gpts(inner_limit, cutoff)
        limits = self._integral_limits(cutoff)

        # def potential_blurred(r, func):
        #     ri = np.linspace(-4, 4, 101)[(None,) * len(r.shape)]
        #     r = r[..., None]
        #     r = np.abs(r + ri)
        #     f = (
        #         func(r) * r[None, None] ** 2 * np.exp(-(ri**2) / 0.1)[None, None]
        #     ).sum(-1)
        #     return f
        #
        # potential_blurred_ = lambda r: potential_blurred(r, potential)
        #
        # projection = lambda z: potential_blurred_(
        #     np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2)
        # )

        # projection = lambda z: potential(
        #    np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2)
        # )

        def project_along_z(z):
            return potential(np.sqrt(radial_gpts[:, None] ** 2 + z[None] ** 2))

        # * np.exp(-(radial_gpts[:, None] ** 2) / 10000)

        table = np.zeros((len(limits) - 1, len(radial_gpts)))
        table[0, :] = integrate.fixed_quad(
            project_along_z, -limits[0] * 2, limits[0], n=self._quad_order
        )[0]

        for j, (a, b) in enumerate(zip(limits[1:-1], limits[2:])):
            table[j + 1] = (
                table[j]
                + integrate.fixed_quad(project_along_z, a, b, n=self._quad_order)[0]
            )

        table = table * self._taper_values(radial_gpts, cutoff, self._taper)[None]

        return ProjectionIntegralTable(radial_gpts, limits[1:], table)

    def get_integral_table(self, symbol, sampling):
        """
        Build table of projection integrals of the radial atomic potential.

        Parameters
        ----------
        symbol : str
            Chemical symbol to build the integral table.
        inner_limit : float, optional
            Smallest radius from the core at which to calculate the projection integral
            [Å].

        Returns
        -------
        projection_integral_table :
            ProjectionIntegralTable
        """
        key = (symbol, tuple(sampling), _precision_key())
        table = self._tables.get(key)
        if table is None:
            table = self._tables.put(
                key, self._calculate_integral_table(symbol, sampling)
            )

        return table

    def integrate_on_grid(
        self,
        atoms: Atoms,
        a: float,
        b: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        device: str = "cpu",
    ) -> np.ndarray:
        xp = get_array_module(device)

        array = xp.zeros(gpts, dtype=get_dtype(complex=False))
        for number in np.unique(atoms.numbers):
            table = self.get_integral_table(chemical_symbols[number], sampling)

            positions = atoms.positions[atoms.numbers == number]

            shifted_a = a - positions[:, 2]
            shifted_b = b - positions[:, 2]

            fp_dtype = get_dtype(complex=False)

            cutoff = table.radial_gpts[-1]
            # Precision is the one key component this needed; the symbol was
            # already here. It belongs for the same reason as everywhere else
            # in this file, though it takes two steps to get there: the disk is
            # sized int(ceil(cutoff / min(sampling))), and `cutoff` is
            # precision-dependent (optimize_cutoff evaluates the parametrization
            # at the configured dtype) -- 5.066261105906332 against
            # 5.066261205200659 for Si, ~2e-8 relative.
            #
            # That is enough to move the integer. Solving cutoff32/m <= s <
            # cutoff64/m gives a counterexample for every element tried; at
            # sampling 0.281458955844481 Si wants radius 18 at float32 and 19
            # at float64. Serving the float32 disk to a float64 build is then
            # measurably wrong -- 4.6e-05 absolute on a peak of 312, i.e. 1.5e-7
            # relative -- because the missing ring lands just inside the last
            # radial gridpoint, where interpolate_radial_functions still
            # contributes (idx == n - 2) rather than clamping to zero.
            #
            # An earlier version of this key omitted precision, on the argument
            # that a 400-sampling scan found no case where the integer moves.
            # It found none because a linear scan cannot: the cases have to be
            # solved for, not stumbled on.
            #
            # The symbol component is older than this commit and was never
            # missing -- see test_the_sorted_disk_is_not_served_across_elements,
            # which covers it for the first time rather than fixing it.
            disk_key = (
                chemical_symbols[number],
                tuple(sampling),
                _precision_key(),
            )
            cached_disk = self._sorted_disks.get(disk_key)
            if cached_disk is not None:
                disk, disk_radii = cached_disk
            else:
                disk = disk_meshgrid(int(np.ceil(cutoff / np.min(sampling))))
                # Sort the disk pixels by physical radial distance so that the
                # interpolation can stop at each atom's lateral cutoff.
                disk_radii = np.hypot(
                    disk[:, 0] * sampling[0], disk[:, 1] * sampling[1]
                )
                order = np.argsort(disk_radii)
                disk = np.ascontiguousarray(disk[order])
                disk_radii = disk_radii[order]
                self._sorted_disks.put(disk_key, (disk, disk_radii))

            # A pixel at lateral distance r only receives contributions from
            # the part of the radial potential at 3D distance
            # sqrt(r ** 2 + dz ** 2), with dz the distance from the atom to the
            # slice interval. Beyond r = sqrt(cutoff ** 2 - dz ** 2) the
            # potential is below the cutoff tolerance, so those pixels are
            # skipped. The margin accounts for the atom position rounding to
            # the nearest pixel.
            dz = np.maximum(np.maximum(shifted_a, -shifted_b), 0.0)
            lateral_cutoff = np.sqrt(np.maximum(cutoff**2 - dz**2, 0.0))
            margin = np.hypot(sampling[0], sampling[1]) / 2
            disk_counts = np.searchsorted(
                disk_radii, lateral_cutoff + margin, side="right"
            )
            # Cheap host-side reduction: the largest per-atom truncation index
            # bounds how much of the sorted disk the kernels need at all.
            max_disk_count = int(disk_counts.max()) if len(disk_counts) else 0

            # Transfer the integral table and radial grid to the compute dtype
            # (float32 or float64 according to the precision config) so that all
            # subsequent GPU operations stay in the configured precision.
            # Without the explicit dtype, table.integrate() returns numpy float64
            # and xp.asarray() would preserve that, creating a full float64 GPU
            # allocation even when precision='float32'.
            radial_potential = xp.asarray(
                table.integrate(shifted_a, shifted_b), dtype=fp_dtype
            )
            radial_gpts_device = xp.asarray(table.radial_gpts, dtype=fp_dtype)

            positions = xp.asarray(positions, dtype=fp_dtype)

            # Compute derivative entirely in the compute dtype.  Using
            # table.radial_gpts (float64 numpy) as the denominator would upcast
            # the division to float64 and then silently downcast back when
            # assigned into the float32 derivative array.
            radial_potential_derivative = xp.zeros_like(radial_potential)
            radial_potential_derivative[:, :-1] = (
                xp.diff(radial_potential, axis=1) / xp.diff(radial_gpts_device)[None]
            )

            if len(self._parametrization.sigmas):
                temp = xp.zeros(gpts, dtype=fp_dtype)
            else:
                temp = array

            if xp is cp:
                # radial_gpts_device already has the correct dtype (computed
                # above); reuse it directly instead of re-converting.
                # The kernel truncates per atom at disk_counts (with the
                # chunk's global offset), so at most the max_disk_count prefix
                # of the radius-sorted disk is ever scanned.
                disk_counts_device = cp.asarray(disk_counts)
                chunk_size = 2_000_000
                if len(disk) <= chunk_size:
                    # Common case: the disk fits comfortably on device, so keep
                    # a cached copy -- re-uploading it every slice measurably
                    # dominated GPU build time (see PR #309 discussion).
                    # disk_counts_device was just allocated for this work, so
                    # it names the device this copy has to live on. Keying on
                    # the plain "gpu" string instead served an array cached for
                    # one GPU to a kernel running on another, whenever a single
                    # process drives several.
                    cache_key = disk_key + (
                        _device_cache_key(device, like=disk_counts_device),
                    )
                    disk_device = self._device_arrays.get(cache_key)
                    if disk_device is None:
                        with disk_counts_device.device:
                            disk_device = self._device_arrays.put(
                                cache_key, cp.asarray(disk)
                            )
                    interpolate_radial_functions_cuda(
                        array=temp,
                        positions=positions,
                        disk_indices=disk_device[:max_disk_count],
                        disk_counts=disk_counts_device,
                        sampling=sampling,
                        radial_gpts=radial_gpts_device,
                        radial_functions=radial_potential,
                        radial_derivative=radial_potential_derivative,
                    )
                else:
                    # For very fine sampling the disk can contain hundreds of
                    # millions of pixels (>5 GB) which would exceed device
                    # memory: stream the needed prefix in bounded chunks so the
                    # full disk never resides on the GPU.  The CUDA kernel
                    # accumulates via atomic adds, so multiple calls produce
                    # the same result.
                    for start in range(0, max_disk_count, chunk_size):
                        disk_chunk = cp.asarray(disk[start : start + chunk_size])
                        interpolate_radial_functions_cuda(
                            array=temp,
                            positions=positions,
                            disk_indices=disk_chunk,
                            disk_counts=disk_counts_device,
                            sampling=sampling,
                            radial_gpts=radial_gpts_device,
                            radial_functions=radial_potential,
                            radial_derivative=radial_potential_derivative,
                            chunk_offset=start,
                        )
                        del disk_chunk
            else:
                _threaded_interpolate_radial_functions(
                    array=temp,
                    positions=positions,
                    disk_indices=disk,
                    disk_counts=disk_counts,
                    sampling=sampling,
                    radial_gpts=np.asarray(table.radial_gpts, dtype=fp_dtype),
                    radial_functions=radial_potential,
                    radial_derivative=radial_potential_derivative,
                )

            symbol = chemical_symbols[number]

            if symbol in self._parametrization.sigmas:
                sigma = self._parametrization.sigmas[symbol] / np.array(sampling)
                temp = get_ndimage_module(temp).gaussian_filter(
                    temp, sigma=sigma, mode="wrap"
                )
                array += temp

        return array
