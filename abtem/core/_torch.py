"""NumPy-like array namespace backed by PyTorch tensors on Apple Metal (MPS).

abTEM dispatches array operations on an "array module" (``xp``) that is NumPy on
the CPU and CuPy on CUDA GPUs. Metal has no CuPy equivalent, so this module
adapts PyTorch's MPS backend to the same interface: :class:`TorchNDArray` wraps a
``torch.Tensor`` and presents the parts of the ``numpy.ndarray`` API that abTEM
relies on, while :data:`torch_numpy` plays the role of the ``numpy`` module
itself.

The adaptation is deliberately partial -- only the operations reached by the
multislice and potential-projection code paths are implemented. Anything else
raises ``AttributeError``, which callers treat as "not supported on this device"
and handle by falling back to NumPy.

MPS supports single precision only, so ``precision`` must be ``float32``.
"""

from __future__ import annotations

import functools
import threading
from types import SimpleNamespace
from typing import Any

import dask.array as da
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


DEVICE = "mps"


# PyTorch's MPS backend keeps a process-wide Metal shader-library cache that is
# not thread-safe: two threads compiling or looking up the same kernel corrupt
# its hash table, which surfaces either as a process spinning forever inside
# ``MetalShaderLibrary::exec_unary_kernel`` or as an outright ``Fatal Python
# error: Aborted``. Being a race, it strikes intermittently and in whichever
# operation happens to collide, so it cannot be chased call site by call site.
#
# dask reaches this backend from its threaded scheduler's worker threads, and a
# library's callers may thread on their own account, so the guard belongs here
# at the boundary: one thread at a time enters torch. The lock is reentrant
# because these wrappers legitimately call one another (``tensordot`` converts
# its operands with ``asarray``, ``pad`` builds index tensors, ...).
#
# The cost is negligible against the operations it guards -- an uncontended
# lock is tens of nanoseconds, a Metal kernel launch is microseconds at best --
# and abTEM additionally steers Metal computations onto dask's synchronous
# scheduler, so in the common case the lock is never contended at all.
_TORCH_LOCK = threading.RLock()


def _serialized(func):
    """Run ``func`` holding the Metal lock; see :data:`_TORCH_LOCK`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _TORCH_LOCK:
            return func(*args, **kwargs)

    return wrapper


# NumPy functions this namespace implements for device arrays, consulted by
# TorchNDArray.__array_function__. Without it, a NumPy function handed a device
# array falls back to __array__ and silently copies to the host -- which is how
# dask's finalize step was quietly turning a Metal result back into a NumPy one
# (it calls np.concatenate to assemble the chunks). Populated below, once the
# implementations exist.
_ARRAY_FUNCTIONS: dict = {}

# NumPy ufuncs this namespace implements, consulted by
# TorchNDArray.__array_ufunc__. Operators and ufuncs dispatch through a
# different protocol than the functions above: `host_array @ device_array`
# reaches np.matmul, not np.__array_function__, and without this it fails
# outright because ndarray does not know what to do with the right operand.
_ARRAY_UFUNCS: dict = {}


_TO_TORCH_DTYPE: dict = {}
_TO_NUMPY_DTYPE: dict = {}

if torch is not None:
    _TO_TORCH_DTYPE = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
    }
    _TO_NUMPY_DTYPE = {v: k for k, v in _TO_TORCH_DTYPE.items()}


def is_available() -> bool:
    """Whether PyTorch is installed and its Metal (MPS) backend is usable."""
    return torch is not None and torch.backends.mps.is_available()


def _check_available() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed, Metal (MPS) calculations are disabled. "
            "Install it from https://pytorch.org, or change the device to 'cpu'."
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "The Metal (MPS) backend is not available in this PyTorch build. "
            "Metal requires macOS on Apple silicon; change the device to 'cpu'."
        )


_DOWNCAST = {
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("complex64"),
}


def to_torch_dtype(dtype, downcast: bool = False) -> Any:
    """Translate a NumPy dtype to the equivalent ``torch`` dtype.

    Metal is a single-precision backend, so double precision has no
    representation on the device. ``downcast`` narrows it to single precision
    instead of raising, for a dtype that was merely inferred from the input --
    NumPy defaults a Python float to float64, and rejecting that would make
    ordinary values like a sampling tuple unusable. An explicitly requested
    double-precision dtype still raises, rather than quietly losing precision
    the caller asked for.
    """
    dtype = np.dtype(dtype)

    if dtype in _DOWNCAST:
        if not downcast:
            raise RuntimeError(
                f"Metal (MPS) does not support {dtype} arrays; it is a "
                "single-precision backend. Set "
                "abtem.config.set({'precision': 'float32'}), or change the device "
                "to 'cpu' or 'gpu' for double precision."
            )
        dtype = _DOWNCAST[dtype]

    try:
        return _TO_TORCH_DTYPE[dtype]
    except KeyError:
        raise RuntimeError(f"dtype {dtype} is not supported on Metal (MPS)") from None


def to_numpy_dtype(dtype) -> np.dtype:
    """Translate a ``torch`` dtype to the equivalent NumPy dtype."""
    return _TO_NUMPY_DTYPE[dtype]


def _unwrap(x):
    """Return the underlying tensor of a :class:`TorchNDArray`, else ``x``."""
    if isinstance(x, TorchNDArray):
        return x._tensor
    return x


def _wrap(x):
    """Wrap a tensor as a :class:`TorchNDArray`, passing anything else through."""
    if torch is not None and isinstance(x, torch.Tensor):
        return TorchNDArray(x)
    return x


def _unwrap_key(key):
    """Unwrap the (possibly nested) index expression of a ``__getitem__``."""
    if isinstance(key, tuple):
        return tuple(_unwrap_key(k) for k in key)
    return _unwrap(key)


def _resolve_reversed_slices(tensor, key):
    """Rewrite negative-step slices, which torch rejects but NumPy allows.

    ``array[::-1]`` is ordinary NumPy; torch raises "step must be greater than
    zero". Each reversed slice is turned into an explicit gather of the indices
    it selects, which reproduces NumPy's semantics exactly (including its
    handling of negative bounds) and leaves every other index form untouched.
    """
    entries = key if isinstance(key, tuple) else (key,)

    if not any(
        isinstance(entry, slice) and entry.step is not None and entry.step < 0
        for entry in entries
    ):
        return tensor, key

    # Ellipsis and None shift the mapping from key entries to tensor axes; the
    # gather below assumes they line up, so leave those keys to torch.
    if any(entry is Ellipsis or entry is None for entry in entries):
        return tensor, key

    resolved = []
    for axis, entry in enumerate(entries):
        if isinstance(entry, slice) and entry.step is not None and entry.step < 0:
            indices = range(*entry.indices(tensor.shape[axis]))
            tensor = torch.index_select(
                tensor,
                axis,
                torch.tensor(list(indices), dtype=torch.int64, device=tensor.device),
            )
            resolved.append(slice(None))
        else:
            resolved.append(entry)

    return tensor, tuple(resolved)


def _forward(name: str):
    """Build a method that applies the tensor's ``name`` and rewraps the result."""

    def method(self, *args, **kwargs):
        args = tuple(_unwrap(arg) for arg in args)
        kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
        return _wrap(getattr(self._tensor, name)(*args, **kwargs))

    method.__name__ = name
    return _serialized(method)


class TorchNDArray:
    """A ``torch.Tensor`` presenting the ``numpy.ndarray`` interface abTEM uses.

    Wrapping rather than subclassing keeps the NumPy-flavored parts of the API
    that ``torch.Tensor`` spells differently -- ``dtype`` as a NumPy dtype,
    ``size`` as an element count, ``astype``, ``copy`` -- from colliding with
    the tensor's own meanings for those names.
    """

    __array_priority__ = 100  # bind ndarray op TorchNDArray to our reflected dunder

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        """The wrapped ``torch.Tensor``."""
        return self._tensor

    @property
    def dtype(self) -> np.dtype:
        return to_numpy_dtype(self._tensor.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def size(self) -> int:
        return self._tensor.numel()

    @property
    def real(self):
        return _wrap(self._tensor.real)

    @property
    @_serialized
    def imag(self):
        # torch raises on .imag of a real tensor, where numpy returns zeros
        if self._tensor.is_complex():
            return _wrap(self._tensor.imag)
        return _wrap(torch.zeros_like(self._tensor))

    @imag.setter
    @_serialized
    def imag(self, value):
        # numpy supports assigning the imaginary part of a complex array in
        # place; torch exposes it only as a view, so write through that.
        self._tensor.imag.copy_(_unwrap(asarray(value)))

    @real.setter
    @_serialized
    def real(self, value):
        self._tensor.real.copy_(_unwrap(asarray(value)))

    @property
    def T(self):
        return _wrap(self._tensor.T)

    @_serialized
    def astype(self, dtype, copy: bool = True):
        torch_dtype = to_torch_dtype(dtype)
        if not copy and self._tensor.dtype == torch_dtype:
            return self
        return _wrap(self._tensor.to(torch_dtype))

    @_serialized
    def copy(self):
        return _wrap(self._tensor.clone())

    @_serialized
    def __deepcopy__(self, memo):
        # Without this, copy.deepcopy recurses into __dict__ and reaches
        # torch.Tensor.__deepcopy__, which touches Metal outside the lock --
        # abtem.core.utils.CopyMixin.copy deepcopies whole objects, so this is
        # on the ordinary path, not an exotic one. Cloning is also cheaper than
        # torch's own deepcopy, which reconstructs the tensor's autograd state.
        clone = TorchNDArray(self._tensor.clone())
        memo[id(self)] = clone
        return clone

    @_serialized
    def conjugate(self):
        return _wrap(torch.conj(self._tensor))

    conj = conjugate

    @_serialized
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _wrap(self._tensor.reshape(shape))

    @_serialized
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            return _wrap(self._tensor.permute(*reversed(range(self._tensor.ndim))))
        return _wrap(self._tensor.permute(*axes))

    @_serialized
    def swapaxes(self, axis1: int, axis2: int):
        return _wrap(self._tensor.transpose(axis1, axis2))

    @_serialized
    def sum(self, axis=None, **kwargs):
        if _is_empty_axis(axis):
            return _wrap(self._tensor.clone())
        if axis is None:
            return _wrap(self._tensor.sum(**kwargs))
        return _wrap(self._tensor.sum(dim=axis, **kwargs))

    @_serialized
    def mean(self, axis=None, **kwargs):
        if _is_empty_axis(axis):
            return _wrap(self._tensor.clone())
        if axis is None:
            return _wrap(self._tensor.mean(**kwargs))
        return _wrap(self._tensor.mean(dim=axis, **kwargs))

    @_serialized
    def max(self, axis=None, **kwargs):
        if axis is None:
            return _wrap(self._tensor.max(**kwargs))
        return _wrap(self._tensor.amax(dim=axis, **kwargs))

    @_serialized
    def min(self, axis=None, **kwargs):
        if axis is None:
            return _wrap(self._tensor.min(**kwargs))
        return _wrap(self._tensor.amin(dim=axis, **kwargs))

    @_serialized
    def get(self):
        """Return this array as a NumPy array (mirrors ``cupy.ndarray.get``)."""
        return asnumpy(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Run NumPy's ufuncs and operators on the device where implemented.

        This is the protocol behind ``host_array * device_array`` and
        ``host_array @ device_array``: the operator resolves to a ufunc, and
        without a say here NumPy either refuses outright or coerces the device
        operand to the host. Operands are moved onto the device, so the device
        side wins, which is what the caller asking for a device array wants.
        """
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        implementation = _ARRAY_UFUNCS.get(ufunc)

        if implementation is None:
            return NotImplemented

        return implementation(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Keep NumPy's dispatched functions on the device where implemented.

        Anything absent from the registry returns ``NotImplemented``, so NumPy
        raises rather than quietly copying the array to the host through
        ``__array__`` -- the same loud behavior CuPy has, and the one that suits
        a simulation library, where a silent host fallback shows up only as
        mysterious slowness.
        """
        implementation = _ARRAY_FUNCTIONS.get(func)

        if implementation is None:
            return NotImplemented

        return implementation(*args, **kwargs)

    @_serialized
    def __array__(self, dtype=None, copy=None):
        array = asnumpy(self)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __len__(self) -> int:
        return len(self._tensor)

    @_serialized
    def __iter__(self):
        return (_wrap(item) for item in self._tensor)

    @_serialized
    def __getitem__(self, key):
        key = _unwrap_key(key)
        tensor, key = _resolve_reversed_slices(self._tensor, key)
        return _wrap(tensor[key])

    @_serialized
    def __setitem__(self, key, value):
        value = _unwrap(value)
        # NumPy casts on assignment (e.g. a real result into a complex array);
        # torch instead refuses a dtype mismatch, so cast to match NumPy.
        if isinstance(value, torch.Tensor) and value.dtype != self._tensor.dtype:
            value = value.to(self._tensor.dtype)
        self._tensor[_unwrap_key(key)] = value

    def __repr__(self) -> str:
        return f"TorchNDArray({self._tensor!r})"

    @_serialized
    def __float__(self) -> float:
        return float(self._tensor)

    @_serialized
    def __int__(self) -> int:
        return int(self._tensor)

    @_serialized
    def __bool__(self) -> bool:
        return bool(self._tensor)

    def __getattr__(self, name):
        # only reached for names not defined above; forwards e.g. .item(), .flatten()
        if name.startswith("_"):
            raise AttributeError(name)
        attribute = getattr(self._tensor, name)
        if callable(attribute):
            return _forward(name).__get__(self, type(self))
        return _wrap(attribute)

    __add__ = _forward("__add__")
    __radd__ = _forward("__radd__")
    __sub__ = _forward("__sub__")
    __rsub__ = _forward("__rsub__")
    __mul__ = _forward("__mul__")
    __rmul__ = _forward("__rmul__")
    __truediv__ = _forward("__truediv__")
    __rtruediv__ = _forward("__rtruediv__")
    __pow__ = _forward("__pow__")
    __rpow__ = _forward("__rpow__")
    __mod__ = _forward("__mod__")
    __rmod__ = _forward("__rmod__")
    __matmul__ = _forward("__matmul__")
    # numpy's ndarray.__matmul__ defers to us (see __array_priority__), so the
    # reflected form is what actually runs for `host_array @ device_array`.
    __rmatmul__ = _forward("__rmatmul__")
    __neg__ = _forward("__neg__")
    __abs__ = _forward("__abs__")
    __and__ = _forward("__and__")
    __rand__ = _forward("__rand__")
    __or__ = _forward("__or__")
    __ror__ = _forward("__ror__")
    __xor__ = _forward("__xor__")
    __rxor__ = _forward("__rxor__")
    __invert__ = _forward("__invert__")
    __eq__ = _forward("__eq__")
    __ne__ = _forward("__ne__")
    __lt__ = _forward("__lt__")
    __le__ = _forward("__le__")
    __gt__ = _forward("__gt__")
    __ge__ = _forward("__ge__")

    # mypy compares these against the *args/**kwargs signatures _forward builds
    # for the matching binary operators, which it cannot see are compatible.
    @_serialized
    def __iadd__(self, other):  # type: ignore[misc]
        self._tensor += _unwrap(other)
        return self

    @_serialized
    def __isub__(self, other):  # type: ignore[misc]
        self._tensor -= _unwrap(other)
        return self

    @_serialized
    def __imul__(self, other):  # type: ignore[misc]
        self._tensor *= _unwrap(other)
        return self

    @_serialized
    def __itruediv__(self, other):  # type: ignore[misc]
        self._tensor /= _unwrap(other)
        return self


@_serialized
def asarray(data, dtype=None):
    """Copy ``data`` onto the Metal device as a :class:`TorchNDArray`."""
    _check_available()

    if isinstance(data, TorchNDArray):
        return data.astype(dtype, copy=False) if dtype is not None else data

    if isinstance(data, torch.Tensor):
        tensor = data.to(DEVICE)
        if dtype is not None:
            tensor = tensor.to(to_torch_dtype(dtype))
        return TorchNDArray(tensor)

    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # A dtype the caller did not ask for is inferred from the data, so double
    # precision there is incidental (NumPy's default for a Python float) and is
    # narrowed rather than rejected.
    inferred = dtype is None
    if inferred:
        dtype = data.dtype
    if not data.flags["C_CONTIGUOUS"] or any(stride < 0 for stride in data.strides):
        # The stride test is not redundant: NumPy calls a size-1 array
        # contiguous whatever its stride sign, so `argsort(...)[::-1]` of a
        # single element passes the flag while still carrying a negative
        # stride, which torch refuses. copy() rather than np.asarray(order="C")
        # for the same reason -- asarray believes the flag and declines to
        # copy -- and rather than np.ascontiguousarray, which would promote a
        # 0-d scalar to shape (1,) and add a dimension to every scalar.
        data = data.copy(order="C")
    # from_numpy avoids a host-side copy before the device transfer
    tensor = torch.from_numpy(data)
    torch_dtype = to_torch_dtype(dtype, downcast=inferred)
    return TorchNDArray(tensor.to(torch_dtype).to(DEVICE))


@_serialized
def array(data, dtype=None):
    """Like :func:`asarray`, but also stacks sequences of device arrays."""
    if isinstance(data, (list, tuple)) and data:
        if any(isinstance(item, (TorchNDArray, torch.Tensor)) for item in data):
            tensor = torch.stack([_unwrap(asarray(item)) for item in data])
            if dtype is not None:
                tensor = tensor.to(to_torch_dtype(dtype))
            return TorchNDArray(tensor)

    return asarray(data, dtype=dtype)


@_serialized
def asnumpy(x):
    """Copy a device array back to the host as a NumPy array."""
    if isinstance(x, TorchNDArray):
        x = x.tensor
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _creation(name: str):
    """Build ``zeros``/``ones``/``empty``, which take a shape and a NumPy dtype."""

    def func(shape, dtype=None):
        _check_available()
        if isinstance(shape, (int, np.integer)):
            shape = (shape,)
        torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
        tensor = getattr(torch, name)(tuple(shape), dtype=torch_dtype, device=DEVICE)
        return TorchNDArray(tensor)

    func.__name__ = name
    return _serialized(func)


def _elementwise(name: str):
    """Build a unary ufunc that forwards to ``torch`` and rewraps the result."""

    def func(x, *args, **kwargs):
        tensor = _unwrap(x)
        if not isinstance(tensor, torch.Tensor):
            # A plain Python or NumPy scalar: torch's ufuncs insist on a tensor,
            # while NumPy's answer here is a host scalar. Keep it on the host --
            # moving a scalar to the device to take its sine would be silly.
            return getattr(np, name)(tensor, *args, **kwargs)

        args = tuple(_unwrap(arg) for arg in args)
        kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
        return _wrap(getattr(torch, name)(tensor, *args, **kwargs))

    func.__name__ = name
    return _serialized(func)


def _is_empty_axis(axis) -> bool:
    """Whether ``axis`` selects no axes at all.

    NumPy reads an empty axis tuple as "reduce nothing" and returns the array
    unchanged; torch reads ``dim=()`` as "reduce every axis" and returns a
    scalar. abTEM reduces over a computed tuple of ensemble axes, which is
    empty whenever there are none, so the two readings differ on a real code
    path rather than a hypothetical one.
    """
    return isinstance(axis, (tuple, list)) and len(axis) == 0


def _reduction(name: str):
    """Build a reduction that accepts NumPy's ``axis=`` rather than ``dim=``."""

    def func(x, axis=None, **kwargs):
        tensor = _unwrap(x)
        if "keepdims" in kwargs:
            kwargs["keepdim"] = kwargs.pop("keepdims")
        if _is_empty_axis(axis):
            return _wrap(tensor.clone())
        if axis is None:
            return _wrap(getattr(torch, name)(tensor, **kwargs))
        return _wrap(getattr(torch, name)(tensor, dim=axis, **kwargs))

    func.__name__ = name
    return _serialized(func)


@_serialized
def tensordot(a, b, axes=2):
    """``numpy.tensordot``, whose ``axes`` torch spells ``dims``.

    Both operands are moved onto the device first: callers routinely contract a
    device array against a host-built one (a detector mask, say), which NumPy
    handles implicitly and torch refuses. The axis pair is also normalized to
    lists of non-negative ints, the only spelling torch's overload accepts.
    """
    a_tensor = _unwrap(asarray(a))
    b_tensor = _unwrap(asarray(b))

    def _axis_list(axis, ndim):
        # NumPy accepts either a single axis or a sequence on each side.
        if isinstance(axis, (int, np.integer)):
            axis = (axis,)
        return [int(a) % ndim for a in axis]

    if isinstance(axes, (tuple, list)) and len(axes) == 2:
        dims = (
            _axis_list(axes[0], a_tensor.ndim),
            _axis_list(axes[1], b_tensor.ndim),
        )
    else:
        dims = int(axes)

    return _wrap(torch.tensordot(a_tensor, b_tensor, dims=dims))


@_serialized
def where(condition, x, y):
    return _wrap(torch.where(_unwrap(condition), _unwrap(x), _unwrap(y)))


def _creation_like(name: str):
    """Build ``zeros_like``/``ones_like``/``empty_like`` over a device array."""

    def func(x, dtype=None, shape=None):
        tensor = _unwrap(x)
        torch_dtype = to_torch_dtype(dtype) if dtype is not None else tensor.dtype
        if shape is not None:
            # numpy's *_like take a shape override, which torch's spell as a
            # plain creation call on the same device and dtype.
            if isinstance(shape, (int, np.integer)):
                shape = (shape,)
            creation = name.replace("_like", "")
            return TorchNDArray(
                getattr(torch, creation)(
                    tuple(shape), dtype=torch_dtype, device=tensor.device
                )
            )
        return TorchNDArray(getattr(torch, name)(tensor, dtype=torch_dtype))

    func.__name__ = name
    return _serialized(func)


zeros_like = _creation_like("zeros_like")


@_serialized
def _add_at(a, indices, values):
    """In-place ``a[indices] += values`` accumulating repeated indices.

    The NumPy/CuPy spelling of this is ``np.add.at`` / ``cupyx.scatter_add``;
    unlike plain fancy-index assignment, contributions to a repeated index are
    summed rather than overwriting each other. That is what makes it correct for
    superposing delta functions of atoms that land in the same pixel.
    """
    tensor = _unwrap(a)
    if isinstance(indices, tuple):
        indices = tuple(_unwrap(asarray(index)).long() for index in indices)
    else:
        indices = (_unwrap(asarray(indices)).long(),)

    values = _unwrap(asarray(values, dtype=to_numpy_dtype(tensor.dtype)))
    tensor.index_put_(indices, values.expand(indices[0].shape), accumulate=True)
    return a


@_serialized
def sum_run_length_encoded(array, result, separators):
    """Sum run-length-encoded data into bins, the Metal counterpart of the
    CuPy kernel in :mod:`abtem.core._cuda` and the Numba loop on the CPU.

    Parameters
    ----------
    array : TorchNDArray, shape (n_batch, n_selected)
        The reindexed diffraction-pattern data (selected pixels only, in bin
        order).
    result : TorchNDArray, shape (n_batch, n_bins)
        Output array, added to in place (must be pre-zeroed).
    separators : TorchNDArray, shape (n_bins + 1,)
        Cumulative bin boundary offsets into the second axis of ``array``.
    """
    array_tensor = _unwrap(array)
    result_tensor = _unwrap(result)
    separators_tensor = _unwrap(separators).to(torch.int64)

    n_bins = result_tensor.shape[1]
    if n_bins == 0:
        return

    # Expand the run-length encoding into a per-column bin index, so the whole
    # reduction is a single scatter-add rather than a loop over bins.
    counts = separators_tensor[1:] - separators_tensor[:-1]
    segments = torch.repeat_interleave(
        torch.arange(n_bins, device=array_tensor.device), counts
    )
    result_tensor.index_add_(1, segments, array_tensor)


@_serialized
def fftfreq(n: int, d: float = 1.0, dtype=None):
    _check_available()
    torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
    return TorchNDArray(torch.fft.fftfreq(n, d=d, dtype=torch_dtype, device=DEVICE))


def synchronize() -> None:
    """Block until all queued Metal work has completed (for timing)."""
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


class RandomState:
    """``numpy.random.RandomState`` returning device-resident arrays.

    Deliberately draws on the host with NumPy and transfers the result rather
    than using torch's own generator: a device-parametrized test that seeds
    identically on 'cpu' and 'mps' and compares the two results needs the same
    numbers on both devices, and torch's RNG stream does not match NumPy's.
    """

    def __init__(self, seed=None):
        self._random_state = np.random.RandomState(seed)

    def rand(self, *shape):
        return asarray(self._random_state.rand(*shape))

    def randn(self, *shape):
        return asarray(self._random_state.randn(*shape))


_random = SimpleNamespace(RandomState=RandomState)


def _fft_func(name: str):
    def func(x, **kwargs):
        # numpy's transformed axes are `axes` (or `axis` for the 1-D
        # transforms); torch spells both `dim`.
        if "axes" in kwargs:
            kwargs["dim"] = kwargs.pop("axes")
        if "axis" in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return _wrap(getattr(torch.fft, name)(_unwrap(x), **kwargs))

    func.__name__ = name
    return _serialized(func)


def _fft_shift_func(name: str):
    """``fftshift``/``ifftshift``, whose ``axes`` torch spells ``dim``."""

    def func(x, axes=None):
        return _wrap(getattr(torch.fft, name)(_unwrap(x), dim=axes))

    func.__name__ = name
    return _serialized(func)


_fft = SimpleNamespace(
    fft2=_fft_func("fft2"),
    ifft2=_fft_func("ifft2"),
    fftn=_fft_func("fftn"),
    ifftn=_fft_func("ifftn"),
    fft=_fft_func("fft"),
    ifft=_fft_func("ifft"),
    fftshift=_fft_shift_func("fftshift"),
    ifftshift=_fft_shift_func("ifftshift"),
    fftfreq=fftfreq,
)


# ``xp.add.at(...)`` is how abTEM spells scatter-add; mirror NumPy's ufunc shape.
_add = SimpleNamespace(at=_add_at)


@_serialized
def arange(*args, dtype=None):
    _check_available()
    torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
    return TorchNDArray(torch.arange(*args, dtype=torch_dtype, device=DEVICE))


@_serialized
def linspace(start, stop, num=50, endpoint=True, dtype=None):
    _check_available()
    torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
    if not endpoint and num > 0:
        # torch.linspace always includes the stop value; drop to the last
        # sample of the half-open interval so the spacing stays (stop-start)/num
        stop = start + (stop - start) * (num - 1) / num
    return TorchNDArray(
        torch.linspace(start, stop, num, dtype=torch_dtype, device=DEVICE)
    )


@_serialized
def squeeze(x, axis=None):
    """``numpy.squeeze``, whose ``axis`` torch spells ``dim``."""
    if axis is not None:
        # numpy treats an empty axis tuple as "squeeze nothing"; torch rejects
        # it. abtem.array.ArrayObject.squeeze passes one whenever no ensemble
        # axis is of length one, which is the common case.
        axis = tuple(int(a) for a in np.atleast_1d(axis).ravel())
        if not axis:
            return x

    # A dask array is a container around device chunks, so the squeeze belongs
    # to dask. NumPy and CuPy get here too, via their __array_function__
    # dispatch; this namespace has to route it explicitly.
    if isinstance(x, da.core.Array):
        return da.squeeze(x, axis=axis)

    tensor = _unwrap(x)
    if axis is None:
        return _wrap(torch.squeeze(tensor))
    return _wrap(torch.squeeze(tensor, dim=axis))


@_serialized
def diff(x, n: int = 1, axis: int = -1):
    """``numpy.diff``, whose ``axis`` torch spells ``dim``."""
    return _wrap(torch.diff(_unwrap(x), n=n, dim=axis))


@_serialized
def full(shape, fill_value, dtype=None):
    _check_available()
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
    return TorchNDArray(
        torch.full(tuple(shape), fill_value, dtype=torch_dtype, device=DEVICE)
    )


@_serialized
def expand_dims(x, axis):
    tensor = _unwrap(x)
    for ax in (axis,) if isinstance(axis, int) else sorted(axis):
        tensor = torch.unsqueeze(tensor, ax)
    return _wrap(tensor)


@_serialized
def concatenate(arrays, axis=0):
    return _wrap(torch.cat([_unwrap(asarray(a)) for a in arrays], dim=axis))


@_serialized
def stack(arrays, axis=0):
    return _wrap(torch.stack([_unwrap(asarray(a)) for a in arrays], dim=axis))


@_serialized
def transpose(x, axes=None):
    tensor = _unwrap(x)
    if axes is None:
        return _wrap(tensor.permute(*reversed(range(tensor.ndim))))
    return _wrap(tensor.permute(*axes))


@_serialized
def meshgrid(*arrays, indexing="xy"):
    tensors = torch.meshgrid(*[_unwrap(asarray(a)) for a in arrays], indexing=indexing)
    return tuple(_wrap(t) for t in tensors)


@_serialized
def tile(x, reps):
    if isinstance(reps, (int, np.integer)):
        reps = (reps,)
    return _wrap(torch.tile(_unwrap(x), tuple(reps)))


@_serialized
def clip(x, a_min=None, a_max=None):
    """``numpy.clip``, whose bounds torch spells ``min``/``max``."""
    return _wrap(torch.clamp(_unwrap(x), min=_unwrap(a_min), max=_unwrap(a_max)))


@_serialized
def _pad_indices(length: int, before: int, after: int, mode: str):
    """Source indices along one axis for a padded output of ``numpy.pad``.

    Expressing every mode as a gather keeps one implementation for any number
    of dimensions. ``torch.nn.functional.pad`` is not usable here: it orders
    its pad widths from the last axis backwards and restricts its non-constant
    modes to the spatial axes of 3-to-5-dimensional tensors.
    """
    positions = torch.arange(-before, length + after, device=DEVICE)

    if mode == "wrap":
        return positions % length

    if mode == "reflect":
        if length == 1:
            return torch.zeros_like(positions)
        # Reflect about the edges without repeating them: a period of
        # 2 * (length - 1), folded back into [0, length).
        period = 2 * (length - 1)
        folded = positions % period
        return torch.where(folded >= length, period - folded, folded)

    raise RuntimeError(f"pad mode {mode!r} is not implemented for Metal (MPS)")


@_serialized
def pad(array, pad_width, mode: str = "constant", constant_values=0):
    """``numpy.pad`` for the modes abTEM uses: constant, wrap and reflect."""
    tensor = _unwrap(array)
    ndim = tensor.ndim

    # numpy accepts a scalar, one (before, after) pair for every axis, or one
    # pair per axis; normalize to the per-axis form.
    if isinstance(pad_width, (int, np.integer)):
        widths = [(int(pad_width), int(pad_width))] * ndim
    else:
        pad_width = list(pad_width)
        if len(pad_width) == 2 and not isinstance(pad_width[0], (tuple, list)):
            widths = [(int(pad_width[0]), int(pad_width[1]))] * ndim
        else:
            widths = [(int(before), int(after)) for before, after in pad_width]

    if len(widths) != ndim:
        raise ValueError(f"pad_width does not match the {ndim} array dimensions")

    if mode == "constant":
        shape = [
            length + before + after
            for length, (before, after) in zip(tensor.shape, widths)
        ]
        out = torch.full(
            shape, constant_values, dtype=tensor.dtype, device=tensor.device
        )
        interior = tuple(
            slice(before, before + length)
            for length, (before, _) in zip(tensor.shape, widths)
        )
        out[interior] = tensor
        return _wrap(out)

    for axis, (before, after) in enumerate(widths):
        if before == 0 and after == 0:
            continue
        indices = _pad_indices(tensor.shape[axis], before, after, mode)
        tensor = torch.index_select(tensor, axis, indices)

    return _wrap(tensor)


def _binary_ufunc(name: str):
    """Build a binary ufunc that brings both operands onto the device first."""

    def func(a, b, **kwargs):
        return _wrap(
            getattr(torch, name)(_unwrap(asarray(a)), _unwrap(asarray(b)), **kwargs)
        )

    func.__name__ = name
    return _serialized(func)


@_serialized
def nonzero(x):
    """``numpy.nonzero`` -- a tuple of index arrays, one per dimension."""
    indices = torch.nonzero(_unwrap(asarray(x)), as_tuple=True)
    return tuple(_wrap(index) for index in indices)


@_serialized
def einsum(subscripts, *operands, **kwargs):
    return _wrap(torch.einsum(subscripts, *[_unwrap(asarray(o)) for o in operands]))


def iscomplexobj(x) -> bool:
    """``numpy.iscomplexobj`` -- a dtype question, answered without the device."""
    return _unwrap(x).is_complex()


def roll(x, shift, axis=None):
    """``numpy.roll``, whose ``shift``/``axis`` torch spells ``shifts``/``dims``."""
    return _wrap(torch.roll(_unwrap(x), shifts=shift, dims=axis))


@_serialized
def eye(n, m=None, dtype=None):
    _check_available()
    torch_dtype = to_torch_dtype(dtype) if dtype is not None else None
    return TorchNDArray(
        torch.eye(n, m if m is not None else n, dtype=torch_dtype, device=DEVICE)
    )


@_serialized
def ascontiguousarray(x):
    return _wrap(_unwrap(asarray(x)).contiguous())


def _linalg_func(name: str):
    def func(*args, **kwargs):
        # Through asarray rather than a bare unwrap: an operand may still be a
        # host array, and a reversed one (``vectors[:, ::-1]``) carries negative
        # strides that torch cannot adopt. asarray makes a contiguous device
        # copy instead.
        args = tuple(
            _unwrap(asarray(arg)) if hasattr(arg, "dtype") else arg for arg in args
        )
        result = getattr(torch.linalg, name)(*args, **kwargs)
        # torch returns named tuples where numpy returns plain ones
        if isinstance(result, tuple):
            return tuple(_wrap(item) for item in result)
        return _wrap(result)

    func.__name__ = name
    return _serialized(func)


# Metal supports these in single precision, which is all abTEM asks of them
# here -- the one place needing double precision streams to the host instead.
_linalg = SimpleNamespace(
    eigh=_linalg_func("eigh"),
    eigvalsh=_linalg_func("eigvalsh"),
    svd=_linalg_func("svd"),
    norm=_linalg_func("norm"),
    solve=_linalg_func("solve"),
    inv=_linalg_func("inv"),
)


@_serialized
def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False) -> bool:
    """``numpy.allclose``, compared on the device rather than on the host."""
    return bool(
        torch.allclose(
            _unwrap(asarray(a)),
            _unwrap(asarray(b)),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    )


def _binary(name: str):
    """Build a binary ufunc (``maximum``, ``minimum``, ...) over two arrays."""

    def func(x, y, **kwargs):
        return _wrap(getattr(torch, name)(_unwrap(x), _unwrap(y), **kwargs))

    func.__name__ = name
    return _serialized(func)


class _TorchNumpyNamespace:
    """The ``numpy``-module stand-in for arrays living on the Metal device.

    Only the operations abTEM's Metal-supported code paths reach are defined.
    Anything else raises :class:`AttributeError` naming the missing operation,
    which is both how callers detect an unsupported path (and fall back to
    NumPy) and how a developer learns exactly what to add here.
    """

    # identity of the namespace and its array type
    ndarray = TorchNDArray
    Tensor = torch.Tensor if torch is not None else None
    fft = _fft
    add = _add
    random = _random
    linalg = _linalg

    # array creation and host transfer
    asarray = staticmethod(asarray)
    array = staticmethod(array)
    asnumpy = staticmethod(asnumpy)
    zeros = staticmethod(_creation("zeros"))
    ones = staticmethod(_creation("ones"))
    empty = staticmethod(_creation("empty"))
    zeros_like = staticmethod(zeros_like)
    ones_like = staticmethod(_creation_like("ones_like"))
    empty_like = staticmethod(_creation_like("empty_like"))
    full = staticmethod(full)
    arange = staticmethod(arange)
    linspace = staticmethod(linspace)
    meshgrid = staticmethod(meshgrid)
    eye = staticmethod(eye)
    nonzero = staticmethod(nonzero)
    einsum = staticmethod(einsum)
    ascontiguousarray = staticmethod(ascontiguousarray)
    where = staticmethod(where)

    # shape manipulation
    expand_dims = staticmethod(expand_dims)
    concatenate = staticmethod(concatenate)
    stack = staticmethod(stack)
    transpose = staticmethod(transpose)
    tile = staticmethod(tile)
    pad = staticmethod(pad)
    squeeze = staticmethod(squeeze)
    moveaxis = staticmethod(_elementwise("moveaxis"))
    swapaxes = staticmethod(_elementwise("swapaxes"))
    roll = staticmethod(roll)

    # elementwise
    exp = staticmethod(_elementwise("exp"))
    log = staticmethod(_elementwise("log"))
    sqrt = staticmethod(_elementwise("sqrt"))
    square = staticmethod(_elementwise("square"))
    abs = staticmethod(_elementwise("abs"))
    sin = staticmethod(_elementwise("sin"))
    cos = staticmethod(_elementwise("cos"))
    tan = staticmethod(_elementwise("tan"))
    sinc = staticmethod(_elementwise("sinc"))
    sign = staticmethod(_elementwise("sign"))
    floor = staticmethod(_elementwise("floor"))
    ceil = staticmethod(_elementwise("ceil"))
    round = staticmethod(_elementwise("round"))
    rint = staticmethod(_elementwise("round"))
    conjugate = staticmethod(_elementwise("conj"))
    conj = staticmethod(_elementwise("conj"))
    angle = staticmethod(_elementwise("angle"))
    real = staticmethod(_elementwise("real"))
    imag = staticmethod(_elementwise("imag"))
    clip = staticmethod(clip)
    maximum = staticmethod(_binary("maximum"))
    minimum = staticmethod(_binary("minimum"))
    arctan2 = staticmethod(_binary("atan2"))

    # reductions
    sum = staticmethod(_reduction("sum"))
    prod = staticmethod(_reduction("prod"))
    mean = staticmethod(_reduction("mean"))
    cumsum = staticmethod(_reduction("cumsum"))
    diff = staticmethod(diff)
    std = staticmethod(_reduction("std"))
    min = staticmethod(_reduction("amin"))
    max = staticmethod(_reduction("amax"))
    tensordot = staticmethod(tensordot)

    # dtypes, mirroring numpy's names so ``xp.int32`` and friends keep working
    float32 = np.float32
    float64 = np.float64
    complex64 = np.complex64
    complex128 = np.complex128
    int32 = np.int32
    int64 = np.int64
    bool_ = np.bool_
    pi = np.pi
    inf = np.inf

    # device control
    synchronize = staticmethod(synchronize)

    def __getattr__(self, name):
        raise AttributeError(
            f"'{name}' is not implemented for the Metal (MPS) backend. Metal "
            "support is experimental and covers only part of abTEM; run this "
            "part of the calculation on the 'cpu' device, or add '{name}' to "
            "abtem/core/_torch.py.".replace("{name}", name)
        )


torch_numpy = _TorchNumpyNamespace()


# See _ARRAY_FUNCTIONS above. np.concatenate is the one dask itself needs (to
# assemble computed chunks); the rest spare callers who reach for the NumPy
# spelling of an operation this namespace already provides.
_ARRAY_FUNCTIONS.update(
    {
        np.concatenate: concatenate,
        np.stack: stack,
        np.squeeze: squeeze,
        np.expand_dims: expand_dims,
        np.transpose: transpose,
        np.tile: tile,
        np.pad: pad,
        np.diff: diff,
        np.tensordot: tensordot,
        np.where: where,
        np.clip: clip,
        np.zeros_like: zeros_like,
        np.ones_like: torch_numpy.ones_like,
        np.empty_like: torch_numpy.empty_like,
        np.sum: torch_numpy.sum,
        np.prod: torch_numpy.prod,
        np.mean: torch_numpy.mean,
        np.abs: torch_numpy.abs,
        np.conjugate: torch_numpy.conjugate,
        np.angle: torch_numpy.angle,
        np.round: torch_numpy.round,
        np.iscomplexobj: iscomplexobj,
        np.nonzero: nonzero,
        np.einsum: einsum,
        np.allclose: allclose,
        np.roll: roll,
        np.fft.fftshift: _fft.fftshift,
        np.fft.ifftshift: _fft.ifftshift,
    }
)


# See _ARRAY_UFUNCS above. Binary entries move both operands onto the device,
# so a host operand on either side of an operator works the way NumPy's own
# mixed-type arithmetic does.
_ARRAY_UFUNCS.update(
    {
        np.matmul: _binary_ufunc("matmul"),
        np.multiply: _binary_ufunc("multiply"),
        np.add: _binary_ufunc("add"),
        np.subtract: _binary_ufunc("subtract"),
        np.true_divide: _binary_ufunc("true_divide"),
        np.power: _binary_ufunc("pow"),
        np.maximum: _binary_ufunc("maximum"),
        np.minimum: _binary_ufunc("minimum"),
        np.exp: torch_numpy.exp,
        np.sqrt: torch_numpy.sqrt,
        np.absolute: torch_numpy.abs,
        np.conjugate: torch_numpy.conjugate,
        np.sin: torch_numpy.sin,
        np.cos: torch_numpy.cos,
    }
)
