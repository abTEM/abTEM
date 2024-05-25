import warnings
from types import SimpleNamespace

import numpy as np

from abtem.core.config import config

try:
    import torch
except ModuleNotFoundError:
    torch = None
except ImportError:
    if config.get("device") == "metal":
        warnings.warn(
            "The PyTorch library could not be imported. Please check your installation, or change your configuration "
            "to use CPU."
        )
    torch = None


dtype_map = {
    np.float32: torch.float32,
    np.dtype("float32"): torch.float32,
    np.dtype("complex64"): torch.complex64,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def _get_tensor(torch_ndarray):
    if isinstance(torch_ndarray, TorchNDArray):
        return torch_ndarray.tensor
    return torch_ndarray


def _binary_operation(name):

    def func(self, other):
        other = _get_tensor(other)
        return self.__class__(getattr(self.tensor, name)(other))

    return func


def _unary_operation(name):
    def func(self):
        return self.__class__(getattr(self.tensor, name)())

    return func


class TorchNDArray:
    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def tensor(self):
        return self._tensor

    def __setitem__(self, key, value):
        value = _get_tensor(value)
        key = _get_tensor(key)
        self._tensor.__setitem__(key, value)

    def __getitem__(self, item):
        return self.__class__(self._tensor[item])

    def __repr__(self):
        return repr(self._tensor)

    __pow__ = _binary_operation("__pow__")
    __add__ = _binary_operation("__add__")
    __radd__ = _binary_operation("__radd__")
    __sub__ = _binary_operation("__sub__")
    __rsub__ = _binary_operation("__rsub__")
    __mul__ = _binary_operation("__mul__")
    __rmul__ = _binary_operation("__rmul__")
    __truediv__ = _binary_operation("__truediv__")
    __rtruediv__ = _binary_operation("__rtruediv__")

    __eq__ = _binary_operation("__eq__")
    __ne__ = _binary_operation("__ne__")
    __lt__ = _binary_operation("__lt__")
    __le__ = _binary_operation("__le__")
    __gt__ = _binary_operation("__gt__")
    __ge__ = _binary_operation("__ge__")

    __neg__ = _unary_operation("__neg__")

    def astype(self, dtype):
        return self.__class__(self._tensor.type(dtype_map[dtype]))

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def torch_ndarray_func(name, namespace=None):
    if namespace is None:
        namespace = torch

    def func(*args, **kwargs):
        args = (args[0].tensor,) + args[1:]

        tensor = getattr(namespace, name)(*args, **kwargs)

        return TorchNDArray(tensor)

    return func


torch__fft = SimpleNamespace()
torch__fft.fft2 = torch_ndarray_func("fft2", torch.fft)
torch__fft.ifft2 = torch_ndarray_func("ifft2", torch.fft)
torch__fft.fftfreq = lambda *args, **kwargs: TorchNDArray(
    torch.fft.fftfreq(*args, **kwargs).to("mps")
)


class TorchPy:

    @property
    def fft(self):
        return torch__fft

    @property
    def ndarray(self):
        return TorchNDArray

    def asarray(self, data, dtype=None):
        if dtype is None:
            dtype = data.dtype

        return TorchNDArray(torch.tensor(data, dtype=dtype_map[dtype]).to("mps"))

    def where(self, condition, x, y):
        condition = _get_tensor(condition)
        x = _get_tensor(x)
        y = _get_tensor(y)
        return TorchNDArray(torch.where(condition, x, y))

    def asnumpy(self, torch_ndarray):
        return torch_ndarray.tensor.cpu().numpy()

    def __getattr__(self, name):
        return torch_ndarray_func(name)
