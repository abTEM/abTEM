from __future__ import annotations

import inspect
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    ParamSpec,
    TypeGuard,
    TypeVar,
    cast,
    get_type_hints,
)

import array_api_compat
import array_api_compat.numpy as np_compat
import numpy as np
import scipy
from dask.array.core import Array as DaskArray

from abtem.core.config import config
from abtem.core.typing import (
    Array,
    ArrayNamespace,
    Device,
    DType,
    Scalar,
)

P = ParamSpec("P")
R = TypeVar("R")
DT = TypeVar("DT", bound=DType, covariant=True)


if TYPE_CHECKING:
    import array_api_compat.cupy as cupy_compat
    import array_api_compat.torch as torch_compat
    import cupy as cp  # type: ignore
    import cupyx  # type: ignore
    import torch  # type: ignore
    # Necessary for fully deterministic outputs! https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    torch.use_deterministic_algorithms(True)
else:
    try:
        import array_api_compat.cupy as cupy_compat
    except ImportError:
        cupy_compat = None

    try:
        import cupy as cp  # type: ignore
    except ImportError:
        cp = None

    try:
        import array_api_compat.torch as torch_compat
    except ImportError:
        torch_compat = None

    try:
        import torch  # type: ignore
        torch.use_deterministic_algorithms(True)
    except ImportError:
        torch = None

    try:
        import cupyx  # type: ignore
    except ImportError:
        assert cp is None
        cupyx = None

BackendName = Literal["numpy", "cupy", "torch"]


class CupyNotPresentError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "CuPy is not installed. Please install CuPy to use GPU calculations."
        )


class TorchNotPresentError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "PyTorch is not installed. Please install CuPy to use GPU calculations."
        )


def is_module_available(module: ModuleType | None) -> bool:
    return module is not None


def is_numpy_array(x: Any) -> bool:
    return array_api_compat.is_numpy_array(x)


def is_torch_array(x: Any) -> bool:
    if torch is None:
        return False

    # if isinstance(x, DaskArray):
    #     if isinstance(x, DaskTensor):
    #         return True

    return array_api_compat.is_torch_array(x)


def is_cupy_array(x: Any) -> bool:
    if cp is None:
        return False

    # if isinstance(x, DaskArray):
    #     return isinstance(x, DaskTensor)

    return array_api_compat.is_cupy_array(x)


def is_any_array(x: Any) -> TypeGuard[Array]:
    return (
        is_numpy_array(x)
        or is_torch_array(x)
        or is_cupy_array(x)
        or isinstance(x, DaskArray)
    )


def is_scalar_array(x: object | Array[Any, DT]) -> TypeGuard[Scalar[DT]]:
    if is_any_array(x) and x.ndim == 0:
        return True
    else:
        return False


def is_array_namespace(x: Any) -> bool:
    if isinstance(x, ArrayNamespace):
        return True
    if x is np_compat or x is np:
        return True
    if x is torch_compat or x is torch:
        return True
    return False


def get_array_namespace(
    x: ArrayNamespace | DaskArray | Array[Any, Any] | BackendName | None = None,
) -> ArrayNamespace:
    xp = getattr(x, "__array_namespace__", None)
    if xp is not None:
        return xp()

    if is_array_namespace(x):
        return cast(ArrayNamespace, x)

    if isinstance(x, DaskArray):
        return get_array_namespace(x._meta)

    if x is None:
        x = config.get("backend", "numpy")

    if isinstance(x, str):
        if x.lower() == "numpy":
            return np_compat  # type: ignore
        elif x.lower() == "torch":
            if is_module_available(torch):
                return torch_compat  # type: ignore
        elif x.lower() == "cupy":
            if is_module_available(cp):
                return cupy_compat  # type: ignore
            else:
                raise CupyNotPresentError()
        else:
            raise ValueError(f"Unknown backend: {x}")

    return array_api_compat.array_namespace(x)  # type: ignore


def array_namespace_name(
    xp: ArrayNamespace | Array[Any, Any] | BackendName | None = None,
) -> BackendName:
    if xp is np_compat or xp is np:
        return "numpy"
    elif xp is torch_compat or xp is torch:
        return "torch"
    elif (cupy_compat is not None and cp is not None) and (
        xp is cupy_compat or xp is cp
    ):
        return "cupy"

    xp = get_array_namespace(xp)

    return array_namespace_name(xp)


def assert_array_namespace(
    xp: ArrayNamespace | Array[Any, Any],
    backend: BackendName | tuple[BackendName, ...],
) -> None:
    name = array_namespace_name(xp)
    if not isinstance(backend, tuple):
        backend = (backend,)
    if name not in backend:
        raise ValueError(
            f"Expected array namespace {backend}, but got {name}."
            f" Please check your configuration or the array type."
        )


def assert_numpy_array(x: Array[Any, Any]) -> TypeGuard[np.ndarray]:
    if not array_api_compat.is_numpy_array(x):
        raise TypeError(
            f"Expected a NumPy array, but got {type(x).__name__}. "
            "Please check your array type."
        )
    return True


def ensure_numpy(x: Any, device: Device | None = None) -> Any:
    if is_numpy_array(x):
        return x
    elif is_torch_array(x):
        return x.cpu().numpy()  # type: ignore
    elif is_cupy_array(x):
        return cp.asnumpy(x)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")


def ensure_torch(x: Array[Any, Any], device: Device | None = None) -> Array:
    if is_torch_array(x):
        return x
    elif is_numpy_array(x):
        return torch.from_numpy(x).to(device)
    elif is_cupy_array(x):
        return torch.from_numpy(cp.asnumpy(x)).to(device)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")


def ensure_cupy(x: Array[Any, Any], device: Device | None = None) -> Array:
    if is_cupy_array(x):
        return x
    elif is_numpy_array(x):
        return cp.asarray(x)
    elif is_torch_array(x):
        return cp.asarray(x.cpu().numpy(), device=device)  # type: ignore
    else:
        raise TypeError(f"Unsupported type for conversion: {type(x)}")


def ensure_array_backend(
    x: Array[Any, Any], backend: str, device: Device | None = None
) -> Array:
    if backend == "numpy":
        return ensure_numpy(x)
    elif backend == "torch":
        return ensure_torch(x, device=device)
    elif backend == "cupy":
        return ensure_cupy(x, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_device(x: Array[Any, Any] | None = None) -> Device:
    if x is None:
        return config.get("device", "cpu")
    return array_api_compat.device(x)


def to_device(
    array: Array[Any, Any], device: Array[Any, Any] | Device | str
) -> Array[Any, Any]:
    try:
        device = array_api_compat.device(device)
    except AttributeError:
        pass

    return array_api_compat.to_device(array, device)


def validate_device(device: str | None = None) -> str:
    if device is None:
        device = config.get("device")
        assert isinstance(device, str)
        return device

    return device


def get_scipy_namespace(
    x: ArrayNamespace | Array[Any, Any] | BackendName | None = None,
):
    xp = get_array_namespace(x)

    if xp is np:
        return scipy

    elif xp is cp:
        assert cupyx is not None
        return cast(scipy, cupyx.scipy)

    else:
        raise ValueError(f"scipy is not available for backend {xp}")


def use_numpy_fallback(func: Callable[P, R]) -> Callable[P, R]:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        backend: str | None = None
        device: Device | None = None

        for name, value in bound.arguments.items():
            if type_hints.get(name) is Array:
                device = getattr(value, "device", None)
                if is_torch_array(value):
                    backend = "torch"
                    break
                elif is_cupy_array(value):
                    backend = "cupy"
                    break
                elif isinstance(value, np.ndarray):
                    backend = "numpy"
                    break

        for name, value in list(bound.arguments.items()):
            if type_hints.get(name) is Array and backend is not None:
                bound.arguments[name] = ensure_array_backend(value, "numpy")

        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if param_name in bound.arguments:
                    positional_args.append(bound.arguments[param_name])
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                var = bound.arguments.get(param_name, ())
                if isinstance(var, (list, tuple)):
                    positional_args.extend(var)
                else:
                    positional_args.append(var)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if param_name in bound.arguments:
                    keyword_args[param_name] = bound.arguments[param_name]
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                var = bound.arguments.get(param_name, {})
                if isinstance(var, dict):
                    keyword_args.update(var)

        result = cast(Callable[..., R], func)(*positional_args, **keyword_args)

        if type_hints.get("return") is Array and backend is not None:
            result = cast(Array, result)
            result = ensure_array_backend(result, backend, device=device)
        return cast(R, result)

    return cast(Callable[P, R], wrapper)
