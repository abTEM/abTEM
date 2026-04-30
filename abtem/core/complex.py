import dask.array as da
import numba as nb  # type: ignore
import numpy as np

from abtem.core.backend import check_cupy_is_installed, cp


@nb.vectorize(
    [nb.complex64(nb.float32), nb.complex128(nb.float64), nb.complex64(nb.complex64)]
)
def _complex_exponential(x):
    """
    Calculate the complex exponential.
    """
    return np.cos(x) + 1.0j * np.sin(x)


@nb.vectorize([nb.float32(nb.complex64), nb.float64(nb.complex128)])
def _abs2(x):
    """
    Calculate the absolute square of a complex number.
    """
    return x.real**2 + x.imag**2


if cp is not None:
    _abs2_cupy_float32 = cp.ElementwiseKernel(
        in_params="float32 x, float32 y",
        out_params="float32 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
    _abs2_cupy_float64 = cp.ElementwiseKernel(
        in_params="float64 x, float64 y",
        out_params="float64 z",
        operation="z = x * x + y * y",
        name="abs_squared",
    )
    # Fused kernels: compute exp(i*x) = cos(x) + i*sin(x) directly from a
    # real-valued input, avoiding the intermediate complex allocation that
    # cp.exp(1j*x) would create.
    _complex_exp_cupy_float32 = cp.ElementwiseKernel(
        in_params="float32 x",
        out_params="complex64 z",
        operation="z = thrust::complex<float>(cosf(x), sinf(x))",
        name="complex_exp_float32",
    )
    _complex_exp_cupy_float64 = cp.ElementwiseKernel(
        in_params="float64 x",
        out_params="complex128 z",
        operation="z = thrust::complex<double>(cos(x), sin(x))",
        name="complex_exp_float64",
    )
    # Fused kernels: compute exp(i * scale * x) in a single GPU pass, avoiding
    # the intermediate real allocation that `complex_exponential(scale * x)`
    # would create (saves one slice-sized buffer per chunk).
    _complex_exp_scaled_cupy_float32 = cp.ElementwiseKernel(
        in_params="float32 x, float32 scale",
        out_params="complex64 z",
        operation="z = thrust::complex<float>(cosf(scale * x), sinf(scale * x))",
        name="complex_exp_scaled_float32",
    )
    _complex_exp_scaled_cupy_float64 = cp.ElementwiseKernel(
        in_params="float64 x, float64 scale",
        out_params="complex128 z",
        operation="z = thrust::complex<double>(cos(scale * x), sin(scale * x))",
        name="complex_exp_scaled_float64",
    )
else:
    _abs2_cupy_float32 = None
    _abs2_cupy_float64 = None
    _complex_exp_cupy_float32 = None
    _complex_exp_cupy_float64 = None
    _complex_exp_scaled_cupy_float32 = None
    _complex_exp_scaled_cupy_float64 = None


def abs2(x: np.ndarray | da.core.Array) -> np.ndarray | da.core.Array:
    """
    Fast calculation of the absolute square of a complex number.
    """
    if isinstance(x, np.ndarray):
        return _abs2(x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(abs2, x)

    check_cupy_is_installed()  # type: ignore

    if isinstance(x, cp.ndarray):
        if x.dtype == "complex64":
            return _abs2_cupy_float32(x.real, x.imag)
        elif x.dtype == "complex128":
            return _abs2_cupy_float64(x.real, x.imag)
        else:
            raise RuntimeError(
                "Unsupported dtype for calculation of abs2 with cupy array."
            )

    raise ValueError("abs2 only supports numpy arrays, dask arrays and cupy arrays.")


def complex_exponential(x: np.ndarray | da.core.Array) -> np.ndarray | da.core.Array:
    """
    Fast calculation of the complex exponential.
    """

    if isinstance(x, np.ndarray):
        return _complex_exponential(x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(complex_exponential, x)

    check_cupy_is_installed()  # type: ignore

    if isinstance(x, cp.ndarray):
        if x.dtype == np.float32:
            return _complex_exp_cupy_float32(x)
        elif x.dtype == np.float64:
            return _complex_exp_cupy_float64(x)
        else:
            # Fallback for complex or other dtypes (e.g. already-complex input)
            return cp.exp(1.0j * x)

    raise ValueError(
        "complex_exponential only supports numpy arrays, dask arrays and cupy arrays."
    )


def complex_exponential_scaled(
    x: np.ndarray | da.core.Array, scale: float
) -> np.ndarray | da.core.Array:
    """Compute ``exp(i * scale * x)`` without creating a temporary real array.

    On GPU, the scale multiplication is fused into the sin/cos kernel, saving
    one slice-sized real buffer allocation per chunk compared to the two-step
    ``complex_exponential(scale * x)``.

    On CPU, falls back to the equivalent two-step computation (NumPy/Numba).

    Parameters
    ----------
    x : array-like
        Real-valued input array.
    scale : float
        Scalar multiplier applied before the exponential.
    """
    if isinstance(x, np.ndarray):
        return _complex_exponential(x.dtype.type(scale) * x)

    if isinstance(x, da.core.Array):
        return da.map_blocks(complex_exponential_scaled, x, scale=scale)

    check_cupy_is_installed()  # type: ignore

    if isinstance(x, cp.ndarray):
        if x.dtype == np.float32:
            return _complex_exp_scaled_cupy_float32(x, cp.float32(scale))
        elif x.dtype == np.float64:
            return _complex_exp_scaled_cupy_float64(x, cp.float64(scale))
        else:
            return cp.exp(1.0j * scale * x)

    raise ValueError(
        "complex_exponential_scaled only supports numpy arrays, dask arrays and cupy arrays."
    )
