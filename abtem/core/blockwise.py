"""Map array kernels over lazy or eager ensembles.

A *kernel* is an ordinary array function (NumPy, CuPy or the Metal backend) with
the contract of a NumPy generalized ufunc:

- it works on the **trailing** ``base_ndim`` axes of its input (one member's image,
  diffraction pattern or spectrum) and may change their number and size;
- it treats any **leading** axes as a stack of members, keeps their shape and never
  mixes members.

:func:`map_kernel` calls such a kernel on an array object's array, lazy or eager. For
an eager array it is a plain call; for a lazy array the kernel runs block by block,
with every base array in one block. Dask's own bookkeeping (dropped and new axes,
output chunks, ``meta``) is derived from the shape the kernel returns, so the call
site only states that shape and the output dtype.

:func:`map_overlapping_kernel` is the counterpart for filters, which can run on
chunks of the axes they act on given enough overlap between neighbouring chunks.

:func:`array_namespace` picks ``dask.array`` or the array module for whole-array
functions (``tile``, ``moveaxis``, ``concatenate``, ...).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Optional

import dask.array as da
import numpy as np

from abtem.core.backend import get_array_module


def array_namespace(array) -> ModuleType:
    """The module whose functions take `array`: `dask.array` for a dask array,
    otherwise its array module (NumPy, CuPy or the Metal backend).

    Use it for whole-array functions such as ``tile``, ``moveaxis``,
    ``concatenate`` or ``fft.fftshift`` on an array that may be lazy. NumPy's
    functions dispatch a dask array to dask themselves, but CuPy's raise a
    ``TypeError``, so ``get_array_module(array).tile(array, ...)`` breaks for a
    lazy array on the GPU.
    """
    if isinstance(array, da.Array):
        return da
    return get_array_module(array)


def _array_and_base_ndim(source: Any, base_ndim: Optional[int]) -> tuple[Any, int]:
    if hasattr(source, "base_shape"):
        array = source.array
        if base_ndim is None:
            base_ndim = len(source.base_shape)
    else:
        array = source
        if base_ndim is None:
            raise ValueError("base_ndim must be given when mapping over a bare array")
    if not 0 <= base_ndim <= array.ndim:
        raise ValueError(
            f"base_ndim must be between 0 and {array.ndim}, got {base_ndim}"
        )
    return array, base_ndim


def merge_base_chunks(array: da.Array, base_ndim: int) -> da.Array:
    """Rechunk a dask array so that its trailing `base_ndim` axes are in one chunk.

    An array whose base axes are already in one chunk is returned as it is.
    Otherwise dask chooses the chunks of the leading (ensemble) axes again, so that
    a block holds the whole base array of as many members as fit dask's
    ``array.chunk-size``.
    """
    n_ensemble = array.ndim - base_ndim
    if all(len(c) == 1 for c in array.chunks[n_ensemble:]):
        return array
    return array.rechunk(("auto",) * n_ensemble + (-1,) * base_ndim)


def _member_chunks(arg: np.ndarray, ensemble_chunks: tuple) -> tuple:
    # A member argument lines up with the leading axes of the array: each of its
    # leading axes is as long as the matching ensemble axis, chunked like it, or 1,
    # broadcast. Its own trailing axes stay in one chunk.
    n_ensemble = len(ensemble_chunks)
    leading = tuple(
        c if n == sum(c) else (n,)
        for n, c in zip(arg.shape[:n_ensemble], ensemble_chunks)
    )
    return leading + tuple((n,) for n in arg.shape[n_ensemble:])


def map_kernel(
    source: Any,
    kernel: Callable,
    *member_args: np.ndarray,
    base_shape_out: Optional[tuple[int, ...]] = None,
    dtype: Optional[np.dtype | type] = None,
    base_ndim: Optional[int] = None,
    device: Optional[str] = None,
    **kwargs: Any,
):
    """Apply a kernel to every member of an array object or array, lazy or eager.

    Parameters
    ----------
    source : ArrayObject or array
        The array object whose array is mapped, or a bare NumPy, CuPy or dask array.
    kernel : callable
        Function called as ``kernel(array, *member_args, **kwargs)``. It acts on the
        trailing `base_ndim` axes of `array` and keeps the leading axes; see the
        module docstring for the full contract.
    member_args : array
        Arrays with one entry per member, e.g. one orientation matrix per pattern.
        Their leading axes line up with the ensemble axes of `source` (each has
        that length or 1); their remaining axes are passed whole. For a lazy
        source they are chunked like its ensemble axes.
    base_shape_out : tuple of int, optional
        The shape of the trailing axes returned by the kernel, which may differ in
        number from the base axes of `source`. Default is the base shape of
        `source`.
    dtype : dtype, optional
        The dtype returned by the kernel. Default is the dtype of `source`.
    base_ndim : int, optional
        The number of base axes; required for a bare array, otherwise the number of
        base axes of the array object.
    device : str, optional
        The device of the array the kernel returns, when it differs from that of
        `source` (e.g. a kernel that moves host input to the GPU). Default is the
        device of `source`.
    kwargs :
        Keyword arguments passed to the kernel. They are baked into every task of a
        lazy array.

    Returns
    -------
    array : array
        The new array, lazy if the array of `source` is lazy.
    """
    array, base_ndim = _array_and_base_ndim(source, base_ndim)

    if not isinstance(array, da.Array):
        return kernel(array, *member_args, **kwargs)

    n_ensemble = array.ndim - base_ndim
    array = merge_base_chunks(array, base_ndim)
    ensemble_chunks = array.chunks[:n_ensemble]

    if base_shape_out is None:
        base_shape_out = array.shape[n_ensemble:]
    base_shape_out = tuple(int(n) for n in base_shape_out)

    if dtype is None:
        dtype = array.dtype
    meta = get_array_module(array if device is None else device).array((), dtype=dtype)

    # A gufunc signature such as "(b0,b1),(m0_0,m0_1)->(o0)": every core dimension
    # its own name, so dask neither aligns nor checks any of them against another.
    def core(prefix: str, ndim: int) -> str:
        return "(" + ",".join(f"{prefix}{i}" for i in range(ndim)) + ")"

    lazy_member_args = []
    in_signatures = [core("b", base_ndim)]
    for i, arg in enumerate(member_args):
        lazy_member_args.append(
            da.from_array(arg, chunks=_member_chunks(arg, ensemble_chunks))
        )
        in_signatures.append(core(f"m{i}_", np.ndim(arg) - n_ensemble))

    signature = ",".join(in_signatures) + "->" + core("o", len(base_shape_out))
    output_sizes = {f"o{i}": n for i, n in enumerate(base_shape_out)}

    return da.apply_gufunc(
        kernel,
        signature,
        array,
        *lazy_member_args,
        output_sizes=output_sizes,
        meta=meta,
        **kwargs,
    )


def _dask_boundary(boundary: str, cval: float):
    # dask's map_overlap takes 'periodic', 'reflect', 'nearest' or 'none', or a
    # number as the constant fill value, but not the string 'constant'.
    if boundary == "constant":
        return cval
    if boundary in ("periodic", "reflect"):
        return boundary
    raise ValueError(
        f"boundary must be 'periodic', 'reflect' or 'constant', got {boundary!r}"
    )


def map_overlapping_kernel(
    array,
    kernel: Callable,
    depth: dict[int, int],
    boundary: str = "periodic",
    cval: float = 0.0,
    dtype: Optional[np.dtype | type] = None,
    **kwargs: Any,
):
    """Apply a filter kernel to an array, lazy or eager, chunked along the filtered
    axes.

    The kernel is called as ``kernel(array, **kwargs)`` and must return an array of
    the same shape. It handles the array edges itself (e.g. with a ``mode`` keyword);
    for a lazy array each chunk is first extended by `depth` pixels from its
    neighbours, or across the array edges as given by `boundary`, so that the
    kernel sees what it would see in the whole array.

    Parameters
    ----------
    array : array
        NumPy, CuPy or dask array.
    kernel : callable
        The filter, returning an array of the same shape.
    depth : dict of int to int
        The overlap, in pixels, for every axis the kernel filters along: at least
        the radius of its filter. Axes not given have no overlap.
    boundary : {'periodic', 'reflect', 'constant'}
        How the overlap is filled beyond the edges of the array, matching the
        kernel's own edge handling.
    cval : float
        The fill value for ``boundary='constant'``.
    dtype : dtype, optional
        The dtype returned by the kernel. Default is the dtype of `array`.
    kwargs :
        Keyword arguments passed to the kernel.
    """
    if not isinstance(array, da.Array):
        return kernel(array, **kwargs)

    if dtype is None:
        dtype = array.dtype

    # Clip to the axis length: dask refuses an overlap larger than a chunk.
    depth = {axis: min(d, array.shape[axis]) for axis, d in depth.items()}

    return da.map_overlap(
        kernel,
        array,
        depth=depth,
        boundary=_dask_boundary(boundary, cval),
        # `meta` follows the given dtype, never a default: a complex array declared
        # as real loses its imaginary part when dask assembles the output.
        meta=get_array_module(array).array((), dtype=dtype),
        **kwargs,
    )
