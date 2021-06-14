from typing import Tuple

import dask
import dask.array as da
import numpy as np

from abtem.utils.backend import get_array_module


def spatial_frequencies(gpts: Tuple[int, int],
                        sampling: Tuple[float, float],
                        return_grid: bool = False,
                        return_radial: bool = False,
                        xp=np):
    """
    Calculate spatial frequencies of a grid.

    Parameters
    ----------
    gpts: tuple of int
        Number of grid points.
    sampling: tuple of float
        Sampling of the potential [1 / Å].

    Returns
    -------
    tuple of arrays
    """

    xp = get_array_module(xp)

    kis = ()
    for n, d in zip(gpts, sampling):
        kis += (da.from_delayed(dask.delayed(np.fft.fftfreq, pure=True)(n, d), shape=(n,),
                                dtype=np.float64).astype(np.float32).map_blocks(xp.asarray),)

    if return_grid:
        out = da.meshgrid(*kis, indexing='ij')
    else:
        out = kis

    if return_radial:
        def expand_dims(x, axis):
            slic = tuple(slice(None) if i == axis else None for i in range(len(kis)))
            return x[slic]

        k = da.sqrt(sum([expand_dims(ki, axis=i) ** 2 for i, ki in enumerate(kis)]))
        out += (k,)

    return out
