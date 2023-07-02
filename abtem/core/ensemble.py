from __future__ import annotations
import itertools
import operator
import warnings
from abc import abstractmethod, ABCMeta
from functools import reduce, partial
from typing import Tuple, List, Union, TYPE_CHECKING

import dask.array as da
import numpy as np

from abtem.core import config
#from abtem.core.axes import AxisMetadata, AxesMetadataList
from abtem.core.chunks import (
    chunk_ranges,
    validate_chunks,
    Chunks,
)
from abtem.core.utils import tuple_range, interleave

if TYPE_CHECKING:
    from abtem.core.axes import AxisMetadata, AxesMetadataList


def _wrap_with_array(x, ndims:int=None):

    if ndims is None:
        ndims = len(x.ensemble_shape)

    wrapped = np.zeros((1,) * ndims, dtype=object)
    wrapped.itemset(0, x)
    return wrapped


def _wrap_args_with_array(*args, ndims):
    args = _wrap_with_array(args, ndims)
    print(args)
    return args


def unpack_blockwise_args(args):
    unpacked = ()
    for arg in args:
        if hasattr(arg, "item"):
            arg = arg.item()

        unpacked += (arg,)
    return unpacked


# def pack_unpack(*args, func, **kwargs):
#     args = unpack_blockwise_args(args)
#
#     new = func(*args, **kwargs)
#
#     if ndims:
#         return _wrap_with_array(new, len(new.ensemble_shape))
#
#     return new
#





class Ensemble(metaclass=ABCMeta):

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the ensemble axes."""
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the ensemble axes."""
        return []

    @property
    def axes_metadata(self) -> AxesMetadataList:
        """List of AxisMetadata."""
        return AxesMetadataList(
            self.ensemble_axes_metadata + self.base_axes_metadata, self.shape
        )

    @property
    def shape(self):
        """Shape of the ensemble."""
        return self.ensemble_shape + self.base_shape

    @property
    def base_shape(self) -> tuple[int, ...]:
        """Shape of the base axes."""
        return ()

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the base axes."""
        return []

    @abstractmethod
    def _default_ensemble_chunks(self):
        pass

    @abstractmethod
    def _partition_args(self, chunks: Chunks = None, lazy: bool = True):
        pass

    def select_block(self, index: Tuple[int, ...], chunks: Chunks):
        """
        Select a block from the ensemble.

        Parameters
        ----------
        index : tuple of ints
            Index of selected block.
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        args = self._partition_args(chunks, lazy=False)
        assert len(args) == len(index)
        selected_args = tuple(arg[index] for arg, index in zip(args, index))
        return self._from_partitioned_args()(*selected_args)

    @abstractmethod
    def _from_partitioned_args(self):
        pass

    # def _wrapped_from_partitioned_args(self):
    #     def wrap_from_partitioned_args(*args, from_partitioned_args, **kwargs):
    #
    #         blocks = tuple(arg.item() for arg in args)
    #
    #         n = sum(len(a.shape) for a in args)
    #
    #         arr = np.empty((1,) * n, dtype=object)
    #         arr.itemset(0, from_partitioned_args(*blocks, **kwargs))
    #         return arr
    #
    #     return partial(
    #         wrap_from_partitioned_args,
    #         from_partitioned_args=self._from_partitioned_args(),
    #     )

    def ensemble_blocks(self, chunks: Chunks = None) -> da.core.Array:
        """
        Split the ensemble into an array of smaller ensembles.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        chunks = self._validate_ensemble_chunks(chunks)

        args = self._partition_args(chunks, lazy=True)

        out_symbols = tuple_range(sum(len(arg.shape) for arg in args))

        #assert len(out_symbols) == max(len(self.ensemble_shape), 1)

        arg_symbols = ()
        offset = 0
        for arg in args:
            arg_symbols += (tuple_range(len(arg.shape), offset),)
            offset += len(arg.shape)

        adjust_chunks = {i: axes_chunks for i, axes_chunks in enumerate(chunks)}

        func = self._from_partitioned_args()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
            return da.blockwise(
                func,
                out_symbols,
                *interleave(args, arg_symbols),
                adjust_chunks=adjust_chunks,
                concatenate=True,
                meta=np.array((), dtype=object),
            )

    def generate_blocks(self, chunks: Chunks = 1):
        """
        Generate chunks of the ensemble.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """
        chunks = self._validate_ensemble_chunks(chunks)

        blocks = self._partition_args(chunks=chunks, lazy=False)

        for block_indices, start_stop in zip(
            itertools.product(*(range(block.shape[0]) for block in blocks)),
            itertools.product(*chunk_ranges(chunks)),
        ):
            block = tuple(block[i] for i, block in zip(block_indices, blocks))
            slics = tuple(slice(start, stop) for start, stop in start_stop)
            yield block_indices, slics, self._from_partitioned_args()(*block)

    def _validate_ensemble_chunks(self, chunks: Chunks, limit: Union[str, int] = "auto"):
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, limit=limit)
        return chunks

    def _ensemble_chunks(
        self,
        max_batch: Union[str, int] = None,
        base_shape: Tuple[int, ...] = (),
        dtype=np.dtype("complex64"),
    ):

        shape = self.ensemble_shape
        chunks = self._default_ensemble_chunks

        if max_batch == "auto":
            max_batch = config.get("dask.chunk-size")

        if base_shape is not None:
            shape += base_shape
            chunks += (-1,) * len(base_shape)

            if isinstance(max_batch, int):
                max_batch = max_batch * reduce(operator.mul, base_shape)

        chunks = validate_chunks(shape, chunks, max_batch, dtype)

        if base_shape is not None:
            chunks = chunks[: -len(base_shape)]

        return chunks


class EmptyEnsemble(Ensemble):
    @property
    def _default_ensemble_chunks(self):
        return ()

    @property
    def ensemble_axes_metadata(self):
        return []

    def _partition_args(self, chunks=None, lazy: bool = True):
        return ()

    def _from_partitioned_args(self):
        return self.__class__

    @property
    def ensemble_shape(self):
        return ()


def concatenate_array_blocks(blocks):
    for i in range(len(blocks.shape)):
        new_blocks = np.empty(blocks.shape[:-1], dtype=object)

        for indices in np.ndindex(blocks.shape):
            concat_index = len(indices) - 1
            indices = indices[:-1]
            new_blocks[indices] = np.concatenate(blocks[indices], axis=concat_index)

        blocks = new_blocks

    return blocks.item()
