import itertools
import operator
from abc import abstractmethod, ABCMeta
from functools import reduce, partial
from typing import Tuple, List, Union

import dask.array as da
import numpy as np

from abtem.core import config
from abtem.core.axes import AxisMetadata
from abtem.core.chunks import (
    chunk_ranges,
    validate_chunks,
    chunk_shape,
    iterate_chunk_ranges,
    Chunks,
)


class Ensemble(metaclass=ABCMeta):
    ensemble_shape: Tuple[int, ...]
    ensemble_axes_metadata: List[AxisMetadata]

    @property
    @abstractmethod
    def _default_ensemble_chunks(self):
        pass

    @abstractmethod
    def _partition_args(self, chunks: Chunks = None, lazy: bool = True):
        pass

    def _partition_ensemble_axes_metadata(
        self, chunks: Chunks = None, lazy: bool = True
    ):
        chunks = self._validate_chunks(chunks)

        ensemble_axes = np.zeros(chunk_shape(chunks), dtype=object)
        for index, slic in iterate_chunk_ranges(chunks):
            ensemble_axes.itemset(
                index,
                [
                    self.ensemble_axes_metadata[i][slic[i]]
                    for i, axis in enumerate(self.ensemble_axes_metadata)
                ],
            )

        if lazy:
            ensemble_axes = da.from_array(ensemble_axes, chunks=1)

        return ensemble_axes

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

    def ensemble_blocks(self, chunks: Chunks = None) -> da.core.Array:
        """
        Split the ensemble into an array of smaller ensembles.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        chunks = self._validate_chunks(chunks)

        args = self._partition_args(chunks, lazy=True)

        assert isinstance(args, tuple)

        symbols = tuple(range(len(args)))
        args = tuple((block, (i,)) for i, block in zip(symbols, args))
        adjust_chunks = {i: c for i, c in enumerate(chunks)}

        return da.blockwise(
            self._wrapped_from_partitioned_args(),
            symbols,
            *tuple(itertools.chain(*args)),
            adjust_chunks=adjust_chunks,
            meta=np.array((), dtype=object)
        )

    def generate_blocks(self, chunks: Chunks = 1):
        """
        Generate chunks of the ensemble.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        chunks = self._validate_chunks(chunks)
        blocks = self._partition_args(chunks=chunks, lazy=False)

        for block_indices, start_stop in zip(
            itertools.product(*(range(block.shape[0]) for block in blocks)),
            itertools.product(*chunk_ranges(chunks)),
        ):
            block = tuple(block[i] for i, block in zip(block_indices, blocks))

            slics = tuple(slice(start, stop) for start, stop in start_stop)

            yield block_indices, slics, self._from_partitioned_args()(*block)

    def _validate_chunks(self, chunks: Chunks, limit: Union[str, int] = "auto"):
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

    @abstractmethod
    def _from_partitioned_args(self):
        pass

    def _wrapped_from_partitioned_args(self):
        def wrap_from_partitioned_args(*args, from_partitioned_args, **kwargs):
            blocks = tuple(arg.item() for arg in args)
            arr = np.empty((1,) * len(args), dtype=object)
            arr.itemset(0, from_partitioned_args(*blocks, **kwargs))
            return arr

        return partial(
            wrap_from_partitioned_args,
            from_partitioned_args=self._from_partitioned_args(),
        )


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
