from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from itertools import accumulate
from typing import Any, Callable, Generator, Optional, Union

import dask.array as da
import numpy as np

from abtem.core.axes import AxesMetadataList, AxisMetadata
from abtem.core.chunks import Chunks, ValidatedChunks, chunk_ranges, validate_chunks
from abtem.core.utils import interleave, itemset


def _wrap_with_array(x: Any, ndims: int | None = None) -> np.ndarray:
    if ndims is None:
        ndims = len(x.ensemble_shape)

    wrapped = np.zeros((1,) * ndims, dtype=object)
    itemset(wrapped, 0, x)
    return wrapped


def unpack_blockwise_args(args) -> tuple:
    unpacked = tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)
    return unpacked


class Ensemble:
    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the ensemble axes."""
        return ()

    @property
    def base_shape(self) -> tuple[int, ...]:
        """Shape of the base axes."""
        return ()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the ensemble."""
        return self.ensemble_shape + self.base_shape

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        """List of AxisMetadata of the base axes."""
        return []

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
    @abstractmethod
    def _default_ensemble_chunks(self) -> Chunks:
        pass

    def _validate_ensemble_chunks(
        self, chunks: Optional[Chunks] = None, limit: Union[str, int] = "auto"
    ) -> ValidatedChunks:
        if chunks is None:
            chunks = self._default_ensemble_chunks

        chunks = validate_chunks(self.ensemble_shape, chunks, max_elements=limit)
        return chunks

    @abstractmethod
    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        pass

    @abstractmethod
    def _from_partitioned_args(self) -> Callable[..., np.ndarray]:
        pass

    def ensemble_blocks(self, chunks: Optional[Chunks] = None) -> da.core.Array:
        """
        Split the ensemble into an array of smaller ensembles.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """

        chunks = self._validate_ensemble_chunks(chunks)

        args = self._partition_args(chunks, lazy=True)
        arg_dims = tuple(len(arg.shape) for arg in args)
        arg_starts = accumulate((0,) + arg_dims[:-1])
        arg_ends = accumulate(arg_dims)
        arg_ind = tuple(
            tuple(range(start, end)) for start, end in zip(arg_starts, arg_ends)
        )

        out_ind = tuple(range(sum(arg_dims)))
        adjust_chunks = {i: axes_chunks for i, axes_chunks in enumerate(chunks)}

        func = self._from_partitioned_args()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Increasing number of chunks")
            blocks = da.blockwise(
                func,
                out_ind,
                *interleave(args, arg_ind),
                adjust_chunks=adjust_chunks,
                concatenate=True,
                meta=np.array((), dtype=object),
            )
            return blocks

    def generate_blocks(
        self, chunks: Chunks = 1
    ) -> Generator[tuple[tuple[int, ...], tuple[slice, ...], np.ndarray], None, None]:
        """
        Generate chunks of the ensemble.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """
        chunks = self._validate_ensemble_chunks(chunks)
        blocks = self._partition_args(chunks=chunks, lazy=False)

        shape = sum((block.shape for block in blocks), ())

        start_stops = chunk_ranges(chunks)

        #print(start_stops)
        #print(shape)
        assert tuple(len(cr) for cr in start_stops) == shape

        for indices, start_stop in zip(
            np.ndindex(shape), itertools.product(*start_stops)
        ):
            block_indices: tuple[tuple[int, ...], ...] = ()
            j = 0
            for block in blocks:
                n = len(block.shape)
                block_indices += (tuple(indices[index] for index in range(j, j + n)),)
                j += n

            args = tuple(block[i] for i, block in zip(block_indices, blocks))
            slics = tuple(slice(start, stop) for start, stop in start_stop)

            yield indices, slics, self._from_partitioned_args()(*args)

        # print(blocks)

        # for block in blocks:
        # if len(block.shape) > 1:
        #    print(block)
        #         raise NotImplementedError
        # axis_indices = tuple(
        #     tuple(range(block.shape[0])) if len(block.shape) else () for block in blocks
        # )

        # if not any(len(indices) for indices in axis_indices):
        #     yield (), (), self._from_partitioned_args()(*blocks)

        # print(len(tuple(itertools.product(*chunk_ranges(chunks)))))
        # for block_indices, start_stop in zip(
        #     itertools.product(*axis_indices),
        #     itertools.product(*chunk_ranges(chunks)),
        # ):

        #     block = tuple(block[i] for i, block in zip(block_indices, blocks))
        #     slics = tuple(slice(start, stop) for start, stop in start_stop)
        #     print(slics)

        #     yield block_indices, slics, self._from_partitioned_args()(*block)


class EmptyEnsemble(Ensemble):
    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        return ()

    def _from_partitioned_args(self) -> type:
        return self.__class__

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()


def concatenate_array_blocks(blocks: np.ndarray) -> np.ndarray:
    for i in range(len(blocks.shape)):
        new_blocks = np.empty(blocks.shape[:-1], dtype=object)

        for indices in np.ndindex(blocks.shape):
            concat_index = len(indices) - 1
            indices = indices[:-1]
            new_blocks[indices] = np.concatenate(blocks[indices], axis=concat_index)

        blocks = new_blocks

    return blocks.item()
