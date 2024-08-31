from __future__ import annotations

import itertools
import warnings
from abc import abstractmethod
from typing import Callable, Generator, Optional, Union, Any

import dask.array as da
import numpy as np

from abtem.core.axes import AxesMetadataList, AxisMetadata
from abtem.core.chunks import Chunks, ValidatedChunks, chunk_ranges, validate_chunks
from abtem.core.utils import interleave, itemset, tuple_range


def _wrap_with_array(x: Any, ndims: int | None = None):
    if ndims is None:
        ndims = len(x.ensemble_shape)

    wrapped = np.zeros((1,) * ndims, dtype=object)
    itemset(wrapped, 0, x)
    return wrapped


def unpack_blockwise_args(args) -> tuple:
    return tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)


class Ensemble:
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
    def shape(self) -> tuple[int, ...]:
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
    def _partition_args(self, chunks: Chunks | None = None, lazy: bool = True) -> tuple:
        pass

    @abstractmethod
    def _from_partitioned_args(self) -> Callable[..., Ensemble]:
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

        out_symbols = tuple_range(sum(len(arg.shape) for arg in args))
        
        arg_symbols = tuple(
            tuple_range(len(arg.shape), sum(len(a.shape) for a in args[:i]))
            for i, arg in enumerate(args)
        )

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

    def generate_blocks(self, chunks: Chunks = 1) -> Generator[tuple[tuple[int, ...], tuple[slice, ...], Ensemble], None, None]:
        """
        Generate chunks of the ensemble.

        Parameters
        ----------
        chunks : iterable of tuples
            Block sizes along each dimension.
        """
        chunks = self._validate_ensemble_chunks(chunks)
        blocks = self._partition_args(chunks=chunks, lazy=False)

        for block in blocks:
            if len(block.shape) > 1:
                raise NotImplementedError

        for block_indices, start_stop in zip(
            itertools.product(*(range(block.shape[0]) for block in blocks)),
            itertools.product(*chunk_ranges(chunks)),
        ):
            block = tuple(block[i] for i, block in zip(block_indices, blocks))
            slics = tuple(slice(start, stop) for start, stop in start_stop)

            yield block_indices, slics, self._from_partitioned_args()(*block)


class EmptyEnsemble(Ensemble):
    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True) -> tuple:
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
