import itertools
import operator
from abc import abstractmethod, ABCMeta
from functools import reduce

import dask.array as da
import numpy as np

from abtem.core import config
from abtem.core.dask import validate_chunks


class Ensemble(metaclass=ABCMeta):

    @property
    def ensemble_dims(self):
        return len(self.ensemble_shape)

    @property
    @abstractmethod
    def ensemble_axes_metadata(self):
        pass

    @property
    @abstractmethod
    def ensemble_shape(self):
        pass

    @property
    @abstractmethod
    def default_ensemble_chunks(self):
        pass

    @abstractmethod
    def ensemble_blocks(self, chunks):
        pass

    @abstractmethod
    def ensemble_partial(self):
        pass

    def _ensemble_blockwise(self, max_batch):
        chunks = validate_chunks(self.ensemble_shape, self.default_ensemble_chunks, limit=max_batch)
        partial = self.ensemble_partial()
        blocks = self.ensemble_blocks(chunks)
        return ensemble_blockwise(partial, blocks)


def concatenate_blocks(blocks):
    for i in range(len(blocks.shape)):
        new_blocks = np.empty(blocks.shape[:-1], dtype=object)

        for indices in np.ndindex(blocks.shape):
            concat_index = len(indices) - 1
            indices = indices[:-1]
            new_blocks[indices] = np.concatenate(blocks[indices], axis=concat_index)

        blocks = new_blocks

    return blocks.item()


def ensemble_blockwise(partial, blocks, ensemble_chunks=None, base_shape=None, dtype=None):
    block_indices = tuple(range(len(blocks)))
    args = tuple((block, (i,)) for i, block in zip(block_indices, blocks))

    if ensemble_chunks is not None:
        adjust_chunks = {i: c for i, c in enumerate(ensemble_chunks)}
    else:
        adjust_chunks = None

    if base_shape is not None:
        new_axes = {i + len(block_indices): n for i, n in enumerate(base_shape)}
        block_indices += tuple(new_axes.keys())
    else:
        new_axes = None

    if dtype is None:
        dtype = object

    return da.blockwise(partial,
                        block_indices,
                        *tuple(itertools.chain(*args)),
                        adjust_chunks=adjust_chunks,
                        new_axes=new_axes,
                        meta=np.array((), dtype=dtype))


def ensemble_chunks(ensembles, max_batch=None, base_shape=None, dtype=None):
    shape = tuple(itertools.chain(*tuple(ensemble.ensemble_shape for ensemble in ensembles)))
    chunks = tuple(itertools.chain(*tuple(ensemble.default_ensemble_chunks for ensemble in ensembles)))

    if max_batch == 'auto':
        max_batch = config.get("dask.chunk-size")

    if base_shape is not None:
        shape += base_shape
        chunks += (-1,) * len(base_shape)

        if isinstance(max_batch, int):
            max_batch = max_batch * reduce(operator.mul, base_shape)

    chunks = validate_chunks(shape, chunks, max_batch, dtype)

    if base_shape is not None:
        chunks = chunks[:-len(base_shape)]

    return chunks


def ensemble_blocks(ensembles, chunks):
    blocks = ()
    start = 0
    stop = ensembles[0].ensemble_dims
    for i, ensemble in enumerate(ensembles):
        blocks += (ensemble.ensemble_blocks(chunks=chunks[start:stop]),)
        start = stop
        try:
            stop = stop + ensembles[i + 1].ensemble_dims
        except IndexError:
            stop = len(chunks)

    return tuple(itertools.chain(*blocks))
