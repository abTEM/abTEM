"""Registers ``dask.sizeof.sizeof`` for abTEM's large payload-carrying types.

abTEM ships large arrays to dask by wrapping them in a small object-dtype
``ndarray`` -- see ``_wrap_with_array``/``shared_constant_arg`` in
``abtem.core.ensemble``, a transport convention used at roughly fifty call
sites across the package (potentials, transition potentials, waves,
scans, detectors, frozen phonons, S-matrices, ...). ``dask.sizeof.sizeof``
drives distributed's spill, rebalance and transfer-cost decisions, but it
has no registration for any of abTEM's classes -- they fall back to
``sys.getsizeof`` (tens of bytes regardless of payload) -- and it does not
look inside object-dtype arrays either, so the *wrapper* is sized instead
of the payload it carries (a handful of bytes for the pointer, no matter
how large the wrapped object is). The net effect: distributed cannot see
these payloads at all, so a worker holding hundreds of MB of them can be
paused or killed on its RSS threshold while dask spills correctly-sized
(and much smaller) data instead.

This module fixes both halves of that gap:

1. Registers ``sizeof`` for the abTEM/ase classes that hold a large
   payload directly, so weighing one of them (wrapped or not) reports the
   payload's size instead of ``sys.getsizeof``'s constant few dozen bytes.
2. Wraps dask's own ``np.ndarray`` handler so that an **object-dtype**
   array reports the summed ``sizeof`` of its elements -- recursing back
   through ``sizeof``, so a registered abTEM/ase payload nested inside is
   picked up by (1) -- instead of ``nbytes``, which for an object array is
   just a pointer per slot. Every other dtype is delegated unchanged to
   dask's own handler, which is captured once, below, before it is
   overridden, so non-object arrays keep exactly the behaviour dask
   already gives them (including its 0-in-strides broadcast-view special
   case).

Both registrations must happen at import time (module level, not lazily
inside a function that might run later): ``dask.sizeof.sizeof`` is a
``dask.utils.Dispatch`` that memoizes its type -> function lookup the
first time a given type is weighed, so a registration made after that
point is silently ignored for that type. In practice this is only a risk
for ``np.ndarray`` itself, a foreign type distributed code can plausibly
weigh before ``import abtem`` runs; the abTEM/ase classes registered below
cannot be instantiated before ``import abtem`` has already executed this
module, so there is no window in which one of their instances could reach
``sizeof`` unregistered.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from ase import Atoms
from dask.sizeof import sizeof

# Force dask's own lazy numpy registration to run now, and capture the
# function it installs, so the override below can delegate to the exact
# upstream implementation for every non-object dtype rather than
# reimplementing its edge cases (e.g. the 0-in-strides broadcast view).
_dask_sizeof_numpy_ndarray = sizeof.dispatch(np.ndarray)


def _sizeof_object_ndarray(x: np.ndarray) -> int:
    """Sum ``sizeof`` of an object-dtype array's elements.

    Mirrors dask's own ``sizeof_python_collection`` (used for
    ``list``/``tuple``/``set``): for more than a handful of elements, a
    random sample is scaled up rather than visiting every element, since
    an object array can carry thousands of scan positions or similar.
    """
    flat = x.reshape(-1)
    n = flat.shape[0]
    if n == 0:
        return int(x.nbytes)

    num_samples = 10
    if n > num_samples:
        import random

        indices = random.sample(range(n), num_samples)
        elements_size = int(n / num_samples * sum(sizeof(flat[i]) for i in indices))
    else:
        elements_size = sum(sizeof(flat[i]) for i in range(n))

    return int(x.nbytes) + elements_size


def _sizeof_ndarray(x: np.ndarray) -> int:
    if x.dtype == object:
        return _sizeof_object_ndarray(x)
    return _dask_sizeof_numpy_ndarray(x)


sizeof.register(np.ndarray)(_sizeof_ndarray)


def _sizeof_array_object(x: Any) -> int:
    """``sizeof`` for abTEM's ``ArrayObject`` (``Waves``, ``PotentialArray``,
    ``TransitionPotentialArray``, ``Images``, ``SMatrixArray``, ...).

    abTEM code hands dask either the whole object or (via
    ``_wrap_with_array``) just its payload array; either way ``_array`` is
    where the bytes are, so this covers both.
    """
    total = sys.getsizeof(x)
    array = getattr(x, "_array", None)
    if array is not None:
        total += int(sizeof(array))
    return total


def _sizeof_compressed_s_matrix_array(x: Any) -> int:
    """``sizeof`` for ``CompressedSMatrixArray``.

    Not an ``ArrayObject`` -- it holds its truncated-SVD factors
    (``u``, ``sigma``, ``vh_dense``, ``dense_indices``) directly, of which
    ``u`` (shape ``(K, gpts_x, gpts_y)``) is normally the dominant one.
    """
    total = sys.getsizeof(x)
    for attr in ("_u", "_sigma", "_vh_dense", "_dense_indices"):
        array = getattr(x, attr, None)
        if array is not None:
            total += int(sizeof(array))
    return total


def _sizeof_ase_atoms(atoms: Atoms) -> int:
    """``sizeof`` for ``ase.Atoms``.

    Not an abTEM class, but carried the same way -- e.g. by
    ``DummyFrozenPhonons``, which wraps an ``Atoms`` object exactly like
    abTEM wraps its own array carriers.
    """
    total = sys.getsizeof(atoms)
    for array in atoms.arrays.values():
        total += int(sizeof(array))
    total += int(sizeof(np.asarray(atoms.cell)))
    return total


def _register() -> None:
    from abtem.array import ArrayObject
    from abtem.prism.s_matrix import CompressedSMatrixArray

    sizeof.register(ArrayObject)(_sizeof_array_object)
    sizeof.register(CompressedSMatrixArray)(_sizeof_compressed_s_matrix_array)
    sizeof.register(Atoms)(_sizeof_ase_atoms)


_register()
