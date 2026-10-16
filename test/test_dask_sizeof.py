"""Regression tests for abtem.core.dask_sizeof.

``dask.sizeof.sizeof`` memoizes its type -> function lookup the first time
a given type is weighed (``dask.utils.Dispatch.dispatch``), so a
registration made after that point is silently ignored for that type --
see abtem.core.dask_sizeof's own docstring. Some other test module in the
same pytest process could plausibly have already constructed an
``ArrayObject`` subclass, an ``ase.Atoms``, or a plain ``ndarray`` and
weighed it before this file's tests run, which would make an in-process
assertion pass or fail for reasons unrelated to the registration's own
correctness. The checks here therefore run in fresh subprocesses that
import abtem cold, so the only thing that can have touched
``dask.sizeof.sizeof`` before the measurement is abtem's own import-time
registration.

A subprocess that imports abtem costs seconds, so every check that only
needs "abtem's registration came first" reads its numbers from one shared
subprocess (``_COLD_SCRIPT``, run once per module), with the reload check
last because reloading re-registers. Only the case where dask weighs an
ndarray *before* abtem is imported needs a process of its own.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# Keeps the module on one xdist worker, so the shared subprocess below runs once.
pytestmark = pytest.mark.xdist_group("dask_sizeof")


def _run(script: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_COLD_SCRIPT = """
import importlib
import json

import numpy as np
from ase import Atoms
from dask.sizeof import sizeof

import abtem  # noqa: F401
import abtem.core.dask_sizeof as m
from abtem.core.axes import OrdinalAxis
from abtem.core.ensemble import _wrap_with_array
from abtem.inelastic.core_loss import TransitionPotentialArray

out = {}

rng = np.random.default_rng(0)
arr = (rng.standard_normal((4, 64, 64))
       + 1j * rng.standard_normal((4, 64, 64))).astype(np.complex64)
tp = TransitionPotentialArray(
    Z=14, array=arr, energy=100e3, extent=(8., 8.),
    ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(4)))],
    metadata={"Z": 14, "n": 1, "l": 0},
)
out["tp_nbytes"] = int(tp.array.nbytes)
out["tp_sizeof"] = int(sizeof(tp))
for ndims in (0, 1):
    out[f"wrapped_sizeof_{ndims}"] = int(sizeof(_wrap_with_array(tp, ndims=ndims)))

big = np.empty((4,), dtype=object)
for i in range(4):
    big[i] = np.zeros(1000)
out["object_array_sizeof"] = int(sizeof(big))

atoms = Atoms("H" * 1000, positions=np.random.rand(1000, 3))
out["atoms_positions_nbytes"] = int(
    atoms.arrays["positions"].nbytes + atoms.arrays["numbers"].nbytes
)
out["atoms_sizeof"] = int(sizeof(atoms))

plain = np.arange(1000, dtype=np.float64)
broadcast = np.broadcast_to(np.arange(5, dtype=np.float64), (1000, 5))
out["plain_sizeof"] = int(sizeof(plain))
out["plain_nbytes"] = int(plain.nbytes)
out["broadcast_sizeof"] = int(sizeof(broadcast))
out["broadcast_full_nbytes"] = int(broadcast.nbytes)

class Foo:
    def __init__(self, payload):
        self.payload = payload

foos = np.empty((3,), dtype=object)
for i in range(3):
    foos[i] = Foo(np.zeros(100))
out["foo_array_sizeof"] = int(sizeof(foos))
out["foo_wrapper_nbytes"] = int(foos.nbytes)
out["foo_element_sizeof_sum"] = int(sum(sizeof(x) for x in foos))

# Last: reloading re-registers abtem's implementations.
out["before_reload"] = int(sizeof(np.zeros(10)))
importlib.reload(m)
out["after_one_reload"] = int(sizeof(np.zeros(10)))
importlib.reload(m)
out["after_two_reloads"] = int(sizeof(np.zeros(10)))

print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def cold():
    """Measurements from one subprocess that imported abtem cold."""
    return _run(_COLD_SCRIPT)


def test_transition_potential_array_sizeof_reflects_payload(cold):
    """A TransitionPotentialArray's sizeof must scale with its array, not
    with sys.getsizeof's constant few dozen bytes (the issue's own
    reproduction: unfixed dev reports 48 regardless of payload size)."""
    # Unfixed: sizeof == 48 no matter what nbytes is. Fixed: sizeof tracks
    # nbytes plus a small constant object overhead.
    assert cold["tp_nbytes"] == 4 * 64 * 64 * 8
    assert cold["tp_sizeof"] >= cold["tp_nbytes"]
    assert cold["tp_sizeof"] < cold["tp_nbytes"] * 1.05


@pytest.mark.parametrize("ndims", [0, 1])
def test_wrapped_object_array_sizeof_reflects_payload(cold, ndims):
    """_wrap_with_array's whole purpose is to carry a payload through a
    dask graph node; the wrapper's own sizeof must reflect what it wraps
    (unfixed dev reports 8 for both ndims, independent of payload size)."""
    wrapped_sizeof = cold[f"wrapped_sizeof_{ndims}"]
    assert wrapped_sizeof >= cold["tp_nbytes"]
    assert wrapped_sizeof < cold["tp_nbytes"] * 1.05


def test_generic_object_array_of_ndarrays_sizeof_reflects_contents(cold):
    """Not abTEM-specific: any object-dtype ndarray of ndarrays should be
    weighed by its contents (unfixed dev reports 32 for this exact case,
    from the issue's reproduction, independent of the 4x1000-float64
    payload actually held)."""
    payload_nbytes = 4 * 1000 * 8  # 4 float64[1000] arrays
    assert cold["object_array_sizeof"] >= payload_nbytes
    assert cold["object_array_sizeof"] < payload_nbytes * 1.01


def test_ase_atoms_sizeof_reflects_positions(cold):
    """ase.Atoms is carried the same way (e.g. by DummyFrozenPhonons) and
    is just as unregistered upstream; unfixed dev reports a constant ~48
    bytes regardless of atom count."""
    assert cold["atoms_sizeof"] >= cold["atoms_positions_nbytes"]
    assert cold["atoms_sizeof"] < cold["atoms_positions_nbytes"] * 1.5


def test_ordinary_ndarray_sizeof_unchanged(cold):
    """The fix must not touch dask's own accounting for non-object arrays,
    including its 0-in-strides broadcast-view special case (a broadcast
    view is weighed by its distinct data, not the shape it presents)."""
    assert cold["plain_sizeof"] == cold["plain_nbytes"]
    # A 1000x5 broadcast of a 5-element array holds only 5 distinct
    # float64s; sizeof must report that, not the presented 1000x5 shape.
    assert cold["broadcast_sizeof"] == 5 * 8
    assert cold["broadcast_sizeof"] < cold["broadcast_full_nbytes"]


def test_ordinary_ndarray_sizeof_unchanged_even_if_dask_registers_first():
    """dask's own numpy sizeof registration is itself lazy (registered on
    first use), so abtem's override must win even in a process where some
    unrelated code already weighed a plain ndarray before `import abtem`
    ran."""
    script = """
import json
import numpy as np
from dask.sizeof import sizeof

a = np.arange(1000, dtype=np.float64)
before = int(sizeof(a))  # forces dask's own lazy numpy registration first

import abtem  # noqa: F401

after = int(sizeof(a))
print(json.dumps({"before": before, "after": after, "nbytes": int(a.nbytes)}))
"""
    result = _run(script)
    assert result["before"] == result["nbytes"]
    assert result["after"] == result["nbytes"]


def test_unrelated_object_array_does_not_overclaim(cold):
    """An object array holding instances of a class abTEM knows nothing
    about must not be inflated beyond its elements' own (generic)
    sizeof -- the fix sums whatever sizeof() already reports for each
    element, it does not reach into arbitrary user classes."""
    # Exactly wrapper nbytes plus the (unrecursed, generic) per-element
    # sizeof -- not inflated by the 100-float64 payload nested inside Foo,
    # which abTEM has no registration for.
    assert cold["foo_array_sizeof"] == (
        cold["foo_wrapper_nbytes"] + cold["foo_element_sizeof_sum"]
    )
    assert cold["foo_array_sizeof"] < 100 * 8  # far less than Foo's hidden payload


def test_module_reload_does_not_recurse(cold):
    """importlib.reload of this module -- what IPython's `%autoreload 2`
    does, which abTEM notebook users routinely have on -- used to make
    _sizeof_ndarray delegate to itself. On reload, sizeof.dispatch(np.ndarray)
    returns the wrapper this module installed the first time; capturing
    that again as "the original non-object implementation" makes every
    non-object array recurse into itself until the stack overflows
    (unfixed: RecursionError the moment any ndarray is weighed after a
    reload, not just abTEM's own types)."""
    assert cold["before_reload"] == 80
    assert cold["after_one_reload"] == 80
    assert cold["after_two_reloads"] == 80
