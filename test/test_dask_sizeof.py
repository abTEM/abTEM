"""Regression tests for abtem.core.dask_sizeof.

``dask.sizeof.sizeof`` memoizes its type -> function lookup the first time
a given type is weighed (``dask.utils.Dispatch.dispatch``), so a
registration made after that point is silently ignored for that type --
see abtem.core.dask_sizeof's own docstring. Some other test module in the
same pytest process could plausibly have already constructed an
``ArrayObject`` subclass, an ``ase.Atoms``, or a plain ``ndarray`` and
weighed it before this file's tests run, which would make an in-process
assertion pass or fail for reasons unrelated to the registration's own
correctness. Every check here therefore runs in a fresh subprocess that
imports abtem cold, so the only thing that can have touched
``dask.sizeof.sizeof`` before the measurement is abtem's own import-time
registration.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _run(script: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_transition_potential_array_sizeof_reflects_payload():
    """A TransitionPotentialArray's sizeof must scale with its array, not
    with sys.getsizeof's constant few dozen bytes (the issue's own
    reproduction: unfixed dev reports 48 regardless of payload size)."""
    script = """
import json
import numpy as np
from dask.sizeof import sizeof
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray

rng = np.random.default_rng(0)
arr = (rng.standard_normal((8, 512, 512))
       + 1j * rng.standard_normal((8, 512, 512))).astype(np.complex64)
tp = TransitionPotentialArray(
    Z=14, array=arr, energy=100e3, extent=(8., 8.),
    ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(8)))],
    metadata={"Z": 14, "n": 1, "l": 0},
)
print(json.dumps({"nbytes": int(tp.array.nbytes), "sizeof": int(sizeof(tp))}))
"""
    result = _run(script)
    # Unfixed: sizeof == 48 no matter what nbytes is. Fixed: sizeof tracks
    # nbytes plus a small constant object overhead.
    assert result["nbytes"] == 16_777_216
    assert result["sizeof"] >= result["nbytes"]
    assert result["sizeof"] < result["nbytes"] * 1.01


@pytest.mark.parametrize("ndims", [0, 1])
def test_wrapped_object_array_sizeof_reflects_payload(ndims):
    """_wrap_with_array's whole purpose is to carry a payload through a
    dask graph node; the wrapper's own sizeof must reflect what it wraps
    (unfixed dev reports 8 for both ndims, independent of payload size)."""
    script = f"""
import json
import numpy as np
from dask.sizeof import sizeof
from abtem.core.axes import OrdinalAxis
from abtem.core.ensemble import _wrap_with_array
from abtem.inelastic.core_loss import TransitionPotentialArray

rng = np.random.default_rng(0)
arr = (rng.standard_normal((8, 512, 512))
       + 1j * rng.standard_normal((8, 512, 512))).astype(np.complex64)
tp = TransitionPotentialArray(
    Z=14, array=arr, energy=100e3, extent=(8., 8.),
    ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(8)))],
    metadata={{"Z": 14, "n": 1, "l": 0}},
)
wrapped = _wrap_with_array(tp, ndims={ndims})
print(json.dumps({{"nbytes": int(tp.array.nbytes), "sizeof": int(sizeof(wrapped))}}))
"""
    result = _run(script)
    assert result["nbytes"] == 16_777_216
    assert result["sizeof"] >= result["nbytes"]
    assert result["sizeof"] < result["nbytes"] * 1.01


def test_generic_object_array_of_ndarrays_sizeof_reflects_contents():
    """Not abTEM-specific: any object-dtype ndarray of ndarrays should be
    weighed by its contents (unfixed dev reports 32 for this exact case,
    from the issue's reproduction, independent of the 4x1000-float64
    payload actually held)."""
    script = """
import json
import numpy as np
from dask.sizeof import sizeof
import abtem  # noqa: F401

big = np.empty((4,), dtype=object)
for i in range(4):
    big[i] = np.zeros(1000)
print(json.dumps({"sizeof": int(sizeof(big))}))
"""
    result = _run(script)
    payload_nbytes = 4 * 1000 * 8  # 4 float64[1000] arrays
    assert result["sizeof"] >= payload_nbytes
    assert result["sizeof"] < payload_nbytes * 1.01


def test_ase_atoms_sizeof_reflects_positions():
    """ase.Atoms is carried the same way (e.g. by DummyFrozenPhonons) and
    is just as unregistered upstream; unfixed dev reports a constant ~48
    bytes regardless of atom count."""
    script = """
import json
import numpy as np
from dask.sizeof import sizeof
from ase import Atoms
import abtem  # noqa: F401

atoms = Atoms("H" * 1000, positions=np.random.rand(1000, 3))
positions_nbytes = atoms.arrays["positions"].nbytes + atoms.arrays["numbers"].nbytes
print(json.dumps({"positions_nbytes": int(positions_nbytes), "sizeof": int(sizeof(atoms))}))
"""
    result = _run(script)
    assert result["sizeof"] >= result["positions_nbytes"]
    assert result["sizeof"] < result["positions_nbytes"] * 1.5


def test_ordinary_ndarray_sizeof_unchanged():
    """The fix must not touch dask's own accounting for non-object arrays,
    including its 0-in-strides broadcast-view special case (a broadcast
    view is weighed by its distinct data, not the shape it presents)."""
    script = """
import json
import numpy as np
from dask.sizeof import sizeof
import abtem  # noqa: F401

plain = np.arange(1000, dtype=np.float64)
broadcast = np.broadcast_to(np.arange(5, dtype=np.float64), (1000, 5))
print(json.dumps({
    "plain_sizeof": int(sizeof(plain)),
    "plain_nbytes": int(plain.nbytes),
    "broadcast_sizeof": int(sizeof(broadcast)),
    "broadcast_full_nbytes": int(broadcast.nbytes),
}))
"""
    result = _run(script)
    assert result["plain_sizeof"] == result["plain_nbytes"]
    # A 1000x5 broadcast of a 5-element array holds only 5 distinct
    # float64s; sizeof must report that, not the presented 1000x5 shape.
    assert result["broadcast_sizeof"] == 5 * 8
    assert result["broadcast_sizeof"] < result["broadcast_full_nbytes"]


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


def test_unrelated_object_array_does_not_overclaim():
    """An object array holding instances of a class abTEM knows nothing
    about must not be inflated beyond its elements' own (generic)
    sizeof -- the fix sums whatever sizeof() already reports for each
    element, it does not reach into arbitrary user classes."""
    script = """
import json
import sys
import numpy as np
from dask.sizeof import sizeof
import abtem  # noqa: F401

class Foo:
    def __init__(self, payload):
        self.payload = payload

arr = np.empty((3,), dtype=object)
for i in range(3):
    arr[i] = Foo(np.zeros(100))

element_sizeof_sum = sum(sizeof(x) for x in arr)
print(json.dumps({
    "array_sizeof": int(sizeof(arr)),
    "wrapper_nbytes": int(arr.nbytes),
    "element_sizeof_sum": int(element_sizeof_sum),
}))
"""
    result = _run(script)
    # Exactly wrapper nbytes plus the (unrecursed, generic) per-element
    # sizeof -- not inflated by the 100-float64 payload nested inside Foo,
    # which abTEM has no registration for.
    assert result["array_sizeof"] == result["wrapper_nbytes"] + result["element_sizeof_sum"]
    assert result["array_sizeof"] < 100 * 8  # far less than Foo's hidden payload


def test_module_reload_does_not_recurse():
    """importlib.reload of this module -- what IPython's `%autoreload 2`
    does, which abTEM notebook users routinely have on -- used to make
    _sizeof_ndarray delegate to itself. On reload, sizeof.dispatch(np.ndarray)
    returns the wrapper this module installed the first time; capturing
    that again as "the original non-object implementation" makes every
    non-object array recurse into itself until the stack overflows
    (unfixed: RecursionError the moment any ndarray is weighed after a
    reload, not just abTEM's own types)."""
    script = """
import json
import importlib
import numpy as np
from dask.sizeof import sizeof
import abtem  # noqa: F401
import abtem.core.dask_sizeof as m

before = int(sizeof(np.zeros(10)))
importlib.reload(m)
after_one_reload = int(sizeof(np.zeros(10)))
importlib.reload(m)
after_two_reloads = int(sizeof(np.zeros(10)))
print(json.dumps({
    "before": before,
    "after_one_reload": after_one_reload,
    "after_two_reloads": after_two_reloads,
}))
"""
    result = _run(script)
    assert result["before"] == 80
    assert result["after_one_reload"] == 80
    assert result["after_two_reloads"] == 80
