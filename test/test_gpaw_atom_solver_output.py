"""GPAW's radial atom solvers, as abTEM runs them, never touch sys.stdout."""

import io
import sys
import threading

import numpy as np
import pytest

aeatom = pytest.importorskip("gpaw.atom.aeatom")
all_electron = pytest.importorskip("gpaw.atom.all_electron")

from abtem.inelastic.core_loss import (  # noqa: E402
    calculate_bound_radial_wavefunction,
    calculate_continuum_radial_wavefunction,
)
from abtem.potentials.gpaw import GPAWParametrization  # noqa: E402

R = np.linspace(0.01, 5.0, 64)


def _charge(symbol):
    return GPAWParametrization().charge(symbol)(R)


def _bound(Z, n, l):
    return np.asarray(calculate_bound_radial_wavefunction(Z, n, l)._radial_values)


def _continuum(Z, n, l, lprime):
    w = calculate_continuum_radial_wavefunction(Z, n, l, lprime, 10.0)
    return np.asarray(w._radial_values)


@pytest.mark.parametrize(
    "solve, cls",
    [
        (lambda: _charge("C"), aeatom.AllElectronAtom),
        (lambda: _bound(6, 1, 0), all_electron.AllElectron),
        (lambda: _continuum(6, 1, 0, 1), aeatom.AllElectronAtom),
    ],
    ids=["charge", "bound", "continuum"],
)
def test_atom_solve_leaves_sys_stdout_alone(monkeypatch, solve, cls):
    sentinel = io.StringIO()
    seen = []
    run = cls.run

    def spy(self, *args, **kwargs):
        # the solver writes to its own stream, never to whatever sys.stdout is
        stream = getattr(self, "fd", None) or getattr(self, "txt", None)
        seen.append(sys.stdout is sentinel and stream is not sentinel)
        return run(self, *args, **kwargs)

    monkeypatch.setattr(cls, "run", spy)
    monkeypatch.setattr(sys, "stdout", sentinel)
    solve()
    assert seen and all(seen)
    assert sys.stdout is sentinel
    assert sentinel.getvalue() == ""


def test_continuum_solve_does_not_silence_other_atom_solvers():
    calculate_continuum_radial_wavefunction(6, 1, 0, 1, 10.0)
    log = io.StringIO()
    aeatom.AllElectronAtom("C", log=log)
    assert log.getvalue()


def test_concurrent_atom_solves():
    jobs = [
        (_charge, ("C",)),
        (_charge, ("Si",)),
        (_bound, (14, 2, 1)),
        (_bound, (6, 1, 0)),
        (_continuum, (6, 1, 0, 1)),
    ]
    serial = [f(*a) for f, a in jobs]
    stdout = sys.stdout
    for _ in range(5):
        results = [None] * len(jobs)

        def work(i):
            f, a = jobs[i]
            results[i] = f(*a)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        restored = sys.stdout is stdout
        sys.stdout = stdout
        assert restored
        for got, expected in zip(results, serial):
            np.testing.assert_array_equal(got, expected)
