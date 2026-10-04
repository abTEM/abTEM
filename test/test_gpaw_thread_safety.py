"""abTEM never runs two GPAW evaluations at once in one process.

GPAW shares FFT scratch arrays between all calculators on the same grid, so two
threads inside GPAW corrupt each other's results. These tests replace GPAW with a
fake whose methods wait at a two-party barrier: if two threads are ever inside
GPAW together, the barrier opens and the overlap is recorded. They need no GPAW
installation.
"""

import threading
import time
from types import SimpleNamespace

import dask
import numpy as np
import pytest
from ase import Atoms

import abtem.potentials.gpaw as gpaw_module
from abtem.magnetism.gpaw import get_vector_potential_from_gpaw
from abtem.potentials.gpaw import _DummyGPAW

# Long enough that a second dask thread reaches the barrier on a slow runner;
# with the lock in place the first thread waits this long once, then the
# barrier is broken and every later wait returns at once.
_BARRIER_TIMEOUT = 2.0


class _OverlapProbe:
    def __init__(self):
        self._barrier = threading.Barrier(2, timeout=_BARRIER_TIMEOUT)
        self.overlapped = False

    def __call__(self):
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            return
        self.overlapped = True


@pytest.fixture
def probe(monkeypatch):
    probe = _OverlapProbe()

    class FakeGPAW:
        def __init__(self, restart=None, txt=None, mode="pw", xc="PBE"):
            probe()
            self.atoms = Atoms("C", cell=(2.0, 2.0, 2.0), pbc=True)
            self.parameters = SimpleNamespace(mode=mode, xc=xc)
            self.density = SimpleNamespace(
                Q_aL={0: np.zeros(1)},
                nt_sG=np.zeros((1, 2, 2, 2)),
                gd=SimpleNamespace(new_descriptor=lambda comm: None),
                D_asp={0: np.zeros((1, 1))},
            )
            self.setups = "setups"

        def get_electrostatic_potential(self):
            probe()
            return np.zeros((2, 2, 2))

        def initialize(self, atoms):
            probe()

    monkeypatch.setattr(gpaw_module, "GPAW", FakeGPAW)
    monkeypatch.setattr(gpaw_module, "SerialCommunicator", lambda: None, raising=False)
    return probe


def _dummy():
    return _DummyGPAW(
        setup_mode="pw",
        setup_xc="PBE",
        nt_sG=np.zeros((1, 2, 2, 2)),
        gd=None,
        D_asp={},
        atoms=Atoms("C", cell=(2.0, 2.0, 2.0), pbc=True),
        Q_aL={},
        valence_potential=np.zeros((2, 2, 2)),
    )


def test_reading_two_gpw_files_on_the_threaded_scheduler_never_overlaps(probe):
    calculators = dask.compute(
        _DummyGPAW.from_file("a.gpw"),
        _DummyGPAW.from_file("b.gpw"),
        scheduler="threads",
        num_workers=2,
    )
    assert all(isinstance(c, _DummyGPAW) for c in calculators)
    assert not probe.overlapped


def test_setups_never_overlap_a_file_read(probe):
    get_setups = dask.delayed(lambda calculator: calculator.setups)
    setups, calculator = dask.compute(
        get_setups(_dummy()),
        _DummyGPAW.from_file("a.gpw"),
        scheduler="threads",
        num_workers=2,
    )
    assert setups == "setups"
    assert isinstance(calculator, _DummyGPAW)
    assert not probe.overlapped


def test_two_setups_never_overlap(probe):
    get_setups = dask.delayed(lambda calculator: calculator.setups)
    dask.compute(
        get_setups(_dummy()), get_setups(_dummy()), scheduler="threads", num_workers=2
    )
    assert not probe.overlapped


def test_the_magnetic_vector_potential_never_overlaps_a_setups_call(probe):
    class FakeCalculator:
        atoms = Atoms("C", cell=(2.0, 2.0, 2.0), pbc=True)

        def get_all_electron_density(self, spin, gridrefinement):
            probe()
            return np.zeros((1, 2, 4, 4, 4))

    get_setups = dask.delayed(lambda calculator: calculator.setups)
    vector_potential = dask.delayed(get_vector_potential_from_gpaw)
    setups, potential = dask.compute(
        get_setups(_dummy()),
        vector_potential(FakeCalculator()),
        scheduler="threads",
        num_workers=2,
    )
    assert setups == "setups"
    assert potential.shape == (3, 4, 4, 4)
    assert not probe.overlapped


def _sleeping(transform):
    def sleeping_transform(self):
        # Sleeping releases the GIL inside every GPAW transform, so two tasks
        # that are in GPAW at the same time interleave their transforms.
        time.sleep(0.002)
        return transform(self)

    return sleeping_transform


@pytest.fixture(scope="module")
def two_gpw_files(tmp_path_factory):
    gpaw = pytest.importorskip("gpaw")
    directory = tmp_path_factory.mktemp("gpw")
    paths = []
    for i, x in enumerate((0.0, 0.3)):
        atoms = Atoms("C", positions=[(x, 0.0, 0.0)], cell=(2.0,) * 3, pbc=True)
        # h=0.18: see the gpaw_calculator_bonding fixture in test_gpaw.py.
        atoms.calc = gpaw.GPAW(mode=gpaw.PW(500), h=0.18, txt=None, kpts=(3, 3, 3))
        atoms.get_potential_energy()
        path = str(directory / f"{i}.gpw")
        atoms.calc.write(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("repeat", range(3))
def test_potential_from_two_files_on_the_threaded_scheduler(
    two_gpw_files, monkeypatch, repeat
):
    import gpaw.fftw

    from abtem.potentials.gpaw import GPAWPotential

    expected = [
        GPAWPotential(path, gpts=(32, 32))
        .build()
        .compute(scheduler="synchronous")
        .array
        for path in two_gpw_files
    ]
    for plans in (gpaw.fftw.FFTWPlans, gpaw.fftw.NumpyFFTPlans):
        for name in ("fft", "ifft"):
            monkeypatch.setattr(plans, name, _sleeping(getattr(plans, name)))

    with dask.config.set(scheduler="threads", num_workers=2):
        potential = GPAWPotential(two_gpw_files, gpts=(32, 32)).build().compute()

    for member, reference in zip(potential.array, expected):
        np.testing.assert_allclose(
            member, reference, rtol=0, atol=1e-12 * np.abs(reference).max()
        )


def test_a_lazy_file_read_pickles():
    # The processes and distributed schedulers pickle every task; the lock
    # itself cannot be pickled.
    import cloudpickle

    task = _DummyGPAW.from_file("a.gpw")
    assert cloudpickle.loads(cloudpickle.dumps(task)).key == task.key
