"""``GPAWPotential`` built or simulated from every kind of input.

A ``.gpw`` path is read lazily: the potential holds a ``dask.delayed`` read in place
of the calculator. A lazy build resolves it as a task argument; an eager build
reads it when it makes the block that needs it, keeping only the calculator read
last. Each eager result is compared with the lazy build of the same
potential on the synchronous scheduler.

A potential from one calculator is one configuration with no ensemble axis, like
``abtem.Potential(atoms)``, whose ``DummyFrozenPhonons`` has ``num_configs=None``
and ``num_configurations == 1``. The built ``GPAWPotential`` (a ``PotentialArray``)
follows it, and every simulation of the unbuilt potential must give the built
potential's result.

The tests that use ``fake_gpaw`` replace GPAW with a minimal stand-in, so they run
where GPAW is not installed (CI). The last test needs GPAW.
"""

import gc
import os
import weakref
from types import SimpleNamespace

import dask
import numpy as np
import pytest
from ase import Atoms
from scipy.interpolate import interp1d

import abtem
import abtem.potentials.gpaw as gpaw_module
from abtem.potentials.gpaw import GPAWPotential, _DummyGPAW
from abtem.potentials.iam import CrystalPotential
from utils import synthetic_transition_potential

GPTS = (16, 16)


def _atoms():
    return Atoms("C", positions=[(0.6, 0.8, 1.0)], cell=(2.0, 2.0, 2.0), pbc=True)


def _valence_potential(seed):
    return np.random.default_rng(seed).standard_normal((12, 10, 8))


@pytest.fixture
def fake_gpaw(monkeypatch):
    """GPAW replaced by a stand-in whose valence potential depends on the path."""

    class FakeGPAW:
        def __init__(self, restart=None, txt=None, mode="pw", xc="PBE"):
            self.seed = sum(map(ord, restart)) if restart else 0
            self.atoms = _atoms()
            self.parameters = SimpleNamespace(mode=mode, xc=xc)
            self.density = SimpleNamespace(
                Q_aL={0: np.zeros(1)},
                nt_sG=np.zeros((1, 2, 2, 2)),
                gd=SimpleNamespace(new_descriptor=lambda comm: None),
                D_asp={0: np.zeros((1, 1))},
            )
            self.setups = None

        def get_electrostatic_potential(self):
            return _valence_potential(self.seed)

        def initialize(self, atoms):
            pass

    r = np.linspace(0.0, 3.0, 200)
    core = interp1d(
        r, -20.0 * np.exp(-(r**2) / 0.1), fill_value=0.0, bounds_error=False
    )
    read_atoms = gpaw_module._safe_read_atoms

    monkeypatch.setattr(gpaw_module, "GPAW", FakeGPAW)
    monkeypatch.setattr(gpaw_module, "SerialCommunicator", lambda: None, raising=False)
    monkeypatch.setattr(
        gpaw_module,
        "get_core_correction_interpolators",
        lambda setups, D_asp, Q_aL, rcgauss: [core],
    )
    # A path is read with gpaw.io.Reader; stand in for that one call.
    monkeypatch.setattr(
        gpaw_module,
        "_safe_read_atoms",
        lambda c, clean=True: _atoms() if isinstance(c, str) else read_atoms(c, clean),
    )


def _frozen_phonons():
    return abtem.FrozenPhonons(_atoms(), num_configs=2, sigmas=0.1, seed=1)


def _loaded(path):
    return _DummyGPAW.from_gpaw(gpaw_module.GPAW(path))


# Every input that has an ensemble axis and at least one path.
_ENSEMBLES = {
    "two paths": lambda: GPAWPotential(["a.gpw", "b.gpw"], gpts=GPTS),
    "one path in a list": lambda: GPAWPotential(["a.gpw"], gpts=GPTS),
    "path and calculator": lambda: GPAWPotential(
        ["a.gpw", _loaded("b.gpw")], gpts=GPTS
    ),
    "path and frozen phonons": lambda: GPAWPotential(
        "a.gpw", gpts=GPTS, frozen_phonons=_frozen_phonons()
    ),
}


def _assert_equal(result, expected):
    assert result.shape == expected.shape
    assert result.ensemble_axes_metadata == expected.ensemble_axes_metadata
    scale = np.abs(expected.array).max()
    assert scale > 0
    np.testing.assert_allclose(result.array, expected.array, rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("case", list(_ENSEMBLES))
def test_an_eager_build_equals_the_lazy_build(fake_gpaw, case):
    expected = _ENSEMBLES[case]().build().compute(scheduler="synchronous")

    result = _ENSEMBLES[case]().build(lazy=False)

    _assert_equal(result, expected)


def test_the_members_of_an_eager_build_come_from_their_own_paths(fake_gpaw):
    result = GPAWPotential(["a.gpw", "b.gpw"], gpts=GPTS).build(lazy=False)
    single = [
        GPAWPotential(path, gpts=GPTS).build(lazy=False).array
        for path in ("a.gpw", "b.gpw")
    ]

    assert not np.allclose(single[0], single[1])
    for member, expected in zip(result.array, single):
        np.testing.assert_allclose(
            member, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
        )


@pytest.mark.parametrize("wave", ["PlaneWave", "Probe"])
@pytest.mark.parametrize("case", list(_ENSEMBLES))
def test_an_eager_multislice_equals_the_lazy_one(fake_gpaw, case, wave):
    def multislice(lazy):
        waves = (
            abtem.PlaneWave(energy=100e3)
            if wave == "PlaneWave"
            else abtem.Probe(energy=100e3, semiangle_cutoff=20)
        )
        result = waves.multislice(_ENSEMBLES[case](), lazy=lazy)
        return result.compute(scheduler="synchronous") if lazy else result

    _assert_equal(multislice(False), multislice(True))


def test_a_single_path_builds_eagerly(fake_gpaw):
    # A path without an ensemble axis is read by ``generate_slices`` itself.
    potential = GPAWPotential("a.gpw", gpts=GPTS)

    _assert_equal(
        potential.build(lazy=False), potential.build().compute(scheduler="synchronous")
    )


def _single_calculator(kind):
    return "single.gpw" if kind == "path" else _loaded("single.gpw")


def _simulate(wave, potential, lazy):
    if wave == "PlaneWave":
        result = abtem.PlaneWave(energy=100e3).multislice(potential, lazy=lazy)
    elif wave == "Probe":
        result = abtem.Probe(energy=100e3, semiangle_cutoff=20).multislice(
            potential, lazy=lazy
        )
    else:
        result = abtem.Probe(energy=100e3, semiangle_cutoff=20).scan(
            potential,
            scan=abtem.GridScan(start=(0, 0), end=(2, 2), gpts=(3, 2)),
            detectors=abtem.AnnularDetector(inner=20, outer=60),
            lazy=lazy,
        )
    return result.compute() if lazy else result


@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_one_calculator_is_one_configuration(fake_gpaw, kind):
    potential = GPAWPotential(_single_calculator(kind), gpts=GPTS)
    reference = abtem.Potential(_atoms(), gpts=GPTS)

    assert potential.num_configurations == reference.num_configurations == 1
    assert potential.num_frozen_phonons == 1
    assert potential.ensemble_shape == reference.ensemble_shape == ()
    assert potential.ensemble_axes_metadata == []


@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_the_number_of_frozen_phonons_is_the_number_of_configurations(fake_gpaw, kind):
    potential = GPAWPotential(
        _single_calculator(kind),
        gpts=GPTS,
        frozen_phonons=abtem.FrozenPhonons(_atoms(), 3, sigmas=0.05, seed=1),
    )

    assert potential.num_configurations == potential.num_frozen_phonons == 3


def test_a_list_of_calculators_has_one_configuration_each(fake_gpaw):
    potential = GPAWPotential(["a.gpw", _loaded("b.gpw")], gpts=GPTS)

    assert potential.num_configurations == potential.num_frozen_phonons == 2


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("wave", ["PlaneWave", "Probe", "scan"])
@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_one_calculator_simulates_like_its_built_potential(fake_gpaw, kind, wave, lazy):
    built = GPAWPotential(_single_calculator(kind), gpts=GPTS).build(lazy=False)
    expected = _simulate(wave, built, lazy=False)

    result = _simulate(wave, GPAWPotential(_single_calculator(kind), gpts=GPTS), lazy)

    _assert_equal(result, expected)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_one_calculator_in_a_core_loss_scan(fake_gpaw, kind, lazy):
    def scan(potential):
        result = abtem.Probe(
            energy=100e3, semiangle_cutoff=20
        ).transition_potential_scan(
            potential=potential,
            transition_potentials=synthetic_transition_potential(
                Z=6, gpts=GPTS, extent=(2.0, 2.0), n_transitions=2
            ),
            scan=abtem.GridScan(start=(0, 0), end=(2, 2), gpts=(3, 2)),
            detectors=abtem.AnnularDetector(inner=0, outer=40),
            double_channel=False,
            sites=_atoms(),
            lazy=lazy,
        )
        return result.compute() if lazy else result

    expected = scan(
        GPAWPotential(_single_calculator(kind), gpts=GPTS).build(lazy=False)
    )
    result = scan(GPAWPotential(_single_calculator(kind), gpts=GPTS))

    assert result.shape == expected.shape == (3, 2)
    _assert_equal(result, expected)


@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_tiling_one_calculator_into_an_ensemble_warns_like_potential(fake_gpaw, kind):
    with pytest.warns(UserWarning, match="does not have frozen phonons"):
        abtem.CrystalPotential(
            abtem.Potential(_atoms(), gpts=GPTS), (2, 1, 1), num_frozen_phonons=2
        )

    with pytest.warns(UserWarning, match="does not have frozen phonons"):
        abtem.CrystalPotential(
            GPAWPotential(_single_calculator(kind), gpts=GPTS),
            (2, 1, 1),
            num_frozen_phonons=2,
        )


@pytest.mark.parametrize("lazy", [True, False])
def test_a_one_element_list_keeps_its_ensemble_axis(fake_gpaw, lazy):
    potential = GPAWPotential(["single.gpw"], gpts=GPTS)

    assert potential.num_configurations == 1
    assert potential.ensemble_shape == (1,)

    result = _simulate("PlaneWave", potential, lazy)
    assert result.shape == (1,) + GPTS


class _CountReads:
    """Counts ``.gpw`` reads and the calculators they return that are still alive.

    ``alive_while_slicing`` is sampled each time slices are generated from a
    calculator, after a garbage collection, so it counts what an eager build
    holds while it works, not objects waiting to be collected.
    """

    def __init__(self, monkeypatch):
        self.reads = 0
        self.alive_while_slicing = []
        self._alive = set()
        read_gpw = gpaw_module._read_gpw
        generate_slices = gpaw_module._generate_slices

        def counting_read(path):
            self.reads += 1
            calculator = read_gpw(path)
            # _DummyGPAW is an unhashable dataclass, so track ids.
            self._alive.add(id(calculator))
            weakref.finalize(calculator, self._alive.discard, id(calculator))
            return calculator

        def sampling_generate_slices(*args, **kwargs):
            gc.collect()
            self.alive_while_slicing.append(len(self._alive))
            yield from generate_slices(*args, **kwargs)

        # Set before the potential is made: ``from_file`` looks ``_read_gpw`` up
        # when it creates the placeholder.
        monkeypatch.setattr(gpaw_module, "_read_gpw", counting_read)
        monkeypatch.setattr(gpaw_module, "_generate_slices", sampling_generate_slices)

    def alive(self):
        gc.collect()
        return len(self._alive)


def _eager(entry_point, potential):
    if entry_point == "build":
        return potential.build(lazy=False)
    if entry_point == "PlaneWave":
        return abtem.PlaneWave(energy=100e3).multislice(potential, lazy=False)
    if entry_point == "scan":
        return _simulate("scan", potential, lazy=False)
    if entry_point == "PRISM":
        return abtem.SMatrix(
            potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1
        ).scan(
            scan=abtem.GridScan(start=(0, 0), end=(2, 2), gpts=(3, 2)),
            detectors=abtem.AnnularDetector(inner=20, outer=60),
            lazy=False,
        )
    raise ValueError(entry_point)


_PATHS = ["a.gpw", "b.gpw", "c.gpw", "d.gpw"]


@pytest.mark.parametrize("entry_point", ["build", "PlaneWave", "scan", "PRISM"])
def test_an_eager_build_holds_one_read_calculator_at_a_time(
    fake_gpaw, monkeypatch, entry_point
):
    counter = _CountReads(monkeypatch)

    _eager(entry_point, GPAWPotential(_PATHS, gpts=GPTS))

    assert counter.reads == len(_PATHS)
    assert counter.alive_while_slicing
    assert max(counter.alive_while_slicing) == 1
    assert counter.alive() == 0


@pytest.mark.parametrize(
    "paths",
    [["a.gpw", "b.gpw", "a.gpw"], ["a.gpw", "a.gpw"]],
    ids=["returns to a file", "repeats a file"],
)
def test_each_list_entry_is_read_once(fake_gpaw, monkeypatch, paths):
    # Every entry of a list is its own read, in the lazy graph as in an eager
    # build; only one is held at a time.
    counter = _CountReads(monkeypatch)

    GPAWPotential(paths, gpts=GPTS).build(lazy=False)

    assert counter.reads == len(paths)
    assert max(counter.alive_while_slicing) == 1


@pytest.mark.parametrize("entry_point", ["build", "PlaneWave", "scan", "PRISM"])
def test_frozen_phonons_of_one_path_read_it_once(fake_gpaw, monkeypatch, entry_point):
    counter = _CountReads(monkeypatch)
    frozen_phonons = abtem.FrozenPhonons(_atoms(), num_configs=4, sigmas=0.1, seed=1)

    _eager(
        entry_point,
        GPAWPotential("a.gpw", gpts=GPTS, frozen_phonons=frozen_phonons),
    )

    assert counter.reads == 1
    assert len(counter.alive_while_slicing) >= 4
    assert max(counter.alive_while_slicing) == 1


def test_loaded_calculators_between_paths(fake_gpaw, monkeypatch):
    counter = _CountReads(monkeypatch)
    potential = GPAWPotential(
        ["a.gpw", _loaded("b.gpw"), "c.gpw", _loaded("d.gpw")], gpts=GPTS
    )

    potential.build(lazy=False)

    assert counter.reads == 2
    assert max(counter.alive_while_slicing) == 1


@pytest.mark.parametrize("lazy", [False, True])
def test_a_crystal_of_a_path_list_holds_one_read_calculator(
    fake_gpaw, monkeypatch, lazy
):
    counter = _CountReads(monkeypatch)
    crystal = abtem.CrystalPotential(
        GPAWPotential(["a.gpw", "b.gpw"], gpts=GPTS),
        (1, 1, 2),
        num_frozen_phonons=3,
        seeds=(1, 2, 3),
    )

    result = abtem.PlaneWave(energy=100e3).multislice(crystal, lazy=lazy)
    if lazy:
        result.compute(scheduler="threads")

    assert counter.reads == 2
    assert max(counter.alive_while_slicing) == 1


@pytest.mark.parametrize("entry_point", ["build", "PlaneWave", "scan", "PRISM"])
def test_eager_reads_of_a_path_list_do_not_use_the_configured_scheduler(
    fake_gpaw, entry_point
):
    # A read of a path list or of a path with frozen phonons inside an eager
    # build runs where the build runs: it starts no thread pool inside a
    # caller's task and sends nothing to a distributed client set as the
    # default scheduler. A single path without an ensemble is read by
    # generate_slices on the configured scheduler.
    def refuse(*args, **kwargs):
        raise AssertionError("an eager read used the configured scheduler")

    with dask.config.set(scheduler=refuse):
        _eager(entry_point, GPAWPotential(["a.gpw", "b.gpw"], gpts=GPTS))


@pytest.fixture(scope="module")
def gpw_paths(tmp_path_factory):
    gpaw = pytest.importorskip("gpaw")
    directory = tmp_path_factory.mktemp("gpw")
    paths = []
    for i, x in enumerate((0.0, 0.3)):
        atoms = Atoms("C", positions=[(x, 0, 0)], cell=(2.0,) * 3, pbc=True)
        # h=0.18: see the gpaw_calculator_bonding fixture in test_gpaw.py.
        atoms.calc = gpaw.GPAW(mode=gpaw.PW(500), h=0.18, txt=None, kpts=(3, 3, 3))
        atoms.get_potential_energy()
        paths.append(os.path.join(directory, f"C{i}.gpw"))
        atoms.calc.write(paths[-1])
    return paths


@pytest.mark.parametrize(
    "case", ["two paths", "one path in a list", "path and calculator", "frozen phonons"]
)
def test_gpaw_eager_build_of_gpw_files(gpw_paths, case):
    from gpaw import GPAW

    atoms = GPAW(gpw_paths[0], txt=None).atoms
    inputs = {
        "two paths": lambda: dict(calculators=gpw_paths),
        "one path in a list": lambda: dict(calculators=gpw_paths[:1]),
        "path and calculator": lambda: dict(
            calculators=[gpw_paths[0], GPAW(gpw_paths[1], txt=None)]
        ),
        "frozen phonons": lambda: dict(
            calculators=gpw_paths[0],
            frozen_phonons=abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1),
        ),
    }

    def potential():
        return GPAWPotential(gpts=(32, 32), **inputs[case]())

    expected = potential().build().compute(scheduler="synchronous")
    result = potential().build(lazy=False)

    assert result.shape == expected.shape
    scale = np.abs(expected.array).max()
    np.testing.assert_allclose(result.array, expected.array, rtol=0, atol=1e-6 * scale)


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("wave", ["PlaneWave", "Probe"])
@pytest.mark.parametrize("kind", ["path", "calculator"])
def test_gpaw_single_calculator_multislice(gpw_paths, kind, wave, lazy):
    from gpaw import GPAW

    def potential():
        source = gpw_paths[0] if kind == "path" else GPAW(gpw_paths[0], txt=None)
        return GPAWPotential(source, gpts=(32, 32))

    expected = _simulate(wave, potential().build(lazy=False), lazy=False)
    result = _simulate(wave, potential(), lazy)

    assert potential().num_configurations == 1
    assert result.shape == expected.shape == (32, 32)
    scale = np.abs(expected.array).max()
    np.testing.assert_allclose(result.array, expected.array, rtol=0, atol=1e-6 * scale)


def _gpaw_crystal(num_configs, repetitions, **kwargs):
    unit = GPAWPotential(
        "a.gpw",
        gpts=GPTS,
        frozen_phonons=abtem.FrozenPhonons(
            _atoms(), num_configs=num_configs, sigmas=0.1, seed=1
        ),
    )
    return CrystalPotential(unit, repetitions, **kwargs)


@pytest.mark.filterwarnings("ignore:frozen-phonon pool .* is smaller:UserWarning")
@pytest.mark.parametrize(
    "num_configs, repetitions, kwargs",
    [
        (4, (2, 1, 2), dict(num_frozen_phonons=2, seeds=(5, 6))),  # reseeded
        (2, (3, 1, 2), dict()),  # unseeded, pool enlarged from 2 to 3
    ],
    ids=["seeded", "enlarged"],
)
def test_a_crystal_of_a_frozen_phonon_gpaw_unit_builds(
    fake_gpaw, num_configs, repetitions, kwargs
):
    crystal = _gpaw_crystal(num_configs, repetitions, **kwargs)

    built = crystal.build(lazy=False)

    assert built.shape[-3] == len(crystal)
    assert np.abs(built.array).max() > 0


def test_the_pool_unit_of_a_member_is_the_unit_with_the_seed_of_the_member(fake_gpaw):
    crystal = _gpaw_crystal(4, (2, 1, 2), num_frozen_phonons=2, seeds=(5, 6))

    for seed in (5, 6):
        pool_unit = crystal._pool_unit_for_member(seed)
        expected = GPAWPotential(
            "a.gpw",
            gpts=GPTS,
            frozen_phonons=abtem.FrozenPhonons(
                _atoms(), num_configs=4, sigmas=0.1, seed=seed
            ),
        )

        assert pool_unit.calculators is crystal.potential_unit.calculators
        np.testing.assert_array_equal(
            pool_unit.build(lazy=False).array, expected.build(lazy=False).array
        )


@pytest.mark.filterwarnings("ignore:frozen-phonon pool .* is smaller:UserWarning")
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "num_configs, repetitions, kwargs",
    [
        (4, (2, 1, 2), dict(num_frozen_phonons=2, seeds=(5, 6), ensemble_mean=False)),
        (2, (3, 1, 2), dict()),
    ],
    ids=["seeded", "enlarged"],
)
def test_a_crystal_of_a_frozen_phonon_gpaw_unit_simulates_as_it_builds(
    fake_gpaw, num_configs, repetitions, kwargs, lazy
):
    expected = abtem.PlaneWave(energy=100e3).multislice(
        _gpaw_crystal(num_configs, repetitions, **kwargs).build(lazy=False),
        lazy=False,
    )
    result = abtem.PlaneWave(energy=100e3).multislice(
        _gpaw_crystal(num_configs, repetitions, **kwargs), lazy=lazy
    )
    result = result.compute(scheduler="synchronous") if lazy else result

    _assert_equal(result, expected)
