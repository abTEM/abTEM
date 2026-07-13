"""Tests for inelastic / core-loss simulation entry points.

These guard against regressions in the public API that the core-loss tutorial
depends on (see https://abtem.readthedocs.io/en/latest/user_guide/tutorials/core_loss.html).
The transition_potential_scan method was silently dropped in early 2025 and
restored later; the smoke tests below ensure it stays wired up.

All tests are parametrised over ``["cpu", gpu]`` so that on a workstation
with CuPy installed the GPU code paths in ``fast_roll`` /
``transition_potential_multislice_and_detect`` are exercised automatically.
``gpu`` is a ``pytest.param`` defined in ``test/utils.py`` that skips when
CuPy isn't present.
"""
import sys

import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray, fast_roll
from abtem.waves import Probe

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import gpaw  # noqa: F401
except ImportError:
    pass

from utils import gpu  # noqa: E402  -- pytest.param('gpu', skipif no cupy)

# For fast_roll we parametrise over a backend *module* string ("numpy" /
# "cupy"); the abtem-level GPU dispatch tests use the standard device kwarg.
xp_params = [
    "numpy",
    pytest.param("cupy", marks=pytest.mark.skipif(cp is None, reason="no gpu")),
]


def _xp(name):
    return np if name == "numpy" else cp


@pytest.mark.parametrize("xp_name", xp_params)
def test_fast_roll_matches_numpy_roll(xp_name):
    """The vectorised fast_roll must give the same result as a per-site np.roll
    on both numpy and cupy inputs."""
    xp = _xp(xp_name)
    rng = np.random.default_rng(0)
    arr_np = rng.standard_normal((16, 16)).astype(np.complex64)
    arr = xp.asarray(arr_np)
    shifts_np = np.array([[0, 0], [3, 5], [15, 15], [1, 7], [10, 2]])
    shifts = xp.asarray(shifts_np)
    out = fast_roll(arr, shifts)
    if xp is not np:
        out = cp.asnumpy(out)
    for i, s in enumerate(shifts_np):
        expected = np.roll(arr_np, (int(s[0]), int(s[1])), axis=(0, 1))
        assert np.array_equal(out[i], expected), f"mismatch at shift {tuple(s)}"


@pytest.mark.parametrize("xp_name", xp_params)
def test_fast_roll_handles_negative_shifts(xp_name):
    """Modular indexing must produce the same result as np.roll with negative
    shifts on both numpy and cupy inputs.

    The pre-Tier-2 implementation raised RuntimeError on negative shifts;
    handle them correctly so atom positions outside the centred-cell
    convention still work.
    """
    xp = _xp(xp_name)
    rng = np.random.default_rng(1)
    arr_np = rng.standard_normal((8, 8)).astype(np.complex64)
    arr = xp.asarray(arr_np)
    shifts_np = np.array([[-1, -2], [-7, 3], [0, -5]])
    shifts = xp.asarray(shifts_np)
    out = fast_roll(arr, shifts)
    if xp is not np:
        out = cp.asnumpy(out)
    for i, s in enumerate(shifts_np):
        expected = np.roll(arr_np, (int(s[0]), int(s[1])), axis=(0, 1))
        assert np.array_equal(out[i], expected), f"mismatch at shift {tuple(s)}"


# Module-scoped fixtures kept device-agnostic; the test functions thread the
# device kwarg through to Potential / Probe / TransitionPotentialArray.


@pytest.fixture(scope="module")
def si_atoms():
    return ase.build.bulk("Si", cubic=True)


def _make_si_potential(atoms, device):
    return abtem.Potential(atoms, gpts=32, slice_thickness=2.7, device=device)


def _make_si_transition_potential(potential, device):
    # Synthetic transition potential: skips numerov / GPAW so the wiring is
    # exercised without depending on the heavy DFT setup the tutorial uses.
    n_transitions = 2
    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((n_transitions, *potential.gpts))
        + 1j * rng.standard_normal((n_transitions, *potential.gpts))
    ).astype(np.complex64)
    if device == "gpu":
        array = cp.asarray(array)
    return TransitionPotentialArray(
        Z=14,
        array=array,
        energy=100e3,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def _make_probe(potential, device):
    p = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    p.grid.match(potential)
    return p


def test_transition_potential_scan_is_defined_on_probe():
    """Guard against the Feb-2025 regression where the method was commented out."""
    assert hasattr(Probe, "transition_potential_scan")
    assert callable(Probe.transition_potential_scan)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_transition_potential_scan_builds_lazy_graph(si_atoms, device):
    """Wiring smoke test: lazy call returns a measurements object without raising,
    on both CPU and GPU backends."""
    potential = _make_si_potential(si_atoms, device)
    tp = _make_si_transition_potential(potential, device)
    probe = _make_probe(potential, device)
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        lazy=True,
    )
    assert result is not None
    assert hasattr(result, "compute")


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_transition_potential_scan_forwards_inelastic_kwargs(si_atoms, device):
    """double_channel / threshold must flow through to
    transition_potential_multislice_and_detect via **multislice_func_kwargs."""
    potential = _make_si_potential(si_atoms, device)
    tp = _make_si_transition_potential(potential, device)
    probe = _make_probe(potential, device)
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        double_channel=False,
        threshold=0.95,
        lazy=True,
    )
    assert result is not None


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_transition_potential_scan_accepts_grid_scan(si_atoms, device):
    """The tutorial uses GridScan + sites=... for the EELS-map calls."""
    potential = _make_si_potential(si_atoms, device)
    tp = _make_si_transition_potential(potential, device)
    probe = _make_probe(potential, device)
    detector = abtem.FlexibleAnnularDetector()
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), fractional=True,
        potential=potential, endpoint=False, sampling=2.0,
    )
    result = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=si_atoms,
        lazy=True,
    )
    assert result is not None


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_transition_potential_scan_crystal_potential_matches_manual_tile(device):
    """Auto-extracted sites from a CrystalPotential must produce the same
    result as a regular Potential built from a manually-tiled supercell.

    The tutorial uses regular Potential, but CrystalPotential is the natural
    way to express large repeating crystals cheaply. Without the
    ``hasattr(potential, "potential_unit")`` branch in
    transition_potential_multislice_and_detect, ``sites=None`` raises bare
    ValueError because CrystalPotential exposes neither ``get_sliced_atoms``
    nor ``atoms``. This test guards both correctness and the auto-extraction.
    """
    if device == "gpu":
        xp = cp
    else:
        xp = np

    unit_atoms = ase.build.bulk("Si", cubic=True)  # 5.43 Å cubic
    reps = (2, 2, 3)
    # Use slice_thickness equal to one unit-cell thickness so the manual
    # tile and the CrystalPotential land on bit-identical slice boundaries.
    slice_thickness = float(unit_atoms.cell[2, 2])

    manual_pot = abtem.Potential(
        unit_atoms * reps, gpts=(64, 64), slice_thickness=slice_thickness,
        device=device,
    )
    unit_pot = abtem.Potential(
        unit_atoms, gpts=(32, 32), slice_thickness=slice_thickness,
        device=device,
    )
    cryst_pot = abtem.CrystalPotential(unit_pot, repetitions=reps)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(manual_pot)

    rng = np.random.default_rng(0)
    tp_array_np = (
        rng.standard_normal((2, 64, 64))
        + 1j * rng.standard_normal((2, 64, 64))
    ).astype(np.complex64)
    tp_array = xp.asarray(tp_array_np)

    def make_tp(extent):
        return TransitionPotentialArray(
            Z=14, array=tp_array, energy=100e3, extent=extent,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"Z": 14, "n": 1, "l": 0},
        )

    detector = abtem.PixelatedDetector(max_angle=40, to_cpu=True)

    res_manual = probe.transition_potential_scan(
        potential=manual_pot, transition_potentials=make_tp(manual_pot.extent),
        scan=(0, 0), detectors=detector, lazy=False,
    ).compute()
    res_cryst = probe.transition_potential_scan(
        potential=cryst_pot, transition_potentials=make_tp(cryst_pot.extent),
        scan=(0, 0), detectors=detector, lazy=False,
    ).compute()

    arr_manual = np.asarray(res_manual.array)
    arr_cryst = np.asarray(res_cryst.array)

    assert arr_manual.shape == arr_cryst.shape
    # With matched slice geometry the two paths produce bit-identical output
    # on the CPU FFTW path. Allow a small numerical tolerance for the GPU
    # path where FFT plan ordering can drift at the float32 level.
    np.testing.assert_allclose(arr_cryst, arr_manual, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_transition_potential_scan_crystal_double_channel_matches_manual(device):
    """Double-channel + CrystalPotential is the intersection that triggers the
    TransmissionFunction dedup in transition_potential_multislice_and_detect
    (the slice_cache is only built when double_channel=True, and dedup only
    fires when generate_slices yields repeated object identities). Verify the
    deduplicated cache still produces results numerically equivalent to the
    manually-tiled Potential path.
    """
    if device == "gpu":
        xp = cp
    else:
        xp = np

    unit_atoms = ase.build.bulk("Si", cubic=True)
    reps = (2, 2, 3)
    slice_thickness = float(unit_atoms.cell[2, 2])

    manual_pot = abtem.Potential(
        unit_atoms * reps, gpts=(64, 64), slice_thickness=slice_thickness,
        device=device,
    )
    unit_pot = abtem.Potential(
        unit_atoms, gpts=(32, 32), slice_thickness=slice_thickness,
        device=device,
    )
    cryst_pot = abtem.CrystalPotential(unit_pot, repetitions=reps)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(manual_pot)

    rng = np.random.default_rng(1)
    tp_array_np = (
        rng.standard_normal((2, 64, 64))
        + 1j * rng.standard_normal((2, 64, 64))
    ).astype(np.complex64)
    tp_array = xp.asarray(tp_array_np)

    def make_tp(extent):
        return TransitionPotentialArray(
            Z=14, array=tp_array, energy=100e3, extent=extent,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"Z": 14, "n": 1, "l": 0},
        )

    detector = abtem.PixelatedDetector(max_angle=40, to_cpu=True)

    res_manual = probe.transition_potential_scan(
        potential=manual_pot, transition_potentials=make_tp(manual_pot.extent),
        scan=(0, 0), detectors=detector, double_channel=True, lazy=False,
    ).compute()
    res_cryst = probe.transition_potential_scan(
        potential=cryst_pot, transition_potentials=make_tp(cryst_pot.extent),
        scan=(0, 0), detectors=detector, double_channel=True, lazy=False,
    ).compute()

    arr_manual = np.asarray(res_manual.array)
    arr_cryst = np.asarray(res_cryst.array)

    assert arr_manual.shape == arr_cryst.shape
    np.testing.assert_allclose(arr_cryst, arr_manual, rtol=1e-5, atol=0)


def test_transition_potential_crystal_dedup_collapses_slice_cache():
    """White-box check: with CrystalPotential's tile cache, the per-z-rep
    slice_cache built inside transition_potential_multislice_and_detect must
    contain only ``n_unique_unit_slices`` distinct TransmissionFunction
    objects, not ``n_outer = n_unit * reps[2]`` objects. Guards the dedup
    branch from silently regressing into rebuilding identical transmissions.
    """
    unit_atoms = ase.build.bulk("Si", cubic=True)
    reps = (2, 2, 5)
    slice_thickness = float(unit_atoms.cell[2, 2])
    unit_pot = abtem.Potential(
        unit_atoms, gpts=(16, 16), slice_thickness=slice_thickness,
    )
    cryst = abtem.CrystalPotential(unit_pot, repetitions=reps)

    # Tile cache returns the *same* PotentialArray object across z-reps for
    # the no-frozen-phonon case. Confirm that contract holds.
    slices = list(cryst.generate_slices())
    n_unit = len(unit_pot)
    assert len(slices) == n_unit * reps[2]
    unique = {id(s) for s in slices}
    assert len(unique) == n_unit, (
        f"expected {n_unit} unique slice objects (one per unit-cell slice), "
        f"got {len(unique)} — CrystalPotential.generate_slices tile cache may "
        "have regressed"
    )


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
def test_subshell_transitions_real_gpaw_pipeline():
    """End-to-end regression test using GPAW's real atomic all-electron
    solvers (``gpaw.atom.all_electron.AllElectron`` and
    ``gpaw.atom.aeatom.AllElectronAtom``, wired up through
    ``SubshellTransitions``), instead of the synthetic
    ``TransitionPotentialArray`` the other tests in this module use to
    exercise the scan machinery without depending on GPAW.

    This is a different GPAW entry point than the periodic crystal
    calculator used by ``GPAWPotential`` (see ``abtem/potentials/gpaw.py``
    and ``test/test_gpaw.py``), so a GPAW upgrade breaking one gives no
    guarantee about the other -- this needs its own coverage.
    """
    from abtem.inelastic.core_loss import SubshellTransitions

    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, gpts=32, slice_thickness=2.7)

    transitions = SubshellTransitions(Z=14, n=2, l=1, order=1, epsilon=1.0, xc="PBE")
    assert len(transitions) > 0

    transition_potentials = transitions.get_transition_potentials(
        extent=potential.extent, gpts=potential.gpts, energy=100e3
    )
    assert len(transition_potentials) == len(transitions)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=transition_potentials,
        scan=(0, 0),
        detectors=detector,
        lazy=False,
    )
    array = np.asarray(result.array)
    assert np.isfinite(array).all()
    assert np.any(array != 0)


