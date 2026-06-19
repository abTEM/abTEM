"""Tests for inelastic / core-loss simulation entry points.

These guard against regressions in the public API that the core-loss tutorial
depends on (see https://abtem.readthedocs.io/en/latest/user_guide/tutorials/core_loss.html).
The transition_potential_scan method was silently dropped in early 2025 and
restored later; the smoke tests below ensure it stays wired up.
"""
import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.waves import Probe

try:
    import cupy as cp
except ImportError:
    cp = None

from utils import gpu  # noqa: E402  -- pytest.param('gpu', skipif no cupy)


@pytest.fixture(scope="module")
def si_potential():
    atoms = ase.build.bulk("Si", cubic=True)
    return abtem.Potential(atoms, gpts=32, slice_thickness=2.7)


@pytest.fixture(scope="module")
def si_transition_potential(si_potential):
    # Synthetic transition potential: skips numerov / GPAW so the wiring is
    # exercised without depending on the heavy DFT setup the tutorial uses.
    n_transitions = 2
    rng = np.random.default_rng(0)
    array = (
        rng.standard_normal((n_transitions, *si_potential.gpts))
        + 1j * rng.standard_normal((n_transitions, *si_potential.gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=14,
        array=array,
        energy=100e3,
        extent=si_potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


@pytest.fixture(scope="module")
def probe(si_potential):
    p = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    p.grid.match(si_potential)
    return p


def test_transition_potential_scan_is_defined_on_probe():
    """Guard against the Feb-2025 regression where the method was commented out."""
    assert hasattr(Probe, "transition_potential_scan")
    assert callable(Probe.transition_potential_scan)


def test_transition_potential_scan_builds_lazy_graph(
    probe, si_potential, si_transition_potential
):
    """Wiring smoke test: lazy call returns a measurements object without raising."""
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=(0, 0),
        detectors=detector,
        lazy=True,
    )
    assert result is not None
    assert hasattr(result, "compute")


def test_transition_potential_scan_forwards_inelastic_kwargs(
    probe, si_potential, si_transition_potential
):
    """double_channel / threshold must flow through to
    transition_potential_multislice_and_detect via **multislice_func_kwargs."""
    detector = abtem.AnnularDetector(inner=0, outer=40)
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=(0, 0),
        detectors=detector,
        double_channel=False,
        threshold=0.95,
        lazy=True,
    )
    assert result is not None


def test_transition_potential_scan_accepts_grid_scan(
    probe, si_potential, si_transition_potential
):
    """The tutorial uses GridScan + sites=... for the EELS-map calls."""
    detector = abtem.FlexibleAnnularDetector()
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), fractional=True,
        potential=si_potential, endpoint=False, sampling=2.0,
    )
    result = probe.transition_potential_scan(
        potential=si_potential,
        transition_potentials=si_transition_potential,
        scan=scan,
        detectors=detector,
        sites=ase.build.bulk("Si", cubic=True),
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


