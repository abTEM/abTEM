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




def test_prism_eels_mvp_matches_multislice_eels_at_interp_1():
    """Stage-1 MVP: SMatrix.transition_potential_scan at interpolation=(1,1)
    reproduces Probe.transition_potential_scan on a small Si cell.
    Both paths express the same physics — forward multislice through the
    potential, scatter via the transition potential at each site, and
    detect — just decomposed in the plane-wave basis vs a probe per
    scan position. Stage 2 (interpolation > 1, S2_crop linear scaling,
    frozen phonons, GPU) is tracked in issue #287.
    """
    unit_atoms = ase.build.bulk("Si", cubic=True)
    reps = (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])
    atoms = unit_atoms * reps

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, device="cpu"
    )

    energy = 100e3
    semiangle_cutoff = 20.0

    rng = np.random.default_rng(0)
    tp_array = (
        rng.standard_normal((2, 32, 32))
        + 1j * rng.standard_normal((2, 32, 32))
    ).astype(np.complex64)
    tp = TransitionPotentialArray(
        Z=14,
        array=tp_array,
        energy=energy,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )

    detector = abtem.PixelatedDetector(max_angle=40)

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu"
    )
    probe.grid.match(potential)
    # ``double_channel=False`` matches the MVP's single-channel scope: scatter
    # at the scatter slice and detect immediately, without propagating the
    # scattered wave through the remaining potential. The multislice EELS
    # default is ``double_channel=True``, which propagates and detects later
    # at exit_planes; comparing against the double-channel path would be
    # apples-to-oranges since the MVP doesn't (yet) do the inner propagation.
    res_multislice = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        sites=atoms,
        double_channel=False,
        lazy=False,
    ).compute()
    arr_multislice = np.asarray(res_multislice.array)

    # ``downsample=False`` keeps the S-matrix on the same gpts as the probe;
    # the default ``downsample="cutoff"`` would resample to the antialias
    # cutoff and break the comparison.
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=1,
        downsample=False,
        device="cpu",
    )
    res_prism = s_matrix.transition_potential_scan(
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        sites=atoms,
    )
    arr_prism = np.asarray(res_prism.array)

    assert arr_multislice.shape == arr_prism.shape, (
        f"shape mismatch: multislice {arr_multislice.shape} vs "
        f"PRISM {arr_prism.shape}"
    )
    # Bit-equivalent at interpolation=(1,1): the two paths express the same
    # numerical work, just in different bases. Measured max rel diff ~5e-7
    # on this setup (float32 FFT plan-ordering noise); 1e-5 leaves
    # comfortable headroom for run-to-run variation without masking real
    # regressions.
    np.testing.assert_allclose(arr_prism, arr_multislice, rtol=1e-5, atol=0)


def test_smatrix_transition_potential_scan_interp_2_produces_windowed_output():
    """Stage-2 ``interpolation > 1`` runs through the cropping pattern from
    SMatrixArray._reduce_to_waves (s_matrix.py:996-1033) and yields a
    diffraction pattern at ``window_gpts`` size rather than the full
    ``gpts``. The same windowing characterises the elastic SMatrix.scan
    path; PRISM-EELS inherits the convention.
    """
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 2)
    slice_thickness = float(atoms.cell[2, 2]) / 2
    # gpts must be divisible by interpolation.
    potential = abtem.Potential(
        atoms, gpts=(64, 64), slice_thickness=slice_thickness
    )

    energy = 100e3
    semiangle_cutoff = 20.0

    rng = np.random.default_rng(0)
    tp_array = (
        rng.standard_normal((2, 64, 64))
        + 1j * rng.standard_normal((2, 64, 64))
    ).astype(np.complex64)
    tp = TransitionPotentialArray(
        Z=14,
        array=tp_array,
        energy=energy,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )

    detector = abtem.PixelatedDetector(max_angle=30)

    # interp=(1, 1) gives the full-grid diffraction shape; interp=(2, 2)
    # gives a smaller pattern derived from the cropped window. Confirm both
    # match the corresponding shapes produced by the elastic SMatrix.scan
    # path, so the windowing wiring is on the correct convention.
    s1 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=1, downsample=False, device="cpu",
    )
    s2 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=2, downsample=False, device="cpu",
    )

    elastic_1 = s1.scan(scan=(0, 0), detectors=detector, lazy=False).compute()
    elastic_2 = s2.scan(scan=(0, 0), detectors=detector, lazy=False).compute()
    eels_1 = s1.transition_potential_scan(
        transition_potentials=tp, scan=(0, 0), detectors=detector, sites=atoms,
    )
    eels_2 = s2.transition_potential_scan(
        transition_potentials=tp, scan=(0, 0), detectors=detector, sites=atoms,
    )

    # The EELS diffraction pattern shape must equal the elastic one at the
    # same interpolation factor — that's the test that the cropping wiring
    # is on the right convention.
    assert eels_1.shape[-2:] == elastic_1.shape[-2:], (
        f"interp=1: EELS shape {eels_1.shape} vs elastic {elastic_1.shape}"
    )
    assert eels_2.shape[-2:] == elastic_2.shape[-2:], (
        f"interp=2: EELS shape {eels_2.shape} vs elastic {elastic_2.shape}"
    )
    # And the two interpolation factors must yield genuinely different
    # window shapes (i.e. the crop path is exercised, not a no-op).
    assert eels_1.shape[-2:] != eels_2.shape[-2:], (
        "interp=1 and interp=2 produced the same shape — crop path may be a "
        f"no-op (both {eels_1.shape[-2:]})"
    )

    # Output is non-zero and finite (sanity).
    arr_2 = np.asarray(eels_2.array)
    assert np.all(np.isfinite(arr_2))
    assert np.abs(arr_2).max() > 0
