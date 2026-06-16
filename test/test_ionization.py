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




@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_eels_matches_multislice_eels_at_interp_1(device):
    """SMatrix.transition_potential_scan at interpolation=(1,1) reproduces
    Probe.transition_potential_scan on a small Si cell."""
    unit_atoms = ase.build.bulk("Si", cubic=True)
    reps = (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])
    atoms = unit_atoms * reps

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, device=device
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

    detector = abtem.PixelatedDetector(max_angle=40, to_cpu=True)

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
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

    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=1,
        downsample=False,
        device=device,
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
    np.testing.assert_allclose(arr_prism, arr_multislice, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_eels_double_channel_matches_multislice_at_interp_1(device):
    """Double-channel PRISM-EELS at interpolation=(1,1) reproduces
    Probe.transition_potential_scan with double_channel=True."""
    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, device=device
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

    detector = abtem.PixelatedDetector(max_angle=40, to_cpu=True)

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    res_multislice = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        sites=atoms,
        double_channel=True,
        lazy=False,
    ).compute()
    arr_multislice = np.asarray(res_multislice.array)

    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=1,
        downsample=False,
        device=device,
    )
    res_prism = s_matrix.transition_potential_scan(
        transition_potentials=tp,
        scan=(0, 0),
        detectors=detector,
        sites=atoms,
        double_channel=True,
    )
    arr_prism = np.asarray(res_prism.array)

    assert arr_multislice.shape == arr_prism.shape
    np.testing.assert_allclose(arr_prism, arr_multislice, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_smatrix_transition_potential_scan_interp_2_produces_windowed_output(device):
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
        atoms, gpts=(64, 64), slice_thickness=slice_thickness, device=device
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

    detector = abtem.PixelatedDetector(max_angle=30, to_cpu=True)

    s1 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=1, downsample=False, device=device,
    )
    s2 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=2, downsample=False, device=device,
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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_eels_interp_2_accuracy_vs_multislice(device):
    """Stage 3b: the total integrated EELS signal at interp=2 should be
    within ~10% of the multislice reference (Brown et al. Sec. IV B).

    We compare the angle-integrated spatial map because the smaller FFT grid
    at interp>1 redistributes intensity among angular bins — the same effect
    as in elastic PRISM.  The total signal is the physically meaningful
    quantity for EELS mapping.

    The window must be large enough to capture the transition potential.
    A random (non-localized) TP requires a large window; with gpts=128 and
    interp=2, window_gpts=64 gives adequate coverage.
    """
    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])
    potential = abtem.Potential(
        atoms, gpts=(128, 128), slice_thickness=slice_thickness, device=device
    )

    energy = 100e3
    semiangle_cutoff = 20.0

    from abtem.inelastic.core_loss import energy2sigma
    rng = np.random.default_rng(42)
    sampling = tuple(e / g for e, g in zip(potential.extent, (128, 128)))
    y = np.arange(128).astype(np.float32) * sampling[0]
    x = np.arange(128).astype(np.float32) * sampling[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    sigma_gauss = 0.5
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_gauss ** 2)).astype(np.float32)
    raw = (
        rng.standard_normal((2, 128, 128))
        + 1j * rng.standard_normal((2, 128, 128))
    ).astype(np.complex64)
    real_space_tp = raw * gauss[None]
    tp_array = np.fft.fft2(real_space_tp) / energy2sigma(energy)
    tp_array = tp_array.astype(np.complex64)
    tp = TransitionPotentialArray(
        Z=14,
        array=tp_array,
        energy=energy,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )

    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0),
        end=(unit_atoms.cell[0, 0], unit_atoms.cell[1, 1]),
        sampling=0.5, endpoint=False,
    )

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    ms = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp,
        scan=scan, detectors=detector, sites=atoms,
        double_channel=False, lazy=False,
    ).compute()

    s2 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=2, downsample=False, device=device,
    )
    pr = s2.transition_potential_scan(
        transition_potentials=tp, scan=scan, detectors=detector, sites=atoms,
        double_channel=False,
    )

    ms_map = np.asarray(ms.array).sum(axis=(-2, -1))
    pr_map = np.asarray(pr.array).sum(axis=(-2, -1))

    total_error = np.sqrt(np.sum((ms_map - pr_map) ** 2) / np.sum(ms_map ** 2))
    assert total_error < 0.10, (
        f"PRISM-EELS interp=2 total integrated error {total_error:.1%} exceeds 10%"
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_eels_inelastic_crop_window(device):
    """The ``inelastic_crop`` knob (Brown et al. Sec. IV B) decouples the
    transition-potential scatter window from the interpolation factor.

    Invariants:
      * a window >= the PRISM cell (``extent / interpolation``) reproduces the
        default ``None`` result exactly — the centered embed is a no-op and
        over-large requests are clamped to the cell (with a warning);
      * a tighter window stays the same shape but changes the values
        (transition-potential truncation), and never crashes.
    """
    import warnings

    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])
    potential = abtem.Potential(
        atoms, gpts=(128, 128), slice_thickness=slice_thickness, device=device
    )

    energy = 100e3
    semiangle_cutoff = 20.0

    # Localized (Gaussian-enveloped) transition potential so that the scatter
    # carries real signal and a tighter window measurably truncates it.
    from abtem.inelastic.core_loss import energy2sigma
    rng = np.random.default_rng(7)
    sampling = tuple(e / g for e, g in zip(potential.extent, (128, 128)))
    yy, xx = np.meshgrid(
        np.arange(128).astype(np.float32) * sampling[0],
        np.arange(128).astype(np.float32) * sampling[1],
        indexing="ij",
    )
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.5 ** 2)).astype(np.float32)
    raw = (
        rng.standard_normal((2, 128, 128))
        + 1j * rng.standard_normal((2, 128, 128))
    ).astype(np.complex64)
    tp_array = (np.fft.fft2(raw * gauss[None]) / energy2sigma(energy)).astype(
        np.complex64
    )
    tp = TransitionPotentialArray(
        Z=14, array=tp_array, energy=energy, extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )

    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0), end=(unit_atoms.cell[0, 0], unit_atoms.cell[1, 1]),
        sampling=0.5, endpoint=False,
    )

    s2 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=2, downsample=False, device=device,
    )
    cell_extent = potential.extent[0] / 2  # extent / interpolation

    def run(inelastic_crop):
        res = s2.transition_potential_scan(
            transition_potentials=tp, scan=scan, detectors=detector,
            sites=atoms, double_channel=False, inelastic_crop=inelastic_crop,
        )
        return np.asarray(res.array)

    base = run(None)

    # >= cell: clamped to the cell, embed is a no-op -> identical to None.
    at_cell = run(cell_extent)
    assert np.allclose(at_cell, base, rtol=1e-6), (
        "inelastic_crop == PRISM cell should reproduce the default"
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        over = run(2 * cell_extent)
    assert np.allclose(over, base, rtol=1e-6), "over-large crop must clamp"
    assert any("PRISM cell" in str(wi.message) for wi in w), (
        "clamping should warn"
    )

    # Tighter window: same shape, but a measurable change (TP truncation).
    tight = run(cell_extent / 2)
    assert tight.shape == base.shape
    rel_change = np.sqrt(
        np.sum((tight - base) ** 2) / np.sum(base ** 2)
    )
    assert rel_change > 1e-3, (
        f"a tighter inelastic_crop should change the result "
        f"(relative change {rel_change:.2e})"
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("double_channel", [False, True])
def test_prism_eels_exit_planes_match_multislice(double_channel, device):
    """PRISM-EELS with exit_planes produces the same thickness-series as
    multislice at interpolation=(1,1)."""
    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness,
        exit_planes=1, device=device,
    )
    n_exit = len(potential.exit_planes)
    assert n_exit > 1, f"expected multiple exit planes, got {potential.exit_planes}"

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

    detector = abtem.PixelatedDetector(max_angle=40, to_cpu=True)

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    res_ms = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp,
        scan=(0, 0), detectors=detector, sites=atoms,
        double_channel=double_channel, lazy=False,
    ).compute()
    arr_ms = np.asarray(res_ms.array)

    s_matrix = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=1, downsample=False, device=device,
    )
    res_prism = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=(0, 0), detectors=detector,
        sites=atoms, double_channel=double_channel,
    )
    arr_prism = np.asarray(res_prism.array)

    assert arr_ms.shape == arr_prism.shape, (
        f"shape mismatch: multislice {arr_ms.shape} vs PRISM {arr_prism.shape}"
    )
    # The entrance plane (exit_planes[0] == -1) records elastic waves in the
    # multislice path but zeros in PRISM (no pre-scatter reduction is wired).
    # Compare only the physical exit planes where inelastic signal is produced.
    if potential.exit_planes[0] == -1:
        arr_ms = arr_ms[1:]
        arr_prism = arr_prism[1:]
    np.testing.assert_allclose(arr_prism, arr_ms, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("ensemble_mean", [True, False])
def test_prism_eels_frozen_phonons_match_multislice(ensemble_mean, device):
    """PRISM-EELS with frozen phonons matches multislice-EELS at interp=1."""
    from abtem import FrozenPhonons

    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])

    fp = FrozenPhonons(
        atoms, num_configs=2, sigmas=0.1, seed=42, ensemble_mean=ensemble_mean
    )
    potential = abtem.Potential(
        fp, gpts=(32, 32), slice_thickness=slice_thickness, device=device,
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

    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0), end=potential.extent,
        gpts=(2, 2), endpoint=False,
    )

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    res_ms = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=False,
        lazy=True,
    ).compute()
    arr_ms = np.asarray(res_ms.array)

    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=1,
        downsample=False,
        device=device,
    )
    res_prism = s_matrix.transition_potential_scan(
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
    )
    arr_prism = np.asarray(res_prism.array)

    assert arr_ms.shape == arr_prism.shape, (
        f"shape mismatch: multislice {arr_ms.shape} vs PRISM {arr_prism.shape}"
    )
    np.testing.assert_allclose(arr_prism, arr_ms, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_eels_lazy_matches_eager(device):
    """PRISM-EELS with lazy=True produces the same result as lazy=False."""
    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, device=device
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

    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0), end=potential.extent,
        gpts=(4, 4), endpoint=False,
    )

    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=1,
        downsample=False,
        device=device,
    )

    res_eager = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan,
        detectors=detector, sites=atoms, lazy=False,
    )
    arr_eager = np.asarray(res_eager.array)

    res_lazy = s_matrix.transition_potential_scan(
        transition_potentials=tp, scan=scan,
        detectors=detector, sites=atoms, lazy=True,
    ).compute()
    arr_lazy = np.asarray(res_lazy.array)

    assert arr_eager.shape == arr_lazy.shape
    np.testing.assert_allclose(arr_lazy, arr_eager, rtol=1e-5, atol=0)
