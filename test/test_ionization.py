"""Tests for inelastic / core-loss simulation entry points.

These guard against regressions in the public API that the core-loss tutorial
depends on (see https://abtem.github.io/doc/user_guide/tutorials/core_loss.html).
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
from abtem.core.axes import OrdinalAxis, ThicknessAxis
from abtem.core.backend import get_array_module
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

from utils import (  # noqa: E402  -- device-gated markers
    devices,
    requires_gpu,
    synthetic_transition_potential,
    to_host_array,
)


# For fast_roll we parametrise over a backend *module* string ("numpy" /
# "cupy"); the abtem-level GPU dispatch tests use the standard device kwarg.
xp_params = [
    "numpy",
    pytest.param("cupy", marks=requires_gpu.marks),
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


def _make_probe(potential, device):
    p = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
    p.grid.match(potential)
    return p


def test_transition_potential_scan_is_defined_on_probe():
    """Guard against the Feb-2025 regression where the method was commented out."""
    assert hasattr(Probe, "transition_potential_scan")
    assert callable(Probe.transition_potential_scan)


@devices
def test_transition_potential_scan_builds_lazy_graph(si_atoms, device):
    """Wiring smoke test: lazy call returns a measurements object without raising,
    on both CPU and GPU backends."""
    potential = _make_si_potential(si_atoms, device)
    tp = synthetic_transition_potential(
        Z=14, gpts=potential.gpts, extent=potential.extent,
        n_transitions=2, energy=100e3, device=device,
    )
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


@devices
def test_transition_potential_scan_forwards_inelastic_kwargs(si_atoms, device):
    """double_channel / threshold must flow through to
    transition_potential_multislice_and_detect via **multislice_func_kwargs.

    Checked by effect rather than by building a graph: if either kwarg were
    dropped on the way, both calls would run with the driver's default and
    produce identical results.
    - threshold=0.5 keeps only the sites carrying half the probe overlap, so
      it must lose signal relative to threshold=1.0 (no filtering) -- the 8
      Si sites of the cell are not all under the probe at (0, 0).
    - double_channel=False discards the inelastic wave's elastic
      re-scattering through the slices below each site; with >1 slice that
      redistributes the detected intensity over the radial bins (the total
      is nearly conserved, the elastic propagation being unitary).
    """
    potential = _make_si_potential(si_atoms, device)
    assert potential.num_slices > 1
    tp = synthetic_transition_potential(
        Z=14, gpts=potential.gpts, extent=potential.extent,
        n_transitions=2, energy=100e3, device=device,
    )
    probe = _make_probe(potential, device)

    def run(**kwargs):
        result = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=tp,
            scan=(0, 0),
            detectors=abtem.FlexibleAnnularDetector(to_cpu=True),
            lazy=False,
            **kwargs,
        )
        return np.asarray(result.array)

    unfiltered = run(double_channel=False, threshold=1.0)
    filtered = run(double_channel=False, threshold=0.5)
    scale = np.abs(unfiltered).max()
    assert scale > 0
    assert filtered.sum() < unfiltered.sum() * (1 - 1e-3), (
        "threshold=0.5 had no effect: was it forwarded to the driver?"
    )

    double = run(double_channel=True, threshold=1.0)
    assert not np.allclose(double, unfiltered, rtol=1e-3, atol=1e-6 * scale), (
        "double_channel=True matches double_channel=False: was it forwarded?"
    )


@devices
def test_transition_potential_scan_accepts_grid_scan(si_atoms, device):
    """The tutorial uses GridScan + sites=... for the EELS-map calls."""
    potential = _make_si_potential(si_atoms, device)
    tp = synthetic_transition_potential(
        Z=14, gpts=potential.gpts, extent=potential.extent,
        n_transitions=2, energy=100e3, device=device,
    )
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


@devices
def test_transition_potential_scan_auto_sites_survive_potential_prebuild(
    si_atoms, device
):
    """Regression test for a bug where ``_prebuild_reused_potential`` (added for
    issue #339/#340) replaced the ``Potential`` with a bare ``PotentialArray``
    before auto-extracted ``sites`` were resolved from it. ``PotentialArray``
    carries no atoms, so ``_extract_scattering_sites`` raised ``ValueError``
    for any multi-chunk lazy scan (e.g. ``max_batch`` small enough to split the
    scan into more than one dask block) that didn't pass ``sites=`` explicitly.

    Also asserts the auto-extracted-sites result is bit-identical to both an
    unchunked scan and an explicit ``sites=`` scan, since the fix must resolve
    sites from the *original* potential rather than change what site set is
    used.
    """
    potential = _make_si_potential(si_atoms, device)
    tp = synthetic_transition_potential(
        Z=14, gpts=potential.gpts, extent=potential.extent,
        n_transitions=2, energy=100e3, device=device,
    )
    probe = _make_probe(potential, device)
    detector = abtem.FlexibleAnnularDetector()
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), fractional=True,
        potential=potential, endpoint=False, sampling=2.0,
    )

    unchunked = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, double_channel=False, lazy=False,
    ).compute()

    chunked_auto_sites = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, double_channel=False, max_batch=1, lazy=True,
    ).compute()

    chunked_explicit_sites = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, sites=si_atoms, double_channel=False,
        max_batch=1, lazy=True,
    ).compute()

    if device != "cpu":
        chunked_auto_sites = chunked_auto_sites.to_cpu()
        chunked_explicit_sites = chunked_explicit_sites.to_cpu()
        unchunked = unchunked.to_cpu()

    # On the CPU the chunked and unchunked paths issue the same operations in
    # the same order, so bit-identity is a real guarantee worth asserting. An
    # accelerator is free to reassociate the accumulation behind a batch, which
    # moves the last ULP without saying anything about the chunking logic this
    # test is about -- compare at the precision the device actually offers.
    if device == "cpu":
        assert_equal = np.testing.assert_array_equal
    else:
        def assert_equal(actual, desired):
            np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=0.0)

    assert_equal(chunked_auto_sites.array, unchunked.array)
    assert_equal(chunked_auto_sites.array, chunked_explicit_sites.array)


@devices
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
    xp = get_array_module(device)

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



@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
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


@devices
@pytest.mark.parametrize("double_channel", [False, True])
def test_prism_eels_beam_basis_matches_multislice_at_interp_1(device, double_channel):
    """The beam-basis reduction (GitHub issue abTEM/abTEM#293) at
    interpolation=(1,1), window=cell=full grid, reproduces
    Probe.transition_potential_scan -- bit-exact for both single- and
    double-channel. This is the validation gate for the normalisation
    derivation recorded in the project_prism_eels_beam_basis_convention
    memory note: ``recip[q] = N * sum_r conj(S2[q, r]) * psi[r]``.
    """
    from abtem.inelastic.core_loss import prism_transition_potential_scan_beam_basis

    unit_atoms = ase.build.bulk("Si", cubic=True)
    atoms = unit_atoms * (1, 1, 2)
    slice_thickness = float(unit_atoms.cell[2, 2])

    potential = abtem.Potential(
        atoms, gpts=(32, 32), slice_thickness=slice_thickness, device=device
    )

    energy = 100e3
    semiangle_cutoff = 20.0

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
    )

    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    scan = abtem.GridScan(
        start=(0, 0), end=(unit_atoms.cell[0, 0], unit_atoms.cell[1, 1]),
        sampling=0.5, endpoint=False,
    )

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    res_multislice = probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=double_channel,
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
    res_beam_basis = prism_transition_potential_scan_beam_basis(
        s_matrix,
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=double_channel,
    )
    arr_beam_basis = np.asarray(res_beam_basis.array)

    assert arr_multislice.shape == arr_beam_basis.shape
    # The signal is O(1e-9) (sigma^2 * |H|^2), so an absolute atol must be
    # scaled to it -- a fixed atol=1e-6 made the comparison vacuous (a x10
    # amplitude error passed). "Bit-exact" here means agreement to float32
    # round-off: both paths are the same linear algebra in a different
    # association order, measured max relative error ~9e-7 (~7 float32
    # ulps) for both channels on CPU. rtol=1e-4 leaves ~100x headroom for
    # the GPU backends' different FFT/GEMM summation order (not measured
    # there), while any physical normalisation error (>= 1e-3) still fails.
    scale = np.abs(arr_multislice).max()
    assert scale > 0
    np.testing.assert_allclose(
        arr_beam_basis, arr_multislice, rtol=1e-4, atol=1e-6 * scale
    )


@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
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


@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(64, 64), extent=potential.extent, energy=energy, n_transitions=2,
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


@devices
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


@devices
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


@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
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
    thickness_axes = [
        i for i, a in enumerate(res_ms.axes_metadata) if isinstance(a, ThicknessAxis)
    ]
    assert len(thickness_axes) == 1
    thickness_axis = thickness_axes[0]
    assert arr_ms.shape[thickness_axis] == n_exit
    # exit_planes[0] == -1 is the entrance plane (t = 0): no material has
    # been traversed, so no ionisation has happened and both drivers must
    # record exactly zero there; every later plane has inelastic signal.
    assert potential.exit_planes[0] == -1
    for arr in (arr_ms, arr_prism):
        assert np.all(np.take(arr, 0, axis=thickness_axis) == 0)
        for i in range(1, n_exit):
            assert np.abs(np.take(arr, i, axis=thickness_axis)).max() > 0
    np.testing.assert_allclose(arr_prism, arr_ms, rtol=1e-5, atol=0)


@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
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


@devices
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

    tp = synthetic_transition_potential(
        Z=14, gpts=(32, 32), extent=potential.extent, energy=energy, n_transitions=2,
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


@devices
def test_transition_potential_scan_crystal_double_channel_matches_manual(device):
    """Double-channel + CrystalPotential is the intersection that triggers the
    TransmissionFunction dedup in transition_potential_multislice_and_detect
    (the slice_cache is only built when double_channel=True, and dedup only
    fires when generate_slices yields repeated object identities). Verify the
    deduplicated cache still produces results numerically equivalent to the
    manually-tiled Potential path.
    """
    xp = get_array_module(device)

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
@pytest.mark.slow
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


@pytest.mark.skipif("gpaw" not in sys.modules, reason="requires gpaw")
@pytest.mark.filterwarnings("ignore:the cell:RuntimeWarning")
@pytest.mark.slow
def test_orbital_filling_factor_is_spin_only_not_full_shell_degeneracy():
    """Regression test for a (2*l+1) orbital-degeneracy double-count.

    ``SubshellTransitions.get_transitions()`` already realises a subshell's
    orbital degeneracy explicitly: it builds one distinct bound state per
    ``ml`` in ``range(-l, l+1)`` and ``TransitionPotential.build()`` sums
    their contributions incoherently. ``orbital_filling_factor`` must
    therefore contribute only the *remaining* spin degeneracy (2), not the
    full subshell occupancy ``4*l+2 = spin(2) * orbital(2*l+1)`` -- applying
    ``4*l+2`` per already-explicit ``ml`` inflates the total by exactly
    ``(2*l+1)``. This was invisible for every K edge (l=0), where
    ``4*l+2`` reduces to the spin-only factor of 2, which is exactly why it
    went undetected: only checked directly against an independent
    tabulation (Bote & Salvat) does a p- or d-subshell reveal it, at 3x and
    5x respectively.

    Checked here without any external data: for a single explicit bound
    ``ml`` state, turning ``orbital_filling_factor`` on must scale the
    intensity by exactly 2 (spin), regardless of ``l`` -- not by ``4*l+2``.
    """
    from abtem.inelastic.core_loss import SubshellTransitions, TransitionPotential

    for Z, n, l in [(14, 1, 0), (14, 2, 1), (22, 3, 2)]:
        transitions = SubshellTransitions(Z, n, l, epsilon=25.0)
        one_ml_transitions = [
            t for t in transitions.get_transitions() if t[0].ml == 0
        ]
        assert len(one_ml_transitions) > 0

        kwargs = dict(extent=6.0, gpts=32, energy=100e3, double_channel=False)
        with_factor = TransitionPotential(
            Z, one_ml_transitions, orbital_filling_factor=True, **kwargs,
        ).build()
        without_factor = TransitionPotential(
            Z, one_ml_transitions, orbital_filling_factor=False, **kwargs,
        ).build()

        intensity_with = float((np.abs(with_factor.array) ** 2).sum())
        intensity_without = float((np.abs(without_factor.array) ** 2).sum())

        assert intensity_with == pytest.approx(2.0 * intensity_without, rel=1e-6), (
            f"l={l}: orbital_filling_factor should apply spin degeneracy (2x "
            f"intensity), not the full shell occupancy "
            f"({4 * l + 2}x intensity, from the old 4*l+2 formula)"
        )


def _probe_waves(gpts=(32, 32), extent=(8.0, 8.0), device="cpu"):
    probe = Probe(
        energy=100e3, extent=extent, gpts=gpts, semiangle_cutoff=30, device=device
    )
    return probe.build(lazy=False)


@devices
def test_filter_sites_aligns_mask_with_mixed_element_atoms(device):
    """An Atoms input is subset to the transition element by validate_sites;
    the survival mask must index that subset, not the original object."""
    tp = synthetic_transition_potential(
        Z=5, gpts=(32, 32), extent=(8.0, 8.0), n_transitions=2, device=device,
    )
    waves = _probe_waves(device=device)
    atoms = ase.Atoms(
        "BN", positions=[(4.0, 4.0, 0.0), (1.0, 1.0, 0.0)], cell=(8, 8, 4)
    )

    filtered = tp.filter_sites(waves, atoms, threshold=1e-12)

    # Expected from the setup, not from a run: validate_sites keeps only the
    # Z=5 (B) atom, whose xy position is (4, 4). The synthetic transition
    # potential is white noise in reciprocal space, so its local potential
    # |ifft2(H)|^2 covers the whole cell and the site's overlap with the
    # (normalised, unit-intensity) probe is O(1) >> 1e-12: the B site must
    # survive, and the N site must never appear. A mask misaligned with the
    # original Atoms (length 2) would raise or return the N position (1, 1);
    # a filter that drops everything returns an empty array.
    assert isinstance(filtered, np.ndarray)
    np.testing.assert_array_equal(filtered, np.array([[4.0, 4.0]]))


def _symmetric_gaussian_transition_potential(gpts, extent, width=0.5, device="cpu"):
    """A single-transition potential whose real-space form is a Gaussian
    centred on pixel (0, 0) of a periodic grid.

    Its local potential |H(r)|^2 is centrosymmetric, V(r) = V(-r), as for
    any real transition potential (|Y_lm|^2 is inversion-symmetric). That
    matters because ``absolute_threshold`` ranks a *convolution* of V with
    |psi|^2 while ``filter_sites`` evaluates a *correlation*; the two agree
    only for centrosymmetric V.
    """
    from abtem.core.backend import copy_to_device
    from abtem.core.utils import get_dtype

    x = np.fft.fftfreq(gpts[0], 1 / gpts[0]) * extent[0] / gpts[0]
    y = np.fft.fftfreq(gpts[1], 1 / gpts[1]) * extent[1] / gpts[1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    H = np.exp(-(X**2 + Y**2) / (2 * width**2))
    array = np.fft.fft2(H)[None].astype(get_dtype(complex=True))
    array = copy_to_device(array, device)
    return TransitionPotentialArray(
        Z=5,
        array=array,
        energy=100e3,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0,))],
        metadata={"Z": 5, "n": 1, "l": 0},
    )


def _pixel_dense_boron_sites(gpts, extent, z=0.5):
    """One B site on every pixel of the grid, all in the first slice."""
    ij = np.stack(
        np.meshgrid(np.arange(gpts[0]), np.arange(gpts[1]), indexing="ij"), -1
    ).reshape(-1, 2)
    sampling = np.array(extent) / np.array(gpts)
    positions = np.concatenate(
        [ij * sampling, np.full((len(ij), 1), z)], axis=1
    )
    return ij, ase.Atoms(
        numbers=[5] * len(ij), positions=positions, cell=(*extent, 1.0)
    )


def _oracle_overlaps(tp, waves, ij):
    """Overlap of each pixel site with the probe, straight from the
    definition: sum_r V(r - s) |psi(r)|^2, with V the transition
    potential's local potential |ifft2(H)|^2, evaluated with np.roll in
    float64 (independent of filter_sites' fast_roll and of
    absolute_threshold's FFT convolution)."""
    V = np.abs(np.fft.ifft2(to_host_array(tp).astype(np.complex128))) ** 2
    V = V.sum(0)
    psi2 = np.abs(to_host_array(waves).astype(np.complex128)) ** 2
    return np.array([(np.roll(V, tuple(s), (0, 1)) * psi2).sum() for s in ij])


@devices
def test_threshold_retains_requested_fraction_of_overlap(device):
    """``threshold=t`` means: keep the sites that together carry at least a
    fraction t of the probe/local-potential overlap (``absolute_threshold``
    ranks the overlap at every pixel, accumulates it in descending order and
    cuts where the cumulative fraction reaches t). With a site on every
    pixel the retained sites are exactly those ranked pixels, so their
    summed overlap must be >= t of the total, and for t < 1 some must be
    dropped.

    The cut used to be the overlap of the pixel *before* the one where the
    cumulative fraction reaches t, kept strictly above, so the retained set
    stopped ~2 pixels short (t=0.5 retained 0.487, t=0.1 retained 0.088);
    and float32 round-off between the FFT ranking and filter_sites' direct
    sum split groups of symmetry-tied pixels at the cut (t=0.99 ~1e-4
    short). 0.99 and 0.999 cover that tie handling."""
    gpts, extent = (32, 32), (8.0, 8.0)
    tp = _symmetric_gaussian_transition_potential(gpts, extent, device=device)
    waves = _probe_waves(gpts=gpts, extent=extent, device=device)
    ij, sites = _pixel_dense_boron_sites(gpts, extent)
    overlaps = _oracle_overlaps(tp, waves, ij)
    total = overlaps.sum()

    for t in (0.001, 0.1, 0.5, 0.9, 0.99, 0.999):
        kept = tp.filter_sites(
            waves, sites, threshold=tp.absolute_threshold(waves, t)
        )
        kept_ij = np.rint(kept / np.array(tp.sampling)).astype(int)
        kept_overlap = overlaps[kept_ij[:, 0] * gpts[1] + kept_ij[:, 1]].sum()
        assert len(kept) < len(ij), f"t={t}: no site was dropped"
        assert kept_overlap >= t * total, (
            f"t={t}: retained only {kept_overlap / total:.4f} of the overlap"
        )


@devices
def test_threshold_below_the_top_pixel_fraction_keeps_the_top_site(device):
    """A threshold below the largest single-pixel overlap fraction asks for
    exactly the single most-overlapping site. The cut index used to wrap to
    -1 there -- the *smallest* overlap -- so filter_sites kept 1023 of 1024
    sites, silently running the full unfiltered cost."""
    gpts, extent = (32, 32), (8.0, 8.0)
    tp = _symmetric_gaussian_transition_potential(gpts, extent, device=device)
    waves = _probe_waves(gpts=gpts, extent=extent, device=device)
    ij, sites = _pixel_dense_boron_sites(gpts, extent)
    overlaps = _oracle_overlaps(tp, waves, ij)
    assert overlaps.max() / overlaps.sum() > 1e-3

    kept = tp.filter_sites(
        waves, sites, threshold=tp.absolute_threshold(waves, 1e-3)
    )
    kept_ij = np.rint(kept / np.array(tp.sampling)).astype(int)
    kept_flat = kept_ij[:, 0] * gpts[1] + kept_ij[:, 1]
    # The top pixel is unique (the next-ranked group of four ties sits ~15%
    # below it), so it survives alone.
    top_two = np.sort(overlaps)[::-1][:2]
    assert top_two[1] < top_two[0] * (1 - 1e-3)
    np.testing.assert_array_equal(kept_flat, [np.argmax(overlaps)])


@devices
def test_threshold_drops_sites_and_bounds_the_eels_signal(device):
    """End-to-end purpose of ``threshold``: through the EELS driver, t < 1
    must actually drop sites (a smaller integrated signal), dropping more
    as t decreases, yet retain at least the fraction t of the signal, while
    t = 1 (the driver's default) is the unfiltered result.

    Setup: one slice of vacuum, a B site on every pixel, a centrosymmetric
    Gaussian transition potential. With one slice and single-channel, each
    site's integrated EELS intensity is sum_q |FT[H(r - s) psi(r)]|^2, i.e.
    by Parseval its overlap sum_r |H(r - s)|^2 |psi(r)|^2 (up to the
    antialias aperture), so the retained signal fraction tracks the
    retained-overlap fraction, which is >= t.

    The driver used to rank the cut once on the entrance wave and apply it
    to the wave at the slice; even one 1 A vacuum slice of propagation then
    pushed the symmetry-tied group of sites at the cut below it together:
    t=0.1 retained 0.048 of the signal, t=0.5 0.446, and t=0.001 nothing.
    """
    gpts, extent = (32, 32), (8.0, 8.0)
    tp = _symmetric_gaussian_transition_potential(gpts, extent, device=device)
    potential = abtem.Potential(
        ase.Atoms(cell=(*extent, 1.0)),
        gpts=gpts,
        slice_thickness=1.0,
        device=device,
    )
    assert potential.num_slices == 1
    ij, sites = _pixel_dense_boron_sites(gpts, extent)
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=30, device=device)
    probe.grid.match(potential)

    # t = 1 disables filtering at the source: the absolute cut is 0 and
    # filter_sites keeps every site. (Comparing a threshold=1.0 run against
    # a default run would be vacuous -- the driver's default *is* 1.0.)
    waves = probe.build(scan=(4.0, 4.0), lazy=False)
    assert tp.absolute_threshold(waves, 1.0) == 0.0
    assert len(tp.filter_sites(waves, sites, threshold=0.0)) == len(ij)

    def integrated(**kwargs):
        result = probe.transition_potential_scan(
            potential=potential,
            transition_potentials=tp,
            scan=(4.0, 4.0),
            detectors=abtem.FlexibleAnnularDetector(to_cpu=True),
            sites=sites,
            double_channel=False,
            lazy=False,
            **kwargs,
        )
        return float(np.asarray(result.array).sum())

    unfiltered = integrated(threshold=1.0)
    assert unfiltered > 0

    thresholds = (0.001, 0.1, 0.5, 0.9, 0.99)
    ratios = [integrated(threshold=t) / unfiltered for t in thresholds]
    assert 0 < ratios[0] and all(np.diff(ratios) > 0) and ratios[-1] < 1
    for t, ratio in zip(thresholds, ratios):
        # Measured: ratio = 0.048, 0.210, 0.613, 0.902, 0.990, each within
        # ~1e-3 (the antialias aperture) of the retained-overlap fraction.
        assert ratio >= t, f"t={t}: retained only {ratio:.4f} of the signal"


def test_local_potential_device_cache_survives_use_but_not_pickle():
    """The per-device local-potential cache must not ride through pickle into
    dask task graphs."""
    import pickle

    tp = synthetic_transition_potential(
        Z=5, gpts=(32, 32), extent=(8.0, 8.0), n_transitions=2,
    )
    waves = _probe_waves()
    tp.filter_sites(waves, np.array([[4.0, 4.0]]), threshold=1e-12)
    assert tp._local_potential_device_cache is not None

    clone = pickle.loads(pickle.dumps(tp))
    assert clone._local_potential_device_cache is None
    # And the clone repopulates its own cache transparently.
    clone.filter_sites(waves, np.array([[4.0, 4.0]]), threshold=1e-12)
    assert clone._local_potential_device_cache is not None


@requires_gpu
def test_local_potential_device_cache_keys_on_the_arrays_device():
    """The cache key must carry the concrete GPU id, read off the array
    itself -- never a bare 'gpu' bucket that could alias devices."""
    tp = synthetic_transition_potential(
        Z=5, gpts=(32, 32), extent=(8.0, 8.0), n_transitions=2,
    )
    like = cp.zeros((32, 32), dtype=cp.complex64)

    on_device = tp._local_potential_on_device(like)
    key, cached = tp._local_potential_device_cache

    assert key == ("gpu", int(like.device.id))
    assert cached is on_device
    assert tp._local_potential_on_device(like) is on_device  # served from cache

    # A cpu request replaces the slot with a cpu-keyed entry.
    tp._local_potential_on_device(np.zeros((32, 32), dtype=np.complex64))
    assert tp._local_potential_device_cache[0] == "cpu"
