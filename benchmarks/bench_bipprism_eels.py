"""Benchmark: multislice vs PRISM-EELS (beam-basis) vs BiP-PRISM on SrTiO3.

Four reduction backends for core-loss STEM-EELS, all detected through the
SAME fixed-angle annular detector so timings/memory are directly comparable:

  - multislice   : abtem.Probe.transition_potential_scan (ground truth)
  - real_space   : SMatrix.transition_potential_scan(reduction="real_space")
                   (the production windowed PRISM-EELS driver, PR #289 MVP)
  - beam_basis   : SMatrix.transition_potential_scan(reduction="beam_basis")
                   (exact dual-scattering-matrix reduction, GitHub #293 --
                   what the abTEM project calls "PRISM-EELS" in its beam-basis
                   form; S2 is built over the *detector-collection disk*, not
                   the full native grid, via ``collection_angle`` -- without
                   that the O(gpts^2) full-grid S2 build is intractable)
  - bipartite    : SMatrix.transition_potential_scan(reduction="bipartite")
                   (BiP-PRISM, PR #334: S1 and S2 built on sparse hex-ring
                   *parent* beams and reconstructed per-atom; ``partitions_s1``
                   / ``partitions_s2`` control the parent count)

Three sweeps (mirrors benchmarks/bench_prism_eels.py's style/infra):
  1. Scan positions (fixed thickness, fixed cell)     -- amortisation of the
     S-matrix build over many probe positions.
  2. Specimen thickness (fixed scan, fixed cell)       -- per-slice cost.
  3. Lateral cell size / field of view (fixed scan,     -- THE stress test for
     fixed thickness, ``px_per_unit`` held constant so   BiP-PRISM: at fixed
     real-space sampling doesn't change with cell size)  ``collection_angle``,
                                                          the number of
     reciprocal-grid pixels inside the detector disk grows with cell size
     (denser k-sampling, dk = 1/extent) for BOTH real_space's window and
     beam_basis's per-pixel reverse-multislice S2 build -- while BiP-PRISM's
     parent count (``partitions_s2``) stays fixed by construction. This is
     where BiP-PRISM's memory/time advantage over exact beam_basis should
     show up most clearly.

Peak memory: tracemalloc (CPU) or a CuPy MemoryHook (GPU), exactly as in
bench_prism_eels.py. Every timed call is preceded by one untimed warm-up
call (see ``_measure``) so CuPy kernel JIT/autotune on the first call of a
new shape doesn't pollute the measurement -- at small problem sizes that
one-time cost can dwarf the real compute and make a sweep look flat/noisy.

Problem sizes (nz, scan grid, sc_values) are set for GPU-scale runs, not
quick smoke tests -- sweep 2 reaches 256 slices (~1000 A) and sweep 3
reaches sc=16 (512x512 gpts), both at a fixed 4096-position (64x64) scan --
above the scan-position crossover sweep 1 shows, so PRISM's per-slice /
per-cell-size behaviour isn't confounded by still being amortisation-bound.
On CPU, or to iterate quickly, pass explicit smaller values to
sweep_scan_positions/sweep_thickness/sweep_cell_size directly.

Each variant is tracked independently within a sweep (see the ``alive``
dict threaded through ``_run_all_variants``): once a variant fails at some
size it's marked dead and skipped (recorded as NaN) for the rest of that
sweep, while variants that haven't failed yet keep going to larger sizes.
A sweep only stops once every variant has failed.

Usage
-----
    python bench_bipprism_eels.py [device] [sc] [interp] \
        [partitions_s1] [partitions_s2] [collection_angle]
    python bench_bipprism_eels.py gpu 4 4 4 4 25.0

Notes
-----
- ``beam_basis`` and ``bipartite`` require: single exit plane, no
  frozen-phonon ensemble, ``downsample=False`` -- all satisfied by the
  CrystalPotential setup below (repetitions only, no frozen phonons).
- ``beam_basis`` without ``collection_angle`` builds S2 over the FULL native
  grid (O(gpts^2) reverse-multislice propagations) -- this script always
  passes ``collection_angle`` to keep it tractable, and uses a fixed-angle
  ``AnnularDetector`` (not ``FlexibleAnnularDetector``) across ALL four
  variants so they detect exactly the same physical quantity.
- The cell-size sweep's top end (sc=16, gpts=512x512) is expected to push
  ``real_space``/``beam_basis`` toward (or past) GPU memory limits before
  ``bipartite`` -- that gap IS the result BiP-PRISM is for. A GPU OOM
  there is caught by ``_run_one`` and recorded as NaN for that variant only
  (see the per-variant ``alive`` tracking above); ``bipartite`` keeps
  running past it. Trim ``sc_values`` in ``sweep_cell_size`` if it's taking
  too long or OOMing earlier than you'd like to see.
"""
import gc
import sys
import time
import tracemalloc

import dask
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray, energy2sigma

# timed_ms uses lazy=True + .compute() so Probe.build's max_batch chunking
# actually kicks in (see timed_ms below); the default threaded scheduler would
# run that computation on a worker thread the CuPy MemoryHook doesn't see,
# silently reporting 0 MB for multislice. Force the synchronous scheduler
# (same fix bench_prism_eels.py's sibling, prism_eels_cell_size.py, uses) so
# .compute() runs in-thread where both the timer and the hook are active.
dask.config.set(scheduler="synchronous")

# abTEM defaults diagnostics.progress_bar to "tqdm" and prints one on every
# .compute() (timed_ms) -- with many sweep points that interleaves garbage
# into piped/redirected stdout. Disable both progress-bar mechanisms
# (dask's, and the separate per-task one some multislice loops use).
abtem.config.set(
    {"diagnostics.progress_bar": False, "diagnostics.task_progress": False}
)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
sc = int(sys.argv[2]) if len(sys.argv) > 2 else (4 if device == "gpu" else 2)
interp = int(sys.argv[3]) if len(sys.argv) > 3 else (4 if device == "gpu" else 4)
partitions_s1 = int(sys.argv[4]) if len(sys.argv) > 4 else 4
partitions_s2 = int(sys.argv[5]) if len(sys.argv) > 5 else 4
detector_outer = 20.0  # mrad, fixed physical detector across all variants
collection_angle = (
    float(sys.argv[6]) if len(sys.argv) > 6 else detector_outer + 5.0
)

# ---------------------------------------------------------------------------
# GPU-specific helpers (conditional import) -- identical to bench_prism_eels.py
# ---------------------------------------------------------------------------
if device == "gpu":
    import cupy as cp

    _mempool = cp.get_default_memory_pool()
    _pinned_pool = cp.get_default_pinned_memory_pool()

    def _gpu_sync():
        cp.cuda.Device().synchronize()

    class _PeakMemoryHook(cp.cuda.MemoryHook):
        def __init__(self, baseline):
            self.peak = baseline

        def malloc_postprocess(self, device_id, size, mem_size, mem_ptr, pmem_id):
            used = _mempool.used_bytes()
            if used > self.peak:
                self.peak = used

    def _measure_peak_gpu(func, *args, **kwargs):
        _gpu_sync()
        gc.collect()
        _mempool.free_all_blocks()
        _pinned_pool.free_all_blocks()
        _gpu_sync()

        baseline = _mempool.used_bytes()
        hook = _PeakMemoryHook(baseline)
        with hook:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            _gpu_sync()
            elapsed = time.perf_counter() - t0

        peak_mb = (hook.peak - baseline) / 1024**2
        return result, elapsed, max(peak_mb, 0.0)


def _measure(func, *args, **kwargs):
    """Dispatch CPU (tracemalloc) vs GPU (CuPy hook) timing + peak memory.

    Runs ``func`` once first and discards the result -- a cold first call
    pays for CuPy kernel JIT/autotune and any lazy CUDA context setup, which
    at small problem sizes can dwarf the actual compute and make timings
    look flat/noisy across a sweep. The warm-up eats that cost once so the
    timed call reflects steady-state performance.
    """
    func(*args, **kwargs)
    if device == "gpu":
        result, elapsed, peak = _measure_peak_gpu(func, *args, **kwargs)
    else:
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = peak_bytes / 1024**2
    return result, elapsed, peak


# ---------------------------------------------------------------------------
# SrTiO3 unit cell (perovskite, a = 3.905 A)
# ---------------------------------------------------------------------------
a = 3.905
energy = 200e3
semiangle_cutoff = 25.0
px_per_unit = 32  # A/px held fixed as sc grows (divisible by interp 1,2,4,8,16,32)

srtio3_unit = Atoms(
    "SrTiO3",
    positions=[
        (0, 0, 0),
        (a / 2, a / 2, a / 2),
        (a / 2, a / 2, 0),
        (a / 2, 0, a / 2),
        (0, a / 2, a / 2),
    ],
    cell=[a, a, a],
    pbc=True,
)


def make_potential_and_tp(nz, sc_local=None):
    """Build CrystalPotential and synthetic Ti K-edge TransitionPotentialArray.

    ``sc_local`` overrides the module-level ``sc`` (used by the cell-size
    sweep); the per-axis grid is ``sc_local * px_per_unit`` so real-space
    sampling (A/px) stays fixed as the cell grows -- this is what makes
    reciprocal-space sampling (dk = 1/extent) get denser with cell size,
    which is the regime BiP-PRISM's beam partitioning targets.
    """
    sc_local = sc if sc_local is None else sc_local
    gpts_local = (sc_local * px_per_unit, sc_local * px_per_unit)
    atoms = srtio3_unit * (sc_local, sc_local, nz)
    pot_unit = abtem.Potential(
        srtio3_unit, gpts=(px_per_unit, px_per_unit),
        slice_thickness=a, device=device,
    )
    crystal_pot = abtem.CrystalPotential(
        potential_unit=pot_unit, repetitions=(sc_local, sc_local, nz)
    )
    extent = crystal_pot.extent
    rng = np.random.default_rng(42)
    sampling = (extent[0] / gpts_local[0], extent[1] / gpts_local[1])
    y = np.arange(gpts_local[0], dtype=np.float32) * sampling[0]
    x = np.arange(gpts_local[1], dtype=np.float32) * sampling[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / (2 * 0.5**2)).astype(np.float32)
    raw = (
        rng.standard_normal((2, *gpts_local))
        + 1j * rng.standard_normal((2, *gpts_local))
    ).astype(np.complex64)
    tp_array = np.fft.fft2(raw * gauss[None]) / energy2sigma(energy)
    tp = TransitionPotentialArray(
        Z=22,
        array=tp_array.astype(np.complex64),
        energy=energy,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 22, "n": 1, "l": 0},
    )
    return crystal_pot, tp, atoms


# ---------------------------------------------------------------------------
# Per-variant run wrappers -- all share the SAME fixed-angle AnnularDetector
# ---------------------------------------------------------------------------
VARIANTS = ["multislice", "real_space", "beam_basis", "bipartite"]
COLORS = {"multislice": "k", "real_space": "C0", "beam_basis": "C1", "bipartite": "C2"}
LABELS = {
    "multislice": "Multislice",
    "real_space": "PRISM (real_space)",
    "beam_basis": "PRISM-EELS (beam_basis)",
    "bipartite": f"BiP-PRISM (s1={partitions_s1}, s2={partitions_s2})",
}


def _detector():
    return abtem.AnnularDetector(inner=0, outer=detector_outer)


def timed_ms(probe, potential, tp, scan, atoms, double_channel):
    def _run():
        # lazy=True (not False): Probe.build's eager path
        # (WavesBuilder._build_validated) computes the ENTIRE scan as one
        # array/kernel call regardless of `max_batch`, ignoring it entirely
        # -- only the lazy=True path actually dask-chunks the scan by
        # `max_batch` (from config['dask.chunk-size-gpu']). At large scan
        # counts the unchunked eager path is both unrepresentative of real
        # usage and, on some GPUs, can trigger a CUDA illegal-address crash
        # from a single oversized batched kernel launch.
        return probe.transition_potential_scan(
            potential=potential,
            transition_potentials=tp,
            scan=scan,
            detectors=_detector(),
            sites=atoms,
            double_channel=double_channel,
            lazy=True,
        ).compute()

    _, elapsed, peak = _measure(_run)
    return elapsed, peak


# The three SMatrix.transition_potential_scan variants below stay lazy=False
# deliberately, unlike timed_ms above: SMatrix's lazy=True path (see
# transition_potential_scan in abtem/prism/s_matrix.py) only dask-chunks
# over ensemble blocks (frozen-phonon/potential configs) -- for our
# single-configuration CrystalPotential that's one block, i.e. no
# chunking at all. The scan dimension is appended to `chunks` unsplit
# (`chunks += scan.shape`), so lazy=True would still build the full scan
# in one call here; it isn't a mitigation for this reduction family the
# way it is for Probe.transition_potential_scan.
def timed_real_space(S, tp, scan, atoms, double_channel):
    def _run():
        return S.transition_potential_scan(
            tp,
            scan=scan,
            detectors=_detector(),
            sites=atoms,
            double_channel=double_channel,
            reduction="real_space",
            lazy=False,
        )

    _, elapsed, peak = _measure(_run)
    return elapsed, peak


def timed_beam_basis(S, tp, scan, atoms, double_channel):
    def _run():
        return S.transition_potential_scan(
            tp,
            scan=scan,
            detectors=_detector(),
            sites=atoms,
            double_channel=double_channel,
            reduction="beam_basis",
            collection_angle=collection_angle,
            lazy=False,
        )

    _, elapsed, peak = _measure(_run)
    return elapsed, peak


def timed_bipartite(S, tp, scan, atoms, double_channel):
    def _run():
        return S.transition_potential_scan(
            tp,
            scan=scan,
            detectors=_detector(),
            sites=atoms,
            double_channel=double_channel,
            reduction="bipartite",
            collection_angle=collection_angle,
            partitions_s1=partitions_s1,
            partitions_s2=partitions_s2,
            lazy=False,
        )

    _, elapsed, peak = _measure(_run)
    return elapsed, peak


TIMED_FUNCS = {
    "real_space": timed_real_space,
    "beam_basis": timed_beam_basis,
    "bipartite": timed_bipartite,
}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def _run_one(name, func, *args):
    """Run one timed_* call; on failure, warn and return (nan, nan) so a
    single bad config doesn't kill an otherwise-long unattended sweep.

    Two known ways this can legitimately fail as problem sizes grow:
    (1) if a scan grid is so sparse relative to ``interpolation`` that NO
    scan position falls inside some atom's PRISM window, that atom's
    per-position mask is all-False -- CPU/NumPy tolerates the resulting
    zero-batch-size FFT, CuPy's GPU FFT does not (see the ``min_scan_n``
    guard in ``sweep_scan_positions``, which avoids this; this is a
    backstop for any other edge case). (2) ``real_space``/``beam_basis``
    build S1/S2 over the full aperture/collection-disk beam set, so at
    large cell sizes (sweep 3) they can genuinely exceed GPU memory --
    that is the exact failure mode BiP-PRISM's partitioning is meant to
    avoid, so seeing ``beam_basis`` OOM while ``bipartite`` keeps working
    at the same cell size is an expected, informative result, not a bug.
    """
    try:
        return func(*args)
    except Exception as exc:  # noqa: BLE001 -- keep the sweep alive
        msg = repr(exc)
        print(f"  [WARN] {name} failed: {msg}")
        # The caught exception's traceback references every frame between
        # here and the OOM site, which keeps that frame's local arrays (the
        # partially-built ones that got as far as they could before the
        # allocator gave up) alive -- `del exc` + a cyclic-GC pass BEFORE
        # `free_all_blocks()` is required, or the pool cannot actually
        # reclaim that memory (frame<->traceback is a reference cycle,
        # not just a refcount, so plain refcounting never collects it).
        # Without this, one OOM leaves the pool permanently holding however
        # much the failed call had allocated, and every later config --
        # even a tiny one -- OOMs again against that same stuck total.
        del exc
        gc.collect()
        if device == "gpu":
            _mempool.free_all_blocks()
            _pinned_pool.free_all_blocks()
            _gpu_sync()
        return float("nan"), float("nan")


def _run_all_variants(probe, S, tp, scan, atoms, double_channel, alive):
    """Run each variant that's still ``alive`` for one config; return
    ``{variant: (t, mem)}``.

    ``alive`` is a ``{variant: bool}`` dict owned by the calling sweep loop
    and mutated in place: once a variant's ``_run_one`` call comes back NaN
    it's marked dead here, and every later size in that sweep skips it
    (recording NaN without even attempting the call) while variants that
    haven't failed yet keep going -- so each variant is pushed to its OWN
    limit rather than the whole sweep stopping at the first one to fail.
    """
    out = {}
    if alive["multislice"]:
        out["multislice"] = _run_one(
            "multislice", timed_ms, probe, S.potential, tp, scan, atoms, double_channel
        )
        if np.isnan(out["multislice"][0]):
            alive["multislice"] = False
    else:
        out["multislice"] = (float("nan"), float("nan"))

    for variant, func in TIMED_FUNCS.items():
        if alive[variant]:
            out[variant] = _run_one(variant, func, S, tp, scan, atoms, double_channel)
            if np.isnan(out[variant][0]):
                alive[variant] = False
        else:
            out[variant] = (float("nan"), float("nan"))
    return out


def sweep_scan_positions(double_channel, nz=16, scan_ns=None):
    """Sweep 1: vary scan grid, fixed thickness and cell size.

    ``nz=16`` (~62 A) rather than a couple of slices so multislice does
    enough real per-position propagation work to not be swamped by fixed
    per-call overhead (kernel launch, warm-up notwithstanding).

    ``scan_ns`` defaults to multiples of ``2 * interp`` -- below that, the
    scan grid can be sparse enough relative to the PRISM cell
    (``extent / interpolation``, and the per-atom valid window is HALF of
    that) that some atoms see zero valid scan positions; see ``_run_one``.
    The top of the range (32x the minimum) reaches into the tens of
    thousands of positions -- realistic full-frame EELS map territory,
    and enough for multislice's O(n_pos) cost to dominate its fixed
    overhead.
    """
    if scan_ns is None:
        min_n = max(2, 2 * interp)
        scan_ns = tuple(min_n * k for k in (1, 2, 4, 8, 16, 32))
    mode = "double" if double_channel else "single"
    potential, tp, atoms = make_potential_and_tp(nz)
    n_slices = len(list(potential.generate_slices()))

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)

    S = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=interp, downsample=False, device=device,
    )
    cell_xy = potential.extent

    print("=" * 88)
    print(
        f"SrTiO3 {sc}x{sc}x{nz}  |  {S.gpts}  |  {n_slices} slices  |  "
        f"{energy / 1e3:.0f} keV {semiangle_cutoff} mrad  |  {mode}-channel  |  "
        f"detector {detector_outer} mrad, collection {collection_angle} mrad"
    )
    print(f"--- Sweep 1: scan positions (nz={nz}, sc={sc}, interp={interp}) ---")
    header = f"{'n':>4s} {'N_pos':>6s}"
    for v in VARIANTS:
        header += f" | {v:>11s} t(s) {'MB':>6s}"
    print(header)

    results = {v: {"t": [], "m": []} for v in VARIANTS}
    n_pos_list = []
    alive = {v: True for v in VARIANTS}
    for n in scan_ns:
        scan = abtem.GridScan(start=(0, 0), end=cell_xy, gpts=(n, n), endpoint=False)
        n_pos_list.append(n * n)
        row = _run_all_variants(probe, S, tp, scan, atoms, double_channel, alive)
        line = f"{n:4d} {n * n:6d}"
        for v in VARIANTS:
            t, m = row[v]
            results[v]["t"].append(t)
            results[v]["m"].append(m)
            line += f" | {t:11.3f}s {m:6.0f}"
        print(line)
        if not any(alive.values()):
            print("  (stopping sweep: every variant has failed)")
            break

    return dict(
        x=n_pos_list, xlabel="Scan positions", n_slices=n_slices, results=results
    )


def sweep_thickness(
    double_channel, scan_n=None, nz_values=(4, 8, 16, 32, 64, 128, 256)
):
    """Sweep 2: vary z-repetitions, fixed scan grid and cell size.

    Up to 256 slices (~1000 A) -- deep enough that per-slice cost actually
    accumulates into a measurable signal instead of being noise-floor.

    ``scan_n`` defaults to 64 (4096 positions) -- comfortably above the
    scan-position crossover from sweep 1, where PRISM's S-matrix reuse
    starts winning over multislice, so this sweep's per-slice comparison
    isn't confounded by being on the wrong side of that crossover (see
    ``sweep_scan_positions``'s ``min_n`` guard for the safety floor this
    is also subject to).
    """
    if scan_n is None:
        scan_n = max(64, 2 * interp)
    n_pos = scan_n**2

    print(
        f"\n--- Sweep 2: thickness (sc={sc}, {scan_n}x{scan_n}={n_pos} positions) ---"
    )
    header = f"{'nz':>4s} {'thick':>7s}"
    for v in VARIANTS:
        header += f" | {v:>11s} t(s) {'MB':>6s}"
    print(header)

    results = {v: {"t": [], "m": []} for v in VARIANTS}
    thickness_list = []
    alive = {v: True for v in VARIANTS}
    for nz in nz_values:
        potential, tp, atoms = make_potential_and_tp(nz)
        thickness_list.append(nz * a)

        probe = abtem.Probe(
            energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
        )
        probe.grid.match(potential)
        S = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
            interpolation=interp, downsample=False, device=device,
        )
        scan = abtem.GridScan(
            start=(0, 0), end=potential.extent, gpts=(scan_n, scan_n), endpoint=False
        )
        row = _run_all_variants(probe, S, tp, scan, atoms, double_channel, alive)
        line = f"{nz:4d} {nz * a:6.1f}A"
        for v in VARIANTS:
            t, m = row[v]
            results[v]["t"].append(t)
            results[v]["m"].append(m)
            line += f" | {t:11.3f}s {m:6.0f}"
        print(line)
        if not any(alive.values()):
            print("  (stopping sweep: every variant has failed)")
            break

    return dict(
        x=thickness_list, xlabel="Specimen thickness (A)", n_pos=n_pos, results=results
    )


def sweep_cell_size(
    double_channel, nz=16, scan_n=None, sc_values=(2, 4, 8, 12, 16)
):
    """Sweep 3 (new): vary lateral cell size at FIXED A/px sampling.

    This is the stress test for BiP-PRISM: as the cell grows, reciprocal-space
    sampling gets denser (dk = 1/extent), so the number of beams inside a
    FIXED physical ``collection_angle`` disk grows -- real_space's window and
    beam_basis's per-pixel S2 reverse-multislice both get more expensive,
    while bipartite's parent count (partitions_s1/partitions_s2) is fixed by
    construction and should stay roughly flat.

    ``sc_values`` reaches gpts=512x512 at the top end (``sc=16``), where
    ``real_space``/``beam_basis`` build S1/S2 over the FULL aperture /
    collection-disk beam set -- expect them to become the memory bottleneck
    (potentially even OOM on GPU) well before ``bipartite`` does, since its
    parent count is fixed by ``partitions_s1``/``partitions_s2`` regardless
    of cell size. That gap is the demonstration, not a bug: each variant is
    tracked independently (see ``_run_all_variants``'s ``alive`` dict), so
    ``real_space``/``beam_basis`` failing early doesn't cut ``bipartite``'s
    run short -- it keeps going against ``sc_values`` until it finds its
    own limit. Extend the list if your GPU has headroom past the built-in
    top end.

    ``scan_n`` defaults to 64 (4096 positions), matching ``sweep_thickness``
    (see its docstring for why: above the scan-position crossover from
    sweep 1). Since ``sc`` scales the extent and the scan grid is fixed,
    the ratio of scan spacing to PRISM-cell size (see
    ``sweep_scan_positions``) stays constant across ``sc_values`` -- so one
    ``scan_n`` covers the whole sweep.

    The x-axis is the supercell size ``sc`` itself (not the derived
    ``sc * px_per_unit`` gpts) -- gpts is still shown in the printed table
    for reference.
    """
    if scan_n is None:
        scan_n = max(64, 2 * interp)
    n_pos = scan_n**2

    print(
        f"\n--- Sweep 3: cell size (nz={nz}, {scan_n}x{scan_n}={n_pos} positions, "
        f"collection_angle={collection_angle} mrad) ---"
    )
    header = f"{'sc':>4s} {'gpts':>9s}"
    for v in VARIANTS:
        header += f" | {v:>11s} t(s) {'MB':>6s}"
    print(header)

    results = {v: {"t": [], "m": []} for v in VARIANTS}
    sc_list = []
    alive = {v: True for v in VARIANTS}
    for sc_local in sc_values:
        potential, tp, atoms = make_potential_and_tp(nz, sc_local=sc_local)
        sc_list.append(sc_local)

        probe = abtem.Probe(
            energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
        )
        probe.grid.match(potential)
        S = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
            interpolation=interp, downsample=False, device=device,
        )
        scan = abtem.GridScan(
            start=(0, 0), end=potential.extent, gpts=(scan_n, scan_n), endpoint=False
        )
        row = _run_all_variants(probe, S, tp, scan, atoms, double_channel, alive)
        line = f"{sc_local:4d} {sc_local * px_per_unit:4d}x{sc_local * px_per_unit:<4d}"
        for v in VARIANTS:
            t, m = row[v]
            results[v]["t"].append(t)
            results[v]["m"].append(m)
            line += f" | {t:11.3f}s {m:6.0f}"
        print(line)
        if not any(alive.values()):
            print("  (stopping sweep: every variant has failed)")
            break

    return dict(
        x=sc_list, xlabel="Supercell size (sc)",
        n_pos=n_pos, results=results,
    )


# ---------------------------------------------------------------------------
# Plotting -- 3 sweeps x 3 panels (time, speedup vs multislice, memory)
# ---------------------------------------------------------------------------
def plot(sweep_results, out_path, double_channel):
    mode = "double" if double_channel else "single"
    mem_label = "Peak GPU memory (MB)" if device == "gpu" else "Peak traced memory (MB)"
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    for row_idx, (title, r) in enumerate(sweep_results.items()):
        x = np.asarray(r["x"], dtype=float)

        ax = axes[row_idx, 0]
        for v in VARIANTS:
            ax.plot(x, r["results"][v]["t"], "o-", ms=5, lw=1.5,
                    color=COLORS[v], label=LABELS[v])
        ax.set_xlabel(r["xlabel"])
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_yscale("log")
        ax.set_title(f"{title}: total time")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[row_idx, 1]
        ms_t = np.asarray(r["results"]["multislice"]["t"])
        for v in VARIANTS:
            if v == "multislice":
                continue
            speedup = ms_t / np.asarray(r["results"][v]["t"])
            ax.plot(x, speedup, "o-", ms=5, lw=1.5, color=COLORS[v], label=LABELS[v])
        ax.axhline(1.0, color="k", ls=":", alpha=0.5)
        ax.set_xlabel(r["xlabel"])
        ax.set_ylabel("Speedup vs multislice")
        ax.set_title(f"{title}: speedup")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[row_idx, 2]
        for v in VARIANTS:
            ax.plot(x, r["results"][v]["m"], "o-", ms=5, lw=1.5,
                    color=COLORS[v], label=LABELS[v])
        ax.set_xlabel(r["xlabel"])
        ax.set_ylabel(mem_label)
        ax.set_title(f"{title}: peak memory")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PRISM-EELS variant benchmark ({device}, {mode}-channel): SrTiO3, "
        f"sc={sc}, interp={interp}, partitions_s1={partitions_s1}, "
        f"partitions_s2={partitions_s2}, collection_angle={collection_angle} mrad, "
        f"{energy / 1e3:.0f} keV, {semiangle_cutoff} mrad aperture, "
        f"{detector_outer} mrad detector",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Validation -- confirm the harness itself is wired correctly before timing
# ---------------------------------------------------------------------------
def validate_interp1():
    """Sanity check at interpolation=1: real_space, beam_basis, and bipartite
    (with NO partitioning, i.e. the exact full-parent limit) must all match
    conventional multislice, for both single- and double-channel."""
    potential, tp, atoms = make_potential_and_tp(nz=2, sc_local=1)

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device=device
    )
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=potential.extent, gpts=(2, 2), endpoint=False
    )
    detector = _detector()

    S1 = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=1, downsample=False, device=device,
    )

    for dc in [False, True]:
        mode = "double" if dc else "single"
        print(f"Validating variants vs multislice at interp=1 ({mode}-channel) ... ",
              end="", flush=True)
        res_ms = probe.transition_potential_scan(
            potential=potential, transition_potentials=tp, scan=scan,
            detectors=detector, sites=atoms, double_channel=dc, lazy=False,
        )
        arr_ms = np.asarray(res_ms.array)

        res_real = S1.transition_potential_scan(
            tp, scan=scan, detectors=detector, sites=atoms,
            double_channel=dc, reduction="real_space", lazy=False,
        )
        res_bb = S1.transition_potential_scan(
            tp, scan=scan, detectors=detector, sites=atoms, double_channel=dc,
            reduction="beam_basis", collection_angle=None, lazy=False,
        )
        res_bip = S1.transition_potential_scan(
            tp, scan=scan, detectors=detector, sites=atoms, double_channel=dc,
            reduction="bipartite", collection_angle=None,
            partitions_s1=None, partitions_s2=None, lazy=False,
        )

        for name, res in [("real_space", res_real), ("beam_basis", res_bb),
                          ("bipartite (no partitions)", res_bip)]:
            arr = np.asarray(res.array)
            assert arr.shape == arr_ms.shape, (
                f"{name}: shape mismatch {arr.shape} vs {arr_ms.shape}"
            )
            np.testing.assert_allclose(arr, arr_ms, rtol=1e-4, atol=1e-6)
        print("OK")


if __name__ == "__main__":
    import pathlib

    stem = pathlib.Path(__file__).with_suffix("")

    validate_interp1()

    for dc in [False, True]:
        mode = "single" if not dc else "double"
        out_path = (
            f"{stem}_{device}_{mode}_sc{sc}_i{interp}_"
            f"p{partitions_s1}-{partitions_s2}.pdf"
        )
        # No explicit nz/scan_n overrides here -- use each sweep function's
        # own tuned defaults (see their docstrings).
        r1 = sweep_scan_positions(double_channel=dc)
        r2 = sweep_thickness(double_channel=dc)
        r3 = sweep_cell_size(double_channel=dc)
        plot(
            {"Scan positions": r1, "Thickness": r2, "Cell size": r3},
            out_path,
            double_channel=dc,
        )
