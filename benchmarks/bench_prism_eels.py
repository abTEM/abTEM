"""Benchmark: PRISM-EELS vs multislice on SrTiO3.

Two sweeps per channeling mode (single + double):
  1. Scan positions (fixed thickness) — shows crossover and linear scaling.
  2. Specimen thickness (fixed scan size above crossover) — shows how the
     per-position saving accumulates with more slices.

Peak traced-memory is recorded for every data point via tracemalloc.

Usage
-----
    python benchmarks/bench_prism_eels.py
        # saves bench_prism_eels_single.pdf and _double.pdf next to script
"""
import gc
import sys
import time
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray, energy2sigma

# ---------------------------------------------------------------------------
# SrTiO3 unit cell (perovskite, a = 3.905 Å)
# ---------------------------------------------------------------------------
a = 3.905
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

# Simulation parameters
energy = 200e3  # 200 keV
semiangle_cutoff = 25.0  # mrad
gpts = (128, 128)
interp = 8
n_k = (gpts[0] // interp) * (gpts[1] // interp)  # 256


def make_potential_and_tp(atoms):
    """Build Potential and synthetic Ti K-edge TransitionPotentialArray."""
    potential = abtem.Potential(
        atoms, gpts=gpts, slice_thickness=a, device="cpu"
    )
    rng = np.random.default_rng(42)
    sampling = (
        potential.extent[0] / gpts[0],
        potential.extent[1] / gpts[1],
    )
    y = np.arange(gpts[0], dtype=np.float32) * sampling[0]
    x = np.arange(gpts[1], dtype=np.float32) * sampling[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / (2 * 0.5**2)).astype(np.float32)
    raw = (
        rng.standard_normal((2, *gpts))
        + 1j * rng.standard_normal((2, *gpts))
    ).astype(np.complex64)
    tp_array = np.fft.fft2(raw * gauss[None]) / energy2sigma(energy)
    tp = TransitionPotentialArray(
        Z=22,
        array=tp_array.astype(np.complex64),
        energy=energy,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 22, "n": 1, "l": 0},
    )
    return potential, tp


def timed_ms(probe, potential, tp, scan, atoms, double_channel):
    """Run multislice EELS and return (time_s, peak_mem_MB)."""
    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    probe.transition_potential_scan(
        potential=potential,
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=double_channel,
        lazy=False,
    ).compute()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024**2


def timed_prism(S, tp, scan, atoms, double_channel):
    """Run PRISM EELS and return (time_s, peak_mem_MB)."""
    detector = abtem.FlexibleAnnularDetector(to_cpu=True)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    S.transition_potential_scan(
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=double_channel,
        lazy=False,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024**2


def sweep_scan_positions(double_channel, nz=8):
    """Sweep 1: vary scan grid, fixed thickness."""
    mode = "double" if double_channel else "single"
    atoms = srtio3_unit * (4, 4, nz)
    potential, tp = make_potential_and_tp(atoms)
    n_slices = len(list(potential.generate_slices()))

    probe = abtem.Probe(
        energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu"
    )
    probe.grid.match(potential)

    S = abtem.SMatrix(
        potential=potential,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=interp,
        downsample=False,
        device="cpu",
    )

    cell_xy = potential.extent

    print("=" * 72)
    print(
        f"SrTiO3 4x4x{nz}  |  {gpts[0]}x{gpts[1]} gpts  |  "
        f"{n_slices} slices ({atoms.cell[2, 2]:.1f} A)  |  "
        f"{energy / 1e3:.0f} keV  {semiangle_cutoff} mrad  |  {mode}-channel"
    )
    print(
        f"PRISM interp={interp}, N_k={n_k}, window={S.window_gpts}"
    )
    print("=" * 72)

    scan_ns = [2, 4, 8, 16, 32, 64]
    n_pos_list = [n * n for n in scan_ns]

    print(
        f"\n--- Sweep 1: scan positions ({mode}-channel) ---\n"
        f"{'n':>4s}  {'N_pos':>6s}  {'MS (s)':>8s} {'ms/pos':>7s} "
        f"{'MB':>6s}  {'PRISM (s)':>9s} {'ms/pos':>7s} {'MB':>6s}  "
        f"{'speedup':>7s}"
    )

    ms_t, ms_m, pr_t, pr_m = [], [], [], []
    for n in scan_ns:
        scan = abtem.GridScan(
            start=(0, 0), end=cell_xy, gpts=(n, n), endpoint=False
        )
        n_pos = n * n
        t_ms, m_ms = timed_ms(probe, potential, tp, scan, atoms, double_channel)
        t_pr, m_pr = timed_prism(S, tp, scan, atoms, double_channel)
        ms_t.append(t_ms)
        ms_m.append(m_ms)
        pr_t.append(t_pr)
        pr_m.append(m_pr)
        sp = t_ms / t_pr
        print(
            f"{n:4d}  {n_pos:6d}  {t_ms:8.2f} {t_ms / n_pos * 1000:7.1f} "
            f"{m_ms:6.0f}  {t_pr:9.2f} {t_pr / n_pos * 1000:7.1f} "
            f"{m_pr:6.0f}  {sp:7.2f}x"
        )

    return dict(
        scan_ns=scan_ns,
        n_pos=n_pos_list,
        ms_t=ms_t,
        ms_m=ms_m,
        pr_t=pr_t,
        pr_m=pr_m,
        n_slices=n_slices,
    )


def sweep_thickness(double_channel, scan_n=32):
    """Sweep 2: vary z-repetitions, fixed scan grid."""
    mode = "double" if double_channel else "single"
    n_pos = scan_n**2
    nz_values = [2, 4, 8, 16, 32, 64]

    print(
        f"\n--- Sweep 2: thickness ({mode}-channel, fixed {scan_n}x{scan_n} "
        f"= {n_pos} positions) ---\n"
        f"{'nz':>4s}  {'slices':>6s}  {'thick':>6s}  {'MS (s)':>8s} "
        f"{'MB':>6s}  {'PRISM (s)':>9s} {'MB':>6s}  {'speedup':>7s}"
    )

    ms_t, ms_m, pr_t, pr_m = [], [], [], []
    n_slices_list = []
    thickness_list = []

    for nz in nz_values:
        atoms = srtio3_unit * (4, 4, nz)
        potential, tp = make_potential_and_tp(atoms)
        n_sl = len(list(potential.generate_slices()))
        n_slices_list.append(n_sl)
        thickness_list.append(nz * a)

        probe = abtem.Probe(
            energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu"
        )
        probe.grid.match(potential)

        S = abtem.SMatrix(
            potential=potential,
            energy=energy,
            semiangle_cutoff=semiangle_cutoff,
            interpolation=interp,
            downsample=False,
            device="cpu",
        )

        scan = abtem.GridScan(
            start=(0, 0),
            end=potential.extent,
            gpts=(scan_n, scan_n),
            endpoint=False,
        )

        t_ms, m_ms = timed_ms(probe, potential, tp, scan, atoms, double_channel)
        t_pr, m_pr = timed_prism(S, tp, scan, atoms, double_channel)
        ms_t.append(t_ms)
        ms_m.append(m_ms)
        pr_t.append(t_pr)
        pr_m.append(m_pr)
        sp = t_ms / t_pr
        thick = nz * a
        print(
            f"{nz:4d}  {n_sl:6d}  {thick:5.1f}A  {t_ms:8.2f} {m_ms:6.0f}"
            f"  {t_pr:9.2f} {m_pr:6.0f}  {sp:7.2f}x"
        )

    return dict(
        nz_values=nz_values,
        thickness=thickness_list,
        n_slices=n_slices_list,
        n_pos=n_pos,
        ms_t=ms_t,
        ms_m=ms_m,
        pr_t=pr_t,
        pr_m=pr_m,
    )


def plot(r1, r2, out_path, double_channel):
    """Six-panel figure: top row = scan sweep, bottom row = thickness sweep."""
    mode = "double" if double_channel else "single"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ---- Row 1: scan positions ----
    x1 = np.array(r1["n_pos"], dtype=float)
    ms_slope1, ms_int1 = np.polyfit(x1, r1["ms_t"], 1)
    pr_slope1, pr_int1 = np.polyfit(x1, r1["pr_t"], 1)
    crossover1 = (
        pr_int1 / (ms_slope1 - pr_slope1)
        if ms_slope1 > pr_slope1
        else float("inf")
    )
    extrap_x1 = np.linspace(0, 3000, 200)

    print(
        f"\nScan-position fit ({mode}-channel): "
        f"MS {ms_slope1 * 1000:.2f} ms/pos + {ms_int1:.3f} s  |  "
        f"PRISM {pr_slope1 * 1000:.2f} ms/pos + {pr_int1:.3f} s"
    )
    print(f"Marginal cost ratio: {pr_slope1 / ms_slope1:.1%}")
    print(f"Crossover: ~{crossover1:.0f} positions")

    ax = axes[0, 0]
    ax.plot(x1, r1["ms_t"], "ko", ms=5, label="Multislice")
    ax.plot(
        extrap_x1, ms_slope1 * extrap_x1 + ms_int1, "k--", alpha=0.4, lw=1
    )
    ax.plot(x1, r1["pr_t"], "C0s", ms=5, label=f"PRISM interp={interp}")
    ax.plot(
        extrap_x1, pr_slope1 * extrap_x1 + pr_int1, "C0--", alpha=0.4, lw=1
    )
    if 0 < crossover1 < 3000:
        ax.axvline(crossover1, color="gray", ls=":", alpha=0.6)
        ax.text(
            crossover1 + 40,
            max(r1["ms_t"]) * 0.1,
            f"~{crossover1:.0f}",
            fontsize=8,
            color="gray",
        )
    ax.set_xlabel("Scan positions")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Total time ({r1['n_slices']} slices)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    sp1 = np.array(r1["ms_t"]) / np.array(r1["pr_t"])
    sp1_ext = (ms_slope1 * extrap_x1 + ms_int1) / (
        pr_slope1 * extrap_x1 + pr_int1
    )
    ax.plot(x1, sp1, "C0s", ms=5, label="Measured")
    ax.plot(extrap_x1, sp1_ext, "C0--", alpha=0.4, lw=1, label="Extrapolated")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    if ms_slope1 > pr_slope1:
        ax.axhline(
            ms_slope1 / pr_slope1,
            color="C0",
            ls=":",
            alpha=0.4,
            label=f"Asymptotic ({ms_slope1 / pr_slope1:.2f}x)",
        )
    if 0 < crossover1 < 3000:
        ax.axvline(crossover1, color="gray", ls=":", alpha=0.6)
    ax.set_xlabel("Scan positions")
    ax.set_ylabel("Speedup (PRISM / MS)")
    ax.set_title("Speedup vs scan positions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(x1, r1["ms_m"], "ko-", ms=5, lw=1.5, label="Multislice")
    ax.plot(
        x1, r1["pr_m"], "C0s-", ms=5, lw=1.5, label=f"PRISM interp={interp}"
    )
    ax.set_xlabel("Scan positions")
    ax.set_ylabel("Peak traced memory (MB)")
    ax.set_title("Peak memory vs scan positions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Row 2: thickness ----
    thick_arr = np.array(r2["thickness"])
    slices_arr = np.array(r2["n_slices"], dtype=float)
    ms_slope2, ms_int2 = np.polyfit(slices_arr, r2["ms_t"], 1)
    pr_slope2, pr_int2 = np.polyfit(slices_arr, r2["pr_t"], 1)

    print(
        f"\nThickness fit ({mode}-channel, {r2['n_pos']} positions): "
        f"MS {ms_slope2:.3f} s/slice + {ms_int2:.3f} s  |  "
        f"PRISM {pr_slope2:.3f} s/slice + {pr_int2:.3f} s"
    )
    if ms_slope2 > pr_slope2:
        print(f"Per-slice cost ratio: {pr_slope2 / ms_slope2:.1%}")

    ax = axes[1, 0]
    ax.plot(thick_arr, r2["ms_t"], "ko-", ms=5, lw=1.5, label="Multislice")
    ax.plot(
        thick_arr,
        r2["pr_t"],
        "C0s-",
        ms=5,
        lw=1.5,
        label=f"PRISM interp={interp}",
    )
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Total time ({r2['n_pos']} positions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sp2 = np.array(r2["ms_t"]) / np.array(r2["pr_t"])
    ax.plot(thick_arr, sp2, "C0s-", ms=5, lw=1.5, label="Measured")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Speedup (PRISM / MS)")
    ax.set_title(f"Speedup vs thickness ({r2['n_pos']} positions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(thick_arr, r2["ms_m"], "ko-", ms=5, lw=1.5, label="Multislice")
    ax.plot(
        thick_arr,
        r2["pr_m"],
        "C0s-",
        ms=5,
        lw=1.5,
        label=f"PRISM interp={interp}",
    )
    ax.set_xlabel("Specimen thickness (Å)")
    ax.set_ylabel("Peak traced memory (MB)")
    ax.set_title(f"Peak memory vs thickness ({r2['n_pos']} positions)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PRISM-EELS benchmark ({mode}-channel): SrTiO3 4×4×N, "
        f"{gpts[0]}×{gpts[1]} gpts, interp={interp}, "
        f"N$_k$={n_k}, {energy / 1e3:.0f} keV, {semiangle_cutoff} mrad",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    import pathlib

    stem = pathlib.Path(__file__).with_suffix("")

    for dc in [False, True]:
        mode = "single" if not dc else "double"
        out_path = f"{stem}_{mode}.pdf"
        r1 = sweep_scan_positions(double_channel=dc, nz=8)
        r2 = sweep_thickness(double_channel=dc, scan_n=32)
        plot(r1, r2, out_path, double_channel=dc)
