"""PRISM-EELS accuracy vs cell size: tiling + interpolation trade-offs.

Demonstrates that the delocalized-edge error depends primarily on
cell = extent/interpolation, and that realistic large-FOV simulations
(big supercell, moderate interpolation) make the error acceptably small
even for delocalized edges like O K.

Uses CrystalPotential to keep memory manageable at large tilings.
Tracks wall-clock time and peak RSS for each configuration.

System: SrTiO3 unit-cell potential tiled NxNx4, 200 keV, 25 mrad,
single-channel, real O K edge (delocalized).
Reference: conventional multislice on the SAME tiled system.
Scan: 4x4 grid over one unit cell (kept modest for tractability).
"""
import gc
import os
import resource
import time

import numpy as np
from ase import Atoms

import abtem
from abtem.inelastic.core_loss import SubshellTransitions

lat = 3.905
energy = 200e3
semiangle_cutoff = 25.0
px_per_unit = 32  # ~0.122 A/px, divisible by interp 2, 4, 8

srtio3_unit = Atoms(
    "SrTiO3",
    positions=[
        (0, 0, 0),
        (lat / 2, lat / 2, lat / 2),
        (lat / 2, lat / 2, 0),
        (lat / 2, 0, lat / 2),
        (0, lat / 2, lat / 2),
    ],
    cell=[lat, lat, lat],
    pbc=True,
)

scan = abtem.GridScan(
    start=(0, 0), end=(lat, lat), sampling=lat / 4, endpoint=False
)
detector = abtem.FlexibleAnnularDetector(to_cpu=True)


def rms(a, b):
    return np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2))


# --- Configurations to benchmark ---
# (tile_xy, interp) -> cell = tile_xy * lat / interp
configs = [
    (1, 1),   # exact baseline, cell=3.9
    (2, 2),   # cell=3.9, extent=7.8
    (4, 4),   # cell=3.9, extent=15.6
    (2, 1),   # cell=7.8, exact
    (4, 2),   # cell=7.8, extent=15.6
    (8, 4),   # cell=7.8, extent=31.2
    (4, 1),   # cell=15.6, exact
    (8, 2),   # cell=15.6, extent=31.2
    (16, 8),  # cell=7.8, extent=62.4
    (16, 4),  # cell=15.6, extent=62.4
]

print(
    f"{'tile':>5} {'interp':>6} {'cell_A':>7} {'extent':>7} {'gpts':>7} "
    f"{'RMS':>8} {'t_ms':>6} {'t_prism':>7} {'peak_MB':>8}",
    flush=True,
)
print("-" * 78, flush=True)

results = []
ms_cache = {}

for tile_xy, interp in configs:
    cell_a = tile_xy * lat / interp
    extent_a = tile_xy * lat
    gpts = (px_per_unit * tile_xy, px_per_unit * tile_xy)

    atoms = srtio3_unit * (tile_xy, tile_xy, 4)

    pot_unit = abtem.Potential(
        srtio3_unit, gpts=(px_per_unit, px_per_unit),
        slice_thickness=lat, device="cpu",
    )
    crystal_pot = abtem.CrystalPotential(
        potential_unit=pot_unit, repetitions=(tile_xy, tile_xy, 4)
    )

    tp = (
        SubshellTransitions(8, 1, 0)
        .get_transition_potentials(
            extent=(extent_a, extent_a),
            gpts=gpts,
            energy=energy,
            double_channel=False,
        )
        .build()
    )

    # --- Multislice reference (cached per tile, since interp=1 is exact) ---
    if tile_xy not in ms_cache:
        probe = abtem.Probe(
            energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu"
        )
        probe.grid.match(crystal_pot)
        t0_ms = time.perf_counter()
        res_ms = probe.transition_potential_scan(
            potential=crystal_pot,
            transition_potentials=tp,
            scan=scan,
            detectors=detector,
            sites=atoms,
            double_channel=False,
            lazy=False,
        )
        dt_ms = time.perf_counter() - t0_ms
        ms_map = np.asarray(res_ms.array).sum(axis=(-2, -1)).flatten()
        ms_cache[tile_xy] = (ms_map, dt_ms)
        del res_ms, probe
    ms_map, dt_ms = ms_cache[tile_xy]

    # --- PRISM ---
    S = abtem.SMatrix(
        potential=crystal_pot,
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        interpolation=interp,
        downsample=False,
        device="cpu",
    )

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()

    res_pr = S.transition_potential_scan(
        transition_potentials=tp,
        scan=scan,
        detectors=detector,
        sites=atoms,
        double_channel=False,
        lazy=False,
    )

    dt_prism = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        peak_mb = rss_after / 1024 / 1024
    else:
        peak_mb = rss_after / 1024

    pr_map = np.asarray(res_pr.array).sum(axis=(-2, -1)).flatten()
    err = rms(pr_map, ms_map)

    row = {
        "tile": tile_xy,
        "interp": interp,
        "cell": cell_a,
        "extent": extent_a,
        "gpts": gpts[0],
        "rms": err,
        "t_ms": dt_ms,
        "t_prism": dt_prism,
        "peak_mb": peak_mb,
    }
    results.append(row)

    tag = " (exact)" if interp == 1 else ""
    print(
        f"{tile_xy:>5} {interp:>6} {cell_a:>7.2f} {extent_a:>7.2f} "
        f"{gpts[0]:>7} {err:>8.4f} {dt_ms:>6.1f} {dt_prism:>7.1f} "
        f"{peak_mb:>8.0f}{tag}",
        flush=True,
    )

    del S, res_pr, tp, crystal_pot, pot_unit, atoms, pr_map
    gc.collect()

# --- Markdown summary table ---
print("\n\n### PRISM-EELS O K edge: accuracy vs cell size (CrystalPotential)\n")
print(
    "| tile | interp | cell (Å) | extent (Å) | gpts "
    "| RMS | t_ms (s) | t_prism (s) | peak RSS (MB) |"
)
print("|---|---|---|---|---|---|---|---|---|")
for r in results:
    tag = " (exact)" if r["interp"] == 1 else ""
    print(
        f"| {r['tile']} | {r['interp']} | {r['cell']:.2f} | {r['extent']:.1f} "
        f"| {r['gpts']} | {r['rms']:.4f}{tag} | {r['t_ms']:.1f} "
        f"| {r['t_prism']:.1f} | {r['peak_mb']:.0f} |"
    )

# --- Grouped by cell ---
from itertools import groupby

print("\n### Grouped by cell size\n")
sorted_results = sorted(results, key=lambda r: (r["cell"], r["extent"]))
for cell_val, group in groupby(sorted_results, key=lambda r: r["cell"]):
    items = list(group)
    entries = ", ".join(
        f"({r['tile']},{r['interp']})={r['rms']:.4f}"
        + (" [exact]" if r["interp"] == 1 else "")
        for r in items
    )
    print(f"  cell={cell_val:.2f} Å: {entries}")
