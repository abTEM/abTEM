"""PRISM-EELS accuracy vs interpolation, per element edge (real transition potentials).

System: SrTiO3 2x2x4, 128x128 gpts, 200 keV, 25 mrad, single-channel.
Reference: conventional multislice (Probe.transition_potential_scan).
Metric: normalized RMS error of the angle-integrated EELS map over an
8x8 scan of one unit cell, PRISM vs multislice.

Edges: O K (Z=8, n=1, l=0), Ti L (Z=22, n=2, l=1), Sr L (Z=38, n=2, l=1).
"""
import numpy as np
import abtem
from ase import Atoms
from abtem.inelastic.core_loss import SubshellTransitions

lat = 3.905
srtio3_unit = Atoms(
    "SrTiO3",
    positions=[(0, 0, 0), (lat/2, lat/2, lat/2),
               (lat/2, lat/2, 0), (lat/2, 0, lat/2), (0, lat/2, lat/2)],
    cell=[lat, lat, lat], pbc=True,
)
atoms = srtio3_unit * (2, 2, 4)

energy = 200e3
semiangle_cutoff = 25.0
gpts = (128, 128)

potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=lat, device="cpu")
extent = potential.extent

detector = abtem.FlexibleAnnularDetector(to_cpu=True)
scan = abtem.GridScan(start=(0, 0), end=(lat, lat), sampling=0.5, endpoint=False)

probe = abtem.Probe(energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu")
probe.grid.match(potential)

edges = [
    ("O",  8, 1, 0),
    ("Ti", 22, 2, 1),
    ("Sr", 38, 2, 1),
]
interps = [1, 2, 3, 4]

# results[interp][element] = rms
results = {iv: {} for iv in interps}

for element, Z, n, l in edges:
    tp = (
        SubshellTransitions(Z, n, l)
        .get_transition_potentials(
            extent=extent, gpts=gpts, energy=energy, double_channel=False
        )
        .build()
    )

    res_ms = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, sites=atoms, double_channel=False, lazy=False,
    ).compute()
    ms_map = np.asarray(res_ms.array).sum(axis=(-2, -1)).flatten()

    for iv in interps:
        S = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
            interpolation=iv, downsample=False, device="cpu",
        )
        res_pr = S.transition_potential_scan(
            transition_potentials=tp, scan=scan, detectors=detector,
            sites=atoms, double_channel=False, lazy=False,
        )
        pr_map = np.asarray(res_pr.array).sum(axis=(-2, -1)).flatten()
        rms = np.sqrt(np.sum((ms_map - pr_map) ** 2) / np.sum(ms_map ** 2))
        results[iv][element] = rms
        print(f"{element:>3} edge | interp={iv}: RMS={rms:.3e}")

# --- Markdown table: rows = interpolation, columns = element ---
print("\n\n### PRISM-EELS RMS error vs multislice\n")
header = "| Interpolation | " + " | ".join(f"{el} ({en} {'KLM'[n-1]})"
        for (el, Z, n, l), en in zip(edges, ["K", "L", "L"])) + " |"
# Simpler header
cols = {"O": "O K", "Ti": "Ti L", "Sr": "Sr L"}
header = "| Interpolation | " + " | ".join(cols[el] for el, *_ in edges) + " |"
sep = "|" + "---|" * (len(edges) + 1)
print(header)
print(sep)
for iv in interps:
    cell = extent[0] / iv
    row = f"| {iv} (cell {cell:.2f} Å) | " + " | ".join(
        f"{results[iv][el]*100:.2f}%" for el, *_ in edges) + " |"
    print(row)
