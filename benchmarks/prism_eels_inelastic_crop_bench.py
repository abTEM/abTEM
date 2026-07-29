"""Accuracy + timing implications of the `inelastic_crop` window (Brown Sec. IV B).

Sweeps the inelastic-crop window for interpolation 1 and 2, single- and
double-channel, reporting:
  * accuracy  -- normalized RMS of the angle-integrated EELS map vs multislice,
  * timing    -- wall-clock of the eager PRISM-EELS run.

System: SrTiO3 2x2x4 (~15.6 A thick), 128x128 gpts, 200 keV, 25 mrad, real O K
edge (Z=8, n=1, l=0) -- a spatially delocalized edge, so the crop tradeoff is
visible. Reference: conventional multislice.

The PRISM cell is extent/interpolation: 7.81 A at interp=1, 3.90 A at interp=2.
`inelastic_crop=None` uses the full cell (current default).
"""
import sys
import time
import numpy as np
import abtem
from ase import Atoms
from abtem.inelastic.core_loss import SubshellTransitions

abtem.config.set({"device": "cpu"})

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

tp = (
    SubshellTransitions(8, 1, 0)  # O K
    .get_transition_potentials(
        extent=extent, gpts=gpts, energy=energy, double_channel=True
    )
    .build()
)

detector = abtem.FlexibleAnnularDetector(to_cpu=True)
scan = abtem.GridScan(start=(0, 0), end=(lat, lat), sampling=0.5, endpoint=False)

probe = abtem.Probe(energy=energy, semiangle_cutoff=semiangle_cutoff, device="cpu")
probe.grid.match(potential)


def ms_map(dc):
    res = probe.transition_potential_scan(
        potential=potential, transition_potentials=tp, scan=scan,
        detectors=detector, sites=atoms, double_channel=dc, lazy=False,
    ).compute()
    return np.asarray(res.array).sum(axis=(-2, -1)).flatten()


def prism_timed(interp, dc, crop):
    S = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=semiangle_cutoff,
        interpolation=interp, downsample=False, device="cpu",
    )
    t0 = time.perf_counter()
    res = S.transition_potential_scan(
        transition_potentials=tp, scan=scan, detectors=detector, sites=atoms,
        double_channel=dc, inelastic_crop=crop, lazy=False,
    )
    dt = time.perf_counter() - t0
    return np.asarray(res.array).sum(axis=(-2, -1)).flatten(), dt


def rms(a, b):
    return np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2))


print("=" * 78, flush=True)
print("SrTiO3 2x2x4 | 128 gpts | 200 keV | 25 mrad | O K edge | %d slices"
      % len(potential), flush=True)
print("PRISM cell = extent/interp: interp1=%.2f A, interp2=%.2f A"
      % (extent[0], extent[0] / 2), flush=True)
print("=" * 78, flush=True)

print("\nMultislice references...", flush=True)
t0 = time.perf_counter(); ms_single = ms_map(False); t_ms_s = time.perf_counter() - t0
t0 = time.perf_counter(); ms_double = ms_map(True);  t_ms_d = time.perf_counter() - t0
print("  multislice single: %6.2f s" % t_ms_s, flush=True)
print("  multislice double: %6.2f s" % t_ms_d, flush=True)

# Warm up FFTW plans so the first timed PRISM run is not penalized.
prism_timed(2, True, 2.0)

sweeps = {
    1: [None, 4.0, 3.0, 2.0],
    2: [None, 3.0, 2.0, 1.5],
}

for interp in [1, 2]:
    cell = extent[0] / interp
    print("\n" + "-" * 78, flush=True)
    print("interpolation = %d  (PRISM cell = %.2f A)" % (interp, cell), flush=True)
    print("%-14s | %-21s | %-21s"
          % ("inelastic_crop", "single-channel", "double-channel"), flush=True)
    print("%-14s | %-9s %-11s | %-9s %-11s"
          % ("(Å)", "RMS", "time(s)", "RMS", "time(s)"), flush=True)
    print("-" * 78, flush=True)
    for crop in sweeps[interp]:
        m_s, t_s = prism_timed(interp, False, crop)
        m_d, t_d = prism_timed(interp, True, crop)
        label = "None (%.2f)" % cell if crop is None else "%.2f" % crop
        print("%-14s | %-9.4f %-11.3f | %-9.4f %-11.3f"
              % (label, rms(m_s, ms_single), t_s, rms(m_d, ms_double), t_d),
              flush=True)

print("\nNotes:", flush=True)
print("  * RMS vs multislice (same channel). 'None' = full PRISM cell (default).",
      flush=True)
print("  * interp=1 None == full grid -> single-channel RMS is the exact baseline.",
      flush=True)
print("  * Double-channel benefits most from a smaller window (inner propagation",
      flush=True)
print("    runs on the smaller grid).", flush=True)
