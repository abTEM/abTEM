#!/usr/bin/env python3
"""
Extended commensurability tests:
  1. Centering / arbitrary translation of bulk periodic crystals
  2. Slab geometry with vacuum (pbc=[T,T,F]), with & without centering
  3. Various input cell sizes (1x1, 2x2, 4x4)
  4. Crystal-system variety (cubic, tetragonal, hexagonal, orthorhombic)
  5. Finite projection
"""
import numpy as np
import abtem
from ase.build import bulk, fcc100, fcc111, fcc110, bcc100, bcc110
from ase.spacegroup import crystal
from abtem.atoms import is_cell_orthogonal

SEP = "─" * 80
THRESHOLD = 1e-3
_results = []


def _potential_at_atoms(sc, projection):
    pot = abtem.Potential(sc, sampling="auto", slice_thickness="auto",
                          projection=projection)
    slices = pot.build().compute()
    arr = slices.array.real.sum(axis=0)          # (gpts_x, gpts_y)
    gpts = pot.gpts
    dx = sc.cell[0, 0] / gpts[0]
    dy = sc.cell[1, 1] / gpts[1]
    vals = []
    for pos in sc.positions:
        ix = int(round(pos[0] / dx)) % gpts[0]
        iy = int(round(pos[1] / dy)) % gpts[1]
        vals.append(float(arr[ix, iy]))
    return np.array(vals), gpts, pot.sampling, pot.slice_thickness


def run_test(name, atoms, projection="infinite"):
    if not is_cell_orthogonal(atoms):
        # Non-orthogonal (hexagonal) cells: the potential grid is built on the
        # orthogonalized cell whose extent differs from sc.cell[i,i], so the
        # simple grid-index mapping used here breaks.  These symmetries are
        # covered by test_commensurate.py which handles them correctly.
        print(f"  ~ {name}")
        print(f"      SKIP — non-orthogonal cell (covered by test_commensurate.py)")
        return

    n_base = len(atoms)
    sc = atoms.repeat([3, 3, 1])
    try:
        vals, gpts, samp, st = _potential_at_atoms(sc, projection)
    except Exception as e:
        print(f"  ✗ {name}")
        print(f"      ERROR: {e}")
        _results.append(False)
        return

    # 9 lateral copies of each primitive-cell atom
    spreads = []
    for base in range(n_base):
        copies = vals[base::n_base]
        if copies.mean() > 1.0:
            rel = (copies.max() - copies.min()) / copies.mean()
            spreads.append(rel)

    max_spread = max(spreads) if spreads else 0.0
    passed = max_spread < THRESHOLD
    flag = "✓" if passed else "✗"
    gx, gy = gpts
    sx, sy = samp
    print(f"  {flag} {name}")
    print(f"      gpts=({gx},{gy})  samp=({sx:.4f},{sy:.4f})  "
          f"slices={len(st)}  spread={max_spread:.2e}")
    _results.append(passed)


def centered(atoms):
    a = atoms.copy(); a.center(); return a

def shifted(atoms, d=(0.13, 0.09, 0.07)):
    a = atoms.copy(); a.translate(list(d)); a.wrap(); return a


# ──────────────────────────────────────────────────────────────────────────────
print(SEP)
print("1. CENTERING & ARBITRARY TRANSLATION — bulk periodic crystals")
print(SEP)

Si  = bulk("Si",  cubic=True)
Al  = bulk("Al",  cubic=True)
Au  = bulk("Au",  cubic=True)
Fe  = bulk("Fe",  cubic=True)
STO = crystal(["Sr","Ti","O"], basis=[(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0)],
              spacegroup=221, cellpar=[3.905]*3+[90]*3)
TiO2 = crystal(["Ti","O"], basis=[(0,0,0),(0.3,0.3,0)],
               spacegroup=136, cellpar=[4.594,4.594,2.959,90,90,90])
GaN = bulk("GaN",  crystalstructure="wurtzite",     a=3.19,  c=5.189)
ZnO = bulk("ZnO",  crystalstructure="wurtzite",     a=3.25,  c=5.207)
from ase.build import mx2
MoS2 = mx2(formula="MoS2", kind="2H", a=3.184, thickness=3.127, vacuum=0)

for label, atoms in [
    ("Si cubic",   Si),
    ("Al FCC",     Al),
    ("Au FCC",     Au),
    ("Fe BCC",     Fe),
    ("SrTiO3 (perovskite, cubic)", STO),
    ("TiO2 rutile (tetragonal)", TiO2),
    ("GaN wurtzite (hexagonal)", GaN),
    ("ZnO wurtzite (hexagonal)", ZnO),
    ("MoS2 (hexagonal)",         MoS2),
]:
    run_test(f"{label}  [standard]",           atoms)
    run_test(f"{label}  [center()]",           centered(atoms))
    run_test(f"{label}  [shift 0.13/0.09/0.07 Å]", shifted(atoms))

# ──────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("2. SLAB GEOMETRY WITH VACUUM — pbc=[T,T,F]")
print(SEP)

slabs = [
    ("Al FCC(100)",  fcc100("Al", size=(1, 1, 4), vacuum=10.0)),
    ("Al FCC(111)",  fcc111("Al", size=(1, 1, 4), vacuum=10.0)),
    ("Al FCC(110)",  fcc110("Al", size=(1, 1, 4), vacuum=10.0)),
    ("Fe BCC(100)",  bcc100("Fe", size=(1, 1, 4), vacuum=10.0)),
    ("Fe BCC(110)",  bcc110("Fe", size=(1, 1, 4), vacuum=10.0)),
    ("Al FCC(100) large vacuum", fcc100("Al", size=(1, 1, 4), vacuum=30.0)),
]

for label, slab in slabs:
    run_test(f"{label}  [standard]",  slab)
    run_test(f"{label}  [center()]",  centered(slab))

# ──────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("3. DIFFERENT INPUT CELL SIZES (directly, not as test supercell)")
print(SEP)

for label, atoms in [
    ("Si cubic 2×2×2",   Si * (2, 2, 2)),
    ("Si cubic 4×4×1",   Si * (4, 4, 1)),
    ("Al FCC  2×2×2",    Al * (2, 2, 2)),
    ("SrTiO3 (perovskite) 2×2×1",  STO * (2, 2, 1)),
    ("TiO2    2×2×1",    TiO2 * (2, 2, 1)),
    ("GaN     2×2×1",    GaN * (2, 2, 1)),
]:
    run_test(f"{label}  [standard]", atoms)
    run_test(f"{label}  [center()]", centered(atoms))

# ──────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("4. FINITE PROJECTION — slabs and centered bulks")
print(SEP)

for label, atoms in [
    ("Al FCC(100) slab std",  fcc100("Al", size=(1, 1, 4), vacuum=10.0)),
    ("Al FCC(100) slab cen",  centered(fcc100("Al", size=(1, 1, 4), vacuum=10.0))),
    ("Fe BCC(100) slab std",  bcc100("Fe", size=(1, 1, 4), vacuum=10.0)),
    ("Fe BCC(100) slab cen",  centered(bcc100("Fe", size=(1, 1, 4), vacuum=10.0))),
    ("Si cubic std",          Si),
    ("Si cubic cen",          centered(Si)),
    ("SrTiO3 std",            STO),
    ("SrTiO3 cen",            centered(STO)),
]:
    run_test(label, atoms, projection="finite")

# ──────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
n_pass = sum(_results)
n_total = len(_results)
# ──────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("5. HEXAGONAL SYSTEMS — gpts translation-invariance check")
print("   (potential-spread check requires orthogonal cell; see test_commensurate.py)")
print(SEP)

from abtem.slicing import commensurate_gpts
from abtem.atoms import best_orthogonal_cell, orthogonalize_cell, plane_to_axes
from abtem.potentials.iam import _require_cell_transform

def gpts_for(atoms):
    """Return the commensurate gpts that Potential would choose for sampling='auto'."""
    cell = np.array(atoms.cell)
    plane, box, origin = "xy", None, (0, 0, 0)
    if _require_cell_transform(cell, box=box, plane=plane, origin=origin):
        axes = plane_to_axes(plane)
        cell_2d = cell[:, list(axes)]
        auto_box = tuple(best_orthogonal_cell(cell_2d))
        extent = auto_box[:2]
        auto_atoms = orthogonalize_cell(atoms, box=auto_box, plane=plane,
                                        origin=origin, return_transform=False,
                                        allow_transform=True)
    else:
        extent = (float(cell[0, 0]), float(cell[1, 1]))
        auto_atoms = atoms
    return commensurate_gpts(extent, auto_atoms.positions)

for label, atoms in [
    ("GaN wurtzite",  GaN),
    ("ZnO wurtzite",  ZnO),
    ("MoS2",          MoS2),
]:
    g_std = gpts_for(atoms)
    g_cen = gpts_for(centered(atoms))
    g_sft = gpts_for(shifted(atoms))
    inv = (g_std == g_cen == g_sft)
    flag = "✓" if inv else "✗"
    print(f"  {flag} {label:20s}  gpts standard={g_std}  centered={g_cen}  shifted={g_sft}")
    _results.append(inv)

print()
print(SEP)
n_pass = sum(_results)
n_total = len(_results)
if n_pass == n_total:
    print(f"Overall: ALL PASS  ({n_pass}/{n_total})")
else:
    print(f"Overall: {n_pass}/{n_total} passed — {n_total - n_pass} FAILED")
print(SEP)
