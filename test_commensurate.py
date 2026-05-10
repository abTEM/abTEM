"""
Test that commensurate sampling gives constant projected potential for equivalent atoms.

Strategy: build each crystal as an N×N×1 supercell (N=3), then check that
atoms related by the primitive-cell translation — i.e., at positions that
differ by integer multiples of the primitive cell vectors — all give the same
projected potential value at their grid point.
"""
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
import abtem
from abtem.atoms import rotate_atoms_to_plane


def test_commensurate(label, prim_atoms, projection, plane=None,
                      supercell=(3, 3, 1), tol=1e-4):
    """
    Build a supercell, compute the projected potential with auto settings,
    and verify that equivalent atoms (same position mod primitive cell) give
    the same projected potential value.
    Returns True if all checks pass.
    """
    sc = prim_atoms * supercell

    if plane is not None:
        sc = rotate_atoms_to_plane(sc, plane)

    pot = abtem.Potential(sc, sampling="auto", slice_thickness="auto",
                          projection=projection)
    pot_arr = pot.build(lazy=False)

    proj = pot_arr.array.sum(axis=0)   # (nx, ny)
    nx, ny = proj.shape
    sx, sy = pot_arr.sampling
    lx, ly = pot_arr.extent

    pos = sc.get_positions()
    nums = sc.numbers

    # Grid indices for each atom
    xi = np.round(pos[:, 0] / sx).astype(int) % nx
    yi = np.round(pos[:, 1] / sy).astype(int) % ny
    pot_vals = proj[xi, yi]

    # Group by (Z, fractional position mod PRIMITIVE CELL)
    # The primitive cell xy extents from the original atoms (before rotation/supercell)
    if plane is not None:
        prim_sc = rotate_atoms_to_plane(prim_atoms.copy(), plane)
    else:
        prim_sc = prim_atoms.copy()
    prim_lx = float(np.linalg.norm(prim_sc.cell[0]))
    prim_ly = float(np.linalg.norm(prim_sc.cell[1]))

    tol_frac = 1e-3
    frac_x = (pos[:, 0] % prim_lx) / prim_lx
    frac_y = (pos[:, 1] % prim_ly) / prim_ly
    frac_id_x = np.round(frac_x / tol_frac).astype(int)
    frac_id_y = np.round(frac_y / tol_frac).astype(int)

    groups = {}
    for i in range(len(nums)):
        key = (int(nums[i]), int(frac_id_x[i]), int(frac_id_y[i]))
        groups.setdefault(key, []).append(float(pot_vals[i]))

    print(f"\n{'─'*60}")
    print(f"{label}  projection={projection}  supercell={supercell}")
    print(f"  gpts={pot.gpts}  "
          f"sampling=({pot.sampling[0]:.4f},{pot.sampling[1]:.4f})  "
          f"slices={pot.num_slices}")

    ok = True
    n_tested = 0
    for (Z, fx, fy), vals in sorted(groups.items()):
        if len(vals) < 2:
            continue
        vals = np.array(vals)
        spread = float(vals.max() - vals.min())
        mean = float(abs(vals.mean()))
        rel = spread / (mean + 1e-30)
        status = "PASS" if rel < tol else "FAIL"
        if status == "FAIL":
            ok = False
        n_tested += 1
        # Only print first and failing groups (otherwise output is huge)
        if status == "FAIL" or n_tested <= 2:
            print(f"  {status}  Z={Z:3d}  n={len(vals):3d}  "
                  f"mean={mean:10.4f}  spread={spread:.2e}  rel={rel:.2e}")
    if n_tested == 0:
        print("  SKIP  no comparable groups found — check supercell size")
        ok = False
    else:
        n_fail = sum(
            1 for (Z, fx, fy), vals in groups.items()
            if len(vals) >= 2 and
            (np.max(vals) - np.min(vals)) / (abs(np.mean(vals)) + 1e-30) >= tol
        )
        n_pass = n_tested - n_fail
        summary = "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL / {n_pass} PASS"
        print(f"  Summary: {summary}  ({n_tested} groups tested)")

    return ok


ALL_PASS = True

def run(*args, **kwargs):
    global ALL_PASS
    ok = test_commensurate(*args, **kwargs)
    if not ok:
        ALL_PASS = False


# ── Silicon cubic ──────────────────────────────────────────────────────────────
si = bulk("Si", cubic=True)
run("Si cubic [001]", si, "infinite")
run("Si cubic [001]", si, "finite")
run("Si cubic [110] (xz plane)", si * (2, 1, 1), "infinite", plane="xz",
    supercell=(3, 3, 1))
run("Si cubic [110] (yz plane)", si * (1, 2, 1), "infinite", plane="yz",
    supercell=(3, 3, 1))

# ── Aluminium FCC ──────────────────────────────────────────────────────────────
al = bulk("Al", cubic=True)
run("Al FCC [001]", al, "infinite")
run("Al FCC [001]", al, "finite")

# ── SrTiO3 perovskite ──────────────────────────────────────────────────────────
srtio3 = crystal(["Sr", "Ti", "O"],
                 basis=[(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0)],
                 spacegroup=221,
                 cellpar=[3.905,3.905,3.905,90,90,90])
run("SrTiO3 [001]", srtio3, "infinite")
run("SrTiO3 [001]", srtio3, "finite")
run("SrTiO3 [110]", srtio3 * (2,1,1), "infinite", plane="xz", supercell=(3,3,1))

# ── MoS2 hexagonal ────────────────────────────────────────────────────────────
mos2 = crystal(["Mo","S"],
               basis=[(1/3,2/3,1/4),(1/3,2/3,0.621)],
               spacegroup=194,
               cellpar=[3.16,3.16,12.3,90,90,120])
run("MoS2 [001]", mos2, "infinite")

# ── Gold FCC ───────────────────────────────────────────────────────────────────
au = bulk("Au", cubic=True)
run("Au FCC [001]", au, "infinite")
run("Au FCC [001]", au, "finite")

# ── GaN wurtzite ───────────────────────────────────────────────────────────────
gan = crystal(["Ga","N"],
              basis=[(1/3,2/3,0),(1/3,2/3,0.376)],
              spacegroup=186,
              cellpar=[3.189,3.189,5.185,90,90,120])
run("GaN wurtzite [001]", gan, "infinite")
run("GaN wurtzite [001]", gan, "finite")

# ── TiO2 rutile ────────────────────────────────────────────────────────────────
tio2 = crystal(["Ti","O"],
               basis=[(0,0,0),(0.306,0.306,0)],
               spacegroup=136,
               cellpar=[4.594,4.594,2.959,90,90,90])
run("TiO2 rutile [001]", tio2, "infinite")
run("TiO2 rutile [001]", tio2, "finite")

# ── BCC Iron ───────────────────────────────────────────────────────────────────
fe = bulk("Fe", cubic=True)
run("Fe BCC [001]", fe, "infinite")
run("Fe BCC [001]", fe, "finite")

# ── Wurtzite ZnO ───────────────────────────────────────────────────────────────
zno = crystal(["Zn","O"],
              basis=[(1/3,2/3,0),(1/3,2/3,0.382)],
              spacegroup=186,
              cellpar=[3.249,3.249,5.206,90,90,120])
run("ZnO wurtzite [001]", zno, "infinite")
run("ZnO wurtzite [001]", zno, "finite")

print(f"\n{'='*60}")
print(f"Overall: {'ALL PASS' if ALL_PASS else 'FAILURES DETECTED'}")
