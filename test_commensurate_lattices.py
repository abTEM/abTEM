#!/usr/bin/env python3
"""
Additional commensurability tests:
  A. New lattice/structure types not covered by earlier tests
  B. All three parametrizations (kirkland, peng, lobato)
"""
import numpy as np
import abtem
from ase.spacegroup import crystal
from abtem.atoms import is_cell_orthogonal

SEP = "─" * 80
THRESHOLD = 1e-3
_results = []


# ── helpers ──────────────────────────────────────────────────────────────────
def _potential_at_atoms(sc, projection, parametrization):
    pot = abtem.Potential(sc, sampling="auto", slice_thickness="auto",
                          projection=projection, parametrization=parametrization)
    arr = pot.build().compute().array.real.sum(axis=0)   # (gpts_x, gpts_y)
    gpts = pot.gpts
    dx = sc.cell[0, 0] / gpts[0]
    dy = sc.cell[1, 1] / gpts[1]
    vals = np.array([
        arr[int(round(p[0]/dx)) % gpts[0],
            int(round(p[1]/dy)) % gpts[1]]
        for p in sc.positions
    ])
    return vals, gpts, pot.sampling, pot.slice_thickness


def run_test(name, atoms, projection="infinite", parametrization="kirkland"):
    if not is_cell_orthogonal(atoms):
        print(f"  ~ {name}  [SKIP — non-orthogonal]")
        return None

    n_base = len(atoms)
    sc = atoms.repeat([3, 3, 1])
    try:
        vals, gpts, samp, st = _potential_at_atoms(sc, projection, parametrization)
    except Exception as e:
        print(f"  ✗ {name}  ERROR: {e}")
        _results.append(False)
        return None

    spreads = [(vals[b::n_base].max() - vals[b::n_base].min()) / vals[b::n_base].mean()
               for b in range(n_base) if vals[b::n_base].mean() > 1.0]
    max_spread = max(spreads) if spreads else 0.0
    passed = max_spread < THRESHOLD
    flag = "✓" if passed else "✗"
    print(f"  {flag} {name}")
    print(f"      gpts=({gpts[0]},{gpts[1]})  samp=({samp[0]:.4f},{samp[1]:.4f})  "
          f"slices={len(st)}  spread={max_spread:.2e}")
    _results.append(passed)
    return gpts


# ── structure builders ────────────────────────────────────────────────────────
def make_MgO():
    return crystal(["Mg","O"], [(0,0,0),(0.5,0,0)],
                   spacegroup=225, cellpar=[4.211]*3+[90]*3)

def make_NaCl():
    return crystal(["Na","Cl"], [(0,0,0),(0.5,0,0)],
                   spacegroup=225, cellpar=[5.640]*3+[90]*3)

def make_CaF2():
    return crystal(["Ca","F"], [(0,0,0),(0.25,0.25,0.25)],
                   spacegroup=225, cellpar=[5.463]*3+[90]*3)

def make_GaAs():
    return crystal(["Ga","As"], [(0,0,0),(0.25,0.25,0.25)],
                   spacegroup=216, cellpar=[5.653]*3+[90]*3)

def make_diamond():
    return crystal(["C"], [(0,0,0)],
                   spacegroup=227, cellpar=[3.567]*3+[90]*3)

def make_BaTiO3_tet():
    # P4mm (sg=99), a=3.992, c=4.032 — off-centre Ti at z≈0.512, O at z≈0.49
    return crystal(["Ba","Ti","O","O"],
                   [(0,0,0),(0.5,0.5,0.512),(0.5,0.5,0.0),(0.5,0.0,0.490)],
                   spacegroup=99, cellpar=[3.992,3.992,4.032,90,90,90])

def make_brookite():
    # TiO2 brookite, Pbca (sg=61), genuinely orthorhombic a≠b≠c
    return crystal(["Ti","O","O"],
                   [(0.1288,0.0993,0.8625),
                    (0.0102,0.1491,0.1834),
                    (0.2309,0.1121,0.5352)],
                   spacegroup=61,
                   cellpar=[9.184,5.447,5.145,90,90,90])

def make_InP():   # zinc-blende, different from GaAs
    return crystal(["In","P"], [(0,0,0),(0.25,0.25,0.25)],
                   spacegroup=216, cellpar=[5.869]*3+[90]*3)

def make_VO2_rutile():
    # VO2 (rutile, P4_2/mnm sg=136), tetragonal a≠c, metallic V
    return crystal(["V","O"], [(0,0,0),(0.3,0.3,0.0)],
                   spacegroup=136, cellpar=[4.554,4.554,2.851,90,90,90])

def make_SrVO3():
    # SrVO3 perovskite (cubic), like SrTiO3 but with V (d-metal)
    return crystal(["Sr","V","O"], [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0)],
                   spacegroup=221, cellpar=[3.841]*3+[90]*3)


# ════════════════════════════════════════════════════════════════════════════
print(SEP)
print("A. NEW LATTICE / STRUCTURE TYPES  (infinite + finite projection)")
print(SEP)

structures = [
    ("MgO rock-salt (cubic)",          make_MgO()),
    ("NaCl rock-salt (cubic)",         make_NaCl()),
    ("CaF2 fluorite (cubic)",          make_CaF2()),
    ("C diamond (cubic)",              make_diamond()),
    ("GaAs zinc-blende (cubic)",       make_GaAs()),
    ("InP  zinc-blende (cubic)",       make_InP()),
    ("BaTiO3 tetragonal P4mm",         make_BaTiO3_tet()),
    ("VO2 rutile tetragonal",          make_VO2_rutile()),
    ("SrVO3 perovskite (cubic)",       make_SrVO3()),
    ("TiO2 brookite (orthorhombic)",   make_brookite()),
]

for label, atoms in structures:
    run_test(f"{label}  [infinite]", atoms, projection="infinite")
    run_test(f"{label}  [finite]",   atoms, projection="finite")

# also test orthorhombic brookite with centering
brookite = make_brookite()
brookite_c = brookite.copy(); brookite_c.center()
run_test("TiO2 brookite  [centered, infinite]", brookite_c, projection="infinite")

# ════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("B. PARAMETRIZATIONS  (kirkland / peng / lobato)")
print("   Testing: gpts is geometric (must be identical), spread must be <1e-3")
print(SEP)

from ase.build import bulk
Si  = bulk("Si",  cubic=True)
Al  = bulk("Al",  cubic=True)
Au  = bulk("Au",  cubic=True)
MgO = make_MgO()

print(f"  {'Structure':25s}  {'param':10s}  gpts           samp       slices  spread")
print(f"  {'-'*25}  {'-'*10}  {'-'*14} {'-'*10} {'-'*6}  {'-'*10}")

for label, atoms in [("Si cubic [001]", Si),
                     ("Al FCC  [001]",  Al),
                     ("Au FCC  [001]",  Au),
                     ("MgO rock-salt",  MgO)]:
    gpts_ref = None
    for param in ("kirkland", "peng", "lobato"):
        if not is_cell_orthogonal(atoms):
            print(f"  ~ {label:25s}  SKIP — non-orthogonal"); break
        n_base = len(atoms)
        sc = atoms.repeat([3, 3, 1])
        try:
            vals, gpts, samp, st = _potential_at_atoms(sc, "infinite", param)
        except Exception as e:
            print(f"  ✗ {label:25s}  {param:10s}  ERROR: {e}")
            _results.append(False)
            continue

        spreads = [(vals[b::n_base].max()-vals[b::n_base].min())/vals[b::n_base].mean()
                   for b in range(n_base) if vals[b::n_base].mean() > 1.0]
        spread = max(spreads) if spreads else 0.0
        passed = spread < THRESHOLD
        if gpts_ref is None: gpts_ref = gpts
        gpts_ok = (gpts == gpts_ref)
        flag = "✓" if (passed and gpts_ok) else "✗"
        gpts_note = "" if gpts_ok else f" ← MISMATCH vs {gpts_ref}"
        print(f"  {flag} {label:25s}  {param:10s}  "
              f"({gpts[0]},{gpts[1]}){gpts_note:4s}  "
              f"({samp[0]:.4f},{samp[1]:.4f})  {len(st):5d}  {spread:.2e}")
        _results.append(passed and gpts_ok)
    print()

# ════════════════════════════════════════════════════════════════════════════
print(SEP)
n_pass = sum(_results)
n_total = len(_results)
if n_pass == n_total:
    print(f"Overall: ALL PASS  ({n_pass}/{n_total})")
else:
    print(f"Overall: {n_pass}/{n_total} passed — {n_total-n_pass} FAILED")
print(SEP)
