"""Tests for Potential(sampling='auto', slice_thickness='auto').

The core physical claim these tests check: atoms related by a lattice
translation must receive an identical discretised potential value, i.e. the
grid chosen by `commensurate_gpts`/`commensurate_slice_thickness` must not
introduce spurious sub-pixel misalignment between symmetry-equivalent atoms.

This migrates the assertions of the standalone verification scripts
test_commensurate.py, test_commensurate_extended.py and
test_commensurate_lattices.py (previously committed at the repo root, where
they were never run by pytest and in fact broke `pytest` collection there)
into real, parametrized pytest coverage. Combined they span:

- many crystal systems/Bravais lattices (cubic, tetragonal, orthorhombic,
  hexagonal; FCC/BCC/rock-salt/fluorite/zinc-blende/diamond/wurtzite/
  perovskite/rutile/brookite), including non-orthogonal input cells that
  exercise the orthogonalize_cell transform path
- translation invariance (centering, arbitrary shifts)
- slab geometry with vacuum (non-periodic z)
- varying input supercell size
- infinite and finite projection
- parametrization (kirkland/peng/lobato) consistency
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bcc100, bcc110, bulk, fcc100, fcc110, fcc111, mx2
from ase.spacegroup import crystal
from utils import gpu

import abtem
from abtem.atoms import rotate_atoms_to_plane
from abtem.slicing import commensurate_gpts, commensurate_slice_thickness

THRESHOLD = 1e-3


# ── structure builders ──────────────────────────────────────────────────────
def _si():
    return bulk("Si", cubic=True)


def _al():
    return bulk("Al", cubic=True)


def _au():
    return bulk("Au", cubic=True)


def _fe():
    return bulk("Fe", cubic=True)


def _srtio3():
    return crystal(
        ["Sr", "Ti", "O"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
    )


def _tio2_rutile():
    return crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.306, 0.306, 0)],
        spacegroup=136,
        cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
    )


def _gan_wurtzite():
    return crystal(
        ["Ga", "N"],
        basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.376)],
        spacegroup=186,
        cellpar=[3.189, 3.189, 5.185, 90, 90, 120],
    )


def _zno_wurtzite():
    return crystal(
        ["Zn", "O"],
        basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.382)],
        spacegroup=186,
        cellpar=[3.249, 3.249, 5.206, 90, 90, 120],
    )


def _mos2():
    return mx2(formula="MoS2", kind="2H", a=3.184, thickness=3.127, vacuum=0)


def _mgo():
    return crystal(
        ["Mg", "O"], [(0, 0, 0), (0.5, 0, 0)], spacegroup=225,
        cellpar=[4.211] * 3 + [90] * 3,
    )


def _nacl():
    return crystal(
        ["Na", "Cl"], [(0, 0, 0), (0.5, 0, 0)], spacegroup=225,
        cellpar=[5.640] * 3 + [90] * 3,
    )


def _caf2():
    return crystal(
        ["Ca", "F"], [(0, 0, 0), (0.25, 0.25, 0.25)], spacegroup=225,
        cellpar=[5.463] * 3 + [90] * 3,
    )


def _gaas():
    return crystal(
        ["Ga", "As"], [(0, 0, 0), (0.25, 0.25, 0.25)], spacegroup=216,
        cellpar=[5.653] * 3 + [90] * 3,
    )


def _inp():
    return crystal(
        ["In", "P"], [(0, 0, 0), (0.25, 0.25, 0.25)], spacegroup=216,
        cellpar=[5.869] * 3 + [90] * 3,
    )


def _diamond():
    return crystal(["C"], [(0, 0, 0)], spacegroup=227, cellpar=[3.567] * 3 + [90] * 3)


def _batio3_tetragonal():
    # off-centre Ti/O positions (P4mm): not exactly commensurate at machine precision
    return crystal(
        ["Ba", "Ti", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.512), (0.5, 0.5, 0.0), (0.5, 0.0, 0.490)],
        spacegroup=99,
        cellpar=[3.992, 3.992, 4.032, 90, 90, 90],
    )


def _vo2_rutile():
    return crystal(
        ["V", "O"], [(0, 0, 0), (0.3, 0.3, 0.0)], spacegroup=136,
        cellpar=[4.554, 4.554, 2.851, 90, 90, 90],
    )


def _srvo3():
    return crystal(
        ["Sr", "V", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
        spacegroup=221,
        cellpar=[3.841] * 3 + [90] * 3,
    )


def _brookite():
    # TiO2 brookite (Pbca): genuinely orthorhombic a != b != c
    return crystal(
        ["Ti", "O", "O"],
        [
            (0.1288, 0.0993, 0.8625),
            (0.0102, 0.1491, 0.1834),
            (0.2309, 0.1121, 0.5352),
        ],
        spacegroup=61,
        cellpar=[9.184, 5.447, 5.145, 90, 90, 90],
    )


LATTICE_STRUCTURES = {
    "Si cubic": _si,
    "Al FCC": _al,
    "Au FCC": _au,
    "Fe BCC": _fe,
    "SrTiO3 perovskite": _srtio3,
    "TiO2 rutile": _tio2_rutile,
    "GaN wurtzite": _gan_wurtzite,
    "ZnO wurtzite": _zno_wurtzite,
    "MoS2": _mos2,
    "MgO rock-salt": _mgo,
    "NaCl rock-salt": _nacl,
    "CaF2 fluorite": _caf2,
    "GaAs zinc-blende": _gaas,
    "InP zinc-blende": _inp,
    "C diamond": _diamond,
    "BaTiO3 tetragonal": _batio3_tetragonal,
    "VO2 rutile": _vo2_rutile,
    "SrVO3 perovskite": _srvo3,
    "TiO2 brookite": _brookite,
}

SLAB_STRUCTURES = {
    "Al FCC(100)": lambda: fcc100("Al", size=(1, 1, 4), vacuum=10.0),
    "Al FCC(111)": lambda: fcc111("Al", size=(1, 1, 4), vacuum=10.0),
    "Al FCC(110)": lambda: fcc110("Al", size=(1, 1, 4), vacuum=10.0),
    "Fe BCC(100)": lambda: bcc100("Fe", size=(1, 1, 4), vacuum=10.0),
    "Fe BCC(110)": lambda: bcc110("Fe", size=(1, 1, 4), vacuum=10.0),
}

# Finite-projection (QuadratureProjectionIntegrals) is much more expensive
# than infinite; only spot-check a representative subset of lattice types.
FINITE_PROJECTION_SUBSET = [
    "Si cubic",
    "SrTiO3 perovskite",
    "TiO2 rutile",
    "Fe BCC",
    "TiO2 brookite",
]


def _centered(atoms):
    a = atoms.copy()
    a.center()
    return a


def _shifted(atoms, d=(0.13, 0.09, 0.07)):
    a = atoms.copy()
    a.translate(list(d))
    a.wrap()
    return a


def _max_relative_spread(
    primitive_atoms,
    projection="infinite",
    parametrization="lobato",
    repeat=(3, 3, 1),
    device="cpu",
    reference=None,
):
    """Repeat `primitive_atoms`, build Potential(sampling='auto',
    slice_thickness='auto'), and return the largest relative spread of the
    projected-potential value across atoms related by a `reference`-cell
    translation -- near-zero for a correctly commensurate grid.

    Atoms are grouped by (atomic number, fractional position mod the
    reference cell) rather than by index in the repeated supercell, since
    orthogonalizing a non-orthogonal (e.g. hexagonal) input cell can
    reorder/replicate atoms internally; grouping by absolute physical
    position is robust to that and works uniformly for orthogonal and
    non-orthogonal input cells.
    """
    reference = primitive_atoms if reference is None else reference
    sc = primitive_atoms.repeat(repeat)

    pot = abtem.Potential(
        sc,
        sampling="auto",
        slice_thickness="auto",
        projection=projection,
        parametrization=parametrization,
        device=device,
    )
    array = pot.build(lazy=False).array.real.sum(axis=0)
    nx, ny = array.shape
    lx, ly = pot.extent
    sx, sy = lx / nx, ly / ny

    pos = sc.get_positions()
    xi = np.round(pos[:, 0] / sx).astype(int) % nx
    yi = np.round(pos[:, 1] / sy).astype(int) % ny
    values = array[xi, yi]

    ref_lx = float(np.linalg.norm(reference.cell[0]))
    ref_ly = float(np.linalg.norm(reference.cell[1]))
    tol_frac = 1e-3
    frac_id_x = np.round(((pos[:, 0] % ref_lx) / ref_lx) / tol_frac).astype(int)
    frac_id_y = np.round(((pos[:, 1] % ref_ly) / ref_ly) / tol_frac).astype(int)

    groups: dict[tuple[int, int, int], list[float]] = {}
    for z, fx, fy, value in zip(sc.numbers, frac_id_x, frac_id_y, values):
        groups.setdefault((int(z), int(fx), int(fy)), []).append(float(value))

    max_spread = 0.0
    for group_values in groups.values():
        if len(group_values) < 2:
            continue
        arr = np.array(group_values)
        if abs(arr.mean()) <= 1.0:
            # Skip near-zero-potential groups (e.g. deep vacuum): relative
            # spread is meaningless there.
            continue
        spread = (arr.max() - arr.min()) / abs(arr.mean())
        max_spread = max(max_spread, float(spread))

    return max_spread, pot


# ── lattice-type sweep ───────────────────────────────────────────────────────
@pytest.mark.parametrize("name", list(LATTICE_STRUCTURES))
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_symmetry_equivalent_atoms_get_identical_potential(name, device):
    atoms = LATTICE_STRUCTURES[name]()
    spread, _ = _max_relative_spread(atoms, device=device)
    assert spread < THRESHOLD


@pytest.mark.parametrize("name", FINITE_PROJECTION_SUBSET)
def test_symmetry_equivalent_atoms_get_identical_potential_finite_projection(name):
    atoms = LATTICE_STRUCTURES[name]()
    spread, _ = _max_relative_spread(atoms, projection="finite")
    assert spread < THRESHOLD


# ── translation invariance (centering / arbitrary shift) ────────────────────
@pytest.mark.parametrize(
    "name", ["Si cubic", "SrTiO3 perovskite", "TiO2 rutile", "Fe BCC", "GaN wurtzite"]
)
@pytest.mark.parametrize("transform", ["standard", "centered", "shifted"])
def test_translation_invariance(name, transform):
    atoms = LATTICE_STRUCTURES[name]()
    if transform == "centered":
        atoms = _centered(atoms)
    elif transform == "shifted":
        atoms = _shifted(atoms)
    spread, _ = _max_relative_spread(atoms)
    assert spread < THRESHOLD


@pytest.mark.parametrize("name", list(LATTICE_STRUCTURES))
def test_gpts_is_translation_invariant(name):
    """commensurate_gpts must give the same grid regardless of how the
    structure is centered or shifted within the cell -- this is the property
    the GCD-based (as opposed to absolute-position-based) algorithm exists
    to guarantee, and is cheap to check for every lattice type since it
    doesn't require building the potential."""
    atoms = LATTICE_STRUCTURES[name]()

    def gpts_for(a):
        return abtem.Potential(a, sampling="auto", slice_thickness="auto").gpts

    g_standard = gpts_for(atoms)
    g_centered = gpts_for(_centered(atoms))
    g_shifted = gpts_for(_shifted(atoms))
    assert g_standard == g_centered == g_shifted


# ── slab geometry with vacuum (non-periodic z) ───────────────────────────────
@pytest.mark.parametrize("name", list(SLAB_STRUCTURES))
@pytest.mark.parametrize("transform", ["standard", "centered"])
def test_slab_with_vacuum(name, transform):
    atoms = SLAB_STRUCTURES[name]()
    if transform == "centered":
        atoms = _centered(atoms)
    spread, _ = _max_relative_spread(atoms)
    assert spread < THRESHOLD


@pytest.mark.parametrize("name", ["Al FCC(100)", "Fe BCC(100)"])
def test_slab_with_vacuum_finite_projection(name):
    atoms = SLAB_STRUCTURES[name]()
    spread, _ = _max_relative_spread(atoms, projection="finite")
    assert spread < THRESHOLD


# ── varying input supercell size ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, tile",
    [
        ("Si cubic", (2, 2, 2)),
        ("Si cubic", (4, 4, 1)),
        ("Al FCC", (2, 2, 2)),
        ("SrTiO3 perovskite", (2, 2, 1)),
        ("TiO2 rutile", (2, 2, 1)),
    ],
)
@pytest.mark.parametrize("transform", ["standard", "centered"])
def test_input_supercell_size_invariance(name, tile, transform):
    atoms = LATTICE_STRUCTURES[name]().repeat(tile)
    if transform == "centered":
        atoms = _centered(atoms)
    spread, _ = _max_relative_spread(atoms)
    assert spread < THRESHOLD


# ── parametrization consistency ──────────────────────────────────────────────
@pytest.mark.parametrize("name", ["Si cubic", "Al FCC", "Au FCC", "MgO rock-salt"])
def test_parametrization_consistency(name):
    """gpts is purely geometric and must not depend on which scattering-
    factor parametrization is used."""
    atoms = LATTICE_STRUCTURES[name]()
    gpts_by_parametrization = {}
    for parametrization in ("kirkland", "peng", "lobato"):
        spread, pot = _max_relative_spread(atoms, parametrization=parametrization)
        assert spread < THRESHOLD
        gpts_by_parametrization[parametrization] = pot.gpts
    assert len(set(gpts_by_parametrization.values())) == 1


# ── non-standard axis orientation ────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, tile, plane",
    [
        ("Si cubic", (2, 1, 1), "xz"),
        ("Si cubic", (1, 2, 1), "yz"),
        ("SrTiO3 perovskite", (2, 1, 1), "xz"),
    ],
)
def test_rotated_plane_orientation(name, tile, plane):
    tiled = LATTICE_STRUCTURES[name]().repeat(tile)
    sc = rotate_atoms_to_plane(tiled.repeat((3, 3, 1)), plane)
    reference = rotate_atoms_to_plane(tiled, plane)
    spread, _ = _max_relative_spread(sc, repeat=(1, 1, 1), reference=reference)
    assert spread < THRESHOLD


# ── commensurate_slice_thickness edge cases ──────────────────────────────────
def _atoms_at_z(z, cell_z, cell_xy=5.0):
    return Atoms(
        "C" * len(z),
        positions=[[0, 0, zz] for zz in z],
        cell=[cell_xy, cell_xy, cell_z],
        pbc=True,
    )


def test_slice_thickness_no_degenerate_trailing_slice():
    """A real plane spacing right before the periodic z boundary (>= tolerance
    but well under half the target thickness) must be merged rather than left
    as its own near-zero final slice: the last and first slices are adjacent
    through the periodic wrap at cell_z = 0."""
    atoms = _atoms_at_z([0.0, 1.0, 2.0, 3.0], cell_z=3.15)

    result = commensurate_slice_thickness(atoms, target_thickness=1.0)

    assert min(result) >= 0.5
    assert result == pytest.approx((1.0, 1.0, 1.15))


def test_slice_thickness_merges_planes_closer_than_tolerance():
    """Two atoms much closer together than `tolerance` must be recognized as
    a single plane even when they straddle what would be a fixed-grid
    rounding boundary."""
    atoms = _atoms_at_z([0.099, 0.101, 1.1, 2.1], cell_z=3.0)

    result = commensurate_slice_thickness(atoms, target_thickness=1.0, tolerance=0.2)

    # A spurious boundary at 0.2 (from rounding 0.099 and 0.101 into
    # different tolerance-wide bins) would split the first slice into two.
    assert len(result) == 3
    assert result == pytest.approx((1.1, 1.0, 0.9))


# ── AtomsEnsemble: commensurability search must be skipped ──────────────────
def test_atoms_ensemble_sampling_and_slice_thickness_auto_skip_commensurate_search():
    """A multi-configuration `AtomsEnsemble` has no single reference lattice to
    detect commensurability against: each configuration is an independent MD
    snapshot, and `frozen_phonons.atoms` is only the first frame. 'auto'
    sampling/slice_thickness must fall back to the plain non-commensurate
    behaviour (as already used for non-periodic structures) rather than
    search for commensurate planes in one arbitrary frame."""
    single = _si().repeat((2, 2, 2))
    # Perturb one axis so the target sampling doesn't already divide the
    # extent evenly, which would make the commensurate and plain fallback
    # gpts coincide by chance and defeat the point of this test.
    single.cell[0, 0] += 0.13

    trajectory = [single, single.copy()]
    # Pin the option: the expected gpts below are the default mode's.
    with abtem.config.set({"grid.round-to-fast-fft": "auto"}):
        ensemble_pot = abtem.Potential(
            trajectory, sampling="auto", slice_thickness="auto"
        )

    extent_x, extent_y = float(single.cell[0, 0]), float(single.cell[1, 1])
    # The ensemble path uses the plain target sampling (no commensurate
    # search), rounded up to a fast FFT size like every auto-derived grid.
    from abtem.core.fft import next_fast_fft_size

    expected_gpts = (
        next_fast_fft_size(int(np.ceil(extent_x / 0.05))),
        next_fast_fft_size(int(np.ceil(extent_y / 0.05))),
    )
    assert ensemble_pot.gpts == expected_gpts

    # commensurate_gpts on the same cell/positions should NOT match the plain
    # target-derived fallback -- confirm they actually differ here, so this
    # test would catch a regression to commensurability search.
    commensurate = commensurate_gpts(
        (extent_x, extent_y), single.positions, target_sampling=0.05
    )
    assert commensurate != expected_gpts

    cell_z = float(single.cell[2, 2])
    n_slices = int(np.ceil(cell_z / 1.0))
    assert ensemble_pot.num_slices == n_slices
    assert np.allclose(ensemble_pot.slice_thickness, cell_z / n_slices)

    # Same behaviour when explicitly wrapped in AtomsEnsemble rather than a
    # plain list.
    with abtem.config.set({"grid.round-to-fast-fft": "auto"}):
        explicit_pot = abtem.Potential(
            abtem.AtomsEnsemble(trajectory), sampling="auto", slice_thickness="auto"
        )
    assert explicit_pot.gpts == expected_gpts
    assert explicit_pot.num_slices == n_slices


def test_atoms_ensemble_single_config_still_uses_commensurate_search():
    """A single-configuration `AtomsEnsemble` is unambiguous -- that one frame
    *is* the configuration, e.g. when a static structure is wrapped in an
    `AtomsEnsemble` purely for API uniformity (as `strategies/potentials.py`'s
    `md_frozen_phonons` does for `min_configs=1`). It must get the same
    commensurate grid/slicing as passing the bare `Atoms` directly, not the
    non-commensurate ensemble fallback."""
    single = _si().repeat((2, 2, 2))
    single.cell[0, 0] += 0.13

    plain_pot = abtem.Potential(single, sampling="auto", slice_thickness="auto")
    ensemble_pot = abtem.Potential(
        abtem.AtomsEnsemble([single]), sampling="auto", slice_thickness="auto"
    )

    assert ensemble_pot.gpts == plain_pot.gpts
    assert ensemble_pot.slice_thickness == plain_pot.slice_thickness
