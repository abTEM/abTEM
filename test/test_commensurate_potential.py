"""Tests for the commensurate grid and slicing of Potential (the default).

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
- translation invariance of the grid (centering, arbitrary shifts)
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
    """Repeat `primitive_atoms`, build a Potential with the default commensurate
    grid and slicing, and return the largest relative spread of the
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
# One structure per branch of commensurate_gpts/commensurate_slice_thickness:
# rational planes at quarters (diamond Si) and halves (perovskite), irrational
# internal parameters that take the fast-FFT fallback (rutile, brookite), a
# hexagonal input cell that is orthogonalized first (wurtzite, MoS2), and
# off-centre z planes (tetragonal BaTiO3). The other cubic lattices of
# LATTICE_STRUCTURES differ from these only in lattice constant and basis,
# which change the numbers but not the branch; all 19 still go through the
# gpts invariance test below. The device only matters for the build, so the
# accelerator runs one structure per kind of grid search.
_SPREAD_SWEEP = {
    "Si cubic": True,
    "SrTiO3 perovskite": False,
    "TiO2 rutile": True,
    "GaN wurtzite": True,
    "MoS2": False,
    "BaTiO3 tetragonal": False,
    "TiO2 brookite": False,
}


@pytest.mark.parametrize(
    "name, device",
    [pytest.param(name, "cpu", id=f"{name}-cpu") for name in _SPREAD_SWEEP]
    + [
        pytest.param(
            name,
            gpu.values[0],
            marks=gpu.marks,
            id=f"{name}-{gpu.id or gpu.values[0]}",
        )
        for name, on_accelerator in _SPREAD_SWEEP.items()
        if on_accelerator
    ],
)
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
def test_gpts_is_translation_invariant():
    """commensurate_gpts must give the same grid regardless of how the
    structure is centered or shifted within the cell -- this is the property
    the GCD-based (as opposed to absolute-position-based) algorithm exists
    to guarantee, and is cheap to check for every lattice type since it
    doesn't require building the potential. The slabs are included: a
    centred slab is their shifted case.

    It is also what makes the symmetry spread translation invariant, so the
    spread is not measured again on centred or shifted structures: the spread
    compares atoms related by a lattice translation, and on a grid commensurate
    with the lattice those sit at the same sub-pixel offset however the whole
    structure is shifted."""

    def gpts_for(a):
        return abtem.Potential(a).gpts

    for name, build in {**LATTICE_STRUCTURES, **SLAB_STRUCTURES}.items():
        atoms = build()
        g_standard = gpts_for(atoms)
        g_centered = gpts_for(_centered(atoms))
        g_shifted = gpts_for(_shifted(atoms))
        assert g_standard == g_centered == g_shifted, (
            name,
            g_standard,
            g_centered,
            g_shifted,
        )


# ── slab geometry with vacuum (non-periodic z) ───────────────────────────────
@pytest.mark.parametrize("name", list(SLAB_STRUCTURES))
def test_slab_with_vacuum(name):
    # Not repeated for a centred slab: centring is a rigid shift, which leaves
    # the spread unchanged (see test_gpts_is_translation_invariant).
    atoms = SLAB_STRUCTURES[name]()
    spread, _ = _max_relative_spread(atoms)
    assert spread < THRESHOLD


@pytest.mark.parametrize("name", ["Al FCC(100)", "Fe BCC(100)"])
def test_slab_with_vacuum_finite_projection(name):
    atoms = SLAB_STRUCTURES[name]()
    spread, _ = _max_relative_spread(atoms, projection="finite")
    assert spread < THRESHOLD


# ── varying input supercell size ─────────────────────────────────────────────
# A centred supercell is a rigid shift (see test_gpts_is_translation_invariant),
# and a (4, 4, 1) Si supercell takes the same GCD search as (2, 2, 2) on a
# grid four times larger, so neither is repeated here.
@pytest.mark.parametrize(
    "name, tile",
    [
        ("Si cubic", (2, 2, 2)),
        ("Al FCC", (2, 2, 2)),
        ("SrTiO3 perovskite", (2, 2, 1)),
        ("TiO2 rutile", (2, 2, 1)),
    ],
)
def test_input_supercell_size_invariance(name, tile):
    atoms = LATTICE_STRUCTURES[name]().repeat(tile)
    spread, _ = _max_relative_spread(atoms)
    assert spread < THRESHOLD


# ── parametrization consistency ──────────────────────────────────────────────
def test_parametrization_consistency():
    """gpts is purely geometric and must not depend on which scattering-
    factor parametrization is used. One structure proves it; two elements make
    each parametrization's values differ between the atoms."""
    atoms = LATTICE_STRUCTURES["MgO rock-salt"]()
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


def test_slice_thickness_never_exceeds_the_target_at_the_periodic_boundary():
    """The plane at 3.0 lies within the tolerance of the top face, so it is the plane
    at 0 of the next cell. Each plane gets the part of the cell closest to it, so the
    slice limits are the midpoints (0.5, 1.5, 2.5) and the last slice is the 0.65 Å
    from the last midpoint to the top face; no slice exceeds the target."""
    atoms = _atoms_at_z([0.0, 1.0, 2.0, 3.0], cell_z=3.15)

    result = commensurate_slice_thickness(atoms, target_thickness=1.0)

    assert result == pytest.approx((0.5, 1.0, 1.0, 0.65))


def test_slice_thickness_splits_gaps_thicker_than_target():
    """Planes further apart than the target (cubic fcc has them 0.5 a apart)
    must be split into the fewest equal slices no thicker than the target, not
    left as one thick slice."""
    atoms = bulk("Au", cubic=True)

    result = commensurate_slice_thickness(atoms, target_thickness=1.0)

    # Each plane gets a slice of the target thickness centered on it (so that no slice
    # limit cuts an atom), and the empty gaps between those are split in two; the
    # plane at the cell face is cut in two by it.
    a = atoms.cell[2, 2]
    assert result == pytest.approx(
        (0.5, a / 4 - 0.5, a / 4 - 0.5, 1.0, a / 4 - 0.5, a / 4 - 0.5, 0.5)
    )
    assert max(result) <= 1.0 + 1e-9
    potential = abtem.Potential(atoms)
    assert potential.slice_thickness == pytest.approx(result)


def test_slice_thickness_merges_planes_closer_than_tolerance():
    """Two atoms much closer together than `tolerance` must be recognized as
    a single plane even when they straddle what would be a fixed-grid
    rounding boundary."""
    atoms = _atoms_at_z([0.099, 0.101, 1.1, 2.1], cell_z=3.0)

    result = commensurate_slice_thickness(atoms, target_thickness=1.2, tolerance=0.2)

    # A spurious boundary at 0.2 (from rounding 0.099 and 0.101 into
    # different tolerance-wide bins) would split the first slice into two.
    assert len(result) == 4
    assert result == pytest.approx((0.6, 1.0, 1.0, 0.4))


@pytest.mark.parametrize("target", [0.5, 1.0, 1.7, 3.0])
@pytest.mark.parametrize(
    "atoms",
    [
        bulk("Si", cubic=True) * (1, 1, 3),
        bulk("Au", cubic=True) * (1, 1, 2),
        bulk("Cu", cubic=True) * (2, 2, 3),
        bulk("Mg", "hcp", orthorhombic=True) * (1, 1, 2),
    ],
    ids=["Si", "Au", "Cu", "Mg"],
)
def test_slice_limits_never_cut_an_atom(atoms, target):
    """A limit on an atom splits it between two slices, which smears its potential
    along the beam when the projection is finite and, under frozen phonons, hands
    a random part of the plane to either side."""
    result = commensurate_slice_thickness(atoms, target_thickness=target)

    cell_z = atoms.cell[2, 2]
    limits = np.cumsum(result)[:-1]
    z = atoms.positions[:, 2] % cell_z
    distance = np.abs(z[None] - limits[:, None])
    distance = np.minimum(distance, cell_z - distance)
    assert distance.min() > 0.05
    assert max(result) <= target * (1 + 1e-9)
    assert sum(result) == pytest.approx(cell_z)
    # equal slices are bit-identical, so that the propagator is computed once
    assert len(set(result)) <= len(result) // 2 + 2


# ── AtomsEnsemble: commensurability search must be skipped ──────────────────
def test_atoms_ensemble_skips_the_commensurate_search():
    """A multi-configuration `AtomsEnsemble` has no single reference lattice to
    detect commensurability against: each configuration is an independent MD
    snapshot, and `frozen_phonons.atoms` is only the first frame. The grid and
    the slicing must fall back to the plain non-commensurate
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
        ensemble_pot = abtem.Potential(trajectory)

    extent_x, extent_y = float(single.cell[0, 0]), float(single.cell[1, 1])
    # The ensemble path uses the plain target sampling (no commensurate
    # search), rounded up to a fast FFT size like every grid abTEM derives.
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
        explicit_pot = abtem.Potential(abtem.AtomsEnsemble(trajectory))
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

    plain_pot = abtem.Potential(single)
    ensemble_pot = abtem.Potential(abtem.AtomsEnsemble([single]))

    assert ensemble_pot.gpts == plain_pot.gpts
    assert ensemble_pot.slice_thickness == plain_pot.slice_thickness


# ── the defaults are commensurate targets that yield to the waves ────────────
def _gold():
    return bulk("Au", cubic=True).repeat((2, 2, 2))


def test_default_slice_thickness_and_sampling_are_the_configured_targets():
    default = abtem.Potential(_gold())
    explicit = abtem.Potential(_gold(), sampling=0.05, slice_thickness=1.0)

    assert default.slice_thickness == pytest.approx(explicit.slice_thickness)
    assert default.slice_thickness == pytest.approx(
        commensurate_slice_thickness(_gold(), 1.0)
    )
    assert default.gpts == explicit.gpts == (168, 168)
    assert default.grid.gpts_is_default
    assert not explicit.grid.gpts_is_default


def test_configured_targets_are_used_when_none_are_given():
    with abtem.config.set(
        {"potential.sampling": 0.1, "potential.slice-thickness": 2.1}
    ):
        configured = abtem.Potential(_gold())
    explicit = abtem.Potential(_gold(), sampling=0.1, slice_thickness=2.1)

    assert configured.gpts == explicit.gpts
    assert configured.slice_thickness == pytest.approx(explicit.slice_thickness)
    assert max(configured.slice_thickness) <= 2.1
    assert configured.grid.gpts_is_default


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_waves_without_a_grid_adopt_the_default_potential_grid(device):
    potential = abtem.Potential(_gold(), device=device)
    waves = abtem.PlaneWave(energy=200e3, device=device).multislice(potential)

    assert waves.gpts == potential.gpts == (168, 168)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "grid_kwargs, expected_gpts",
    [({"gpts": 100}, (100, 100)), ({"sampling": 0.1}, (82, 82))],
)
def test_explicit_waves_grid_overrides_the_default_potential_grid(
    device, grid_kwargs, expected_gpts
):
    potential = abtem.Potential(_gold(), device=device)
    waves = abtem.PlaneWave(energy=200e3, device=device, **grid_kwargs).multislice(
        potential
    )

    assert waves.gpts == potential.gpts == expected_gpts


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("grid_kwargs", [{"gpts": 100}, {"sampling": 0.1}])
def test_explicit_potential_sampling_is_not_overridden_by_the_waves(
    device, grid_kwargs, recwarn
):
    potential = abtem.Potential(_gold(), sampling=0.05, device=device)
    waves = abtem.PlaneWave(energy=200e3, device=device, **grid_kwargs)

    exit_waves = waves.multislice(potential)

    assert exit_waves.gpts == (168, 168)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_waves_that_adopted_a_grid_do_not_count_as_explicit(device, recwarn):
    """A probe reused with a second potential has a grid only because it took one
    from the first; that is not an explicit request to override the grid of a
    default potential."""
    first = abtem.Potential(_gold(), device=device)
    second = abtem.Potential(bulk("Au", cubic=True).repeat((3, 3, 1)), device=device)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20, device=device)

    probe.grid.match(first)
    probe.grid.match(second)

    assert probe.gpts == second.gpts
    assert second.grid.gpts_is_default
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_probe_matched_to_a_default_potential_adopts_its_grid(device):
    potential = abtem.Potential(_gold(), device=device)
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=20, device=device)

    probe.grid.match(potential)
    measurement = probe.scan(
        potential,
        scan=abtem.GridScan((0, 0), (4, 4), sampling=2.0),
        detectors=abtem.AnnularDetector(50, 200),
    )

    assert probe.gpts == potential.gpts == (168, 168)
    assert measurement.shape == (2, 2)


def test_default_slicing_and_grid_of_a_structure_without_atoms():
    empty = _gold()[:0]

    potential = abtem.Potential(empty)

    assert sum(potential.slice_thickness) == pytest.approx(empty.cell[2, 2])
    assert potential.gpts is not None


# ── targets are upper bounds ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "make_atoms",
    [
        lambda: _si() * (3, 3, 2),
        lambda: _au() * (2, 2, 2),
        lambda: _tio2_rutile() * (3, 3, 2),
        lambda: _gan_wurtzite(),
        lambda: _mos2(),
        lambda: _srtio3() * (2, 2, 2),
    ],
    ids=["Si", "Au", "rutile", "GaN", "MoS2", "SrTiO3"],
)
@pytest.mark.parametrize(
    "sampling, slice_thickness", [(0.02, 0.4), (0.05, 1.0), (0.2, 2.5)]
)
def test_grid_and_slices_are_never_coarser_than_requested(
    make_atoms, sampling, slice_thickness
):
    potential = abtem.Potential(
        make_atoms(), sampling=sampling, slice_thickness=slice_thickness
    )

    assert max(potential.sampling) <= sampling * (1 + 1e-9)
    assert max(potential.slice_thickness) <= slice_thickness * (1 + 1e-9)


@pytest.mark.parametrize("n_periods", [150, 180, 199, 250, 450])
def test_commensurate_gpts_never_coarser_and_bounded_in_cost(n_periods):
    # Planes about as fine as the target: the commensurate grid nearest the
    # target was up to 1.5 times coarser than it. The coarsest commensurate grid
    # that meets the target can have twice the points, so beyond 15 % more the
    # plain grid is used.
    extent = 10.0
    planes = np.arange(n_periods) * extent / n_periods
    positions = np.stack([planes, planes, np.zeros_like(planes)], axis=1)

    gpts = commensurate_gpts(
        (extent, extent), positions, target_sampling=0.05, round_to_fast_fft=False
    )

    for n in gpts:
        assert extent / n <= 0.05 * (1 + 1e-9)
        assert n <= 200 * 1.15
        assert n % n_periods == 0 or n == 200


# ── explicit control: gpts and num_slices ────────────────────────────────────
def test_num_slices_divides_the_height_into_equal_slices():
    atoms = _si() * (2, 2, 3)

    potential = abtem.Potential(atoms, num_slices=7)

    assert potential.slice_thickness == pytest.approx((atoms.cell[2, 2] / 7,) * 7)


def test_num_slices_follows_the_plane_and_the_box():
    atoms = _srtio3() * (2, 3, 4)
    a = _srtio3().cell[0, 0]

    in_plane = abtem.Potential(atoms, plane="xz", num_slices=5)
    in_box = abtem.Potential(_srtio3(), box=(2 * a, 3 * a, 4 * a), num_slices=5)

    assert sum(in_plane.slice_thickness) == pytest.approx(atoms.cell[1, 1])
    assert in_box.slice_thickness == pytest.approx((4 * a / 5,) * 5)


@pytest.mark.parametrize("num_slices", [0, -1, 2.5, True, "4"])
def test_invalid_num_slices_raises(num_slices):
    with pytest.raises(ValueError, match="num_slices"):
        abtem.Potential(_si(), num_slices=num_slices)


def test_slice_thickness_and_num_slices_are_exclusive():
    with pytest.raises(ValueError, match="either slice_thickness or num_slices"):
        abtem.Potential(_si(), slice_thickness=1.0, num_slices=4)


def test_gpts_and_slice_thickness_sequences_are_used_exactly():
    atoms = _si() * (2, 2, 1)
    thicknesses = (1.0, 2.0, atoms.cell[2, 2] - 3.0)

    potential = abtem.Potential(atoms, gpts=(101, 103), slice_thickness=thicknesses)

    assert potential.gpts == (101, 103)
    assert potential.slice_thickness == pytest.approx(thicknesses)


# ── commensurate=False ───────────────────────────────────────────────────────
def test_without_commensurability_the_grid_and_slices_are_plain():
    atoms = _si() * (3, 3, 2)
    extent = atoms.cell[0, 0]

    explicit = abtem.Potential(
        atoms, sampling=0.05, slice_thickness=1.0, commensurate=False
    )
    default = abtem.Potential(atoms, commensurate=False)

    # As `Grid` derives it from the sampling, and `slice_thickness` divides the
    # height, before commensurability was introduced.
    plain = abtem.core.grid.Grid(extent=(extent, extent), sampling=0.05).gpts
    n = int(np.ceil(atoms.cell[2, 2] / 1.0))
    for potential in (explicit, default):
        assert potential.gpts == plain
        assert potential.slice_thickness == pytest.approx((atoms.cell[2, 2] / n,) * n)
    assert default.grid.gpts_is_default
    assert not explicit.grid.gpts_is_default


# ── input forms ──────────────────────────────────────────────────────────────
def test_array_sampling_and_slice_thickness_are_accepted():
    atoms = _si() * (2, 2, 2)
    reference = abtem.Potential(atoms, sampling=(0.05, 0.04), slice_thickness=0.9)
    thicknesses = np.full(4, atoms.cell[2, 2] / 4)

    from_arrays = abtem.Potential(
        atoms, sampling=np.array([0.05, 0.04]), slice_thickness=np.float64(0.9)
    )
    sequence = abtem.Potential(atoms, slice_thickness=thicknesses)

    assert from_arrays.gpts == reference.gpts
    assert from_arrays.slice_thickness == reference.slice_thickness
    assert sequence.slice_thickness == pytest.approx(tuple(thicknesses))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sampling": "auto"}, "sampling must be one or two positive"),
        ({"slice_thickness": "auto"}, "slice_thickness must be a positive number"),
        ({"sampling": True}, "sampling must be one or two positive"),
        ({"slice_thickness": True}, "slice_thickness must be a positive number"),
        ({"sampling": -0.05}, "positive"),
        ({"sampling": (0.05, 0.05, 0.05)}, "one or two"),
        ({"slice_thickness": 0.0}, "positive"),
    ],
)
def test_invalid_targets_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        abtem.Potential(_si(), **kwargs)


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("potential.slice-thickness", [0.05, 0.05], "potential.slice-thickness"),
        ("potential.slice-thickness", True, "potential.slice-thickness"),
        ("potential.slice-thickness", "auto", "potential.slice-thickness"),
        ("potential.sampling", True, "potential.sampling"),
        ("potential.sampling", -1.0, "potential.sampling"),
    ],
)
def test_invalid_configured_targets_name_the_configuration_key(key, value, match):
    with abtem.config.set({key: value}):
        with pytest.raises(ValueError, match=match):
            abtem.Potential(_si())


# ── periodicity is that of the potential's axes ──────────────────────────────
def _si_wire():
    # Periodic along x only, with vacuum around it.
    wire = _si() * (3, 1, 1)
    wire.center(vacuum=6.0, axis=(1, 2))
    wire.pbc = (True, False, False)
    return wire


def test_slices_are_commensurate_along_a_periodic_beam_direction_of_a_plane():
    # Viewed along the wire ("yz"), the beam runs along the periodic x of the
    # atoms: the slices are commensurate with its planes (a / 4 apart), which they
    # enclose rather than end on, and no thicker than the default target.
    wire = _si_wire()

    potential = abtem.Potential(wire, plane="yz")

    a = _si().cell[0, 0]
    assert sum(potential.slice_thickness) == pytest.approx(3 * a)
    assert max(potential.slice_thickness) <= 1.0 + 1e-9
    limits = np.cumsum((0.0,) + potential.slice_thickness)
    planes = a / 4 * np.arange(12)
    assert np.min(np.abs(limits[1:-1, None] - planes[None])) > 0.1


def test_grid_is_commensurate_only_along_periodic_axes():
    wire = _si_wire()
    from abtem.core.fft import next_fast_fft_size

    potential = abtem.Potential(wire)

    x, y = potential.extent
    commensurate = commensurate_gpts((x, y), wire.positions, target_sampling=0.05)
    assert potential.gpts[0] == commensurate[0]
    assert potential.gpts[1] == next_fast_fft_size(int(np.ceil(y / 0.05)))
    # Along z the wire is not periodic, so its slices are uniform.
    n = int(np.ceil(wire.cell[2, 2]))
    assert potential.slice_thickness == pytest.approx((wire.cell[2, 2] / n,) * n)


# ── other builders ───────────────────────────────────────────────────────────
def test_lazily_read_trajectory_gets_the_plain_default_grid():
    import dask

    atoms = _si() * (2, 2, 1)
    trajectory = abtem.AtomsEnsemble(
        [dask.delayed(atoms.copy)() for _ in range(2)], cell=atoms.cell
    )

    potential = abtem.Potential(trajectory)

    assert potential.grid.gpts_is_default
    assert potential.slice_thickness == pytest.approx((atoms.cell[2, 2] / 6,) * 6)


@pytest.mark.parametrize("kwargs", [{}, {"sampling": 0.1, "slice_thickness": 0.5}])
def test_magnetic_fields_are_sliced_and_sampled_as_the_potential(kwargs):
    from abtem.magnetism.iam import MagneticField, VectorPotential

    atoms = _fe() * (2, 2, 2)
    potential = abtem.Potential(atoms, **kwargs)

    for field in (MagneticField(atoms, **kwargs), VectorPotential(atoms, **kwargs)):
        assert field.gpts == potential.gpts
        assert field.slice_thickness == potential.slice_thickness
        assert field.grid.gpts_is_default == potential.grid.gpts_is_default


def test_a_rebuilt_potential_keeps_the_resolved_grid_and_slices():
    atoms = _si() * (2, 2, 2)
    potential = abtem.Potential(
        abtem.FrozenPhonons(atoms, 2, sigmas=0.05, seed=1), num_slices=5
    )

    array = potential.build(lazy=True).compute()

    assert array.shape[:2] == (2, 5)
    assert array.gpts == potential.gpts
