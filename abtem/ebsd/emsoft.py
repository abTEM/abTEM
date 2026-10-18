"""Write reference patterns in EMsoft's EBSD master pattern format.

EMsoft's master pattern files are the de-facto interchange format for EBSD
simulations: kikuchipy, EMsoft's own indexing programs and others read them. A
pattern written here can be projected onto a detector and indexed by those
tools, which is the cheapest route to a real EBSD patch since none of the
detector geometry has to be reimplemented.

What such a file holds is the *modulation* -- the diffraction contrast as a
function of exit direction -- stored on a square Lambert grid. The smooth
background, the energy spectrum and the depth distribution of backscattered
electrons all come from a Monte Carlo simulation that neither abTEM nor
kikuchipy performs; EMsoft keeps them in a separate group of the same file.
A file written here therefore carries no background, which is what a consumer
wants anyway: the standard indexing workflow removes the background from the
*experimental* patterns and matches them against background-free simulations.

.. note::
    EMsoft stores lengths in **nanometres**, not Ångström: lattice parameters
    in nm and Debye-Waller factors in nm². This module converts on the way out.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from ase import Atoms
from ase.cell import Cell

from abtem.ebsd.measurements import SparseProjectionWarning, SphericalPattern

__all__ = ["write_emsoft_master_pattern"]

#: EMsoft's integer code for each crystal system, by space group number.
_CRYSTAL_SYSTEMS = [
    (2, 7),  # triclinic
    (15, 6),  # monoclinic
    (74, 3),  # orthorhombic
    (142, 2),  # tetragonal
    (167, 5),  # trigonal
    (194, 4),  # hexagonal
    (230, 1),  # cubic
]


def _string(value: str) -> np.ndarray:
    """A one-element byte-string array, as Fortran writes character data."""
    return np.array([value.encode()], dtype=f"S{len(value) + 1}")


def _scalar(value, dtype) -> np.ndarray:
    """A one-element array.

    Every scalar in a genuine EMsoft file is a shape-(1,) array rather than an
    HDF5 scalar, because Fortran writes them as rank-1 arrays. Readers index
    them as ``dataset[:][0]``, which raises on a scalar dataspace, so writing
    true scalars produces a file that looks right and cannot be read.
    """
    return np.array([value], dtype=dtype)


def _crystal_system(space_group: int) -> int:
    for maximum, code in _CRYSTAL_SYSTEMS:
        if space_group <= maximum:
            return code
    raise ValueError(f"space group must be between 1 and 230, got {space_group}")


def _field(dataset, name: str):
    """spglib >= 2.5 returns a dataclass, older versions a dict."""
    if isinstance(dataset, dict):
        return dataset[name]
    return getattr(dataset, name)


def _conventional_cell(
    atoms: Atoms, space_group: Optional[int]
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Space group, conventional lattice, and the symmetry-unique sites.

    EMsoft pairs a space group number with lattice parameters in that group's
    *conventional* setting, and lists only the asymmetric unit, expanding it by
    symmetry itself. An ASE cell is often neither -- ``ase.build.bulk`` returns
    the primitive cell, whose rhombohedral axes would contradict the cubic
    space group -- so standardize before writing.

    Returns
    -------
    space_group, lattice, scaled_positions, numbers
    """
    if space_group is not None and not 1 <= int(space_group) <= 230:
        raise ValueError(f"space group must be between 1 and 230, got {space_group}")

    try:
        import spglib
    except ImportError:
        if space_group is None:
            raise ValueError(
                "the space group could not be determined because spglib is not "
                "installed; pass space_group=... explicitly or install spglib"
            ) from None
        warnings.warn(
            "spglib is not installed, so the cell is written as given rather "
            "than in the conventional setting of the space group; this may "
            "disagree with the space group number recorded alongside it"
        )
        return (
            space_group,
            np.asarray(atoms.cell[:], dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.numbers, dtype=int),
        )

    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)

    with warnings.catch_warnings():
        # spglib deprecates its own internal error handling; nothing the
        # caller can act on, and it would otherwise fail under -W error.
        warnings.simplefilter("ignore", DeprecationWarning)
        standardized = spglib.standardize_cell(cell, to_primitive=False)
    if standardized is None:
        raise ValueError(
            "spglib could not standardize the cell; pass space_group=... and "
            "supply an already conventional cell"
        )
    lattice = np.asarray(standardized[0], dtype=float)
    positions = np.asarray(standardized[1], dtype=float)
    numbers = np.asarray(standardized[2], dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset = spglib.get_symmetry_dataset(
            (lattice, positions, numbers)  # type: ignore[arg-type]
        )
    if dataset is None:
        raise ValueError("spglib could not determine the symmetry of the cell")

    found = int(_field(dataset, "number"))
    if space_group is not None and int(space_group) != found:
        warnings.warn(
            f"the given space group {space_group} disagrees with the "
            f"{found} spglib finds for this cell; writing the given one"
        )
        found = int(space_group)

    # Only the asymmetric unit is listed; a consumer applies the symmetry.
    unique = np.unique(_field(dataset, "equivalent_atoms"))

    return found, lattice, positions[unique], numbers[unique]


def _atom_data(
    scaled: np.ndarray, numbers: np.ndarray, debye_waller
) -> tuple[np.ndarray, np.ndarray]:
    """EMsoft's AtomData/Atomtypes: one column per symmetry-unique site.

    Rows are the fractional coordinates, the occupancy, and the Debye-Waller
    factor in nm².
    """
    n_sites = len(numbers)
    debye_waller = np.broadcast_to(np.asarray(debye_waller, dtype=float), (n_sites,))

    data = np.zeros((5, n_sites), dtype=np.float32)
    data[:3] = scaled.T
    data[3] = 1.0
    data[4] = debye_waller

    return data, np.asarray(numbers, dtype=np.int32)


def write_emsoft_master_pattern(
    path: str,
    pattern: SphericalPattern,
    atoms: Atoms,
    npx: Optional[int] = None,
    energy: Optional[float] = None,
    space_group: Optional[int] = None,
    sample_tilt: float = 70.0,
    debye_waller: float | Sequence[float] = 0.005,
    name: str = "abTEM reciprocity multislice",
    overwrite: bool = False,
) -> None:
    """Write a reference pattern as an EMsoft EBSD master pattern file.

    Parameters
    ----------
    path : str
        Destination ``.h5`` file.
    pattern : SphericalPattern
        The reference pattern. Must not have ensemble axes. Its southern
        hemisphere is written from the pattern itself when it covers it (built
        with ``hemisphere='both'``), and otherwise from the inversion image of
        the northern one -- see :attr:`SphericalPattern.southern`.
    atoms : ase.Atoms
        The unit cell the pattern was calculated from, used for the crystal
        metadata a consumer needs to build a phase.
    npx : int, optional
        Half-width of the square Lambert grid; the stored arrays are
        ``2 * npx + 1`` on a side. Defaults to a grid matching the pattern's
        own sampling density, so that no detail is invented or discarded.
    energy : float, optional
        Electron energy [eV]. Defaults to the pattern's metadata.
    space_group : int, optional
        Space group number. Determined with spglib if not given.
    sample_tilt : float, optional
        Sample tilt [degrees] recorded in the file (default 70.0). Metadata
        only -- no part of this calculation depends on it.
    debye_waller : float or sequence of float, optional
        Debye-Waller factor per atom [nm²] (default 0.005, i.e. 0.5 Å²).
        Recorded for the consumer; the pattern itself already accounts for
        whatever displacements `atoms` carries.
    name : str, optional
        Phase name recorded in the file.
    overwrite : bool, optional
        If True, replace an existing file.

    Notes
    -----
    A pattern covering only the northern hemisphere fills the southern one by
    assuming the crystal is centrosymmetric, ``I(-k) = I(k)``: on the Lambert
    square, a 180° rotation of the northern array. For a crystal without an
    inversion centre that is wrong, and the pattern warns when it records as
    much; build it with ``hemisphere='both'`` to write the southern hemisphere
    that was actually calculated.

    No background, energy spectrum or depth distribution is written: those are
    Monte Carlo products. Consumers that expect them will fall back to a flat
    weighting.
    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "writing EMsoft master patterns requires h5py; install it with "
            "'pip install h5py'"
        ) from None

    if pattern.ensemble_shape != ():
        raise ValueError(
            f"only a single pattern can be written, but this one has ensemble "
            f"shape {pattern.ensemble_shape}"
        )

    if energy is None:
        # EMsoft's EkeV is the beam energy; older patterns record only "energy"
        energy = pattern.metadata.get("beam_energy", pattern.metadata.get("energy"))
        if energy is None:
            raise ValueError(
                "the pattern carries no beam energy in its metadata; pass energy=..."
            )

    if npx is None:
        # Match the pattern's own sampling: a hemisphere sampled with N
        # directions supports a grid of about sqrt(N) on a side.
        npx = max(1, int(round((np.sqrt(len(pattern)) - 1) / 2)))

    space_group, lattice, scaled, numbers = _conventional_cell(atoms, space_group)

    # EMsoft's Rosca-Lambert square is exactly abTEM's SquareLambertProjection,
    # up to the edge length of the square, so this is a resampling and not a
    # change of projection.
    #
    # It is stored transposed, though. EMsoft swaps the two Lambert
    # coordinates when it looks a direction up (`ixy(1), ixy(2) = ixy(2),
    # -ixy(1)` in CalcEBSDPatternSingleFull) and writes the array in Fortran
    # order besides, which together amount to a transpose of the projection
    # here. Writing it unchanged produces a file that loads, looks like a
    # plausible Kikuchi pattern, and indexes to the wrong orientation --
    # verified against kikuchipy, which reproduces this module's own detector
    # projection only once the stored array is transposed.
    northern = np.asarray(pattern.interpolate(2 * npx + 1, "lambert").array).T
    # the node at (X, Y) of the southern square is the direction (x, y, -z);
    # for a pattern of the northern hemisphere alone that is the inversion
    # image, a 180 degree rotation of the northern square
    southern = np.asarray(
        pattern.interpolate(2 * npx + 1, "lambert", hemisphere="south").array
    ).T

    # The stereographic arrays are a display convenience -- EMsoft and
    # kikuchipy both project detector patterns from the Lambert ones. A pattern
    # sampled in Lambert cannot also land exactly on a stereographic grid, so
    # the resulting fill warning is unactionable here and is silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseProjectionWarning)
        stereographic_n = np.asarray(
            pattern.interpolate(2 * npx + 1, "stereographic").array
        ).T
        stereographic_s = np.asarray(
            pattern.interpolate(2 * npx + 1, "stereographic", hemisphere="south").array
        ).T

    if np.any(northern < 0.0):
        warnings.warn(
            "the pattern contains negative intensities, which a master pattern "
            "should not; check the calculation before indexing against it"
        )

    energies_kev = np.array([energy / 1e3], dtype=np.float32)
    atom_data, atom_types = _atom_data(scaled, numbers, debye_waller)

    conventional = Cell(lattice)
    lengths = conventional.lengths() / 10.0  # Å -> nm, EMsoft's convention
    angles = conventional.angles()

    mode = "w" if overwrite else "w-"
    with h5py.File(path, mode) as f:
        crystal = f.create_group("CrystalData")
        crystal.create_dataset("AtomData", data=atom_data)
        crystal.create_dataset("Atomtypes", data=atom_types)
        crystal.create_dataset(
            "CrystalSystem", data=_scalar(_crystal_system(space_group), np.int32)
        )
        crystal.create_dataset(
            "LatticeParameters",
            data=np.concatenate([lengths, angles]).astype(np.float64),
        )
        crystal.create_dataset("Natomtypes", data=_scalar(len(atom_types), np.int32))
        crystal.create_dataset("SpaceGroupNumber", data=_scalar(space_group, np.int32))
        crystal.create_dataset("SpaceGroupSetting", data=_scalar(1, np.int32))
        crystal.create_dataset("Source", data=_string(name))
        crystal.create_dataset("ProgramName", data=_string("abtem.ebsd"))

        master = f.create_group("EMData/EBSDmaster")
        master.create_dataset(
            "BetheParameters", data=np.array([4.0, 8.0, 50.0, 1.0], dtype=np.float32)
        )
        master.create_dataset("EkeVs", data=energies_kev)
        # (numset, numEbins, ny, nx)
        master.create_dataset("mLPNH", data=northern[None, None].astype(np.float32))
        master.create_dataset("mLPSH", data=southern[None, None].astype(np.float32))
        # (numEbins, ny, nx)
        master.create_dataset(
            "masterSPNH", data=stereographic_n[None].astype(np.float32)
        )
        master.create_dataset(
            "masterSPSH", data=stereographic_s[None].astype(np.float32)
        )
        master.create_dataset("numEbins", data=_scalar(1, np.int32))
        master.create_dataset("numset", data=_scalar(1, np.int32))
        master.create_dataset("lastEnergy", data=_scalar(1, np.int32))
        master.create_dataset("xtalname", data=_string(f"{name}.xtal"))

        nml = f.create_group("NMLparameters")
        master_nml = nml.create_group("EBSDMasterNameList")
        master_nml.create_dataset("dmin", data=_scalar(0.05, np.float32))
        master_nml.create_dataset("latgridtype", data=_string("Lambert"))
        master_nml.create_dataset("npx", data=_scalar(npx, np.int32))
        master_nml.create_dataset("combinesites", data=_scalar(0, np.int32))
        master_nml.create_dataset("uniform", data=_scalar(0, np.int32))
        master_nml.create_dataset("energyfile", data=_string(path))

        mc_nml = nml.create_group("MCCLNameList")
        mc_nml.create_dataset("Ebinsize", data=_scalar(1.0, np.float64))
        mc_nml.create_dataset(
            "Ehistmin", data=_scalar(float(energies_kev[0]), np.float64)
        )
        mc_nml.create_dataset("EkeV", data=_scalar(float(energies_kev[0]), np.float64))
        mc_nml.create_dataset("MCmode", data=_string("CSDA"))
        mc_nml.create_dataset("dataname", data=_string(path))
        mc_nml.create_dataset("depthmax", data=_scalar(100.0, np.float64))
        mc_nml.create_dataset("depthstep", data=_scalar(1.0, np.float64))
        mc_nml.create_dataset("numsx", data=_scalar(2 * npx + 1, np.int32))
        mc_nml.create_dataset("sig", data=_scalar(float(sample_tilt), np.float64))
        mc_nml.create_dataset("omega", data=_scalar(0.0, np.float64))
        mc_nml.create_dataset("totnum_el", data=_scalar(0, np.int32))
        mc_nml.create_dataset("multiplier", data=_scalar(1, np.int32))
        mc_nml.create_dataset("xtalname", data=_string(f"{name}.xtal"))

        bethe = nml.create_group("BetheList")
        for key, value in [("c1", 4.0), ("c2", 8.0), ("c3", 50.0), ("sgdbdiff", 1.0)]:
            bethe.create_dataset(key, data=_scalar(value, np.float32))

        header = f.create_group("EMheader/EBSDmaster")
        header.create_dataset("ProgramName", data=_string("EMEBSDmaster.f90"))
        header.create_dataset("Version", data=_string("abTEM"))
