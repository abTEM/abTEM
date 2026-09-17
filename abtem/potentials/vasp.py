"""Module to build the electrostatic potential directly from VASP output files.

:class:`.VASPPotential` combines VASP's self-consistent valence electron density
(`AECCAR2`) with an accurate per-species core electron correction parsed
directly out of the corresponding `POTCAR` file. VASP's PAW datasets tabulate
the core electron density on a fine (typically 300+ point) logarithmic radial
grid; this is used exactly as it is -- no auxiliary DFT calculation (VASP,
GPAW, or otherwise) is required to obtain it.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np
from scipy.interpolate import interp1d

from abtem.core.ensemble import _wrap_with_array
from abtem.inelastic.phonons import DummyFrozenPhonons
from abtem.potentials.charge_density import ChargeDensityPotential, _generate_slices

_SQRT_4PI = np.sqrt(4 * np.pi)


def _read_radial_block(lines: list, i: int, n: int):
    """Read `n` whitespace-separated floats starting right after `lines[i]` (the
    block's header line). Returns (values, index of the first line after the
    block)."""
    values = []
    j = i + 1
    while len(values) < n:
        values.extend(float(x) for x in lines[j].split())
        j += 1
    if len(values) != n:
        raise ValueError(
            f"expected {n} values in POTCAR radial block, got {len(values)}"
        )
    return np.array(values[:n]), j


def parse_potcar(path: Union[str, Path]) -> Dict[str, dict]:
    """
    Parse a VASP `POTCAR` file and extract, for each element it contains, the PAW
    radial grid and the tabulated all-electron core charge-density.

    Parameters
    ----------
    path : str or Path
        Path to the `POTCAR` file. May contain the concatenation of several
        elements' PAW datasets, as VASP requires.

    Returns
    -------
    elements : dict
        Maps chemical symbol to a dict with keys:

        - `"nmax"` : number of radial grid points.
        - `"grid"` : radial grid `r` [Å], shape `(nmax,)`. VASP tabulates the
          PAW radial sets in its own internal units (Ångström), not atomic units.
        - `"core_density"` : the POTCAR's tabulated `"core charge-density"` block,
          shape `(nmax,)`. This is `sqrt(4 pi) * r**2 * n_core(r)` (VASP tabulates
          radial densities expanded in real spherical harmonics, for which
          `Y_00 = 1 / sqrt(4 pi)`) -- use :func:`core_density_fourier_transform` or
          :func:`get_core_density_fourier_interpolator` rather than using this
          array directly.
    """
    with open(path) as f:
        lines = f.readlines()

    elements: Dict[str, dict] = {}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("VRHFIN"):
            symbol = line.split("=")[1].split(":")[0].strip()
        elif line == "PAW radial sets":
            nmax = int(lines[i + 1].split()[0])
            j = i + 3  # skip the "(5E20.12)"-style format line
            grid = None
            core_density = None
            while j < n:
                tag = lines[j].strip()
                if tag == "End of Dataset" or tag.startswith("VRHFIN"):
                    break
                if tag == "grid":
                    grid, j = _read_radial_block(lines, j, nmax)
                    continue
                if tag == "core charge-density":
                    core_density, j = _read_radial_block(lines, j, nmax)
                    continue
                j += 1

            if grid is None or core_density is None:
                raise ValueError(
                    f"could not find 'grid' and 'core charge-density' blocks for "
                    f"element '{symbol}' in {path}"
                )

            elements[symbol] = {
                "nmax": nmax,
                "grid": grid,
                "core_density": core_density,
            }
            i = j
            continue
        i += 1

    return elements


def core_density_fourier_transform(
    r: np.ndarray, core_density: np.ndarray, G: np.ndarray
) -> np.ndarray:
    """
    l=0 spherical Fourier transform of a POTCAR-tabulated core charge-density,
    evaluated at angular wavenumbers `G`.

    `core_density` is the POTCAR's `"core charge-density"` array
    `sqrt(4 pi) * r**2 * n_core(r)` (see :func:`parse_potcar`); with this
    normalization the transform simplifies to

    `f(G) = sqrt(4 pi) * int core_density(r) * sinc(G r) dr`,

    which equals the core electron count `Nc` at `G=0`.

    Parameters
    ----------
    r : numpy.ndarray
        Radial grid, shape `(nmax,)`.
    core_density : numpy.ndarray
        POTCAR core charge-density array, shape `(nmax,)`.
    G : numpy.ndarray
        Angular wavenumbers at which to evaluate the transform, in the same
        inverse-length unit as `1 / r`.

    Returns
    -------
    f : numpy.ndarray
        The Fourier transform, same shape as `G`.
    """
    G = np.asarray(G)
    Gr = np.multiply.outer(G, r)
    sinc = np.sinc(Gr / np.pi)  # np.sinc(x) = sin(pi x) / (pi x)
    return _SQRT_4PI * np.trapezoid(sinc * core_density, r, axis=-1)


def get_core_density_fourier_interpolator(
    symbol: str,
    elements: Dict[str, dict],
    n_G: int = 4000,
    G_max: float = 200.0,
) -> "tuple[Callable[[np.ndarray], np.ndarray], float]":
    """
    Build an interpolator for the l=0 spherical Fourier transform of one
    element's POTCAR core charge-density.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    elements : dict
        Parsed POTCAR data, as returned by :func:`parse_potcar`.
    n_G : int
        Number of points used to tabulate the transform before interpolating.
    G_max : float
        Maximum angular wavenumber [1 / Å] used to tabulate the transform.
        The default comfortably covers the spatial frequencies of typical
        multislice grids; increasing it (and `n_G` to match) leaves converged
        results unchanged.

    Returns
    -------
    interpolator : callable
        Maps an array of angular wavenumbers `G` [1 / Å] to the core density's
        Fourier transform (in electrons).
    Nc : float
        The element's core electron count (the transform's value at `G=0`).
    """
    d = elements[symbol]
    r = d["grid"]
    core_density = d["core_density"]

    # `r` is in Å, so the conjugate variable is already in 1 / Å -- no unit
    # conversion belongs here. Reading the POTCAR grid as Bohr instead would
    # compress the core density by 1 / 0.529, pushing V_core(0) ~1.9x too high.
    G = np.linspace(0.0, G_max, n_G)
    f_k = core_density_fourier_transform(r, core_density, G)
    Nc = float(f_k[0])

    interpolator = interp1d(G, f_k, bounds_error=False, fill_value=(f_k[0], 0.0))
    return interpolator, Nc


class VASPPotential(ChargeDensityPotential):
    """
    The VASP potential calculates the electrostatic potential from VASP's
    self-consistent valence electron density (`AECCAR2`) plus an accurate
    per-species core electron correction parsed directly from the
    corresponding `POTCAR` file -- no auxiliary DFT calculation is needed for
    the core correction.

    `AECCAR2` is written when the VASP run sets `LAECHG = .TRUE.`, which
    reconstructs the all-electron charge density on the fine
    (`NGXF` x `NGYF` x `NGZF`) FFT grid and writes it as three files: the core
    density (`AECCAR0`), the proto-atomic valence density (`AECCAR1`), and the
    self-consistent valence density (`AECCAR2`). Only the last is wanted here --
    this class supplies the core contribution itself, from `POTCAR`.

    In PAW terminology "all-electron" does not mean "the density of all
    electrons"; it means a density that retains the nodal structure near the
    nucleus belonging to the true one-electron orbitals, rather than the
    pseudized ones. A plain `CHGCAR` is the pseudo charge density and lacks that
    structure, so it will work but is less accurate near each nucleus.

    This refines :class:`.ChargeDensityPotential`'s crude, single
    Gaussian-broadened point-charge correction (equal to each atom's full
    atomic number) by instead injecting each species' actual radial core
    electron density, given by its `POTCAR` entry, in reciprocal space. See
    :func:`parse_potcar` and :func:`get_core_density_fourier_interpolator`.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons
        Atomic configuration(s) used in the independent atom model for calculating
        the electrostatic potential(s).
    charge_density : numpy.ndarray
        Valence-only electron density as a 3D NumPy array [electrons / Å^3] --
        VASP's `AECCAR2`, written when the run sets `LAECHG = .TRUE.`. A plain
        `CHGCAR` (the pseudo charge density) is accepted but is less accurate near
        each nucleus. Must not be an all-electron density covering core *and*
        valence, such as the `AECCAR0+AECCAR2` sum some VASP workflows produce for
        Bader charge analysis -- this class adds the core contribution itself.
    potcar : str, Path, or dict
        Path to the VASP `POTCAR` file used to generate `charge_density`, or an
        already-parsed dict as returned by :func:`parse_potcar`. Must contain an
        entry for every chemical species present in `atoms`.
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential
        calculated by specifying either `sampling` or `gpts`. The core-density
        correction is resolved at this grid -- unlike the crude
        :class:`.ChargeDensityPotential` correction, its accuracy near each nucleus
        keeps improving with finer `gpts`/`sampling`, even beyond `charge_density`'s
        own native resolution.
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å] calculated by specifying either
        `sampling` or `gpts`.
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices [Å] (default is 1.0 Å). If given as a float,
        the number of slices are calculated by dividing the slice thickness into the
        `z`-height of the cell. The slice thickness may be given as a sequence of values
        for each slice, in which case an error will be thrown if the sum of slice
        thicknesses is not equal to the height of the atoms.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer,
        a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to the `xy` plane of the
        potential, i.e. the propagation direction will be perpendicular to the provided
        plane. If str, must be a concatenation of two of 'x', 'y' and 'z'; the default
        value 'xy' indicates that potential slices are cuts parallel to the 'xy'-plane.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped
        to the `x` and `y` directions of the potential, respectively. The length of the
        vectors has no influence. If the vectors are not perpendicular, the second
        vector is rotated in the plane to become perpendicular to the first. A value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to 'xy'.
    origin : three float, optional
        The origin relative to the provided atoms mapped to the origin of the potential.
        This is equivalent to translating the atoms.
        The default is (0., 0., 0.).
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined
        from the atoms. If the box size does not match an integer number of the atoms'
        cell, an affine transformation may be necessary to preserve periodicity,
        determined by the `periodic` keyword.
    periodic : bool, True
        If a transformation of the atomic structure is required, `periodic` determines
        how the atomic structure is transformed. If True, the periodicity of the atoms
        is preserved, which may require applying a small affine transformation to the
        atoms. If False, the transformed potential is effectively cut out of a larger
        repeated potential, which may not preserve periodicity.
    repetitions : three int, optional
        Repeats the atoms and the charge density by integer amounts in the `x`, `y`
        and `z` directions. The default is (1, 1, 1).
    device : str, optional
        The device used for calculating the potential. The default is determined by the
        user configuration file.
    subtract_min : bool, optional
        If True, each slice's own minimum value is subtracted from it. This constant,
        spatially uniform shift doesn't change multislice-simulated intensities (a
        per-slice constant only contributes an overall, unobservable phase), but it
        does mean the potential's absolute value -- e.g. in vacuum -- is not the
        physical electrostatic reference and differs slice-to-slice, and from other
        potential builders (e.g. :class:`.GPAWPotential`) that don't do this. Default
        is False.
    """

    def __init__(
        self,
        atoms,
        charge_density: np.ndarray = None,
        potcar: Union[str, Path, Dict[str, dict]] = None,
        gpts=None,
        sampling=None,
        slice_thickness=1.0,
        plane: str = "xy",
        box=None,
        origin=(0.0, 0.0, 0.0),
        periodic: bool = True,
        exit_planes: int = None,
        repetitions=(1, 1, 1),
        device: str = None,
        subtract_min: bool = False,
    ):
        super().__init__(
            atoms=atoms,
            charge_density=charge_density,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            plane=plane,
            box=box,
            origin=origin,
            periodic=periodic,
            exit_planes=exit_planes,
            repetitions=repetitions,
            device=device,
            subtract_min=subtract_min,
        )

        if isinstance(potcar, (str, Path)):
            elements = parse_potcar(potcar)
        elif potcar is None:
            raise ValueError("potcar must be given -- a path to a POTCAR file")
        else:
            elements = potcar

        symbols = set(self.frozen_phonons.atoms.get_chemical_symbols())
        missing = symbols - set(elements)
        if missing:
            raise ValueError(
                f"potcar is missing the element(s) {sorted(missing)} present in atoms"
            )

        self._potcar = elements
        self._core_density_correction = {
            symbol: get_core_density_fourier_interpolator(symbol, elements)[0]
            for symbol in symbols
        }

    @property
    def potcar(self):
        return self._potcar

    @staticmethod
    def _vasp_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        if args["atoms"] is not None:
            atoms = frozen_phonons_partial(args["atoms"])
        else:
            atoms = DummyFrozenPhonons(kwargs.pop("_default_atoms"))

        charge_density = args["charge_density"]
        potential = VASPPotential(
            atoms=atoms, charge_density=charge_density, **kwargs
        )
        return _wrap_with_array(potential)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("atoms", "charge_density"), cls=VASPPotential
        )
        # See ChargeDensityPotential._from_partitioned_args: self.box has already
        # been resolved to a concrete tuple, which would silently skip
        # non-orthogonal (skew) auto-detection on every lazy/dask block
        # reconstruction. Reset it to None so the reconstructed instance
        # re-detects the skew from the atoms.
        if self._non_orthogonal:
            kwargs["box"] = None
        frozen_phonons_partial = (
            self._get_ewald_potential().frozen_phonons._from_partitioned_args()
        )
        return partial(
            self._vasp_potential,
            frozen_phonons_partial=frozen_phonons_partial,
            _default_atoms=self.frozen_phonons.atoms,
            **kwargs,
        )

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        """
        Generate the slices for the potential.

        Parameters
        ----------
        first_slice : int, optional
            Index of the first slice of the generated potential.
        last_slice : int, optional
            Index of the last slice of the generated potential.
        Returns
        -------
        slices : generator of numpy.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        array, ewald_potential = self._prepare_array_and_ewald_potential()

        for slic in _generate_slices(
            array,
            ewald_potential,
            first_slice=first_slice,
            last_slice=last_slice,
            core_density_correction=self._core_density_correction,
            subtract_min=self.subtract_min,
        ):
            yield slic
