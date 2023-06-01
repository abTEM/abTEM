"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from math import pi
from typing import Any
from typing import TYPE_CHECKING
from typing import Tuple, Union, List

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase import units
from ase.data import chemical_symbols, atomic_numbers
from ase.units import Bohr
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from abtem.core.axes import AxisMetadata
from abtem.core.constants import eps0
from abtem.core.electron_configurations import (
    electron_configurations,
    config_str_to_config_tuples,
)
from abtem.core.fft import fft_crop
from abtem.potentials.charge_density import _interpolate_slice
from abtem.core.parametrizations.ewald import EwaldParametrization
from abtem.inelastic.phonons import (
    DummyFrozenPhonons,
    FrozenPhonons,
    BaseFrozenPhonons,
    _safe_read_atoms,
)
from abtem.potentials.charge_density import _generate_slices
from abtem.potentials.iam import _PotentialBuilder, Potential

try:
    from gpaw import GPAW
    from gpaw.lfc import LFC, BasisFunctions
    from gpaw.transformers import Transformer
    from gpaw.utilities import unpack2
    from gpaw.atom.aeatom import AllElectronAtom
    from gpaw.io import Reader
    from gpaw.density import RealSpaceDensity
    from gpaw.mpi import SerialCommunicator
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.utilities import unpack_atomic_matrices
    from gpaw.atom.shapefunc import shape_functions

    if TYPE_CHECKING:
        from gpaw.setup import Setups
except:
    GPAW = None
    LFC = None
    BasisFunctions = None
    Transformer = None
    unpack2 = None
    Setups = None
    AllElectronAtom = None
    Reader = None
    SerialCommunicator = None
    GridDescriptor = None
    unpack_atomic_matrices = None


def _get_gpaw_setups(atoms, mode, xc):
    gpaw = GPAW(txt=None, mode=mode, xc=xc)
    gpaw.initialize(atoms)

    return gpaw.setups


@dataclass
class _DummyGPAW:
    setup_mode: str
    setup_xc: str
    nt_sG: np.ndarray
    gd: Any
    D_asp: np.ndarray
    atoms: Atoms
    Q_aL: dict
    valence_potential: np.ndarray

    def __len__(self):
        return 1

    @property
    def setups(self):
        gpaw = GPAW(txt=None, mode=self.setup_mode, xc=self.setup_xc)
        gpaw.initialize(self.atoms)
        return gpaw.setups

    @classmethod
    def from_gpaw(cls, gpaw, lazy: bool = True):
        # if lazy:
        #    return dask.delayed(cls.from_gpaw)(gpaw, lazy=False)

        atoms = gpaw.atoms.copy()
        atoms.calc = None

        kwargs = {
            "setup_mode": gpaw.parameters["mode"],
            "setup_xc": gpaw.parameters["xc"],
            "nt_sG": gpaw.density.nt_sG.copy(),
            "gd": gpaw.density.gd.new_descriptor(comm=SerialCommunicator()),
            "D_asp": dict(gpaw.density.D_asp),
            "atoms": atoms,
            "valence_potential": gpaw.get_electrostatic_potential(),
            "Q_aL": dict(gpaw.density.Q_aL),
        }

        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str, lazy: bool = True):
        if lazy:
            return dask.delayed(cls.from_file)(path, lazy=False)

        calc = GPAW(path)

        return cls.from_gpaw(calc)

        # with Reader(path) as reader:
        #     atoms = read_atoms(reader.atoms)
        #
        #     from gpaw.calculator import GPAW
        #
        #     parameters = copy.copy(GPAW.default_parameters)
        #     parameters.update(reader.parameters.asdict())
        #
        #     setup_mode = parameters["mode"]
        #     setup_xc = parameters["xc"]
        #
        #     if isinstance(setup_xc, dict) and "setup_name" in setup_xc:
        #         setup_xc = setup_xc["setup_name"]
        #
        #     assert isinstance(setup_xc, str)
        #
        #     density = reader.density.density * units.Bohr**3
        #     gd = GridDescriptor(
        #         N_c=density.shape[-3:],
        #         cell_cv=atoms.get_cell() / Bohr,
        #         comm=SerialCommunicator(),
        #     )
        #
        #     setups = _get_gpaw_setups(atoms, setup_mode, setup_xc)
        #
        #     D_asp = unpack_atomic_matrices(
        #         reader.density.atomic_density_matrices, setups
        #     )
        #
        #     hamiltonian = reader.hamiltonian
        #
        # kwargs = {
        #     "setup_mode": setup_mode,
        #     "setup_xc": setup_xc,
        #     "nt_sG": density,
        #     "gd": gd,
        #     "D_asp": D_asp,
        #     "atoms": atoms,
        # }
        # return cls(**kwargs)

    @classmethod
    def from_generic(cls, calculator, lazy: bool = True):
        if isinstance(calculator, str):
            return cls.from_file(calculator, lazy=lazy)
        elif hasattr(calculator, "density"):
            return cls.from_gpaw(calculator, lazy=lazy)
        elif isinstance(calculator, cls):
            return calculator
        else:
            raise RuntimeError()


def get_core_correction_interpolators(setups, D_asp, Q_aL, rcgauss):
    interpolators = []
    for a, D_sp in D_asp.items():
        setup = setups[a]
        c = setup.xc_correction

        rgd = c.rgd
        params = setup.data.shape_function.copy()
        params["lmax"] = 0
        ghat_g = shape_functions(rgd, **params)[0]
        Z_g = shape_functions(rgd, "gauss", rcgauss, lmax=0)[0] * setup.Z
        D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])

        dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * np.sqrt(4 * np.pi)
        dn_g += 4 * np.pi * (c.nc_g - c.nct_g)
        dn_g -= Z_g
        dn_g -= Q_aL[a][0] * ghat_g * np.sqrt(4 * np.pi)

        dv_g = rgd.poisson(dn_g) / np.sqrt(4 * np.pi)
        dv_g[1:] /= rgd.r_g[1:]
        dv_g[0] = dv_g[1]
        dv_g[-1] = 0.0

        interpolator = interp1d(
            units.Bohr * rgd.r_g,
            -dv_g / np.sqrt(4 * np.pi) * units.Ha,
            fill_value=0,
            bounds_error=False,
        )
        interpolators.append(interpolator)

    return interpolators


def integrate_slice(array, gpts, a, b, thickness):
    dz = thickness / array.shape[2]
    na = int(np.floor(a / dz))
    nb = int(np.floor(b / dz))
    slice_array = np.sum(array[..., na:nb], axis=-1) * dz
    new_shape = (nb - na,) + gpts
    old_shape = (nb - na,) + slice_array.shape
    slice_array = np.fft.fftn(slice_array)
    slice_array = fft_crop(slice_array, gpts)
    slice_array = (
        np.fft.ifftn(slice_array).real * np.prod(new_shape) / np.prod(old_shape)
    )
    return slice_array


class _DummyParametrization:
    def __init__(self, potential):
        self._potential = potential
        super().__init__()

    def potential(self, symbol):
        return self._potential


def _generate_slices(
    interpolators,
    valence_potential,
    atoms,
    gpts,
    slice_thickness,
    first_slice=0,
    last_slice=None,
):

    potential_generators = []
    for i, interpolator in enumerate(interpolators):
        parametrization = _DummyParametrization(interpolator)

        potential = Potential(
            gpts=gpts,
            atoms=atoms[i : i + 1],
            parametrization=parametrization,
            slice_thickness=slice_thickness,
            projection="finite",
        )
        potential_generators.append(potential.generate_slices())

    transformed_atoms = potential.get_transformed_atoms()

    if np.allclose(transformed_atoms.cell, atoms.cell):
        transform_valence_potential = False
    else:
        transform_valence_potential = True

    if last_slice is None:
        last_slice = len(potential)

    for i, slice_idx in enumerate(range(first_slice, last_slice)):
        slic = next(potential_generators[0])

        a, b = potential.get_sliced_atoms().slice_limits[slice_idx]

        for potential_generator in potential_generators[1:]:
            slic.array[:] += next(potential_generator).array

        if transform_valence_potential:
            slic.array[:] -= _interpolate_slice(
                valence_potential, atoms.cell, potential.gpts, potential.sampling, a, b
            )
        else:
            slic.array[:] -= integrate_slice(
                valence_potential, potential.gpts, a, b, potential.thickness
            )

        yield slic


class GPAWPotential(_PotentialBuilder):
    """
    Calculate the electrostatic potential from a (set of) converged GPAW DFT calculation(s). Frozen phonons can be
    included either by specifying multiple GPAW calculators corresponding to the different phonon configurations, or
    approximately for a single calculator by using the `frozen_phonons` keyword.

    Parameters
    ----------
    calculators : (list of) gpaw.calculator.GPAW or (list of) str
        GPAW calculator or path to GPAW calculator or list of calculators or paths. Atoms are read from the calculator.
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential. Provide either "sampling" (spacing
        between consecutive grid points) or "gpts" (total number of grid points).
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å]. Provide either "sampling" or "gpts".
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices in the propagation direction in [Å] (default is 0.5 Å).
        If given as a float the number of slices is calculated by dividing the slice thickness into the `z`-height
        of supercell. The slice thickness may be given as a sequence of values for each slice, in which case an
        error will be thrown if the sum of slice thicknesses is not equal to the height of the atoms.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the slice indices after which an
        exit plane is desired, and hence during a multislice simulation a measurement is created. If `exit_planes` is
        an integer a measurement will be collected every `exit_planes` number of slices.
    plane : str or two tuples of three float, optional
        The plane relative to the provided atoms mapped to `xy` plane of the potential, i.e. provided plane is
        perpendicular to the propagation direction. If string, it must be a concatenation of two of 'x', 'y' and 'z';
        the default value 'xy' indicates that potential slices are cuts along the `xy`-plane of the atoms.
        The plane may also be specified with two arbitrary 3D vectors, which are mapped to the `x` and `y` directions of
        the potential, respectively. The length of the vectors has no influence. If the vectors are not perpendicular,
        the second vector is rotated in the plane to become perpendicular to the first. Providing a value of
        ((1., 0., 0.), (0., 1., 0.)) is equivalent to providing 'xy'.
    origin : three float, optional
        The origin relative to the provided Atoms mapped to the origin of the Potential. This is equivalent to translating
        the atoms. The default is (0., 0., 0.)
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. If not given this is determined from the atoms' cell.
        If the box size does not match an integer number of the atoms' supercell, an affine transformation may be
        necessary to preserve periodicity, determined by the `periodic` keyword
    periodic : bool
        If a transformation of the atomic structure is required, `periodic` determines how the atomic structure is
        transformed. If True (default), the periodicity of the Atoms is preserved, which may require applying a small affine
        transformation to the atoms. If False, the transformed potential is effectively cut out of a larger repeated
        potential, which may not preserve periodicity.
    frozen_phonons : abtem.AbstractFrozenPhonons, optional
        Approximates frozen phonons for a single GPAW calculator by displacing only the nuclear core potentials.
        Supercedes the atoms from the calculator.
    repetitions : tuple of int
        Repeats the atoms by integer amounts in the `x`, `y` and `z` directions before applying frozen phonon displacements
        to calculate the potential contribution of the nuclear cores. Necessary when using frozen phonons.
    gridrefinement : int
        Necessary interpolation of the charge density into a finer grid for improved numerical precision.
        Allowed values are '2' and '4'.
    device : str, optional
        The device used for calculating the potential, 'cpu' or 'gpu'. The default is determined by the user
        configuration file.
    """

    def __init__(
        self,
        calculators: Union["GPAW", List["GPAW"], List[str], str],
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        slice_thickness: float = 1.0,
        exit_planes: int = None,
        plane: str = "xy",
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Tuple[float, float, float] = None,
        periodic: bool = True,
        frozen_phonons: BaseFrozenPhonons = None,
        repetitions: Tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 4,
        device: str = None,
    ):

        if GPAW is None:
            raise RuntimeError(
                "This functionality of abTEM requires GPAW, see https://wiki.fysik.dtu.dk/gpaw/."
            )

        if isinstance(calculators, (tuple, list)):
            atoms = _safe_read_atoms(calculators[0])

            num_configs = len(calculators)

            if frozen_phonons is not None:
                raise ValueError()

            calculators = [
                _DummyGPAW.from_generic(calculator) for calculator in calculators
            ]

            frozen_phonons = DummyFrozenPhonons(atoms, num_configs=num_configs)

        else:
            atoms = _safe_read_atoms(calculators)

            calculators = _DummyGPAW.from_generic(calculators)

            if frozen_phonons is None:
                frozen_phonons = DummyFrozenPhonons(atoms, num_configs=None)

        self._calculators = calculators
        self._frozen_phonons = frozen_phonons
        self._gridrefinement = gridrefinement
        self._repetitions = repetitions

        cell = frozen_phonons.atoms.cell * repetitions
        frozen_phonons.atoms.calc = None

        super().__init__(
            gpts=gpts,
            sampling=sampling,
            cell=cell,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
        )

    @property
    def frozen_phonons(self):
        return self._frozen_phonons

    @property
    def repetitions(self):
        return self._repetitions

    @property
    def gridrefinement(self):
        return self._gridrefinement

    @property
    def calculators(self):
        return self._calculators

    def _get_all_electron_density(self):

        calculator = (
            self.calculators[0]
            if isinstance(self.calculators, list)
            else self.calculators
        )

        try:
            calculator = calculator.compute()
        except AttributeError:
            pass

        # assert len(self.calculators) == 1

        calculator = _DummyGPAW.from_generic(calculator)

        atoms = self.frozen_phonons.atoms

        if self.repetitions != (1, 1, 1):
            cell_cv = calculator.gd.cell_cv * self.repetitions
            N_c = tuple(
                n_c * rep for n_c, rep in zip(calculator.gd.N_c, self.repetitions)
            )
            gd = calculator.gd.new_descriptor(N_c=N_c, cell_cv=cell_cv)
            atoms = atoms * self.repetitions
            nt_sG = np.tile(calculator.nt_sG, self.repetitions)
        else:
            gd = calculator.gd
            nt_sG = calculator.nt_sG

        random_atoms = self.frozen_phonons.randomize(atoms)

        calc = GPAW(txt=None, mode=calculator.setup_mode, xc=calculator.setup_xc)
        calc.initialize(random_atoms)

        return _get_all_electron_density(
            nt_sG=nt_sG,
            gd=gd,
            D_asp=calculator.D_asp,
            setups=calc.setups,
            gridrefinement=self.gridrefinement,
            atoms=random_atoms,
        )

    def _get_ewald_potential(self):
        ewald_parametrization = EwaldParametrization(width=3)

        atoms = self.frozen_phonons.atoms * self.repetitions

        atoms = self.frozen_phonons.randomize(atoms)

        ewald_potential = Potential(
            atoms=atoms,
            gpts=self.gpts,
            sampling=self.sampling,
            parametrization=ewald_parametrization,
            slice_thickness=self.slice_thickness,
            projection="finite",
            integral_method="quadrature",
            plane=self.plane,
            box=self.box,
            origin=self.origin,
            exit_planes=self.exit_planes,
            device=self.device,
        )

        return ewald_potential

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
        slices : generator of np.ndarray
            Generator for the array of slices.
        """
        if last_slice is None:
            last_slice = len(self)

        calculator = (
            self.calculators[0]
            if isinstance(self.calculators, list)
            else self.calculators
        )

        try:
            calculator = calculator.compute()
        except AttributeError:
            pass

        # assert len(self.calculators) == 1

        calculator = _DummyGPAW.from_generic(calculator)

        # print(calculator)

        atoms = self.frozen_phonons.atoms

        if self.repetitions != (1, 1, 1):
            cell_cv = calculator.gd.cell_cv * self.repetitions
            N_c = tuple(
                n_c * rep for n_c, rep in zip(calculator.gd.N_c, self.repetitions)
            )
            gd = calculator.gd.new_descriptor(N_c=N_c, cell_cv=cell_cv)
            atoms = atoms * self.repetitions
            nt_sG = np.tile(calculator.nt_sG, self.repetitions)
        else:
            gd = calculator.gd
            nt_sG = calculator.nt_sG

        random_atoms = self.frozen_phonons.randomize(atoms)

        interpolators = get_core_correction_interpolators(
            calculator.setups, calculator.D_asp, calculator.Q_aL, 0.02
        )

        # calc = GPAW(txt=None, mode=calculator.setup_mode, xc=calculator.setup_xc)
        # calc.initialize(random_atoms)

        # ewald_potential = self._get_ewald_potential()
        #
        # array = self._get_all_electron_density()

        for slic in _generate_slices(
            interpolators,
            valence_potential=calculator.valence_potential,
            atoms=random_atoms,
            gpts=self.gpts,
            slice_thickness=self.slice_thickness,
            first_slice=first_slice,
            last_slice=last_slice,
        ):
            yield slic
        # for slic in _generate_slices(
        #     array, ewald_potential, first_slice=first_slice, last_slice=last_slice
        # ):
        #     yield slic

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return self._frozen_phonons.ensemble_axes_metadata

    @property
    def num_frozen_phonons(self):
        return len(self.calculators)

    @property
    def ensemble_shape(self):
        return self._frozen_phonons.ensemble_shape

    @staticmethod
    def _gpaw_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        if args["frozen_phonons"] is not None:
            frozen_phonons = frozen_phonons_partial(args["frozen_phonons"])
        else:
            frozen_phonons = None

        calculators = args["calculators"]

        return GPAWPotential(calculators, frozen_phonons=frozen_phonons, **kwargs)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(exclude=("calculators", "frozen_phonons"))

        frozen_phonons_partial = self.frozen_phonons._from_partitioned_args()

        return partial(
            self._gpaw_potential,
            frozen_phonons_partial=frozen_phonons_partial,
            **kwargs
        )

    def _partition_args(self, chunks: int = 1, lazy: bool = True):

        chunks = self._validate_chunks(chunks)

        def frozen_phonons(calculators, frozen_phonons):
            arr = np.zeros((1,), dtype=object)
            arr.itemset(
                0, {"calculators": calculators, "frozen_phonons": frozen_phonons}
            )
            return arr

        calculators = self.calculators

        if isinstance(self.frozen_phonons, FrozenPhonons):
            array = np.zeros(len(self.frozen_phonons), dtype=object)
            for i, fp in enumerate(
                self.frozen_phonons._partition_args(chunks, lazy=lazy)[0]
            ):
                if lazy:
                    block = dask.delayed(frozen_phonons)(calculators, fp)

                    array.itemset(i, da.from_delayed(block, shape=(1,), dtype=object))
                else:
                    array.itemset(i, frozen_phonons(calculators, fp))

            if lazy:
                array = da.concatenate(list(array))

            return (array,)

        else:
            if len(self.ensemble_shape) == 0:
                array = np.zeros((1,), dtype=object)
                calculators = [calculators]
            else:
                array = np.zeros(self.ensemble_shape[0], dtype=object)

            for i, calculator in enumerate(calculators):

                if len(self.ensemble_shape) > 0:
                    calculator = [calculator]

                if lazy:
                    calculator = dask.delayed(calculator)
                    block = da.from_delayed(
                        dask.delayed(frozen_phonons)(calculator, None),
                        shape=(1,),
                        dtype=object,
                    )
                else:
                    block = frozen_phonons(calculator, None)

                array.itemset(i, block)

            if lazy:
                return (da.concatenate(list(array)),)
            else:
                return (array,)


class GPAWParametrization:
    """
    Calculate an Independent Atomic Model (IAM) potential based on a GPAW DFT calculation.
    """

    def __init__(self):
        self._potential_functions = {}

    def _get_added_electrons(self, symbol, charge):
        if not charge:
            return []

        charge = int(np.sign(charge) * np.ceil(np.abs(charge)))

        number = atomic_numbers[symbol]
        config = config_str_to_config_tuples(
            electron_configurations[chemical_symbols[number]]
        )
        ionic_config = config_str_to_config_tuples(
            electron_configurations[chemical_symbols[number - charge]]
        )

        config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in config})

        ionic_config = defaultdict(
            lambda: 0, {shell[:2]: shell[2] for shell in ionic_config}
        )

        # sss

        electrons = []
        for key in set(config.keys()).union(set(ionic_config.keys())):

            difference = config[key] - ionic_config[key]

            for i in range(np.abs(difference)):
                electrons.append(key + (-np.sign(difference),))

        return electrons

    def _get_all_electron_atom(self, symbol, charge=0.0):

        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectronAtom(symbol, spinpol=False, xc="PBE")

        added_electrons = self._get_added_electrons(symbol, charge)

        for added_electron in added_electrons:
            ae.add(*added_electron[:2], added_electron[-1])

        # # ae.run()
        # ae.run(mix=0.005, maxiter=5000, dnmax=1e-5)
        ae.run(maxiter=5000, mix=0.005, dnmax=1e-5)
        # ae.refine()

        return ae

        # vr_e = interp1d(radial_coord, electron_potential, fill_value='extrapolate', bounds_error=False)
        # vr = lambda r: atomic_numbers[symbol] / r / (4 * np.pi * eps0) + vr_e(r) / r * units.Hartree * units.Bohr

    def charge(self, symbol: str, charge: float = 0.0):
        """
        Calculate the radial charge density for an atom.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the atomic element.
        charge : float, optional
            Charge the atom by the given fractional number of electrons.

        Returns
        -------
        charge : callable
            Function of the radial charge density with parameter 'r' corresponding to the radial distance from the core.
        """
        ae = self._get_all_electron_atom(symbol, charge)
        r = ae.rgd.r_g * units.Bohr
        n = ae.n_sg.sum(0) / units.Bohr**3
        return interp1d(r, n, fill_value="extrapolate", bounds_error=False)

    def potential(self, symbol: str, charge: float = 0.0):
        """
        Calculate the radial electrostatic potential for an atom.

        Parameters
        ----------
        symbol : str
            Chemical symbol of the atomic element.
        charge : float, optional
            Charge the atom by the given fractional number of electrons.

        Returns
        -------
        potential : callable
            Function of the radial electrostatic potential with parameter 'r' corresponding to the radial distance from the core.
        """

        ae = self._get_all_electron_atom(symbol, charge)
        r = ae.rgd.r_g * units.Bohr

        ve = -ae.rgd.poisson(ae.n_sg.sum(0))
        ve = interp1d(r, ve, fill_value="extrapolate", bounds_error=False)

        vr = (
            lambda r: atomic_numbers[symbol] / r / (4 * np.pi * eps0)
            + ve(r) / r * units.Hartree * units.Bohr
        )
        return vr
