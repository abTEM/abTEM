"""Module to handle ab initio electrostatic potentials from the DFT code GPAW."""
import copy
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
from ase.io.trajectory import read_atoms
from ase.units import Bohr
from scipy.interpolate import interp1d

from abtem.potentials.charge_density import _generate_slices
from abtem.core.axes import AxisMetadata
from abtem.core.constants import eps0
from abtem.core.electron_configurations import (
    electron_configurations,
    config_str_to_config_tuples,
)
from abtem.core.parametrizations.ewald import EwaldParametrization
from abtem.inelastic.phonons import (
    DummyFrozenPhonons,
    FrozenPhonons,
    BaseFrozenPhonons, _safe_read_atoms,
)
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
        }

        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str, lazy: bool = True):
        if lazy:
            return dask.delayed(cls.from_file)(path, lazy=False)

        with Reader(path) as reader:
            atoms = read_atoms(reader.atoms)

            from gpaw.calculator import GPAW

            parameters = copy.copy(GPAW.default_parameters)
            parameters.update(reader.parameters.asdict())

            setup_mode = parameters["mode"]
            setup_xc = parameters["xc"]

            if isinstance(setup_xc, dict) and "setup_name" in setup_xc:
                setup_xc = setup_xc["setup_name"]

            assert isinstance(setup_xc, str)

            density = reader.density.density * units.Bohr**3
            gd = GridDescriptor(
                N_c=density.shape[-3:],
                cell_cv=atoms.get_cell() / Bohr,
                comm=SerialCommunicator(),
            )

            setups = _get_gpaw_setups(atoms, setup_mode, setup_xc)

            D_asp = unpack_atomic_matrices(
                reader.density.atomic_density_matrices, setups
            )

        kwargs = {
            "setup_mode": setup_mode,
            "setup_xc": setup_xc,
            "nt_sG": density,
            "gd": gd,
            "D_asp": D_asp,
            "atoms": atoms,
        }
        return cls(**kwargs)

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


def _interpolate_pseudo_density(nt_sg, gd, gridrefinement=1):
    if gridrefinement == 1:
        return nt_sg, gd

    assert gridrefinement % 2 == 0

    iterations = int(np.log(gridrefinement) / np.log(2))

    finegd = gd
    n_sg = nt_sg

    for i in range(iterations):
        finegd = gd.refine()
        interpolator = Transformer(gd, finegd, 3)

        n_sg = finegd.empty(nt_sg.shape[0])

        for s in range(nt_sg.shape[0]):
            interpolator.apply(nt_sg[s], n_sg[s])

        nt_sg = n_sg
        gd = finegd

    return n_sg, finegd


def _calculate_I_a(nspins, D_asp, setups):
    I_sa = np.zeros((nspins, len(D_asp)))

    for s in range(nspins):
        for a, setup in enumerate(setups):
            D_sp = D_asp.get(a % len(D_asp))

            if D_sp is not None:
                I_sa[s, a] = setup.Nct / nspins - np.sqrt(4 * pi) * np.dot(
                    D_sp[s], setup.Delta_pL[:, 0]
                )
                I_sa[s, a] -= setup.Nc / nspins

    return I_sa


def _add_valence_density_correction(n_sg, gd, D_asp, setups, atoms):
    nspins = n_sg.shape[0]
    spos_ac = atoms.get_scaled_positions() % 1.0

    phi_aj = []
    phit_aj = []
    for setup in setups:
        phi_j, phit_j = setup.get_partial_waves()[:2]
        phi_aj.append(phi_j)
        phit_aj.append(phit_j)

    phi = BasisFunctions(gd, phi_aj)
    phit = BasisFunctions(gd, phit_aj)
    phi.set_positions(spos_ac)
    phit.set_positions(spos_ac)

    a_W = np.empty(len(phi.M_W), np.intc)
    W = 0
    for a in phi.atom_indices:
        nw = len(phi.sphere_a[a].M_w)
        a_W[W: W + nw] = a
        W += nw

    x_W = phi.create_displacement_arrays()[0]

    I_sa = _calculate_I_a(nspins, D_asp, setups)

    rho_MM = np.zeros((phi.Mmax, phi.Mmax))

    for s in range(nspins):
        M1 = 0
        for a, setup in enumerate(setups):
            ni = setup.ni
            D_sp = D_asp.get(a % len(D_asp))

            if D_sp is None:
                D_sp = np.zeros((nspins, ni * (ni + 1) // 2))

            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = unpack2(D_sp[s])
            M1 = M2

        assert np.all(n_sg[s].shape == phi.gd.n_c)
        phi.lfc.ae_valence_density_correction(rho_MM, n_sg[s], a_W, I_sa[s], x_W)
        phit.lfc.ae_valence_density_correction(-rho_MM, n_sg[s], a_W, I_sa[s], x_W)

    return n_sg, I_sa


def _add_core_density_correction(n_sg, gd, I_sa, setups, atoms):
    nspins = n_sg.shape[0]
    spos_ac = atoms.get_scaled_positions() % 1.0

    nc_a = []
    nct_a = []
    for setup in setups:
        nc, nct = setup.get_partial_waves()[2:4]
        nc_a.append([nc])
        nct_a.append([nct])

    nc = LFC(gd, nc_a)
    nct = LFC(gd, nct_a)
    nc.set_positions(spos_ac)
    nct.set_positions(spos_ac)

    a_W = np.empty(len(nc.M_W), np.intc)
    W = 0
    for a in nc.atom_indices:
        nw = len(nc.sphere_a[a].M_w)
        a_W[W: W + nw] = a
        W += nw
    scale = 1.0 / nspins

    for s, I_a in enumerate(I_sa):
        nc.lfc.ae_core_density_correction(scale, n_sg[s], a_W, I_a)
        nct.lfc.ae_core_density_correction(-scale, n_sg[s], a_W, I_a)

        N_c = gd.N_c
        g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c

        for I, g_c in zip(I_a, g_ac):
            if np.all(g_c >= 0) and np.all(g_c < gd.n_c):
                n_sg[s][tuple(g_c)] -= I / gd.dv

    return n_sg


def _get_all_electron_density(
    nt_sG, gd, D_asp: dict, setups, atoms: Atoms, gridrefinement: int = 1
):
    nspins = nt_sG.shape[0]
    spos_ac = atoms.get_scaled_positions() % 1.0

    n_sg, gd = _interpolate_pseudo_density(nt_sG, gd, gridrefinement)

    phi_aj = []
    phit_aj = []
    nc_a = []
    nct_a = []
    for setup in setups:
        phi_j, phit_j, nc, nct = setup.get_partial_waves()[:4]
        phi_aj.append(phi_j)
        phit_aj.append(phit_j)
        nc_a.append([nc])
        nct_a.append([nct])

    # Create localized functions from splines
    phi = BasisFunctions(gd, phi_aj)
    phit = BasisFunctions(gd, phit_aj)
    nc = LFC(gd, nc_a)
    nct = LFC(gd, nct_a)
    phi.set_positions(spos_ac)
    phit.set_positions(spos_ac)
    nc.set_positions(spos_ac)
    nct.set_positions(spos_ac)

    I_sa = np.zeros((nspins, len(spos_ac)))
    a_W = np.empty(len(phi.M_W), np.intc)
    W = 0
    for a in phi.atom_indices:
        nw = len(phi.sphere_a[a].M_w)
        a_W[W : W + nw] = a
        W += nw

    x_W = phi.create_displacement_arrays()[0]

    rho_MM = np.zeros((phi.Mmax, phi.Mmax))

    for s, I_a in enumerate(I_sa):
        M1 = 0
        for a, setup in enumerate(setups):
            ni = setup.ni
            D_sp = D_asp.get(a % len(D_asp))

            if D_sp is None:
                D_sp = np.zeros((nspins, ni * (ni + 1) // 2))
            else:
                I_a[a] = setup.Nct / nspins - np.sqrt(4 * pi) * np.dot(
                    D_sp[s], setup.Delta_pL[:, 0]
                )
                I_a[a] -= setup.Nc / nspins

            M2 = M1 + ni
            rho_MM[M1:M2, M1:M2] = unpack2(D_sp[s])
            M1 = M2

        assert np.all(n_sg[s].shape == phi.gd.n_c)
        phi.lfc.ae_valence_density_correction(rho_MM, n_sg[s], a_W, I_a, x_W)
        phit.lfc.ae_valence_density_correction(-rho_MM, n_sg[s], a_W, I_a, x_W)

    a_W = np.empty(len(nc.M_W), np.intc)
    W = 0
    for a in nc.atom_indices:
        nw = len(nc.sphere_a[a].M_w)
        a_W[W : W + nw] = a
        W += nw
    scale = 1.0 / nspins

    for s, I_a in enumerate(I_sa):
        nc.lfc.ae_core_density_correction(scale, n_sg[s], a_W, I_a)
        nct.lfc.ae_core_density_correction(-scale, n_sg[s], a_W, I_a)
        # D_asp.partition.comm.sum(I_a)

        N_c = gd.N_c
        g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c

        for I, g_c in zip(I_a, g_ac):
            if np.all(g_c >= 0) and np.all(g_c < gd.n_c):
                n_sg[s][tuple(g_c)] -= I / gd.dv

    return n_sg.sum(0) / Bohr**3


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

        atoms = self.frozen_phonons.atoms * self.repetitions
        random_atoms = self.frozen_phonons.randomize(atoms)

        ewald_parametrization = EwaldParametrization(width=1)

        ewald_potential = Potential(
            atoms=random_atoms,
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

        array = self._get_all_electron_density()

        for slic in _generate_slices(
            array, ewald_potential, first_slice=first_slice, last_slice=last_slice
        ):
            yield slic

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
        #print(electron_configurations[chemical_symbols[number]])
        #print(electron_configurations[chemical_symbols[number - charge]])

        config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in config})

        ionic_config = defaultdict(
            lambda: 0, {shell[:2]: shell[2] for shell in ionic_config}
        )

        #sss

        electrons = []
        for key in set(config.keys()).union(set(ionic_config.keys())):

            difference = config[key] - ionic_config[key]

            for i in range(np.abs(difference)):
                electrons.append(key + (-np.sign(difference),))

        return electrons

    def _get_all_electron_atom(self, symbol, charge=0.0):

        #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        ae = AllElectronAtom(symbol, spinpol=False, xc="PBE")

        added_electrons = self._get_added_electrons(symbol, charge)

        for added_electron in added_electrons:
            ae.add(*added_electron[:2], added_electron[-1])

        # # ae.run()
        # ae.run(mix=0.005, maxiter=5000, dnmax=1e-5)
        ae.run(maxiter=5000, mix=0.005, dnmax=1e-5)
        #ae.refine()

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
