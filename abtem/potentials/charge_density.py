"""Module for describing the nuclear and electronic charge density used as the
electrostatic potential in multislice simulations."""

from __future__ import annotations

from functools import partial
from typing import Tuple, Union

import dask
import dask.array as da
import numpy as np
from ase import Atoms
from ase.cell import Cell
from scipy.ndimage import map_coordinates

from abtem.atoms import _box_strain_warning_silenced, plane_to_axes
from abtem.core.backend import copy_to_device
from abtem.core.constants import eps0
from abtem.core.ensemble import _wrap_with_array
from abtem.core.fft import fft_crop
from abtem.core.utils import get_dtype, itemset
from abtem.inelastic.phonons import AtomsEnsemble, DummyFrozenPhonons
from abtem.parametrizations import EwaldParametrization
from abtem.potentials.iam import Potential, PotentialArray, _PotentialBuilder


def _spatial_frequencies_orthorhombic(shape, cell: Cell):
    if not cell.orthorhombic:
        raise RuntimeError()

    kx, ky, kz = (np.fft.fftfreq(n, d=1 / n) for n in shape)
    lengths = cell.reciprocal().lengths()
    kx = kx[:, None, None] * lengths[0]
    ky = ky[None, :, None] * lengths[1]
    kz = kz[None, None, :] * lengths[2]
    return kx, ky, kz


def _spatial_frequencies_meshgrid(shape, cell):
    kx, ky, kz = (np.fft.fftfreq(n, d=1 / n) for n in shape)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    kp = np.array([kx.ravel(), ky.ravel(), kz.ravel()]).T
    kx, ky, kz = np.dot(kp, cell.reciprocal().array).T
    return kx.reshape(shape), ky.reshape(shape), kz.reshape(shape)


def _spatial_frequencies(shape, cell):
    if cell.orthorhombic:
        return _spatial_frequencies_orthorhombic(shape, cell)
    else:
        return _spatial_frequencies_meshgrid(shape, cell)


def _spatial_frequencies_squared(shape, cell: Cell):
    kx, ky, kz = _spatial_frequencies(shape, cell)
    return kx**2 + ky**2 + kz**2


def curl_fourier(vector_field: np.ndarray, cell: Cell) -> np.ndarray:
    """
    Calculate the curl of a vector field in 3D using Fourier-space differentiation.

    Parameters
    ----------
    vector_field : numpy.ndarray
        Array representing a vector field of dimension 3.
    cell : ase.cell.Cell
        ASE `Cell` object defining the region of space where the vector field is
        defined.

    Returns
    -------
    curl : numpy.ndarray
        Array representing the curl of the vector field.
    """

    F_hat = np.fft.fftn(vector_field, axes=(-3, -2, -1))

    shape = vector_field.shape[1:]
    kx, ky, kz = _spatial_frequencies(shape, cell)

    curl_x_hat = 2 * np.pi * 1.0j * (ky * F_hat[2] - kz * F_hat[1])
    curl_y_hat = 2 * np.pi * 1.0j * (kz * F_hat[0] - kx * F_hat[2])
    curl_z_hat = 2 * np.pi * 1.0j * (kx * F_hat[1] - ky * F_hat[0])

    curl_hat = np.stack((curl_x_hat, curl_y_hat, curl_z_hat))
    curl = np.fft.ifftn(curl_hat, axes=(-3, -2, -1)).real
    return curl


def integrate_gradient_fourier(
    array: np.ndarray, cell: Cell, in_space: str = "real", out_space: str = "real"
) -> np.ndarray:
    """
    Integrate an array representation of a gradient in 3D using Fourier-space
    integration.

    Parameters
    ----------
    array : numpy.ndarray
        Array representing a gradient of dimension 3.
    cell : ase.cell.Cell
        ASE `Cell` object defining the region of space where the gradient is integrated.
    in_space : str
        Space in which the gradient is defined (either "real" or "fourier").
    out_space : str
        Space in which the integrated gradient is defined ("real" or "fourier").

    Returns
    -------
    integrated : numpy.ndarray
        Integrated gradient.
    """

    if in_space == "real":
        array = np.fft.fftn(array, axes=(-3, -2, -1))

    k2 = _spatial_frequencies_squared(array.shape[-3:], cell)

    k2 = -(2**2) * np.pi**2 * k2
    k2[0, 0, 0] = 1.0
    array /= k2

    if out_space == "real":
        array = np.fft.ifftn(array, axes=(-3, -2, -1)).real

    return array


def _superpose_deltas(positions, array, scale=1):
    corners = np.floor(positions).astype(int)
    shape = array.shape

    xi = np.array([corners[:, 0] % shape[0], (corners[:, 0] + 1) % shape[0]]).T[
        :, :, None, None
    ]
    xi = np.tile(xi, (1, 1, 2, 2)).reshape((len(positions), -1))
    yi = np.array([corners[:, 1] % shape[1], (corners[:, 1] + 1) % shape[1]]).T[
        :, None, :, None
    ]
    yi = np.tile(yi, (1, 2, 1, 2)).reshape((len(positions), -1))
    zi = np.array([corners[:, 2] % shape[2], (corners[:, 2] + 1) % shape[2]]).T[
        :, None, None, :
    ]
    zi = np.tile(zi, (1, 2, 2, 1)).reshape((len(positions), -1))

    x, y, z = (positions - corners).T
    x = np.array([1 - x, x]).T[:, :, None, None]
    y = np.array([1 - y, y]).T[:, None, :, None]
    z = np.array([1 - z, z]).T[:, None, None, :]

    values = (x * y * z).reshape((len(positions), -1)) * scale
    array[xi, yi, zi] += values
    return array


def _add_point_charges_real_space(array, atoms):
    pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)

    inverse_cell = np.linalg.inv(np.array(atoms.cell))
    positions = np.dot(atoms.positions, inverse_cell)
    positions *= array.shape

    for number in np.unique(atoms.numbers):
        array = _superpose_deltas(
            positions[atoms.numbers == number],
            array,
            scale=number / pixel_volume,
        )

    return array


def _fourier_space_delta(kx, ky, kz, x, y, z):
    return np.exp(-2 * np.pi * 1j * (kx * x + ky * y + kz * z))


def _fourier_space_gaussian(k2, width):
    a = np.sqrt(1 / (2 * width**2)) / (2 * np.pi)
    return np.exp(-1 / (4 * a**2) * k2)


def add_point_charges_fourier(
    array: np.ndarray, atoms: Atoms, broadening: float = 0.05
) -> np.ndarray:
    """
    Add the nuclear point charges in Reciprocal space.

    Parameters
    ----------
    array : numpy.ndarray
        Array representing 3D charge density to which the point charges are added.
    atoms : ase.Atoms
        Atoms from which the nuclear charges with magnitudes and positions are
        determined.
    broadening : float
        Gaussian broadening of the point charges (default is 0.05).

    Returns
    -------
    density : numpy.ndarray
        3D charge density with added nuclear charges in reciprocal space.
    """
    pixel_volume = np.prod(np.diag(atoms.cell)) / np.prod(array.shape)

    kx, ky, kz = _spatial_frequencies(array.shape, atoms.cell)

    if broadening:
        broadening = _fourier_space_gaussian(kx**2 + ky**2 + kz**2, broadening)
    else:
        broadening = 0.05

    if hasattr(atoms, "atoms"):
        atoms = atoms.atoms

    for atom in atoms:
        scale = atom.number / pixel_volume
        array += scale * broadening * _fourier_space_delta(kx, ky, kz, *atom.position)

    return array


def _interpolate_between_cells(
    array, new_shape, old_cell, new_cell, offset=(0.0, 0.0, 0.0), order=2
):
    x = np.linspace(0, 1, new_shape[0], endpoint=False)
    y = np.linspace(0, 1, new_shape[1], endpoint=False)
    z = np.linspace(0, 1, new_shape[2], endpoint=False)

    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    coordinates = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    coordinates = np.dot(coordinates, new_cell) + offset

    padding = 3
    padded_array = np.pad(array, ((padding,) * 2,) * 3, mode="wrap")

    inverse_old_cell = np.linalg.inv(np.array(old_cell))
    mapped_coordinates = np.dot(coordinates, inverse_old_cell) % 1.0
    mapped_coordinates *= array.shape
    mapped_coordinates += padding

    interpolated = map_coordinates(
        padded_array, mapped_coordinates.T, mode="wrap", order=order
    )
    interpolated = interpolated.reshape(new_shape)
    return interpolated


def _interpolate_slice(array, cell, gpts, sampling, a, b):
    dz_charge = cell[2, 2] / array.shape[2]
    nz = max(int(np.ceil((b - a) / dz_charge)), 2)
    slice_shape = gpts + (nz,)

    slice_box = np.diag((gpts[0] * sampling[0], gpts[1] * sampling[1]) + (b - a,))

    dz = (b - a) / slice_shape[-1]

    # Midpoint rule: sample at a + (i + 1/2) dz. The samples used to start at
    # z = a (a left Riemann sum), which centres the slice integral dz/2 below
    # the slice centre -- a quarter of the slice thickness at the minimum of
    # two samples. Summed over a full period the bias cancels, but per slice it
    # shifts the long-range part of ChargeDensityPotential against the
    # short-range quadrature part, which is placed correctly: for a neutral
    # carbon atom on a slice boundary the slices above and below differed by
    # 2x and the potential 3 A from the atom was ~1 eV A instead of ~0.
    slice_array = _interpolate_between_cells(
        array, slice_shape, cell, slice_box, (0, 0, a + dz / 2)
    )

    return np.sum(slice_array, axis=-1) * dz


def _generate_slices(
    charge, ewald_potential, first_slice: int = 0, last_slice: int = None
):
    if last_slice is None:
        last_slice = len(ewald_potential)

    if ewald_potential.plane != "xy":
        axes = plane_to_axes(ewald_potential.plane)
        charge = np.moveaxis(charge, axes[:2], (0, 1))
        atoms, _, frame = ewald_potential._transform_atoms()
    else:
        atoms, frame = ewald_potential.frozen_phonons.atoms, np.eye(3)

    atoms = ewald_potential.frozen_phonons._randomize_transformed(atoms, frame)

    charge = -np.fft.fftn(charge)

    charge = fft_crop(
        charge, charge.shape[:2] + (ewald_potential.num_slices,), normalize=True
    )

    charge = add_point_charges_fourier(
        charge, atoms, ewald_potential.integrator.parametrization.width
    )

    potential = -(
        integrate_gradient_fourier(
            charge, atoms.cell, in_space="fourier", out_space="real"
        )
        / eps0
    )

    for i, ((a, b), slic) in enumerate(
        zip(
            ewald_potential.slice_limits[first_slice:last_slice],
            ewald_potential.generate_slices(first_slice, last_slice),
        )
    ):
        slice_array = _interpolate_slice(
            potential, atoms.cell, ewald_potential.gpts, ewald_potential.sampling, a, b
        )

        slic._array = slic._array + copy_to_device(slice_array[None], slic.array)

        slic._array -= slic._array.min()
        yield slic


class ChargeDensityPotential(_PotentialBuilder):
    """
    The charge density potential is used to calculate the electrostatic potential from a
    set of core charges defined by an ASE `Atoms` object and corresponding electron
    charge density defined by a NumPy array.

    Parameters
    ----------
    atoms : Atoms or FrozenPhonons
        Atomic configuration(s) used in the independent atom model for calculating the
        electrostatic potential(s).
    charge_density : numpy.ndarray
        Charge density as a 3D NumPy array [electrons / Å^3], shared by every
        configuration. A 4D array stacks charge densities along its first axis: one
        for each configuration, or a single one (a first axis of length one) shared by
        all of them. Any other number of charge densities raises a `ValueError`.
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential
        calculated by specifying either `sampling` or `gpts`.
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å] calculated by specifying either
        `sampling` or `gpts`.
    slice_thickness : float or sequence of float, optional
        Thickness of the potential slices [Å] (default is 0.5 Å). If given as a float,
        the number of slices are calculated by dividing the slice thickness into the
        `z`-height of the cell. The slice thickness may be given as a sequence of values
        for each slice, in which case an error will be thrown if the sum of slice
        thicknesses is not equal to the height of the atoms.
    exit_planes : int or tuple of int, optional
        The `exit_planes` argument can be used to calculate thickness series.
        Providing `exit_planes` as a tuple of int indicates that the tuple contains the
        slice indices after which an exit plane is desired, and hence during a
        multislice simulation a measurement is created. If `exit_planes` is an integer
        `n`, a measurement is collected every `n` slices and after the last slice. The
        first measurement is then taken at the entrance surface (zero thickness),
        before any scattering, so the thickness series has ``1 + ceil(num_slices / n)``
        planes; index it with ``[-1]`` for the exit surface. If `n` exceeds the number
        of slices, only the exit surface is returned and no thickness axis is added.
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
        Only the default (0., 0., 0.) is supported; any other origin raises a
        `NotImplementedError`, because the valence part is interpolated from the
        charge density of the atoms' own cell.
    box : three float, optional
        The extent of the potential in `x`, `y` and `z`. Only the default is
        supported, the atoms' cell repeated by `repetitions` (rotated to `plane`, and
        for a non-orthogonal cell its best orthogonal cell); any other box raises a
        `NotImplementedError`.
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
    """

    _supports_box_and_origin = False

    def __init__(
        self,
        atoms: Union[Atoms, AtomsEnsemble],
        charge_density: np.ndarray = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        slice_thickness: Union[float, Tuple[float]] = 1.0,
        plane: str = "xy",
        box: Tuple[float, float, float] = None,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        periodic: bool = True,
        exit_planes: int = None,
        repetitions: Tuple[int, int, int] = (1, 1, 1),
        device: str = None,
    ):
        if hasattr(atoms, "randomize"):
            self._frozen_phonons = atoms
        elif isinstance(atoms, Atoms):
            self._frozen_phonons = DummyFrozenPhonons(atoms)
        else:
            raise RuntimeError()

        self._charge_density = charge_density.astype(get_dtype(complex=False))
        self._repetitions = repetitions

        # A 3D charge density, or a stack of them along a first axis: one for
        # every configuration, or one shared by all of them.
        if self._charge_density.ndim == 4:
            num_configurations = (
                self._frozen_phonons.ensemble_shape[0]
                if len(self._frozen_phonons.ensemble_shape)
                else 1
            )
            num_densities = self._charge_density.shape[0]
            if num_densities not in (1, num_configurations):
                raise ValueError(
                    f"{num_densities} charge densities were given for "
                    f"{num_configurations} atomic configurations; give one, or one "
                    "for each configuration."
                )

        # ``Cell * repetitions`` broadcasts over columns, which only scales lattice
        # vectors correctly for an orthogonal cell. For a skewed cell with
        # anisotropic repetitions, each row (lattice vector) must be scaled by its
        # own repetition factor instead.
        cell = np.array(self._frozen_phonons.atoms.cell, dtype=float)
        cell = cell * np.array(repetitions, dtype=float)[:, None]

        super().__init__(
            array_object=PotentialArray,
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
    def num_configurations(self):
        return len(self.frozen_phonons)

    @property
    def ensemble_axes_metadata(self):
        return self.frozen_phonons.ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        return self.frozen_phonons.ensemble_shape

    @property
    def repetitions(self):
        return self._repetitions

    @property
    def num_frozen_phonons(self):
        return len(self.frozen_phonons)

    @property
    def is_lazy(self):
        return isinstance(self.charge_density, da.core.Array)

    @property
    def charge_density(self):
        return self._charge_density

    def _partition_args(self, chunks: int = 1, lazy: bool = True):
        chunks = self._validate_ensemble_chunks(chunks)

        charge_densities = self.charge_density

        if len(charge_densities.shape) == 3:
            charge_densities = charge_densities[None]
        elif len(charge_densities.shape) != 4:
            raise RuntimeError()

        def wrap_args(charge_density, frozen_phonon):
            return np.array(
                [{"charge_density": charge_density, "atoms": frozen_phonon}],
                dtype=object,
            )

        if len(self.ensemble_shape) == 0:
            # No ensemble dimension (DummyFrozenPhonons, single config)
            if lazy:
                if not isinstance(charge_densities, da.core.Array):
                    charge_densities = da.from_array(
                        charge_densities, chunks=(1, -1, -1, -1)
                    )
                block = dask.delayed(wrap_args)(charge_densities[0], None)
                array = da.from_delayed(block, shape=(), dtype=object)
            else:
                array = wrap_args(charge_densities[0], None)
                array = _wrap_with_array(array, ndims=0)
        else:
            # Ensemble case (FrozenPhonons or AtomsEnsemble)
            n_ensemble = self.ensemble_shape[0]

            if lazy:
                if not isinstance(charge_densities, da.core.Array):
                    charge_densities = da.from_array(
                        charge_densities, chunks=(1, -1, -1, -1)
                    )
                if charge_densities.shape[0] == 1 and n_ensemble > 1:
                    charge_densities = da.tile(
                        charge_densities, (n_ensemble,) + (1,) * 3
                    )
                charge_densities = charge_densities.to_delayed().ravel()
            elif hasattr(charge_densities, "compute"):
                raise RuntimeError
            elif charge_densities.shape[0] == 1 and n_ensemble > 1:
                # One charge density shared by every configuration.
                charge_densities = [charge_densities[0]] * n_ensemble

            frozen_phonon_blocks = (
                self._get_ewald_potential()
                .frozen_phonons._partition_args(chunks, lazy=lazy)[0]
            )
            if lazy:
                # Iterating a dask array yields dask array slices, whose shapes
                # dask cannot infer once wrapped by dask.delayed below; the
                # delayed blocks of to_delayed() compose correctly.
                frozen_phonon_blocks = frozen_phonon_blocks.to_delayed().ravel()

            array = np.zeros((len(chunks[0]),), dtype=object)
            for i, (cd, fp) in enumerate(
                zip(charge_densities, frozen_phonon_blocks)
            ):
                if lazy:
                    block = dask.delayed(wrap_args)(cd, fp)
                    itemset(
                        array, i, da.from_delayed(block, shape=(1,), dtype=object)
                    )
                else:
                    itemset(array, i, wrap_args(cd, fp))

            if lazy:
                array = da.concatenate(list(array))

        return (array,)

    # The box was reported when the user left it to abTEM.
    @staticmethod
    @_box_strain_warning_silenced()
    def _charge_density_potential(*args, frozen_phonons_partial, **kwargs):
        args = args[0]
        if hasattr(args, "item"):
            args = args.item()

        default_atoms = kwargs.pop("_default_atoms")
        if args["atoms"] is not None:
            atoms = frozen_phonons_partial(args["atoms"]).item()
        else:
            atoms = DummyFrozenPhonons(default_atoms)

        charge_density = args["charge_density"]
        potential = ChargeDensityPotential(
            atoms=atoms, charge_density=charge_density, **kwargs
        )
        return _wrap_with_array(potential)

    def _from_partitioned_args(self):
        kwargs = self._copy_kwargs(
            exclude=("atoms", "charge_density"), cls=ChargeDensityPotential
        )
        frozen_phonons_partial = (
            self._get_ewald_potential().frozen_phonons._from_partitioned_args()
        )
        return partial(
            self._charge_density_potential,
            frozen_phonons_partial=frozen_phonons_partial,
            _default_atoms=self.frozen_phonons.atoms,
            **kwargs,
        )

    # The box it is given is this potential's own, not one the user gave.
    @_box_strain_warning_silenced()
    def _get_ewald_potential(self):
        ewald_parametrization = EwaldParametrization(width=3)

        return Potential(
            atoms=self.frozen_phonons,
            gpts=self.gpts,
            sampling=self.sampling,
            parametrization=ewald_parametrization,
            slice_thickness=self.slice_thickness,
            projection="finite",
            plane=self.plane,
            box=self.box,
            origin=self.origin,
            exit_planes=self.exit_planes,
            device=self.device,
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

        if len(self.charge_density.shape) == 4:
            if self.charge_density.shape[0] > 1:
                raise RuntimeError()

            array = self.charge_density[0]
        elif len(self.charge_density.shape) == 3:
            array = self.charge_density
        else:
            raise RuntimeError()

        if self.repetitions != (1, 1, 1):
            array = np.tile(array, self.repetitions)
            atoms = self.frozen_phonons.atoms * self.repetitions
            ewald_parametrization = EwaldParametrization(width=3)
            ewald_potential = Potential(
                atoms=atoms,
                gpts=self.gpts,
                sampling=self.sampling,
                parametrization=ewald_parametrization,
                slice_thickness=self.slice_thickness,
                projection="finite",
                plane=self.plane,
                box=self.box,
                origin=self.origin,
                exit_planes=self.exit_planes,
                device=self.device,
            )
        else:
            ewald_potential = self._get_ewald_potential()

        for slic in _generate_slices(
            array, ewald_potential, first_slice=first_slice, last_slice=last_slice
        ):
            yield slic
