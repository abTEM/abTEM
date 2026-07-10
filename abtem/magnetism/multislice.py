"""
Paraxial Pauli multislice for magnetic scattering.

Implements the two-component (spinor) paraxial Pauli equation of
Edström, Lubk and Rusz [PRB 94, 174414 (2016), Eq. (12); PRL 116, 127203
(2016), Eq. (3)] as a generalization of the real-space multislice method:

    d/dz |psi> = i * H |psi>,

    H = lambda/(4*pi) * laplace_xy      (paraxial propagation)
      + sigma_e * V / dz                (electrostatic transmission)
      - (e/hbar) * A_z                  (magnetic phase; collinear physics)
      + i * (e/(hbar*k)) * A_xy . grad_xy   (orbital coupling, L.B)
      - (e/(2*hbar*k)) * sigma . B      (spin Zeeman; sigma_x/sigma_y mix
                                         the two spin components)

with k = 2*pi/lambda the relativistically corrected wave number. All
magnetic coefficients are mass-independent (the gamma factors cancel
against the (hbar*k)**-1 prefactor of the Pauli equation), so relativistic
correctness follows from the corrected wavelength alone.

Each slice applies exp(1j * dz * H) by Taylor series — the same expansion
as `RealSpaceMultislice` — which handles the non-commuting A.grad term
without operator splitting. Fields follow the projected (slice-integrated)
convention of `PotentialArray`: vector potential slices in Å²T, magnetic
field slices in ÅT.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from ase import units

from abtem.antialias import AntialiasAperture
from abtem.core.axes import SpinAxis
from abtem.core.backend import get_array_module
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.utils import get_dtype
from abtem.detectors import BaseDetector, validate_detectors
from abtem.finite_difference import LaplaceOperator, _multislice_exponential_series
from abtem.magnetism.gpaw import GPAWMagneticFields
from abtem.magnetism.iam import MagneticFieldArray, VectorPotentialArray
from abtem.magnetism.pauli import ADotGradientOperator
from abtem.measurements import BaseMeasurements
from abtem.multislice import (
    MultisliceTransform,
    _generate_potential_configurations,
    _potential_ensemble_shape_and_metadata,
    _update_measurements,
    _validate_potential_ensemble_indices,
    allocate_multislice_measurements,
)
from abtem.potentials.iam import BasePotential, validate_potential

if TYPE_CHECKING:
    from abtem.waves import Waves

#: e / hbar in units of 1/(T Å²), so that (e/hbar) * A[ÅT] has units 1/Å.
e_over_hbar = units._e / (units._hplanck / (2 * np.pi)) * 1e-20


def _find_spin_axis(waves: "Waves") -> int:
    spin_axes = [
        i
        for i, axis in enumerate(waves.ensemble_axes_metadata)
        if isinstance(axis, SpinAxis)
    ]

    if len(spin_axes) != 1:
        raise ValueError(
            "the Pauli multislice solver requires spinor waves with exactly "
            "one SpinAxis ensemble axis; create them with Waves.to_spinor()"
        )

    spin_axis = spin_axes[0]

    if waves.shape[spin_axis] != 2:
        raise ValueError(
            f"the spin axis must have length 2, got {waves.shape[spin_axis]}"
        )

    return spin_axis


def _validate_field(field, potential, name: str, expected_cls) -> None:
    if not isinstance(field, expected_cls):
        raise ValueError(
            f"{name} must be a built {expected_cls.__name__}, got {type(field)}"
        )

    if field.num_slices != potential.num_slices:
        raise ValueError(
            f"{name} has {field.num_slices} slices, but the potential has "
            f"{potential.num_slices}; build them with the same slice_thickness"
        )

    if tuple(field.gpts) != tuple(potential.gpts):
        raise ValueError(
            f"{name} gpts {tuple(field.gpts)} do not match the potential "
            f"gpts {tuple(potential.gpts)}"
        )

    if not np.allclose(field.extent, potential.extent):
        raise ValueError(
            f"{name} extent {tuple(field.extent)} does not match the "
            f"potential extent {tuple(potential.extent)}"
        )


def _non_periodic_vector_potential_slice(
    average_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A_np(r) = 1/2 * B_avg x (r - r0) evaluated on the 2D grid of one slice
    [ÅT]; x, y are broadcastable 2D coordinate arrays and z the slice
    center.
    """
    Bx, By, Bz = average_field
    dx = x - origin[0]
    dy = y - origin[1]
    dz = z - origin[2]

    A_x = 0.5 * (By * dz - Bz * dy)
    A_y = 0.5 * (Bz * dx - Bx * dz)
    A_z = 0.5 * (Bx * dy - By * dx)
    return A_x, A_y, A_z


def _pauli_multislice_step(
    waves: "Waves",
    potential_slice,
    vector_potential_slice,
    magnetic_field_slice,
    laplace: LaplaceOperator,
    a_dot_gradient: ADotGradientOperator,
    antialias_aperture: AntialiasAperture,
    spin_axis: int,
    tolerance: float = 1e-16,
    max_terms: int = 300,
) -> "Waves":
    """
    Apply exp(1j * dz * H) for one slice, with H the paraxial Pauli
    operator. `vector_potential_slice` and `magnetic_field_slice` are
    device arrays of shape (3,) + gpts holding the *unprojected* per-slice
    rates: the vector potential in ÅT and the magnetic field in T (i.e.
    the projected slice arrays divided by the slice thickness, plus any
    uniform contributions).
    """
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    energy = waves._valid_energy
    wavelength = energy2wavelength(energy)
    thickness = potential_slice.thickness

    complex_dtype = get_dtype(complex=True)

    # Spin-diagonal multiplicative terms: sigma_e V / dz - (e/hbar) A_z.
    t_eff = potential_slice.array[0] * (energy2sigma(energy) / thickness)
    t_eff = t_eff - e_over_hbar * vector_potential_slice[2]
    t_eff = t_eff.astype(complex_dtype)

    # i * (e/(hbar k)) * A_xy . grad_xy with k = 2 pi / lambda.
    gradient_coefficient = complex_dtype(
        1.0j * e_over_hbar * wavelength / (2 * np.pi)
    )
    A_x = vector_potential_slice[0].astype(complex_dtype)
    A_y = vector_potential_slice[1].astype(complex_dtype)

    # -(e/(2 hbar k)) * sigma . B.
    zeeman_coefficient = e_over_hbar * wavelength / (4 * np.pi)
    B_z = None
    if magnetic_field_slice is not None:
        B_z = (zeeman_coefficient * magnetic_field_slice[2]).astype(complex_dtype)
        B_minus = (
            zeeman_coefficient
            * (magnetic_field_slice[0] - 1.0j * magnetic_field_slice[1])
        ).astype(complex_dtype)
        B_plus = (
            zeeman_coefficient
            * (magnetic_field_slice[0] + 1.0j * magnetic_field_slice[1])
        ).astype(complex_dtype)

    laplace_stencil = laplace.get_stencil(waves, device=waves.device)
    gradient_stencil = a_dot_gradient.get_stencil(waves, device=waves.device)

    laplace_prefactor = wavelength / (4 * np.pi)

    up = (slice(None),) * spin_axis + (0,)
    down = (slice(None),) * spin_axis + (1,)

    def pauli_operator(array):
        out = laplace_stencil(array) * laplace_prefactor
        out += t_eff * array
        out += gradient_coefficient * gradient_stencil(array, A_x, A_y)

        if B_z is not None:
            zeeman_up = B_z * array[up] + B_minus * array[down]
            zeeman_down = B_plus * array[up] - B_z * array[down]
            out[up] -= zeeman_up
            out[down] -= zeeman_down

        return out

    waves._array = _multislice_exponential_series(
        waves._array,
        None,
        laplace_stencil,
        wavelength,
        thickness,
        tolerance=tolerance,
        max_terms=max_terms,
        operator=pauli_operator,
    )

    waves = antialias_aperture.bandlimit(waves)

    return waves


def pauli_multislice_and_detect(
    waves: "Waves",
    potential: BasePotential,
    detectors: Optional[list[BaseDetector]] = None,
    vector_potential: Optional[VectorPotentialArray] = None,
    magnetic_field: Optional[MagneticFieldArray] = None,
    average_field: Optional[np.ndarray] = None,
    derivative_accuracy: int = 6,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    pbar: bool = False,
) -> BaseMeasurements | "Waves" | list[BaseMeasurements | "Waves"]:
    """
    Run the paraxial Pauli multislice algorithm for a batch of spinor wave
    functions through an electrostatic potential and magnetic
    vector-potential/field, detecting at the exit planes of the potential.

    See `pauli_multislice` for the user-facing entry point and parameter
    documentation.
    """
    waves = waves.ensure_real_space()
    detectors = validate_detectors(detectors)
    waves = waves.copy()

    spin_axis = _find_spin_axis(waves)

    if vector_potential is None:
        raise ValueError("vector_potential is required for the Pauli multislice")

    _validate_field(
        vector_potential, potential, "vector_potential", VectorPotentialArray
    )
    if magnetic_field is not None:
        _validate_field(
            magnetic_field, potential, "magnetic_field", MagneticFieldArray
        )

    xp = get_array_module(waves.device)
    real_dtype = get_dtype(complex=False)

    vector_potential_arrays = xp.asarray(
        vector_potential.array, dtype=real_dtype
    )
    magnetic_field_arrays = None
    if magnetic_field is not None:
        magnetic_field_arrays = xp.asarray(magnetic_field.array, dtype=real_dtype)

    if average_field is not None:
        average_field = np.asarray(average_field, dtype=float)
        if average_field.shape != (3,):
            raise ValueError(
                f"average_field must be a vector of shape (3,), got "
                f"{average_field.shape}"
            )
        # Coordinates for the non-periodic A_np = 1/2 B_avg x (r - r0); the
        # gauge origin is the supercell center, minimizing |A_np| over the
        # cell.
        extent = waves.extent
        gpts = waves.gpts
        x = xp.asarray(
            np.arange(gpts[0])[:, None] * (extent[0] / gpts[0]), dtype=real_dtype
        )
        y = xp.asarray(
            np.arange(gpts[1])[None] * (extent[1] / gpts[1]), dtype=real_dtype
        )
        total_thickness = potential.thickness
        origin = np.array([extent[0] / 2, extent[1] / 2, total_thickness / 2])
        average_field_device = xp.asarray(average_field, dtype=real_dtype)

    laplace = LaplaceOperator(derivative_accuracy)
    a_dot_gradient = ADotGradientOperator(derivative_accuracy)
    antialias_aperture = AntialiasAperture()

    (
        extra_ensemble_axes_shape,
        extra_ensemble_axes_metadata,
    ) = _potential_ensemble_shape_and_metadata(potential)

    if sum(extra_ensemble_axes_shape) == 1:
        measurements = None
    else:
        measurements = allocate_multislice_measurements(
            waves,
            detectors,
            extra_ensemble_axes_shape,
            extra_ensemble_axes_metadata,
        )

    n_waves = np.prod(waves.shape[:-2])
    n_slices = n_waves * potential.num_slices * potential.num_configurations

    tqdm_pbar = TqdmWrapper(
        enabled=pbar, total=int(n_slices), leave=False, desc="multislice"
    )

    waves_input = waves.copy()

    for potential_index, potential_configuration in _generate_potential_configurations(
        potential
    ):
        waves = waves_input.copy()
        exit_plane_index = 0

        if potential.exit_planes[0] == -1:
            measurement_index = _validate_potential_ensemble_indices(
                potential_index, exit_plane_index, potential
            )

            if measurements is not None:
                _update_measurements(waves, detectors, measurements, measurement_index)

            exit_plane_index += 1

        depth = 0.0

        for slice_index, potential_slice in enumerate(
            potential_configuration.generate_slices()
        ):
            thickness = potential_slice.thickness

            # The stored field slices are projected (slice-integrated);
            # divide by the thickness to get the per-slice rates the Pauli
            # operator uses.
            vector_potential_slice = vector_potential_arrays[slice_index] / thickness

            magnetic_field_slice = None
            if magnetic_field_arrays is not None:
                magnetic_field_slice = magnetic_field_arrays[slice_index] / thickness

            if average_field is not None:
                z = depth + thickness / 2
                A_np = _non_periodic_vector_potential_slice(
                    average_field_device, x, y, z, origin
                )
                vector_potential_slice = xp.stack(
                    [
                        vector_potential_slice[0] + A_np[0],
                        vector_potential_slice[1] + A_np[1],
                        vector_potential_slice[2] + A_np[2],
                    ]
                )
                if magnetic_field_slice is not None:
                    magnetic_field_slice = (
                        magnetic_field_slice + average_field_device[:, None, None]
                    )

            waves = _pauli_multislice_step(
                waves,
                potential_slice,
                vector_potential_slice,
                magnetic_field_slice,
                laplace=laplace,
                a_dot_gradient=a_dot_gradient,
                antialias_aperture=antialias_aperture,
                spin_axis=spin_axis,
                tolerance=tolerance,
                max_terms=max_terms,
            )
            tqdm_pbar.update_if_exists(int(n_waves))

            depth += thickness

            if potential_slice.exit_planes:
                measurement_index = _validate_potential_ensemble_indices(
                    potential_index, exit_plane_index, potential
                )

                if measurements is not None:
                    _update_measurements(
                        waves, detectors, measurements, measurement_index
                    )
                exit_plane_index += 1

    if measurements is None:
        measurements = [
            detector.detect(waves)[(None,) * len(potential.ensemble_shape)]
            for detector in detectors
        ]

    tqdm_pbar.close_if_exists()

    return measurements


def pauli_multislice(
    waves: "Waves",
    potential: Optional[BasePotential] = None,
    vector_potential: Optional[VectorPotentialArray] = None,
    magnetic_field: Optional[MagneticFieldArray] = None,
    fields: Optional[GPAWMagneticFields] = None,
    detectors: Optional[BaseDetector | list[BaseDetector]] = None,
    average_field: Optional[np.ndarray] = None,
    derivative_accuracy: int = 6,
    tolerance: float = 1e-16,
    max_terms: int = 300,
) -> BaseMeasurements | "Waves" | list[BaseMeasurements | "Waves"]:
    """
    Run the paraxial Pauli multislice algorithm [PRB 94, 174414 (2016),
    Eq. (12)] for spinor wave functions through an electrostatic potential
    and a magnetic vector potential/field, supporting fully non-collinear
    magnetization.

    The wave functions must carry a spin axis created with
    `Waves.to_spinor()`. Detectors produce per-spin measurements (the spin
    axis behaves as a labeled ensemble axis); sum over it for
    spin-averaged signals or take differences for spin contrast.

    Magnetic signals are of relative order 1e-4 to 1e-8, so float64
    precision (``abtem.config.set({"precision": "float64"})``) is strongly
    recommended.

    Parameters
    ----------
    waves : Waves
        Spinor wave functions (see `Waves.to_spinor`).
    potential : BasePotential
        Electrostatic potential. May carry a frozen-phonon ensemble; the
        magnetic fields are shared between configurations.
    vector_potential : VectorPotentialArray
        Built magnetic vector potential with the same grid and slicing as
        the potential (periodic part; slices in Å²T, slice-integrated).
    magnetic_field : MagneticFieldArray, optional
        Built magnetic field (periodic part; slices in ÅT,
        slice-integrated). Required for the spin Zeeman term — without it
        only the orbital (A) coupling is applied.
    fields : GPAWMagneticFields, optional
        Bundle providing any of the three components above that were not
        given explicitly (build the field with
        ``include_magnetic_field=True`` for the Zeeman term).
    detectors : (list of) BaseDetector, optional
        Detectors applied at the exit plane(s). Defaults to returning the
        exit wave functions.
    average_field : array of three float, optional
        The uniform part of the magnetic field B_avg [T] (e.g. from
        `calculate_constant_magnetic_field`), which a periodic vector
        potential cannot represent — nonzero for ferromagnets. Adds the
        non-periodic vector potential A_np = 1/2 B_avg x (r - r0) and a
        constant Zeeman field. Since A_np is non-periodic, results are
        only physical while the wave function has negligible amplitude at
        the supercell boundary.
    derivative_accuracy : int, optional
        Finite-difference accuracy for the Laplace and gradient operators
        (default 6).
    tolerance : float, optional
        Convergence tolerance for the per-slice exponential Taylor series
        (default 1e-16).
    max_terms : int, optional
        Maximum number of terms in the exponential Taylor series
        (default 300).

    Returns
    -------
    measurements : BaseMeasurements, Waves, or list of those
        One output per detector, each with the spin axis and any potential
        ensemble/exit-plane axes as ensemble axes.
    """
    if fields is not None:
        potential = potential if potential is not None else fields.potential
        if vector_potential is None:
            vector_potential = fields.vector_potential
        if magnetic_field is None:
            magnetic_field = fields.magnetic_field

    if potential is None or vector_potential is None:
        raise ValueError(
            "provide potential and vector_potential, either directly or "
            "through fields"
        )

    if magnetic_field is None:
        warnings.warn(
            "no magnetic_field given: the spin Zeeman term is omitted and "
            "only the orbital (vector-potential) coupling is applied. Build "
            "the fields with include_magnetic_field=True to include it."
        )

    if get_dtype(complex=True) == np.complex64:
        warnings.warn(
            "magnetic scattering signals are of relative order 1e-4 to 1e-8, "
            "below single precision accuracy; set "
            "abtem.config.set({'precision': 'float64'}) for meaningful "
            "results."
        )

    potential = validate_potential(potential, waves)
    waves.grid.match(potential.grid)

    _find_spin_axis(waves)

    if waves.is_lazy:
        # The Pauli step mixes the two spin components inside each dask
        # block, so the spin axis must stay in a single chunk.
        spin_axis = _find_spin_axis(waves)
        chunks = waves._lazy_array.chunks
        if chunks[spin_axis] != (2,):
            waves = waves.rechunk(
                chunks[:spin_axis] + ((2,),) + chunks[spin_axis + 1 :]
            )

    transform = MultisliceTransform(
        potential=potential,
        detectors=detectors,
        multislice_func=pauli_multislice_and_detect,
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
        average_field=average_field,
        derivative_accuracy=derivative_accuracy,
        tolerance=tolerance,
        max_terms=max_terms,
    )

    return transform.apply(waves)
