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

Two per-slice evolution schemes are provided, selected with the same
algorithm objects as the scalar multislice (see `pauli_multislice`'s
`algorithm` parameter): `RealSpaceMultislice` (the default) applies a
full Taylor expansion of exp(1j * dz * H), exact within a slice, while
`FourierMultislice` applies a fast Strang-symmetrized operator-splitting
scheme with the spectral Fourier propagator and exact pointwise
transmission/Zeeman factors, trading a dz^3-commutator splitting error
(requiring slice-thickness convergence, like conventional FFT
multislice) for a 7-15x speedup. Fields follow the projected
(slice-integrated) convention of `PotentialArray`: vector potential
slices in Å²T, magnetic field slices in ÅT.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from ase import units

from abtem.antialias import AntialiasAperture, antialias_aperture
from abtem.core.axes import PositionsAxis, ScanAxis, SpinAxis
from abtem.core.backend import get_array_module
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import energy2sigma, energy2wavelength
from abtem.core.fft import fft2, ifft2
from abtem.core.utils import get_dtype
from abtem.detectors import BaseDetector, validate_detectors
from abtem.finite_difference import LaplaceOperator, _multislice_exponential_series
from abtem.magnetism.gpaw import GPAWMagneticFields
from abtem.magnetism.iam import MagneticFieldArray, VectorPotentialArray
from abtem.magnetism.pauli import ADotGradientOperator
from abtem.measurements import BaseMeasurements
from abtem.multislice import (
    FourierMultislice,
    MultisliceTransform,
    RealSpaceMultislice,
    _fresnel_propagator_array,
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


def _validate_algorithm(
    algorithm: Optional[FourierMultislice | RealSpaceMultislice],
) -> FourierMultislice | RealSpaceMultislice:
    if algorithm is None:
        algorithm = RealSpaceMultislice()

    if isinstance(algorithm, RealSpaceMultislice):
        if algorithm.order != 1 or algorithm.expansion_scope != "propagator":
            raise ValueError(
                "the Pauli operator supports only order=1 and "
                "expansion_scope='propagator' for RealSpaceMultislice"
            )
    elif isinstance(algorithm, FourierMultislice):
        if algorithm.conjugate or algorithm.transpose:
            raise NotImplementedError(
                "conjugate/transpose are not implemented for the Pauli "
                "multislice"
            )
    else:
        raise ValueError(
            f"algorithm must be a RealSpaceMultislice or FourierMultislice, "
            f"got {type(algorithm)}"
        )

    return algorithm


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
    """
    Validate a field against the potential. The field may cover the full
    potential depth, or — for z-periodic samples (e.g. a `CrystalPotential`
    repeating a unit cell) — exactly one period of it: when the field's
    slice count divides the potential's, its slices are cycled modulo its
    length, so unit-cell field arrays never need tiling in z.
    """
    if not isinstance(field, expected_cls):
        raise ValueError(
            f"{name} must be a built {expected_cls.__name__}, got {type(field)}"
        )

    if potential.num_slices % field.num_slices != 0:
        raise ValueError(
            f"{name} has {field.num_slices} slices, which neither matches nor "
            f"divides the potential's {potential.num_slices}; build them with "
            f"the same slice_thickness (a field covering one z-period of a "
            f"repeating potential is cycled automatically)"
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

    # Mismatched thicknesses would silently mis-scale the per-slice rates
    # (the projected field slices are divided by the potential's slice
    # thickness). Compare cyclically to cover the z-periodic case.
    field_thickness = np.asarray(field.slice_thickness)
    potential_thickness = np.asarray(potential.slice_thickness)
    if not np.allclose(
        np.resize(field_thickness, potential_thickness.shape), potential_thickness
    ):
        raise ValueError(
            f"{name} slice thicknesses do not match the potential's; build "
            f"them with the same slice_thickness"
        )


def _probe_gauge_origins(waves: "Waves") -> Optional[np.ndarray]:
    """Per-ensemble-member gauge origins (x, y) for the probe-centred gauge.

    Returns an array of shape (prod(ensemble_shape), 2) ordered to match the
    flattening the gradient stencil applies to the wave array (C order over
    the ensemble axes), or None if the waves carry no scan/position axis.
    """
    shape = tuple(waves.ensemble_shape)
    metadata = waves.ensemble_axes_metadata

    scan_axes = [i for i, m in enumerate(metadata) if isinstance(m, ScanAxis)]
    position_axes = [i for i, m in enumerate(metadata) if isinstance(m, PositionsAxis)]

    xy = np.zeros(shape + (2,), dtype=float)

    if len(scan_axes) == 2:
        ax, ay = scan_axes
        x = np.asarray(metadata[ax].coordinates(shape[ax]), dtype=float)
        y = np.asarray(metadata[ay].coordinates(shape[ay]), dtype=float)
        x_shape = [1] * len(shape)
        x_shape[ax] = shape[ax]
        y_shape = [1] * len(shape)
        y_shape[ay] = shape[ay]
        xy[..., 0] = x.reshape(x_shape)
        xy[..., 1] = y.reshape(y_shape)
    elif len(position_axes) == 1:
        p = position_axes[0]
        values = np.asarray(metadata[p].values, dtype=float)
        if values.ndim != 2 or values.shape[-1] != 2:
            return None
        value_shape = [1] * len(shape) + [2]
        value_shape[p] = shape[p]
        xy[...] = values.reshape(value_shape)
    else:
        return None

    return xy.reshape(-1, 2)


def _uniform_vector_potential_ramps(
    average_field, origins, x, y, extent, padding, xp, dtype
):
    """Per-member uniform A_x(y) / A_y(x) ramps for the probe-centred gauge.

    A_np = 1/2 B_avg x (r - r0) with r0 the member's own probe position. For
    an axial B_avg this is separable -- A_x depends only on y, A_y only on x
    -- so each member needs two 1-D ramps rather than a full 2-D field, which
    is what keeps a per-position gauge affordable.

    The displacement r - r0 is taken in the MINIMUM-IMAGE convention. The wave
    is periodic and the stencil pads it by wrapping, so each periodic image of
    the wave must see the field centred on the corresponding image of its own
    probe; a ramp continued linearly into the padding would instead multiply
    wrapped wave content by a value belonging to the unwrapped position. The
    minimum image makes the ramp periodic -- consistent with the padding by
    construction -- at the cost of a discontinuity on the line antipodal to
    the probe, which is exactly where that probe has least amplitude.
    """
    Bx, By, Bz = (float(component) for component in average_field)

    if Bx != 0.0 or By != 0.0:
        raise NotImplementedError(
            "gauge_origin='probe' currently supports an axial average_field "
            "(B_avg along z) only; for a general direction A_np is not "
            "separable and would need a full per-position field. Got "
            f"average_field={tuple(average_field)}."
        )

    sampling_x = float(x[1, 0] - x[0, 0]) if x.shape[0] > 1 else 1.0
    sampling_y = float(y[0, 1] - y[0, 0]) if y.shape[1] > 1 else 1.0

    x_padded = (
        np.arange(-padding, x.shape[0] + padding) * sampling_x + float(x[0, 0])
    )
    y_padded = (
        np.arange(-padding, y.shape[1] + padding) * sampling_y + float(y[0, 0])
    )

    def minimum_image(coordinates, origin, period):
        delta = coordinates[None] - origin[:, None]
        return (delta + period / 2.0) % period - period / 2.0

    # A_x = -1/2 B_z (y - y0);  A_y = +1/2 B_z (x - x0)
    A_x = -0.5 * Bz * minimum_image(y_padded, origins[:, 1], extent[1])
    A_y = 0.5 * Bz * minimum_image(x_padded, origins[:, 0], extent[0])

    return xp.asarray(A_x, dtype=dtype), xp.asarray(A_y, dtype=dtype)


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


class _SplitStepPropagator:
    """
    Cached Fourier-space kernels for the split-step Pauli method: Fresnel
    propagators of the requested order (see `FourierMultislice.order`;
    order 1 is the paraxial exponential of the same Laplacian term the
    series method expands, so that choice solves the strictly paraxial
    Pauli equation of the papers), optionally with the antialiasing
    aperture folded in (saving the separate per-slice bandlimiting FFT
    pair the series method needs). The kernel itself is built by the same
    `_fresnel_propagator_array` the conventional multislice uses.

    Kernels are cached per (thickness, mask) so the Strang half/merged
    steps (dz/2 at entry and exit planes, (dz_i + dz_{i+1})/2 between
    slices) do not thrash a single-slot cache.
    """

    def __init__(self, order=1):
        self._order = order
        self._grid_key = None
        self._kernels = {}

    def propagate(
        self, waves: "Waves", thickness: float, mask: bool
    ) -> "Waves":
        grid_key = (
            waves._valid_gpts,
            waves._valid_sampling,
            waves.wavelength,
            waves.device,
            waves.array.dtype,
        )
        if grid_key != self._grid_key:
            self._kernels = {}
            self._grid_key = grid_key

        key = (thickness, mask)
        kernel = self._kernels.get(key)
        if kernel is None:
            xp = get_array_module(waves.device)
            kernel = _fresnel_propagator_array(
                thickness,
                waves._valid_gpts,
                waves._valid_sampling,
                waves._valid_energy,
                waves.device,
                order=self._order,
            )
            if mask:
                kernel = kernel * antialias_aperture(
                    waves._valid_gpts, waves._valid_sampling, xp
                )
            kernel = kernel.astype(waves.array.dtype)
            self._kernels[key] = kernel

        # In-place kernel multiply, matching the scalar path's
        # _fft2_convolve -- the out-of-place product would allocate an
        # extra full wave-block on every slice.
        array = fft2(waves._array, overwrite_x=True)
        try:
            array *= kernel
        except ValueError:
            array = array * kernel
        waves._array = ifft2(array, overwrite_x=True)
        return waves


def _pauli_multislice_step_split(
    waves: "Waves",
    potential_slice,
    vector_potential_slice,
    magnetic_field_slice,
    a_dot_gradient: ADotGradientOperator,
    spin_axis: int,
    tolerance: float = 1e-16,
    max_terms: int = 300,
    uniform_ramps: Optional[tuple] = None,
) -> "Waves":
    """
    Apply the transmission part of one slice of the paraxial Pauli
    evolution for the split-step method:

        exp(i dz (T + Z + G)) ≈ T_zeeman . exp(i dz G)

    with T_zeeman the *exact* pointwise electrostatic/A_z/Zeeman factor
    (the Zeeman part is a per-pixel 2x2 spin rotation computed in closed
    form) and G = i (e/(hbar k)) A_xy . grad_xy expanded in a short Taylor
    series (its per-slice magnitude is ~1e-3, so 2-3 terms reach machine
    precision). Propagation is applied by the caller in Strang-symmetrized
    order (half-steps around each transmission), so the leading splitting
    error is of order dz^3 double commutators, one order better than the
    conventional (non-magnetic) FFT multislice algorithm.
    """
    if waves.device != potential_slice.device:
        potential_slice = potential_slice.copy_to_device(device=waves.device)

    xp = get_array_module(waves.device)
    energy = waves._valid_energy
    wavelength = energy2wavelength(energy)
    thickness = potential_slice.thickness

    complex_dtype = get_dtype(complex=True)

    # Exact pointwise transmission: exp(i (sigma_e V_proj - (e/hbar) A_z_proj)).
    phase = potential_slice.array[0] * energy2sigma(energy)
    phase = phase - (e_over_hbar * thickness) * vector_potential_slice[2]
    transmission = complex_exponential(phase).astype(complex_dtype)

    array = waves._array

    up = (slice(None),) * spin_axis + (0,)
    down = (slice(None),) * spin_axis + (1,)

    if magnetic_field_slice is not None:
        # Exact per-pixel Zeeman rotation exp(-i theta . sigma) with
        # theta = dz (e lambda / 4 pi hbar) B; Pauli-matrix identity
        # exp(-i theta n.sigma) = cos(theta) - i sin(theta) n.sigma.
        zeeman_coefficient = e_over_hbar * wavelength / (4 * np.pi)
        theta = (zeeman_coefficient * thickness) * magnetic_field_slice

        if xp.any(theta[0]) or xp.any(theta[1]):
            theta_mag = xp.sqrt(theta[0] ** 2 + theta[1] ** 2 + theta[2] ** 2)
            cos = xp.cos(theta_mag)
            # sin(theta)/theta via sinc, exact and finite at theta = 0.
            sinc = xp.sinc(theta_mag / np.pi).astype(complex_dtype)
            u_diag = (cos - 1.0j * sinc * theta[2]).astype(complex_dtype)
            u_diag_conj = (cos + 1.0j * sinc * theta[2]).astype(complex_dtype)
            u_up_down = (-1.0j * sinc * (theta[0] - 1.0j * theta[1])).astype(
                complex_dtype
            )
            u_down_up = (-1.0j * sinc * (theta[0] + 1.0j * theta[1])).astype(
                complex_dtype
            )
            # Accumulate each output component in place instead of via the
            # nested expression transmission * (a * up + b * down), which
            # holds several wave-sized temporaries at once; per-slice
            # allocation churn at that scale fragments the CuPy memory
            # pool over many slices.
            new_up = u_diag * array[up]
            new_up += u_up_down * array[down]
            new_up *= transmission
            new_down = u_diag_conj * array[down]
            new_down += u_down_up * array[up]
            new_down *= transmission
            array[up] = new_up
            array[down] = new_down
            # Release the wave-sized buffers before the gradient series
            # below allocates its own.
            del new_up, new_down
        else:
            # Collinear fast path: the rotation is diagonal, exp(∓i theta_z).
            array[up] *= transmission * complex_exponential(-theta[2]).astype(
                complex_dtype
            )
            array[down] *= transmission * complex_exponential(theta[2]).astype(
                complex_dtype
            )
    else:
        array *= transmission

    # exp(i dz G) with i dz G = -dz (e lambda / 2 pi hbar) A_xy . grad_xy,
    # by Taylor series; converges in 2-3 terms for physical field strengths.
    gradient_stencil = a_dot_gradient.get_stencil(waves, device=waves.device)
    A_x = vector_potential_slice[0].astype(complex_dtype)
    A_y = vector_potential_slice[1].astype(complex_dtype)

    A_x_uniform, A_y_uniform = uniform_ramps if uniform_ramps is not None else (None, None)

    if xp.any(A_x) or xp.any(A_y) or A_x_uniform is not None:
        A_x_padded = gradient_stencil.pad_field(A_x)
        A_y_padded = gradient_stencil.pad_field(A_y)
        gradient_exponent = complex_dtype(
            -thickness * e_over_hbar * wavelength / (2 * np.pi)
        )

        initial_amplitude = xp.abs(array).sum()
        term = gradient_exponent * gradient_stencil(
            array, A_x_padded, A_y_padded, A_x_uniform, A_y_uniform
        )
        array += term
        for i in range(2, max_terms + 1):
            term = (
                gradient_exponent
                * gradient_stencil(term, A_x_padded, A_y_padded,
                                   A_x_uniform, A_y_uniform)
                / i
            )
            array += term
            if xp.abs(term).sum() / initial_amplitude <= tolerance:
                break

    waves._array = array
    return waves


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
    xp = get_array_module(waves.device)
    B_z = None
    B_minus = None
    if magnetic_field_slice is not None:
        B_z = (zeeman_coefficient * magnetic_field_slice[2]).astype(complex_dtype)
        # Collinear fast path: skip the spin-off-diagonal work when the
        # in-plane field vanishes (the common collinear-DFT case).
        if xp.any(magnetic_field_slice[0]) or xp.any(magnetic_field_slice[1]):
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

    # A_x, A_y are constant across every Taylor-series term of this slice's
    # exponential (only the wave function changes between terms), so pad
    # them once here instead of on every gradient_stencil call.
    A_x_padded = gradient_stencil.pad_field(A_x)
    A_y_padded = gradient_stencil.pad_field(A_y)

    def pauli_operator(array):
        out = laplace_stencil(array) * laplace_prefactor
        out += t_eff * array
        out += gradient_coefficient * gradient_stencil(array, A_x_padded, A_y_padded)

        if B_z is not None:
            if B_minus is not None:
                zeeman_up = B_z * array[up] + B_minus * array[down]
                zeeman_down = B_plus * array[up] - B_z * array[down]
                out[up] -= zeeman_up
                out[down] -= zeeman_down
            else:
                out[up] -= B_z * array[up]
                out[down] += B_z * array[down]

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
    algorithm: Optional[FourierMultislice | RealSpaceMultislice] = None,
    pbar: bool = False,
    gauge_origin: Optional[str | tuple[float, float, float]] = None,
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

    algorithm = _validate_algorithm(algorithm)
    use_split = isinstance(algorithm, FourierMultislice)

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

    # Kept on their native (usually host) device; each slice is transferred
    # on demand in the loop below, so the full field stacks never need to
    # fit in device memory alongside the waves.
    vector_potential_arrays = vector_potential.array
    magnetic_field_arrays = None
    if magnetic_field is not None:
        magnetic_field_arrays = magnetic_field.array

    if average_field is not None:
        average_field = np.asarray(average_field, dtype=float)
        if average_field.shape != (3,):
            raise ValueError(
                f"average_field must be a vector of shape (3,), got "
                f"{average_field.shape}"
            )
        if not np.any(average_field):
            average_field = None

    probe_gauge = isinstance(gauge_origin, str) and gauge_origin == "probe"
    if isinstance(gauge_origin, str) and not probe_gauge:
        raise ValueError(
            f"gauge_origin must be 'probe', a length-3 sequence, or None, "
            f"got {gauge_origin!r}"
        )
    probe_origins = None

    if average_field is not None:
        # Coordinates for the non-periodic A_np = 1/2 B_avg x (r - r0).
        extent = waves.extent
        gpts = waves.gpts
        x = xp.asarray(
            np.arange(gpts[0])[:, None] * (extent[0] / gpts[0]), dtype=real_dtype
        )
        y = xp.asarray(
            np.arange(gpts[1])[None] * (extent[1] / gpts[1]), dtype=real_dtype
        )
        total_thickness = potential.thickness
        if gauge_origin is None or probe_gauge:
            origin = np.array([extent[0] / 2, extent[1] / 2, total_thickness / 2])
        else:
            origin = np.asarray(gauge_origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError(
                    f"an explicit gauge_origin must have shape (3,), got "
                    f"{origin.shape}"
                )
        average_field_device = xp.asarray(average_field, dtype=real_dtype)

        if probe_gauge:
            probe_origins = _probe_gauge_origins(waves)
            if probe_origins is None:
                raise ValueError(
                    "gauge_origin='probe' needs the waves to carry scan or "
                    "position axes so each member's own probe position is "
                    "known; got ensemble axes "
                    f"{[type(m).__name__ for m in waves.ensemble_axes_metadata]}."
                )
    elif probe_gauge:
        # No uniform field -> no A_np -> the gauge origin is irrelevant.
        probe_gauge = False

    if use_split:
        # The split method's A_xy gradient series has per-slice magnitude
        # ~1e-3, so these fixed settings reach machine precision in a few
        # terms; the scheme knobs users tune live on the algorithm object.
        derivative_accuracy = 6
        tolerance = 1e-16
        max_terms = 300
        laplace = None
        split_step_propagator = _SplitStepPropagator(order=algorithm.order)
    else:
        derivative_accuracy = algorithm.derivative_accuracy
        tolerance = algorithm.tolerance
        max_terms = algorithm.max_terms
        laplace = LaplaceOperator(derivative_accuracy)
        split_step_propagator = None

    a_dot_gradient = ADotGradientOperator(derivative_accuracy)
    antialias_aperture = AntialiasAperture()

    # Probe-centred gauge: the uniform A_np is carried as per-member 1-D ramps
    # instead of being folded into the shared 2-D vector-potential slice, so
    # every probe sits at its own gauge origin. Built once -- the ramps are
    # z-independent for an axial B_avg.
    uniform_ramps = None
    if probe_gauge:
        if not use_split:
            raise NotImplementedError(
                "gauge_origin='probe' is currently implemented for the "
                "split-step scheme only; pass algorithm=FourierMultislice()."
            )
        stencil = a_dot_gradient.get_stencil(waves, device=waves.device)
        uniform_ramps = _uniform_vector_potential_ramps(
            average_field,
            probe_origins,
            x,
            y,
            extent,
            stencil.padding,
            xp,
            get_dtype(complex=True),
        )

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

    # The multislice step mutates the wave array in place; each
    # configuration starts from a fresh copy, leaving the input untouched.
    waves_input = waves

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

        # Strang bookkeeping for the split method: each slice's evolution
        # is P(dz/2) T P(dz/2); consecutive half-propagations merge into
        # one FFT pair, so the sweep is P(dz_1/2), T_1, P((dz_1+dz_2)/2),
        # T_2, ..., T_n, P(dz_n/2). `previous_half` tracks the trailing
        # half-thickness still owed before the next transmission. The
        # antialias mask is folded into every kernel except the entry one
        # (one bandlimit per slice, matching the series method).
        previous_half = None

        for slice_index, potential_slice in enumerate(
            potential_configuration.generate_slices()
        ):
            thickness = potential_slice.thickness

            # The stored field slices are projected (slice-integrated);
            # divide by the thickness to get the per-slice rates the Pauli
            # operator uses. The modulo cycles z-periodic (unit-cell)
            # fields over a repeating potential (see _validate_field).
            vector_potential_slice = (
                xp.asarray(
                    vector_potential_arrays[
                        slice_index % len(vector_potential_arrays)
                    ],
                    dtype=real_dtype,
                )
                / thickness
            )

            magnetic_field_slice = None
            if magnetic_field_arrays is not None:
                magnetic_field_slice = (
                    xp.asarray(
                        magnetic_field_arrays[
                            slice_index % len(magnetic_field_arrays)
                        ],
                        dtype=real_dtype,
                    )
                    / thickness
                )

            if average_field is not None:
                if not probe_gauge:
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
                # With the probe-centred gauge the transverse A_np is applied
                # per member through `uniform_ramps`; A_z vanishes for the
                # axial B_avg that gauge supports, so nothing is added here.
                if magnetic_field_slice is not None:
                    magnetic_field_slice = (
                        magnetic_field_slice + average_field_device[:, None, None]
                    )
                else:
                    # No periodic field given, but average_field promises a
                    # constant Zeeman term.
                    magnetic_field_slice = xp.broadcast_to(
                        average_field_device[:, None, None], (3,) + tuple(gpts)
                    )

            if use_split:
                if previous_half is None:
                    waves = split_step_propagator.propagate(
                        waves, thickness / 2, mask=False
                    )
                else:
                    waves = split_step_propagator.propagate(
                        waves, previous_half + thickness / 2, mask=True
                    )
                waves = _pauli_multislice_step_split(
                    waves,
                    potential_slice,
                    vector_potential_slice,
                    magnetic_field_slice,
                    a_dot_gradient=a_dot_gradient,
                    spin_axis=spin_axis,
                    tolerance=tolerance,
                    max_terms=max_terms,
                    uniform_ramps=uniform_ramps,
                )
                previous_half = thickness / 2
            else:
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
                    if use_split:
                        # Complete the owed trailing half-propagation on a
                        # copy for detection; the sweep itself continues
                        # uncorrected (the next merged kernel includes it).
                        exit_waves = split_step_propagator.propagate(
                            waves.copy(), previous_half, mask=True
                        )
                    else:
                        exit_waves = waves
                    _update_measurements(
                        exit_waves, detectors, measurements, measurement_index
                    )
                exit_plane_index += 1

        if use_split and previous_half is not None:
            # Finish the sweep for this configuration so the fallback
            # (single exit plane, measurements allocated lazily below)
            # detects a fully evolved wave.
            waves = split_step_propagator.propagate(
                waves, previous_half, mask=True
            )

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
    algorithm: Optional[FourierMultislice | RealSpaceMultislice] = None,
    gauge_origin: Optional[str | tuple[float, float, float]] = None,
    max_batch: int | str = "auto",
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
        For a z-periodic sample (e.g. a `CrystalPotential` repeating a
        unit cell in z), a field covering exactly one period may be given
        instead — its slices are cycled automatically, avoiding tiling
        the field arrays in z.
    magnetic_field : MagneticFieldArray, optional
        Built magnetic field (periodic part; slices in ÅT,
        slice-integrated; may cover one z-period, like
        `vector_potential`). Required for the spin Zeeman term — without
        it only the orbital (A) coupling is applied.
    fields : GPAWMagneticFields, optional
        Bundle providing any of the components above (including
        `average_field`) that were not given explicitly; build the field
        with ``include_magnetic_field=True`` for the periodic Zeeman term.
        Pass ``average_field=(0, 0, 0)`` explicitly to suppress the
        bundle's uniform field.
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

        Enlarging the supercell does NOT remove the resulting boundary
        error: |A_np| grows linearly with the cell size while the boundary
        amplitude falls, so the artifact saturates (measured: a symmetry
        violation that is flat from 10x10 unit cells upward, and ~500x
        above the average_field=0 baseline). It shows up predominantly in
        orbital quantities -- for a vortex-beam OAM difference it can be
        comparable to the signal, while leaving total intensity and spin
        differences almost untouched. Note also that mu_0 * M is the
        3D-periodic value with no demagnetizing field, which is the wrong
        uniform field for a slab magnetized along its normal. See
        `abtem.magnetism.gpaw.calculate_non_periodic_magnetic_vector_potential`
        for the quantitative details and for mitigations.

        For a beam significantly wider than the unit cell (e.g. a
        large-OAM vortex probe), the interaction with this uniform field
        dominates the orbital (L.B) coupling — the periodic field alone
        gives a near-vanishing, misleadingly small orbital signal for a
        ferromagnet. Omit `average_field` only for compensated textures
        (e.g. antiferromagnets, where B_avg = 0) or when deliberately
        isolating the periodic-field contribution.
    gauge_origin : "probe" or three float, optional
        Gauge origin r0 for the non-periodic A_np = 1/2 B_avg x (r - r0).
        Ignored unless `average_field` is nonzero.

        Default (None) puts r0 at the supercell centre, shared by every
        probe position. That choice is not innocuous for a scan: a probe at
        p then sits in a constant A_np(p) ~ (p - r0), and since the paraxial
        operator is linear in A (the A^2 term is dropped) a constant vector
        potential is not a pure gauge phase -- it tilts the beam, which
        drifts laterally as it propagates by an amount that depends on where
        in the cell the probe sits. The scanned image is then not
        lattice-periodic and loses the crystal's point symmetry, at a
        relative size of ~1e-5 for FePt at B_avg = 1.385 T, comparable to
        the magnetic signal itself.

        Pass "probe" to give every scan position its own gauge origin,
        r0 = p. A beam entering parallel to a uniform axial B feels no
        force (v x B = 0) and must travel straight wherever it enters;
        r0 = p makes the canonical and kinetic transverse momenta agree at
        the probe, so no spurious tilt is introduced, and lattice
        periodicity (hence the crystal symmetry of the scan) is restored.
        Verified on FePt: the four-fold symmetry violation of the scanned
        intensity drops from 1.4e-5 to 7.8e-8, the latter being the
        average_field=0 baseline.

        Requires the split-step scheme (`FourierMultislice`), waves carrying
        scan or position axes, and an axial B_avg (for a general direction
        A_np is not separable). A length-3 sequence sets an explicit fixed
        origin instead.
    algorithm : RealSpaceMultislice or FourierMultislice, optional
        Per-slice evolution scheme, selected with the same algorithm
        objects as the scalar `Waves.multislice`:

        - `RealSpaceMultislice` (the default): full Taylor expansion of
          the per-slice exponential of the complete Pauli operator
          [PRB 94, 174414, Eq. (14)]. Exact within a slice (no
          operator-splitting error), but its propagation term uses a
          finite-difference Laplacian (`derivative_accuracy` field)
          whose k-space dispersion error at coarse transverse sampling
          does not shrink with slice thickness. Only ``order=1`` and
          ``expansion_scope="propagator"`` are supported for the Pauli
          operator. Its ``tolerance`` field directly trades speed for
          per-slice accuracy: the magnetic signal is of relative order
          1e-4 to 1e-8, so a tolerance a few orders below the signal
          (rather than the 1e-16 default) reduces the per-slice term
          count substantially at no meaningful cost (measured: ~1.9x
          faster at 1e-7 for a relative signal change of ~8e-8).
        - `FourierMultislice`: Strang-symmetrized operator splitting
          with the spectral Fourier propagator of the requested
          ``order`` (half-steps around each transmission, merged into
          one FFT pair per slice; ``order=1`` solves the strictly
          paraxial equation of the papers), exact pointwise
          transmission/Zeeman factors (the Zeeman part as a closed-form
          per-pixel spin rotation), and a short internal Taylor series
          for the A_xy gradient coupling only. Per-slice splitting
          error is of order dz^3 double commutators — one order better
          than the standard (non-magnetic) FFT multislice algorithm —
          at 7-15x the speed of the default. ``conjugate`` and
          ``transpose`` are not supported for the Pauli operator.

        The two schemes differ in which error each carries: the
        real-space series is exact in dz but approximates transverse
        propagation with a stencil; the Fourier split-step propagates
        spectrally (exact on the grid) but splits the slice operator
        (error vanishing as dz -> 0). At converged transverse sampling
        they agree (measured: 2.7% signal-level discrepancy at 12 grid
        points per FePt unit cell collapsing to 0.03% at 18) — so
        converge the transverse grid first; a residual difference at
        coarse sampling mostly reflects the stencil dispersion of the
        series scheme, not a split-step deficiency.
    max_batch : int or str, optional
        Maximum ensemble members (e.g. scan positions) per dask block, as
        for `Waves.multislice`. The default `"auto"` sizes each block from
        the wave array alone, without accounting for the extra device
        memory this solver needs beyond it: the A_xy gradient's Taylor
        series holds several wave-sized temporaries per term (padded
        fields, the running term, the accumulator), and both the potential
        and the tiled field arrays add their own footprint independent of
        block size. On a memory-constrained GPU this combination can OOM
        even where "auto" alone would not; pass a smaller integer (an
        explicit member count, not a byte budget) to force smaller blocks.

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
        if average_field is None:
            average_field = fields.average_field

    if potential is None or vector_potential is None:
        raise ValueError(
            "provide potential and vector_potential, either directly or "
            "through fields"
        )

    if magnetic_field is None:
        warnings.warn(
            "no magnetic_field given: the spin Zeeman term of the periodic "
            "field is omitted (a constant Zeeman term is still applied if "
            "average_field is given). Build the fields with "
            "include_magnetic_field=True to include the periodic part."
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

    # gauge_origin="probe" relies on `_probe_gauge_origins` reading each dask
    # block's own probe positions from that block's `ensemble_axes_metadata`
    # (called per block inside the multislice loop). This is correct
    # regardless of how the scan is chunked as long as `LinearAxis.__getitem__`
    # (abtem/core/axes.py) shifts `offset` per block -- which is what makes
    # `ScanAxis` (GridScan) chunk-safe the same way `PositionsAxis`
    # (CustomScan) already was. No special-case rechunking is needed here.

    transform = MultisliceTransform(
        potential=potential,
        detectors=detectors,
        multislice_func=pauli_multislice_and_detect,
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
        average_field=average_field,
        algorithm=algorithm,
        gauge_origin=gauge_origin,
    )

    return transform.apply(waves, max_batch=max_batch)
