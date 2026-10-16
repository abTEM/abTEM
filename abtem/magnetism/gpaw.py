from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from ase import Atoms
from scipy.spatial.transform import Rotation as R  # type: ignore

from abtem.atoms import (
    _box_repetitions,
    _box_strain_warning_silenced,
    _cell_in_plane_frame,
    _vector_field_to_plane,
    plane_to_axes,
)
from abtem.bloch.dynamical import equal_slice_thicknesses
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.fft import fft_interpolate
from abtem.core.utils import get_dtype
from abtem.inelastic.phonons import BaseFrozenPhonons
from abtem.magnetism.iam import (
    BaseMagneticField,
    BaseVectorPotential,
    MagneticFieldArray,
    VectorPotentialArray,
)
from abtem.magnetism.utils import bohr_magneton, vacuum_permeability
from abtem.potentials.charge_density import (
    band_limit_fourier,
    curl_fourier,
    integrate_gradient_fourier,
)
from abtem.potentials.gpaw import _GPAW_LOCK, GPAWPotential
from abtem.potentials.iam import PotentialArray, _default_box, _FieldBuilder
from abtem.slicing import _integrate_planes, is_number


#: Default Fourier-space band limit [1/Å] applied to a magnetization density
#: before the vector potential and magnetic field are constructed from it.
#:
#: An all-electron magnetization density reconstructed pointwise on a uniform
#: grid contains near-nucleus structure far sharper than any such grid can
#: represent: for a heavy element the peak value does not converge with grid
#: refinement (measured for L1_0 FePt at the Pt site: the peak spin density
#: grows 170 -> 500 -> 900 -> 3100 μB/Å³ as the grid is refined 4x -> 12x,
#: while the volume integral -- the physically meaningful magnetic moment --
#: stays constant to five digits). Constructing B from that density is
#: ill-posed in the pointwise sense, and because B = curl(A) multiplies each
#: Fourier component by its frequency, the unrepresentable high-frequency
#: content is amplified rather than averaged away: the resulting near-nucleus
#: field values are grid-dependent and can exceed the physical field by
#: orders of magnitude.
#:
#: Band-limiting the magnetization first makes the construction well-posed --
#: the resulting fields become grid-refinement independent -- and reflects how
#: a density is represented in practice: an LAPW density (as used in
#: Edström, Lubk & Rusz, PRB 94, 174414 (2016)) is a Fourier series with a
#: finite cutoff in the interstitial region and a radial expansion inside the
#: muffin-tin spheres, so placing it on a uniform grid band-limits it
#: inherently -- without making it any less all-electron.
#:
#: The default of 4 1/Å is set to match the density cutoff of the LAPW code
#: used for the reference calculations. WIEN2k expands rho and V in the
#: interstitial region up to a largest wavevector GMAX given in bohr^-1,
#: conventionally 12-14; converting to the spatial-frequency convention used
#: here, GMAX = 12-14 bohr^-1 is 3.6-4.2 1/Å, so 4 1/Å sits in the middle of
#: the range the reference calculation itself would have used.
#:
#: Two independent checks support it. (1) Magnitudes: it brings every panel of
#: PRB 94, 174414 Fig. 2 and Fig. 3 within the paper's own colorbar bounds,
#: with the near-nucleus features appearing as the small unsaturated dots seen
#: there. (2) Sign: the spin density at a heavy nucleus is *negative* from
#: core polarization (measured at the Pt site of L1_0 FePt:
#: -895 μB/Å³) even though the site's total moment is positive (+0.36 μB), so
#: the sign of the smoothed value flips depending on how hard it is smoothed --
#: the narrow negative core spike wins at a high cutoff, the broad positive
#: valence cloud wins at a low one. The paper shows positive spin density and
#: B_z at the Pt site; reproducing that sign requires a cutoff below ~5 1/Å
#: with the default taper, which 4 1/Å satisfies while 6 1/Å does not. The
#: sign is thus a sharper constraint on the cutoff than the magnitudes are.
#:
#: Note that the band limit is applied with a smooth (raised-cosine) roll-off
#: by default, not a brick-wall cutoff -- see the `taper` argument of
#: `band_limit_fourier`. A sharp cutoff fixes the magnitudes just as well but
#: leaves conspicuous concentric Gibbs fringes around each atom, since a
#: brick-wall filter is a ringing (sinc-like) kernel in real space. Because
#: the taper starts rolling off at `g_max / 2`, a given `g_max` removes
#: somewhat more than a sharp cutoff at the same value would.
#:
#: Pass `g_max=None` to disable band-limiting entirely and recover the raw
#: pointwise construction.
DEFAULT_MAGNETIZATION_G_MAX = 4.0


def calculate_constant_magnetic_field(magnetization: np.ndarray) -> np.ndarray:
    """
    Uniform part of the magnetic field, B_avg = mu_0 * mu_B * <m> [T].

    A periodic vector potential can only represent a magnetic field with
    zero cell average [PRB 94, 174414 (2016), Appendix], so the field of a
    magnetic solid decomposes as B = B_p + B_avg, where B_p (zero average)
    comes from the periodic vector potential and B_avg = mu_0 * M is the
    volume average of the field, with M the magnetization of the material.
    B_avg vanishes for antiferromagnets and other compensated textures.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.

    Returns
    -------
    average_field : np.ndarray
        The uniform magnetic field as a vector of shape (3,) [T].
    """
    magnetization = np.asarray(magnetization)
    average_magnetization = magnetization.reshape(3, -1).mean(axis=1)
    return vacuum_permeability * bohr_magneton * average_magnetization


def calculate_non_periodic_magnetic_vector_potential(
    average_field: np.ndarray,
    cell,
    gpts: tuple[int, int, int],
    origin: str | tuple[float, float, float] = "center",
) -> np.ndarray:
    """
    Non-periodic vector potential of a uniform magnetic field on a 3D grid.

    In the Coulomb gauge, A_np(r) = 1/2 * B_avg x (r - r0) represents the
    uniform field component B_avg = curl(A_np) that a periodic vector
    potential cannot carry [PRB 94, 174414 (2016), Appendix]. A_np grows
    linearly with distance from `origin` and is *not* periodic: simulations
    including it are only physical while the wave function has negligible
    amplitude at the supercell boundary, so use a laterally large supercell
    (in the reference calculations of PRB 94, 174414 this was 48x48 to
    60x60 unit cells).

    Two caveats that a larger supercell does NOT fix:

    1. Enlarging the supercell does not make the boundary error go away.
       |A_np| grows linearly with the cell size L at fixed B_avg, while the
       wave amplitude reaching the boundary falls, so their product -- which
       is what the boundary artifact scales with -- saturates. Measured for
       a 30 mrad l=+1 vortex on FePt (B_avg = 1.385 T, 20 u.c. thick),
       as the four-fold symmetry violation of the scanned intensity, which
       the crystal symmetry requires to vanish:

           supercell    boundary intensity fraction    C4 violation
           6x6                    1.26e-2                2.11e-5
           10x10                  5.90e-3                1.47e-5
           14x14                  3.96e-3                1.45e-5

       i.e. flat from 10x10 on, while the boundary amplitude keeps falling.
       With average_field=(0, 0, 0) the same quantity is ~3e-8, so this is
       entirely an A_np artifact. It is an orbital effect: for the same
       system it contributes a relative antisymmetric component of 1.2e-5
       to the total intensity, 7.3e-3 to the spin (I_up - I_down) signal,
       but 0.97 to the OAM (l=+1 minus mirrored l=-1) signal, which is a
       small difference of large numbers. If the simulated structure has a
       known point symmetry, projecting the result onto the symmetric
       subspace removes it; the symmetry-preserving component is
       supercell-converged (0.3% over 6x6 to 14x14 at 4 mrad).

    2. B_avg = mu_0 * M is the 3D-periodic result, which carries no
       demagnetizing field. A multislice sample is an infinite slab --
       periodic in x and y, finite in z -- so the shape-dependent
       demagnetizing field must be added to get the macroscopic B inside
       the sample. For M along the slab normal (the common case of a
       perpendicularly magnetized film viewed along the easy axis) the
       bound currents vanish identically, curl(M) = 0 and M x n = 0, giving
       H = -M and B = 0 inside: the correct uniform field is then zero, not
       mu_0 * M. For M in the film plane the demagnetizing factor is ~0 and
       B_avg = mu_0 * M stands. Choose `average_field` for the geometry
       being modelled rather than passing this function's output blindly.

       This choice is not cosmetic. For the FePt case above, dropping to
       B_avg = 0 removes the artifact at its source -- the C4 violation
       falls from 1.14 to 0.026 without any symmetrization -- but the OAM
       signal itself drops to 18-32% of its B_avg = mu_0 * M value, since
       the uniform field dominates the orbital coupling for a probe wider
       than the unit cell. The two conventions are physically different
       predictions, not numerically equivalent, so the geometry has to be
       decided deliberately.

    A general alternative that avoids the artifact without giving up the
    uniform field: for a *uniform* B the paraxial evolution is analytic, so
    it need not be gridded at all. A uniform axial B_z acts on a paraxial
    beam as a rigid Larmor rotation about the beam axis,
    theta_L = (e / 2 m v) * integral(B_z dz), plus a quadratic (weak-lens)
    phase and, for a vortex probe, an orbital Zeeman phase proportional to
    l * B_z. Applied as operators these are exact for a uniform field and
    symmetry-preserving by construction -- a rigid rotation about a
    symmetry axis cannot break that symmetry, whereas sampling A_np's
    linear ramp on a periodic grid does. For the FePt case theta_L is only
    1.65e-5 rad (0.008 wave pixels at the supercell edge), yet the OAM
    signal it carries is a factor ~5, which is precisely why the ramp's
    discretization error competes with the signal. This is not implemented
    here; it is the recommended direction if the A_np artifact becomes
    limiting.

    Parameters
    ----------
    average_field : np.ndarray
        The uniform magnetic field as a vector of shape (3,) [T], e.g. from
        `calculate_constant_magnetic_field`.
    cell : ase.cell.Cell or np.ndarray
        Cell defining the region of space the grid spans.
    gpts : tuple of three int
        Number of grid points along each cell vector.
    origin : "center" or tuple of three float
        Gauge origin r0 [Å]. Defaults to the cell center, which minimizes
        |A_np| over the cell.

    Returns
    -------
    vector_potential : np.ndarray
        The non-periodic vector potential of shape (3,) + gpts [ÅT].
    """
    average_field = np.asarray(average_field, dtype=float)
    cell = np.asarray(cell, dtype=float)

    fractional = np.stack(
        np.meshgrid(*(np.arange(n) / n for n in gpts), indexing="ij"), axis=-1
    )
    r = fractional @ cell

    if isinstance(origin, str) and origin == "center":
        r0 = cell.sum(axis=0) / 2
    else:
        r0 = np.asarray(origin, dtype=float)

    A = 0.5 * np.cross(average_field, r - r0)
    return np.moveaxis(A, -1, 0)


def _apply_rotation_matrix(
    vector_field: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    shape = vector_field.shape[1:]
    vector_field_reshaped = vector_field.reshape(3, -1)

    rotated_field_reshaped = rotation_matrix @ vector_field_reshaped

    return rotated_field_reshaped.reshape((3,) + shape)


def rotate_vector_field(
    vector_field: np.ndarray, euler_angles: tuple[float, float, float]
) -> np.ndarray:
    """
    Rotate a 3D vector field defined on a grid using Euler angles.

    Parameters
    ----------
    vector_field : numpy.ndarray
        3xNxMxK array representing the 3D vector field.
    euler_angles : tuple
        Euler angles (xyz) for the rotation.

    Returns
    -------
    rotated_field : numpy.ndarray
        Rotated 3D vector field.
    """
    rotation_matrix = R.from_euler("xyz", euler_angles).as_matrix()
    return _apply_rotation_matrix(vector_field, rotation_matrix)


def calculate_vector_potential_from_magnetization(
    magnetization: np.ndarray,
    cell,
    g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
    taper: str = "cosine",
) -> np.ndarray:
    """
    Periodic part of the magnetic vector potential from a magnetization
    density.

    The spin current density j = curl(mu_B * m) sources the periodic vector
    potential through the Poisson equation laplace(A_p) = -mu_0 * j, solved
    here in Fourier space (Coulomb gauge) [PRB 94, 174414 (2016), Sec. II D].

    A periodic vector potential can only represent a magnetic field with
    zero cell average, so for a ferromagnet the uniform remainder must be
    added separately; see `calculate_constant_magnetic_field` and
    `calculate_non_periodic_magnetic_vector_potential`.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.
    cell : ase.cell.Cell
        ASE `Cell` object defining the region of space where the
        magnetization is defined.
    g_max : float, optional
        Band limit [1/Å] applied to the magnetization before the vector
        potential is constructed, removing near-nucleus structure too sharp
        for the grid to represent. Defaults to
        `DEFAULT_MAGNETIZATION_G_MAX`; see that constant for why this
        matters and how the default was chosen. Pass `None` to disable.

    Returns
    -------
    vector_potential : np.ndarray
        The periodic vector potential of shape (3,) + (nx, ny, nz) [ÅT].
    """
    magnetization = band_limit_fourier(magnetization, cell, g_max, taper=taper)
    j = bohr_magneton * curl_fourier(magnetization, cell)
    A = -vacuum_permeability * integrate_gradient_fourier(j, cell)
    return A


def calculate_magnetic_vector_potential(
    spin_density, cell, g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX
):
    """
    Periodic vector potential [ÅT] of a collinear spin density (m along z).

    See `calculate_vector_potential_from_magnetization` for the general
    (non-collinear) form and for the meaning of `g_max`.
    """
    m = np.stack(
        [np.zeros_like(spin_density), np.zeros_like(spin_density), spin_density]
    )
    return calculate_vector_potential_from_magnetization(m, cell, g_max=g_max)


def get_magnetization_from_gpaw(calc, gridrefinement: int = 2):
    """
    Extract the all-electron magnetization density from a converged GPAW
    calculation.

    Supports collinear spin-polarized calculations (two spin channels; the
    magnetization is placed along z by convention, since collinear spin has
    no real-space direction) and non-collinear calculations (four density
    components n, mx, my, mz; requires GPAW's new-style calculator with
    plane-wave mode, ``symmetry='off'`` and ``magmoms`` given as an (N, 3)
    array).

    Parameters
    ----------
    calc : gpaw.new.ase_interface.ASECalculator or gpaw.calculator.GPAW
        Converged GPAW calculator. Non-collinear extraction requires the
        new-style calculator (GPAW >= 25).
    gridrefinement : int
        Grid refinement factor for the all-electron density.

    Returns
    -------
    magnetization : np.ndarray
        Magnetization density of shape (3,) + (nx, ny, nz) in units of
        Bohr magnetons per cubic Ångström.
    collinear : bool
        True when extracted from a collinear calculation, where the spin
        axis is arbitrary (represented along z); False for genuinely
        directional non-collinear magnetization.
    """
    dft = getattr(calc, "dft", None)

    if dft is not None:
        with _GPAW_LOCK:
            n = dft.densities().all_electron_densities(
                grid_refinement=gridrefinement
            )
            density = n.gather(broadcast=True).data

        if density.shape[0] == 1:
            raise ValueError(
                "The GPAW calculation is spin-paired and has no magnetization; "
                "run a spin-polarized (or non-collinear) calculation."
            )
        elif density.shape[0] == 2:
            rho = density[0] - density[1]
            zeros = np.zeros_like(rho)
            return np.stack([zeros, zeros, rho]), True
        else:
            assert density.shape[0] == 4  # (n, mx, my, mz)
            return np.ascontiguousarray(density[1:]), False

    # Legacy calculator API: only collinear calculations are supported.
    # One call for both spin channels: `spin=True` indexes GPAW's (spin, x, y, z)
    # density with a boolean, which adds a leading axis of length one.
    with _GPAW_LOCK:
        n = calc.get_all_electron_density(spin=True, gridrefinement=gridrefinement)
    rho = n[0][0] - n[0][1]
    zeros = np.zeros_like(rho)
    return np.stack([zeros, zeros, rho]), True


def get_vector_potential_from_gpaw(
    calc, gridrefinement=2, g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX
):
    """Periodic vector potential [ÅT] from a converged GPAW calculation.

    See `DEFAULT_MAGNETIZATION_G_MAX` for the meaning of `g_max`.
    """
    magnetization, _ = get_magnetization_from_gpaw(
        calc, gridrefinement=gridrefinement
    )
    return calculate_vector_potential_from_magnetization(
        magnetization, calc.atoms.cell, g_max=g_max
    )


def get_magnetic_field_from_gpaw(
    calc, gridrefinement=2, g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX
):
    """Periodic magnetic field [T] from a converged GPAW calculation.

    See `DEFAULT_MAGNETIZATION_G_MAX` for the meaning of `g_max`.
    """
    A = get_vector_potential_from_gpaw(
        calc, gridrefinement=gridrefinement, g_max=g_max
    )
    B = curl_fourier(A, calc.atoms.cell)
    return B


#: Sentinel default for `rotate_field`: automatically rotate the
#: largest-magnitude in-plane component into z (see
#: `_auto_rotation_matrix_for_vector_field`). Pass an explicit Euler-angle
#: tuple to pick a specific orientation, or `None` to disable rotation and
#: see the raw (Az == 0) output.
_AUTO_ROTATE_FIELD = "auto"

#: Swaps x into z: (Ax, Ay, 0) -> (0, Ay, -Ax). Euler angles (0, pi/2, 0).
_ROTATION_X_INTO_Z = R.from_euler("xyz", (0.0, np.pi / 2, 0.0)).as_matrix()

#: Swaps y into z: (Ax, Ay, 0) -> (Ax, 0, Ay). Euler angles (pi/2, 0, 0).
_ROTATION_Y_INTO_Z = R.from_euler("xyz", (np.pi / 2, 0.0, 0.0)).as_matrix()


def _auto_rotation_matrix_for_vector_field(vector_field: np.ndarray) -> np.ndarray:
    """
    Rotation matrix that swaps whichever of the in-plane (x, y) components
    of `vector_field` has the larger aggregate magnitude into z.

    `calculate_magnetic_vector_potential` always builds the magnetization as
    m = (0, 0, rho): collinear spin has no real-space direction, so GPAW's
    internal spin axis is arbitrary. Because curl(m) and the subsequent
    Poisson solve are applied component-wise, this makes the z-component of
    `vector_field` (and hence the only component `adjust_coulomb_potential`
    uses) identically zero for every collinear calculation, not just some.

    Only the two 90-degree swaps (x into z, or y into z) are considered,
    not an arbitrary in-plane rotation angle: x, y and z are the only
    directions with a physical meaning here (the orthogonal axes of the
    simulation cell), so a continuous "optimal" blend of Ax and Ay has no
    real-space interpretation as a magnetization direction -- it would just
    fit whatever numerical asymmetry happens to be in the grid.
    """
    Ax = vector_field[0].astype(np.float64, copy=False)
    Ay = vector_field[1].astype(np.float64, copy=False)

    Sxx = float(np.sum(Ax * Ax))
    Syy = float(np.sum(Ay * Ay))

    return _ROTATION_X_INTO_Z if Sxx >= Syy else _ROTATION_Y_INTO_Z


def _check_unsupported_ensemble_params(frozen_phonons, repetitions):
    if frozen_phonons is not None:
        raise NotImplementedError(
            "frozen_phonons is not supported for magnetic fields/vector "
            "potentials; build the field from a single calculator and combine "
            "it with an electrostatic FrozenPhonons ensemble instead."
        )
    if tuple(repetitions) != (1, 1, 1):
        raise NotImplementedError(
            "repetitions is not supported for magnetic fields/vector "
            "potentials; build the field for a single unit cell and call "
            ".tile() on the resulting array instead."
        )


def _real_space_slicing(
    slice_thickness: float | tuple[float, ...], num_planes: int, depth: float
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """
    The slice thicknesses of a real-space projection and the number of z plane
    spacings of the density in each slice, which add up to `num_planes`.

    A single thickness is divided into slices of whole planes as evenly as the
    planes allow. A sequence of thicknesses must put every slice boundary on a
    plane, as the thicknesses returned here do, so that a builder made with the
    `slice_thickness` of another slices the same way.
    """
    if is_number(slice_thickness):
        thicknesses, planes = equal_slice_thicknesses(
            num_planes, float(slice_thickness), depth=depth
        )
        return tuple(thicknesses), tuple(int(n) for n in planes)

    dz = depth / num_planes
    thicknesses = np.array([float(t) for t in slice_thickness])
    planes = np.rint(thicknesses / dz).astype(int)
    if (
        planes.min() < 1
        or planes.sum() != num_planes
        or not np.allclose(planes * dz, thicknesses, rtol=1e-6, atol=0.0)
    ):
        raise NotImplementedError(
            "The slice thicknesses of the real-space projection must be whole "
            f"numbers of the z grid spacing, {dz:.6g} Å, that add up to the depth, "
            f"{depth:.6g} Å; the nearest are {tuple(float(n * dz) for n in planes)}."
        )
    return tuple(float(n * dz) for n in planes), tuple(int(n) for n in planes)


def _fourier_slice_integrals(
    array: np.ndarray, slice_limits: list[tuple[float, float]], depth: float
) -> np.ndarray:
    """
    The integrals of `array` along its last axis between each pair of
    `slice_limits` [Å], stacked along a new first axis.

    The last axis holds evenly spaced samples of a field that is periodic over
    `depth`, the first at height 0. The field between the samples is their
    band-limited (trigonometric) interpolant, whose integral over a slice is exact
    in Fourier space for any slice limits. The Nyquist term of an even number of
    samples is taken as a cosine, as for a real field.
    """
    n = array.shape[-1]
    q = (2 * np.pi * np.arange(n // 2 + 1) / depth)[:, None]
    a, b = (np.array(limits, dtype=float) for limits in zip(*slice_limits))

    # The field is real, so the half spectrum of `rfft` holds it: the term of each
    # positive frequency stands for itself and its conjugate at the negative one,
    # twice its real part. The zero frequency and the Nyquist term of an even
    # number of samples stand for themselves.
    weights = np.empty((len(q), len(a)), dtype=complex)
    weights[0] = b - a
    weights[1:] = (np.exp(1j * q[1:] * b) - np.exp(1j * q[1:] * a)) / (1j * q[1:])
    weights[1:] *= 2
    if n % 2 == 0:
        q_nyquist = q[-1]
        weights[-1] = (np.sin(q_nyquist * b) - np.sin(q_nyquist * a)) / q_nyquist
    weights /= n

    # One component at a time, so that only one of their spectra is held at once.
    integrals = np.empty(array.shape[:-1] + (len(a),))
    for component in np.ndindex(array.shape[:-3]):
        integrals[component] = (np.fft.rfft(array[component], axis=-1) @ weights).real
    return np.moveaxis(integrals, -1, 0)


def _is_skewed(cell) -> bool:
    """Whether `cell` is not orthogonal as `orthogonalize_cell` sees it, which
    zeroes components below 1e-6 Å (`_cell_in_plane_frame`). `_supercell_of_box`
    maps the field from that same frame, so the two agree on every cell."""
    cell = _cell_in_plane_frame(cell, "xy")
    return bool(np.any(cell[~np.eye(3, dtype=bool)] != 0.0))


def _supercell_of_box(cell, box) -> tuple[np.ndarray, np.ndarray]:
    """
    The lattice vectors of the supercell of `cell` that fills `box` (an integer
    matrix, rows along x, y and z, in units of the lattice vectors of `cell`), and
    the rotation that `orthogonalize_cell` gives the vectors when it fits that
    supercell into the box (acting on row vectors).

    `orthogonalize_cell` keeps the fractional coordinates of the supercell: a point
    at r goes to r @ A with A the map from the supercell onto the box, which is a
    rotation times a strain. A vector field placed the same way turns its vectors
    by the rotation; a strain has no single rule for the vectors and leaves them.
    """
    cell = _cell_in_plane_frame(cell, "xy")
    vectors = _box_repetitions(cell, box)
    transform = np.linalg.solve(vectors @ cell, np.diag(box))
    u, _, vt = np.linalg.svd(transform)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise NotImplementedError(
            "The cell of the calculator is left-handed, so its box mirrors the "
            "field, which a magnetic field does not follow as a vector field does."
        )
    return vectors.astype(int), rotation


def _box_grid(vectors: np.ndarray, shape: tuple[int, int, int]):
    """
    The grid of the box filled by the supercell with lattice vectors `vectors` that
    holds every Fourier component of a field sampled on a grid of `shape` over the
    cell, and where each component goes on it.

    The component of the cell's frequency m (in units of its reciprocal lattice
    vectors) is the component of frequency `vectors @ m` of the box. A box axis
    made of one lattice vector that no other box axis uses takes the frequencies
    of that axis alone, so it keeps their number of samples per period. The other
    axes mix frequencies of several axes of the cell; the Nyquist component of an
    even number of samples along these is split into halves at +N/2 and -N/2,
    which is the same real field, and the box axis holds every frequency the mix
    gives, so that no two components share a frequency of the box.

    A smaller grid can keep the components apart too (for a bcc or fcc primitive
    cell in its cubic box, one more sample per axis than the cell has), but only
    by folding frequencies past its Nyquist frequency: its samples are those of the
    field, yet the band-limited interpolant between them, which the slice
    integrals of the `fft` projection and the interpolation to `gpts` both use, is
    not. Holding every frequency of the mix keeps that interpolant the field.

    Returns the shape of the box grid and, for each axis of the cell, the indices
    of the FFT output taken, their frequencies and their weights.
    """
    vectors = np.asarray(vectors, dtype=int)

    alone = [False] * 3
    box_shape = [0] * 3
    for k in range(3):
        (used,) = np.nonzero(vectors[k])
        if len(used) == 1 and np.count_nonzero(vectors[:, used[0]]) == 1:
            alone[used[0]] = True
            box_shape[k] = abs(int(vectors[k, used[0]])) * shape[used[0]]

    indices, frequencies, weights = [], [], []
    for j, n in enumerate(shape):
        index = np.arange(n)
        frequency = np.fft.fftfreq(n, d=1.0 / n).astype(int)
        weight = np.ones(n)
        if not alone[j] and n % 2 == 0:
            index = np.append(index, n // 2)
            frequency = np.append(frequency, n // 2)
            weight = np.append(weight, 0.5)
            weight[n // 2] = 0.5
        indices.append(index)
        frequencies.append(frequency)
        weights.append(weight)

    for k in range(3):
        if box_shape[k] == 0:
            box_shape[k] = 1 + sum(
                abs(int(vectors[k, j])) * int(np.ptp(frequencies[j])) for j in range(3)
            )

    return tuple(box_shape), indices, frequencies, weights


def _map_to_box_grid(array: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    The field `array` (components first, then the three axes of a grid over the
    cell along its lattice vectors) on the grid of the box filled by the supercell
    with lattice vectors `vectors`, as `_box_grid` lays it out.

    The field is placed in the fractional coordinates of the supercell, as
    `orthogonalize_cell` places atoms. Each Fourier component moves to its
    frequency on the box without interpolation, so the band-limited field is the
    same at every point.
    """
    grid_shape = array.shape[-3:]
    box_shape, indices, frequencies, weights = _box_grid(vectors, grid_shape)

    # The spectrum of the cell's grid is as large as the field; the box's, built
    # below, is the largest array here.
    spectrum = np.fft.fftn(array, axes=(-3, -2, -1))[(...,) + np.ix_(*indices)]
    spectrum *= (
        weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None]
    )

    target = tuple(
        (
            vectors[k, 0] * frequencies[0][:, None, None]
            + vectors[k, 1] * frequencies[1][None, :, None]
            + vectors[k, 2] * frequencies[2][None, None]
        )
        % box_shape[k]
        for k in range(3)
    )

    # One component at a time, so that only one complex spectrum of the box, the
    # largest array here, is held at once.
    scale = np.prod(box_shape) / np.prod(grid_shape)
    mapped = np.empty(array.shape[:-3] + box_shape)
    for component in np.ndindex(array.shape[:-3]):
        box_spectrum = np.zeros(box_shape, dtype=complex)
        box_spectrum[target] = spectrum[component]
        box_spectrum *= scale
        mapped[component] = np.fft.ifftn(box_spectrum).real
    return mapped


@runtime_checkable
class GPAW(Protocol):
    @property
    def atoms(self) -> Atoms: ...

    def get_number_of_grid_points(self) -> np.ndarray: ...


class _MagnetizationMagnetics(_FieldBuilder):
    """
    Base builder for sliced magnetic fields/vector potentials computed from
    a magnetization density. Subclasses provide `_get_magnetization`,
    returning the (3,) + (nx, ny, nz) magnetization [μB/Å³] and a flag
    marking it as collinear (arbitrary spin axis, represented along z).
    """

    _supports_box_and_origin = False
    # The slices computed from the magnetization on first use, kept for the later
    # calls of `generate_slices`, and the rotation resolved while computing them;
    # incidental state, never identity.
    _eq_exclude = ("_slices", "_resolved_rotation_matrix")

    def __init__(
        self,
        array_object,
        cell,
        grid_shape: tuple[int, int, int],
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        self._cell3d = cell

        self.g_max = g_max
        self.taper = taper

        self._rotate_field = rotate_field
        #: The rotation applied to the field's vector components by
        #: `rotate_field` when the slices were last computed (None until then,
        #: or when no rotation was applied). Consumers needing consistency with
        #: a separately computed constant field (see `gpaw_magnetic_fields`)
        #: use `_orient_constant_vector`, which applies it.
        self._resolved_rotation_matrix: Optional[np.ndarray] = None

        if projection not in ("fft", "real_space"):
            raise ValueError(
                f"projection must be 'fft' or 'real_space', got {projection!r}"
            )

        grid_shape = tuple(int(n) for n in grid_shape)

        # The grid of the magnetization runs along the lattice vectors. For a
        # skewed cell `generate_slices` moves the field onto a grid of the default
        # box, whose lattice vectors in units of the cell's are `_box_vectors`.
        self._box_vectors: Optional[np.ndarray] = None
        self._box_rotation: Optional[np.ndarray] = None
        if _is_skewed(cell):
            # Raises the RuntimeError of a cell that cannot be rotated to `plane`.
            default_box = _default_box(cell, plane)
            if plane != "xy":
                raise NotImplementedError(
                    f"plane={plane!r} is not supported for the magnetic field of a "
                    "non-orthogonal cell, whose vectors would have to be rotated "
                    "with it; use plane='xy', or a calculation of an orthogonal cell."
                )
            self._box_vectors, self._box_rotation = _supercell_of_box(cell, default_box)
            grid_shape = _box_grid(self._box_vectors, grid_shape)[0]
            depth = float(default_box[2])
        else:
            depth = float(np.diag(cell)[plane_to_axes(plane)[2]])

        # The number of z plane spacings of the density in each real-space slice.
        self._planes_per_slice: Optional[tuple[int, ...]] = None

        if projection == "real_space":
            # The slices are stacked along the third axis of `plane`, which is the
            # last axis of the field once `generate_slices` has moved its axes.
            slice_thickness, self._planes_per_slice = _real_space_slicing(
                slice_thickness,
                num_planes=grid_shape[plane_to_axes(plane)[2]],
                depth=depth,
            )

        self._projection = projection

        # The slices of the whole field, (slice, component, x, y) on the host in
        # the configured precision, computed from the magnetization on the first
        # call of `generate_slices`.
        self._slices: Optional[np.ndarray] = None

        self._quantity = quantity

        super().__init__(
            array_object=array_object,
            gpts=gpts,
            cell=cell,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            origin=origin,
            box=box,
            periodic=periodic,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        raise NotImplementedError

    @property
    def num_configurations(self):
        return 1

    @property
    def base_axes_metadata(self):
        pass

    @property
    def plane(self):
        assert isinstance(self._plane, str)
        return self._plane

    def _field(self) -> np.ndarray:
        """The field on the host, its components and axes in the frame of `plane`
        and on a grid over the box, the slicing axis last."""
        magnetization, collinear = self._get_magnetization()

        vector_potential = calculate_vector_potential_from_magnetization(
            magnetization, self._cell3d, g_max=self.g_max, taper=self.taper
        )

        if self._quantity == "vector_potential":
            array = vector_potential
        elif self._quantity == "magnetic_field":
            array = curl_fourier(vector_potential, self._cell3d)
        else:
            raise ValueError(f"Unknown quantity: {self._quantity}")

        if self._box_rotation is not None and not np.allclose(
            self._box_rotation, np.eye(3), rtol=0.0, atol=1e-12
        ):
            # `_apply_rotation_matrix` acts on column vectors.
            array = _apply_rotation_matrix(array, self._box_rotation.T)
            vector_potential = _apply_rotation_matrix(
                vector_potential, self._box_rotation.T
            )

        if self.plane != "xy":
            # The fields are rotated with the atoms: their grids are moved and
            # their components rotated by the same proper rotation, which reverses
            # one axis and its component for the planes that need it.
            array = _vector_field_to_plane(array, self.plane)
            vector_potential = _vector_field_to_plane(vector_potential, self.plane)

        rotate_field = self._rotate_field
        self._resolved_rotation_matrix = None
        if isinstance(rotate_field, str) and rotate_field == _AUTO_ROTATE_FIELD:
            # The auto swap exists because collinear spin has no real-space
            # direction; a non-collinear magnetization is genuinely
            # directional, so it is never auto-rotated.
            if collinear:
                rotation_matrix = _auto_rotation_matrix_for_vector_field(
                    vector_potential
                )
                array = _apply_rotation_matrix(array, rotation_matrix)
                self._resolved_rotation_matrix = rotation_matrix
        elif rotate_field:
            array = rotate_vector_field(array, rotate_field)
            self._resolved_rotation_matrix = R.from_euler(
                "xyz", rotate_field
            ).as_matrix()

        if self._box_vectors is not None:
            array = _map_to_box_grid(array, self._box_vectors)

        return array

    def _orient_constant_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        A constant vector, such as the uniform field B_avg, with its components
        turned as `_field` turns those of the field: by the rotation of a skewed
        cell into its box, to `plane`, and by `rotate_field` as resolved when the
        slices were computed (so the slices must have been computed first).
        """
        vector = np.asarray(vector, dtype=float).reshape((3, 1, 1, 1))
        if self._box_rotation is not None:
            vector = _apply_rotation_matrix(vector, self._box_rotation.T)
        if self.plane != "xy":
            vector = _vector_field_to_plane(vector, self.plane)
        if self._resolved_rotation_matrix is not None:
            vector = _apply_rotation_matrix(vector, self._resolved_rotation_matrix)
        return vector[:, 0, 0, 0]

    def _project(self) -> np.ndarray:
        """
        The field integrated through each slice, (slice, component, x, y) on the
        host, in field units times Å.

        The real-space projection takes plane k, the sample at z = k dz, as the
        field from k dz - dz / 2 to k dz + dz / 2, and integrates each slice over
        the planes it covers. A slice boundary lies on a plane, which is split
        between the slices on either side (the trapezoidal rule), so the integral
        of a slice is centred on the slice; the last slice takes half of plane 0,
        as the field is periodic along z. The fft projection integrates the
        band-limited field between the limits of each slice in Fourier space, for
        any slice thicknesses.
        """
        array = self._field()

        if self._projection == "real_space":
            planes_per_slice = self._planes_per_slice
            assert planes_per_slice is not None
            if sum(planes_per_slice) != array.shape[-1]:
                raise RuntimeError(
                    f"The slices span {sum(planes_per_slice)} z planes, but the "
                    f"magnetization density has {array.shape[-1]}."
                )
            bounds = np.cumsum((0,) + planes_per_slice)
            dz = sum(self.slice_thickness) / array.shape[-1]
            return np.stack(
                [
                    _integrate_planes(array, start, stop) * dz
                    for start, stop in zip(bounds[:-1], bounds[1:])
                ]
            )

        return _fourier_slice_integrals(array, self.slice_limits, self.box[2])

    def generate_slices(self, first_slice: int = 0, last_slice: Optional[int] = None):
        if last_slice is None:
            last_slice = self.num_slices

        # The field is computed from the magnetization once and its slices kept,
        # so that building or chunking the slices range by range does not repeat
        # it. Kept in the configured precision, the precision of the slices handed
        # out, rather than the float64 of the density's grid; computed again if
        # the configured precision has changed since.
        if self._slices is None or self._slices.dtype != get_dtype():
            self._slices = self._project().astype(get_dtype(), copy=False)

        slice_thicknesses = np.array(self.slice_thickness)
        slice_shape = (3,) + self._valid_gpts
        # The slices are computed on the host; each is moved to `device`.
        xp = get_array_module(self.device)

        for slice_idx in range(first_slice, last_slice):
            slice_array = self._slices[slice_idx]

            if self._valid_gpts != slice_array.shape[1:]:
                slice_array = fft_interpolate(slice_array, slice_shape)
            else:
                # Not a view of the kept slices, which a caller could write into.
                slice_array = slice_array.copy()

            yield self._array_object(
                xp.asarray(slice_array[None]),
                extent=self.extent,
                slice_thickness=slice_thicknesses[slice_idx],
            )

    def build(
        self,
        first_slice: int = 0,
        last_slice: Optional[int] = None,
        max_batch: int | str = 1,
        lazy: Optional[bool] = None,
    ):
        if lazy:
            raise ValueError("Lazy not supported for magnetics.")
        return super().build(
            first_slice=first_slice,
            last_slice=last_slice,
            max_batch=max_batch,
            lazy=False,
        )


class _GPAWMagnetics(_MagnetizationMagnetics):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        array_object,
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 4,
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        self.gridrefinement = gridrefinement

        assert isinstance(calculators, GPAW)
        self._calculators = calculators

        super().__init__(
            array_object=array_object,
            cell=calculators.atoms.cell,
            grid_shape=tuple(
                int(n) * gridrefinement
                for n in calculators.get_number_of_grid_points()
            ),
            quantity=quantity,
            projection=projection,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            g_max=g_max,
            taper=taper,
            device=device,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        return get_magnetization_from_gpaw(
            self._calculators, gridrefinement=self.gridrefinement
        )


class _ArrayMagnetics(_MagnetizationMagnetics):
    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        array_object,
        quantity: str = "magnetic_field",
        projection: str = "fft",
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        from ase.cell import Cell

        magnetization = np.asarray(magnetization)

        if magnetization.ndim == 3:
            # A scalar spin density: collinear, along z by convention.
            zeros = np.zeros_like(magnetization)
            magnetization = np.stack([zeros, zeros, magnetization])
            self._collinear = True
        elif magnetization.ndim == 4 and magnetization.shape[0] == 3:
            self._collinear = False
        else:
            raise ValueError(
                "magnetization must have shape (nx, ny, nz) (a collinear "
                "spin density) or (3, nx, ny, nz) (a vector magnetization "
                f"density), got {magnetization.shape}"
            )

        self._magnetization = magnetization

        cell = Cell(np.diag(cell) if np.ndim(cell) == 1 else cell)

        super().__init__(
            array_object=array_object,
            cell=cell,
            grid_shape=magnetization.shape[-3:],
            quantity=quantity,
            projection=projection,
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            g_max=g_max,
            taper=taper,
            device=device,
        )

    def _get_magnetization(self) -> tuple[np.ndarray, bool]:
        return self._magnetization, self._collinear


class MagnetizationMagneticField(_ArrayMagnetics, BaseMagneticField):
    """
    Sliced magnetic field built from a magnetization density on a 3D grid.

    The magnetization may come from any source (e.g. a micromagnetic model
    or a DFT code other than GPAW), making this the DFT-agnostic
    counterpart of `GPAWMagneticField`. Only the periodic part of the
    field is built; see `calculate_constant_magnetic_field` for the
    uniform remainder of a ferromagnet.

    Parameters
    ----------
    magnetization : np.ndarray
        Magnetization density in units of Bohr magnetons per cubic
        Ångström. Either shape (3,) + (nx, ny, nz) for a vector
        (non-collinear) magnetization, or (nx, ny, nz) for a collinear
        spin density (placed along z by convention).
    cell : ase.cell.Cell or np.ndarray
        Cell defining the region of space where the magnetization is
        defined.
    rotate_field : tuple of three float, "auto", or None
        Euler-angle rotation applied to the built field. Defaults to None:
        an explicitly given magnetization is genuinely directional. "auto"
        only applies to collinear (scalar) input.

    Other parameters follow `GPAWMagneticField`.
    """

    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        projection: str = "fft",
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        super().__init__(
            magnetization=magnetization,
            cell=cell,
            array_object=MagneticFieldArray,
            quantity="magnetic_field",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            projection=projection,
            g_max=g_max,
            taper=taper,
            device=device,
        )


class MagnetizationVectorPotential(_ArrayMagnetics, BaseVectorPotential):
    """
    Sliced magnetic vector potential built from a magnetization density on
    a 3D grid; the DFT-agnostic counterpart of `GPAWVectorPotential`.

    See `MagnetizationMagneticField` for the parameters.
    """

    def __init__(
        self,
        magnetization: np.ndarray,
        cell,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        projection: str = "fft",
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        super().__init__(
            magnetization=magnetization,
            cell=cell,
            array_object=VectorPotentialArray,
            quantity="vector_potential",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            periodic=periodic,
            projection=projection,
            g_max=g_max,
            taper=taper,
            device=device,
        )


class SpinDensityMagneticField(MagnetizationMagneticField):
    """
    Sliced magnetic field built from a collinear spin density on a 3D grid.

    Thin convenience wrapper around `MagnetizationMagneticField` for the
    collinear case: the magnetization is `spin_density` along z. Like the
    GPAW builders, `rotate_field` defaults to "auto" here, since a
    collinear spin axis has no real-space meaning.
    """

    def __init__(
        self,
        spin_density: np.ndarray,
        cell,
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        **kwargs,
    ):
        spin_density = np.asarray(spin_density)
        if spin_density.ndim != 3:
            raise ValueError(
                f"spin_density must have shape (nx, ny, nz), got "
                f"{spin_density.shape}"
            )
        super().__init__(
            magnetization=spin_density,
            cell=cell,
            rotate_field=rotate_field,
            **kwargs,
        )


class GPAWMagneticField(_GPAWMagnetics, BaseMagneticField):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        super().__init__(
            calculators=calculators,
            array_object=MagneticFieldArray,
            quantity="magnetic_field",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            gridrefinement=gridrefinement,
            projection=projection,
            periodic=periodic,
            g_max=g_max,
            taper=taper,
        )


class GPAWVectorPotential(_GPAWMagnetics, BaseVectorPotential):
    def __init__(
        self,
        calculators: GPAW | list[GPAW] | list[str] | str,
        gpts: Optional[int | tuple[int, int]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        slice_thickness: float | tuple[float, ...] = 1.0,
        exit_planes: Optional[int | tuple[int, ...]] = None,
        plane: str = "xy",
        rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box: Optional[tuple[float, float, float]] = None,
        periodic: bool = True,
        frozen_phonons: Optional[BaseFrozenPhonons] = None,
        repetitions: tuple[int, int, int] = (1, 1, 1),
        gridrefinement: int = 1,
        projection: str = "fft",
        g_max: Optional[float] = DEFAULT_MAGNETIZATION_G_MAX,
        taper: str = "cosine",
        device: Optional[str] = None,
    ):
        _check_unsupported_ensemble_params(frozen_phonons, repetitions)

        super().__init__(
            calculators=calculators,
            array_object=VectorPotentialArray,
            quantity="vector_potential",
            gpts=gpts,
            sampling=sampling,
            slice_thickness=slice_thickness,
            exit_planes=exit_planes,
            device=device,
            plane=plane,
            rotate_field=rotate_field,
            origin=origin,
            box=box,
            gridrefinement=gridrefinement,
            projection=projection,
            periodic=periodic,
            g_max=g_max,
            taper=taper,
        )


@dataclass
class GPAWMagneticFields:
    """
    Bundles the electrostatic potential, magnetic vector potential and
    (optionally) magnetic field built from the same GPAW calculator(s) by
    `gpaw_magnetic_fields`.

    `potential` may carry a frozen-phonon ensemble axis (or, after tiling,
    come from a `CrystalPotential` build); `vector_potential` and
    `magnetic_field` are always for a single, rigid configuration -- see
    `_check_unsupported_ensemble_params`. Use `.tile()` to bring the
    magnetic components up to a repeated crystal's size, and
    `.combined_potential()` to fold the vector potential into an
    electrostatic potential via `adjust_coulomb_potential`.

    `average_field` (B_avg, the uniform part of the field that a periodic
    vector potential cannot represent -- see `calculate_constant_magnetic_field`)
    is *not* recoverable from `magnetic_field`, since the latter is built
    from the periodic vector potential and integrates to zero average by
    construction. It is computed once from the calculator's magnetization
    while `gpaw_magnetic_fields` still holds a reference to it, permuted by
    the same `plane`-dependent axis order as the field arrays. Pass it as
    `pauli_multislice`'s `average_field` argument; it is left untouched by
    `.tile()` since it is a single global constant, not a per-slice field.
    """

    potential: PotentialArray
    vector_potential: VectorPotentialArray
    magnetic_field: Optional[MagneticFieldArray] = None
    average_field: Optional[np.ndarray] = None

    def tile(
        self, repetitions: tuple[int, int] | tuple[int, int, int]
    ) -> "GPAWMagneticFields":
        """
        Tile `vector_potential` (and `magnetic_field`, if present) to match
        a separately tiled/repeated electrostatic potential, e.g. built via
        `abtem.CrystalPotential`.

        `potential` is left untouched here -- tile or rebuild it separately
        (e.g. `CrystalPotential(electrostatic_ensemble, repetitions=...)`)
        before calling `combined_potential`.
        """
        return replace(
            self,
            vector_potential=self.vector_potential.tile(repetitions),
            magnetic_field=(
                self.magnetic_field.tile(repetitions)
                if self.magnetic_field is not None
                else None
            ),
        )

    def combined_potential(
        self, energy: float, potential: Optional[PotentialArray] = None
    ) -> PotentialArray:
        """
        Combine an electrostatic potential with `vector_potential` via
        `VectorPotentialArray.adjust_coulomb_potential`.

        Parameters
        ----------
        energy : float
            Electron energy [eV].
        potential : PotentialArray, optional
            The electrostatic potential to combine with `vector_potential`.
            Defaults to `self.potential`; pass a separately
            tiled/ensembled potential (e.g. a `CrystalPotential` build)
            after calling `.tile()` for the frozen-phonon workflow.

        Returns
        -------
        PotentialArray
        """
        if potential is None:
            potential = self.potential
        return self.vector_potential.adjust_coulomb_potential(potential, energy=energy)

    def show(
        self,
        tile: tuple[int, int] = (1, 1),
        figsize: tuple[int, int] = (8, 8),
    ):
        """
        Show side-by-side projections of the electrostatic potential and
        the x and z components of the vector potential -- and, if built,
        the magnetic field. `Az`/`Bz` are the components that matter for
        `combined_potential`; `Ax`/`Bx` are shown alongside for a sanity
        check that the in-plane and z-swapped components look sensible
        relative to each other.

        Parameters
        ----------
        tile : two int, optional
            Tile the projected images before plotting, e.g. to preview the
            periodicity of a repeated unit cell.
        figsize : two int, optional
            Figure size passed to `matplotlib.pyplot.figure`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        panels = [
            (self.potential.project(), "$V$", "$Å^{-3}$"),
            (self.vector_potential.project()[0], "$A_x$", "Å²T"),
            (self.vector_potential.project()[2], "$A_z$", "Å²T"),
        ]
        if self.magnetic_field is not None:
            panels += [
                (self.magnetic_field.project()[0], "$B_x$", "ÅT"),
                (self.magnetic_field.project()[2], "$B_z$", "ÅT"),
            ]

        # Create the figure with pyplot's interactive auto-display off, then
        # return it: otherwise Jupyter's inline backend renders it once from
        # the auto-display hook and a second time from the returned value,
        # showing the same plot twice.
        with plt.ioff():
            fig = plt.figure(figsize=figsize)
            grid = ImageGrid(
                fig,
                111,
                nrows_ncols=(1, len(panels)),
                cbar_mode="edge",
                cbar_location="bottom",
                cbar_pad=0.1,
                cbar_size=0.1,
                axes_pad=0.1,
            )

            for ax, (image, name, unit) in zip(grid, panels):
                array = asnumpy(image.tile(tile).compute().array)
                if np.abs(array).max() < 1e-5:
                    vmin, vmax = -1e-5, 1e-5
                else:
                    vmin, vmax = None, None
                im = ax.imshow(array.T, vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(name)
                ax.cax.colorbar(im, label=unit)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        return fig


def gpaw_magnetic_fields(
    calculators: GPAW | list[GPAW] | list[str] | str,
    gpts: Optional[int | tuple[int, int]] = None,
    sampling: Optional[float | tuple[float, float]] = None,
    slice_thickness: float | tuple[float, ...] = 1.0,
    exit_planes: Optional[int | tuple[int, ...]] = None,
    plane: str = "xy",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    box: Optional[tuple[float, float, float]] = None,
    periodic: bool = True,
    frozen_phonons: Optional[BaseFrozenPhonons] = None,
    rotate_field: Optional[tuple[float, float, float]] | str = _AUTO_ROTATE_FIELD,
    include_magnetic_field: bool = False,
    magnetic_calculator: Optional[GPAW] = None,
    device: Optional[str] = None,
    lazy: Optional[bool] = None,
    potential_kwargs: Optional[dict] = None,
    field_kwargs: Optional[dict] = None,
) -> GPAWMagneticFields:
    """
    Build the electrostatic potential, magnetic vector potential and
    (optionally) magnetic field from the same GPAW calculator(s) in one
    call.

    `frozen_phonons` (an ensemble of atomic-displacement configurations) is
    only supported for the electrostatic `potential` -- the magnetic
    components come from a single, rigid ab initio calculation and cannot
    vary per configuration. Tile the returned object with `.tile()` to
    match a separately built, possibly-ensembled electrostatic potential
    (e.g. from `abtem.CrystalPotential`), then combine with
    `.combined_potential()`.

    Parameters
    ----------
    calculators : (list of) gpaw.calculator.GPAW or (list of) str
        One or more converged GPAW calculators (or paths to `.gpw` files).
        Forwarded to `GPAWPotential`. If a list (a frozen-phonon ensemble),
        `magnetic_calculator` must be given explicitly, since
        `GPAWVectorPotential`/`GPAWMagneticField` only support a single
        calculator.
    gpts : one or two int, optional
        Forwarded to all built components. See `GPAWPotential`.
    sampling : one or two float, optional
        Forwarded to all built components. See `GPAWPotential`.
    slice_thickness : float or sequence of float, optional
        Forwarded to all built components. See `GPAWPotential`.
    exit_planes : int or tuple of int, optional
        Forwarded to all built components. See `GPAWPotential`.
    plane : str or two tuples of three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    origin : three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    box : three float, optional
        Forwarded to all built components. See `GPAWPotential`.
    periodic : bool
        Forwarded to all built components. See `GPAWPotential`.
    device : str, optional
        Forwarded to all built components. See `GPAWPotential`.
    frozen_phonons : BaseFrozenPhonons, optional
        Forwarded to `GPAWPotential` only.
    rotate_field : tuple of three float, "auto", or None
        Forwarded to `GPAWVectorPotential`/`GPAWMagneticField`. Defaults to
        `"auto"`: automatically swap the larger-magnitude in-plane
        component into z (see `_auto_rotation_matrix_for_vector_field`).
    include_magnetic_field : bool
        If True, also build the magnetic field `B` (not used by
        `combined_potential`, only for inspection/visualization). Roughly
        doubles the GPAW-side cost of the magnetic part, so it is off by
        default.
    magnetic_calculator : gpaw.calculator.GPAW, optional
        The single calculator representing the (rigid) magnetic
        contribution. Defaults to `calculators` when that is a single
        calculator; required when `calculators` is a list.
    lazy : bool, optional
        Passed to the electrostatic potential's `.build()`. The magnetic
        components are always built eagerly, since
        `GPAWVectorPotential`/`GPAWMagneticField` do not support lazy
        building.
    potential_kwargs, field_kwargs : dict, optional
        Extra keyword arguments forwarded only to `GPAWPotential`, or only
        to `GPAWVectorPotential`/`GPAWMagneticField`, respectively (e.g.
        their differing `gridrefinement` defaults).

    Returns
    -------
    GPAWMagneticFields
        Its `average_field` (B_avg, the uniform field a periodic vector
        potential cannot represent) is computed from `magnetic_calculator`'s
        magnetization and permuted by the same `plane`-dependent axis order
        as `vector_potential`/`magnetic_field`, so it stays consistent with
        them under a non-default `plane`. Pass it to `pauli_multislice`'s
        `average_field` argument.
    """
    if isinstance(calculators, (list, tuple)):
        if magnetic_calculator is None:
            raise ValueError(
                "calculators is a list (a frozen-phonon ensemble); "
                "GPAWVectorPotential/GPAWMagneticField only support a "
                "single calculator. Pass magnetic_calculator explicitly to "
                "pick which one represents the (rigid) magnetic "
                "contribution."
            )
    elif magnetic_calculator is None:
        magnetic_calculator = calculators

    potential_kwargs = dict(potential_kwargs or {})
    field_kwargs = dict(field_kwargs or {})

    shared = dict(
        gpts=gpts,
        sampling=sampling,
        slice_thickness=slice_thickness,
        exit_planes=exit_planes,
        plane=plane,
        origin=origin,
        box=box,
        periodic=periodic,
        device=device,
    )

    potential = GPAWPotential(
        calculators,
        frozen_phonons=frozen_phonons,
        **shared,
        **potential_kwargs,
    ).build(lazy=lazy)
    if not lazy:
        potential = potential.compute()

    # A default box that strains the atoms was reported by the potential above.
    with _box_strain_warning_silenced():
        vector_potential_builder = GPAWVectorPotential(
            magnetic_calculator,
            rotate_field=rotate_field,
            **shared,
            **field_kwargs,
        )
        vector_potential = vector_potential_builder.build().compute()

        magnetic_field = None
        if include_magnetic_field:
            magnetic_field = (
                GPAWMagneticField(
                    magnetic_calculator,
                    rotate_field=rotate_field,
                    **shared,
                    **field_kwargs,
                )
                .build()
                .compute()
            )

    magnetization, _ = get_magnetization_from_gpaw(
        magnetic_calculator, gridrefinement=field_kwargs.get("gridrefinement", 1)
    )
    average_field = calculate_constant_magnetic_field(magnetization)
    # Keep the constant field consistent with the built arrays: its components
    # turned as the builder turned those of the field (to the box of a skewed
    # cell, to `plane`, and by the rotation `rotate_field` resolved to).
    average_field = vector_potential_builder._orient_constant_vector(average_field)

    return GPAWMagneticFields(
        potential=potential,
        vector_potential=vector_potential,
        magnetic_field=magnetic_field,
        average_field=average_field,
    )
