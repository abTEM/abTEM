"""Convergent-beam electron diffraction from Bloch waves.

The probe is a coherent sum of plane waves, one per pixel inside its aperture, and
the crystal responds to each independently. Each incident direction is solved as a
Bloch-wave calculation of the crystal rotated to bring that direction onto the
optic axis, and its beams are placed back on the probe's reciprocal-space grid:
beam g of the plane wave with transverse wave vector k exits at k + g_t, with g in
the crystal frame, as in multislice. Averaged over probe positions, the pattern is
the incoherent sum over directions; at a given position, the amplitudes add
coherently, weighted by the probe's aperture, aberrations and position, as in the
reduction step of PRISM.

Members of the probe's aberration and aperture ensembles share the Bloch-wave
solutions and differ only in these weights. Members of its tilt and energy
ensembles change the incident directions, and are solved separately.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Sequence

import dask.array as da
import numpy as np

from abtem.bloch.utils import reciprocal_cell
from abtem.core import config
from abtem.core.axes import AxisMetadata, EnergyAxis, ThicknessAxis
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2, complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.utils import get_dtype
from abtem.distributions import BaseDistribution
from abtem.measurements import DiffractionPatterns
from abtem.prism.utils import _planewave_shift_coefficients
from abtem.scan import BaseScan, validate_scan
from abtem.tilt import BeamTilt2D
from abtem.waves import Probe, _squeeze_flagged_axes

if TYPE_CHECKING:
    from abtem.bloch.dynamical import BlochWaves


def _distribution_values(value) -> np.ndarray:
    if isinstance(value, BaseDistribution):
        return np.asarray(value.values, dtype=float).ravel()
    return np.atleast_1d(np.asarray(value, dtype=float))


def _tilts(probe: Probe) -> np.ndarray:
    """The probe's tilts [mrad], with the shape of its tilt ensemble + (2,)."""
    tilt = probe.tilt
    if isinstance(tilt, BeamTilt2D):
        values = np.stack(
            np.meshgrid(
                _distribution_values(tilt.tilt_x),
                _distribution_values(tilt.tilt_y),
                indexing="ij",
            ),
            -1,
        )
    elif isinstance(tilt.tilt, BaseDistribution):
        values = np.asarray(tilt.tilt.values, dtype=float)
    else:
        values = np.asarray(tilt.tilt, dtype=float)
    return values.reshape(tilt.ensemble_shape + (2,))


def _energies(probe: Probe, bloch_waves: BlochWaves) -> tuple[np.ndarray, bool]:
    """The energies to calculate [eV], and whether they form an ensemble axis."""
    available = bloch_waves._energies
    if probe.energy is None:
        return available, len(available) > 1

    energies = _distribution_values(probe.energy)
    missing = [e for e in energies if not np.isclose(available, e).any()]
    if missing:
        raise ValueError(
            "the probe's energies ("
            + ", ".join(f"{e:g}" for e in missing)
            + " eV) are not among those of the Bloch waves ("
            + ", ".join(f"{e:g}" for e in available)
            + " eV)"
        )
    return energies, isinstance(probe.energy, BaseDistribution)


def _incident_plane_waves(probe: Probe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The probe's plane waves: their pixel indices (fftshifted), wave vectors
    [1/Å] and complex coefficients. The coefficients have the shape of the
    aberration and aperture ensembles, followed by the plane waves, and each
    member is normalized to unit total intensity."""
    aperture = asnumpy(probe.aperture._evaluate_kernel(probe))
    aberrations = asnumpy(probe.aberrations._evaluate_kernel(probe))
    aperture_shape, aberrations_shape = aperture.shape[:-2], aberrations.shape[:-2]

    ix, iy = np.nonzero(np.any(aperture, axis=tuple(range(len(aperture_shape)))))
    nx, ny = probe.gpts
    kx = np.fft.fftfreq(nx, probe.sampling[0])[ix]
    ky = np.fft.fftfreq(ny, probe.sampling[1])[iy]

    aperture = aperture[..., ix, iy].reshape(
        (1,) * len(aberrations_shape) + aperture_shape + (-1,)
    )
    aberrations = aberrations[..., ix, iy].reshape(
        aberrations_shape + (1,) * len(aperture_shape) + (-1,)
    )
    coefficients = (aberrations * aperture).astype(get_dtype(complex=True))
    coefficients /= np.sqrt(np.sum(np.abs(coefficients) ** 2, -1, keepdims=True))

    pixels = np.stack([(ix + nx // 2) % nx, (iy + ny // 2) % ny], -1)
    return pixels, np.stack([kx, ky], -1), coefficients


def _disc_box(pixels: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """The lower corner and the shape of the box around the illuminated pixels."""
    lower = pixels.min(0)
    return lower, tuple(pixels.max(0) - lower + 1)


def _orientations(
    bloch_waves: BlochWaves, directions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The orientation matrix of `bloch_waves` relative to its structure factor's
    cell, and the 'xy' Euler angles [rad] of the rotations that then bring each
    direction, given in the frame of `bloch_waves`, onto +z."""
    base = np.linalg.solve(
        np.array(bloch_waves.structure_factor.cell), np.array(bloch_waves.cell)
    ).T

    # R_y(b) R_x(a) maps d to +z for a = atan2(d_y, d_z), b = -asin(d_x)
    angles = np.stack(
        [np.arctan2(directions[:, 1], directions[:, 2]), -np.arcsin(directions[:, 0])],
        -1,
    )
    return base, angles


def _beam_offsets(
    bloch_waves: BlochWaves, hkl: np.ndarray, extent: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The offsets of the beams from their plane wave in pixels, g_t * extent
    rounded, and the beams' z-components [1/Å]."""
    g = hkl @ reciprocal_cell(bloch_waves.cell)
    offsets = g[:, :2] * np.array(extent)
    rounded = np.rint(offsets)

    # a rotated crystal moves its higher-order Laue zones off the grid; only the
    # zero-order zone tells whether the extent matches the projected lattice
    zero_order = np.abs(g[:, 2]) <= 0.05 * np.linalg.norm(g, axis=1)
    off_grid = np.abs(offsets - rounded)[zero_order].max(initial=0.0)
    if off_grid > 0.1:
        warnings.warn(
            "the crystal's projected lattice does not fall on the probe's "
            "reciprocal-space grid; beams of the zero-order Laue zone are placed "
            f"up to {off_grid:.2f} pixels from their positions. Match the probe's "
            "extent to a whole number of (orthogonal) unit cells to avoid this."
        )
    return rounded.astype(int), g[:, 2]


def _cbed_array(
    amplitudes: np.ndarray,
    positions: Optional[np.ndarray],
    *,
    pixels: np.ndarray,
    wave_vectors: np.ndarray,
    directions: np.ndarray,
    coefficients: np.ndarray,
    groups: np.ndarray,
    group_index: np.ndarray,
    g_z: np.ndarray,
    thicknesses: np.ndarray,
    wavelength: float,
    gpts: tuple[int, int],
) -> np.ndarray:
    """Place the beams of every plane wave on the detector.

    `amplitudes` has shape (plane waves, thicknesses, beams), and `coefficients`
    the shape of the probe's aberration and aperture ensembles followed by the
    plane waves. Without `positions`, the intensities of the plane waves are
    summed, giving an array of shape ensembles + (thicknesses,) + gpts; with
    positions of shape (..., 2), their amplitudes are summed at each position,
    giving ensembles + (...,) + (thicknesses,) + gpts.
    """
    xp = get_array_module(amplitudes)
    complex_dtype = get_dtype(complex=True)
    thicknesses = xp.asarray(thicknesses, dtype=get_dtype())

    # beams sharing a transverse wave vector form one exit plane wave; sorted by
    # group, the beams of each group are a slice
    order = np.argsort(group_index, kind="stable")
    bounds = np.searchsorted(group_index[order], np.arange(len(groups) + 1))
    g_z = xp.asarray(g_z[order], dtype=get_dtype())
    exit_phases = complex_exponential(2 * np.pi * g_z[None] * thicknesses[:, None])
    amplitudes = amplitudes[..., xp.asarray(order)] * exit_phases[None]

    ensemble_shape = coefficients.shape[:-1]
    coefficients = xp.asarray(
        coefficients.reshape(-1, coefficients.shape[-1]), dtype=complex_dtype
    )
    if positions is None:
        weights = abs2(coefficients)[:, None]
        leading_shape: tuple[int, ...] = ()
        canvas_shape: tuple[int, ...] = (len(coefficients), len(thicknesses))
        dtype = get_dtype()
    else:
        # the free-space phase of each plane wave across the thickness, which the
        # crystal-rotated Bloch-wave calculation leaves out
        transverse = np.sum(directions[:, :2] ** 2, axis=1)
        k_z = -transverse / (wavelength * (1 + directions[:, 2]))
        carrier = complex_exponential(
            2 * np.pi * xp.asarray(k_z, dtype=get_dtype())[:, None] * thicknesses[None]
        )
        amplitudes = amplitudes * carrier[..., None]

        leading_shape = positions.shape[:-1]
        positions = xp.asarray(positions, dtype=get_dtype()).reshape(-1, 2)
        shifts = _planewave_shift_coefficients(
            positions, xp.asarray(wave_vectors, dtype=get_dtype())
        )
        weights = (coefficients[:, None] * shifts[None])[:, :, None]
        canvas_shape = (len(coefficients), len(positions), len(thicknesses))
        dtype = complex_dtype

    # each group of beams is the illuminated disc, shifted by its offset; add
    # the discs through a padded canvas, dropping those that miss the detector
    lower, box = _disc_box(pixels)
    nx, ny = gpts
    canvas = xp.zeros(canvas_shape + (nx + 2 * box[0], ny + 2 * box[1]), dtype=dtype)
    disc = xp.zeros(canvas_shape + box, dtype=dtype)
    rows = xp.asarray(pixels[:, 0] - lower[0])
    cols = xp.asarray(pixels[:, 1] - lower[1])
    for i, (ox, oy) in enumerate(groups):
        x0, y0 = lower[0] + ox + box[0], lower[1] + oy + box[1]
        if not (0 <= x0 < nx + box[0] and 0 <= y0 < ny + box[1]):
            continue
        exit_waves = amplitudes[..., bounds[i] : bounds[i + 1]].sum(-1).T
        if positions is None:
            disc[..., rows, cols] = weights * abs2(exit_waves)
        else:
            disc[..., rows, cols] = weights * exit_waves
        canvas[..., x0 : x0 + box[0], y0 : y0 + box[1]] += disc

    array = canvas[..., box[0] : box[0] + nx, box[1] : box[1] + ny]
    if positions is not None:
        array = abs2(array)
    return array.reshape(ensemble_shape + leading_shape + array.shape[-3:])


def _cbed_member(
    bloch_waves: BlochWaves,
    probe: Probe,
    thicknesses: np.ndarray,
    positions: Optional[np.ndarray],
    lazy: bool,
    pbar: bool,
) -> np.ndarray | da.core.Array:
    """The patterns of a probe of one energy and tilt, with the shape of its
    aberration and aperture ensembles + scan + (thicknesses,) + gpts."""
    from abtem.bloch.dynamical import BlochwaveEnsemble

    wavelength = energy2wavelength(probe.energy)
    pixels, wave_vectors, coefficients = _incident_plane_waves(probe)

    # a tilted probe is a tilted beam: the plane waves keep their pixels, but
    # meet the crystal along directions offset by the tilt, as in multislice
    transverse = wave_vectors * wavelength + _tilts(probe).reshape(2) * 1e-3
    directions = np.concatenate(
        [transverse, np.sqrt(1 - np.sum(transverse**2, 1))[:, None]], axis=1
    )
    base, angles = _orientations(bloch_waves, directions)
    ensemble = BlochwaveEnsemble(
        "xy",
        angles,
        structure_factor=bloch_waves.structure_factor,
        energy=bloch_waves.energy,
        sg_max=bloch_waves.sg_max,
        g_max=bloch_waves.g_max,
        centering=bloch_waves._centering,
        device=bloch_waves._device,
        use_wave_eq=bloch_waves.use_wave_eq,
        orientation_matrix=base,
    )
    # a lazy calculation runs one task per plane wave; its progress is dask's
    amplitudes = ensemble.calculate_diffraction_patterns(
        thicknesses, return_complex=True, lazy=lazy, pbar=pbar and not lazy
    )

    offsets, g_z = _beam_offsets(bloch_waves, amplitudes.miller_indices, probe.extent)
    groups, group_index = np.unique(offsets, axis=0, return_inverse=True)
    kwargs = dict(
        pixels=pixels,
        wave_vectors=wave_vectors,
        directions=directions,
        coefficients=coefficients,
        groups=groups,
        group_index=group_index.ravel(),
        g_z=g_z,
        thicknesses=thicknesses,
        wavelength=wavelength,
        gpts=probe.gpts,
    )
    ensemble_shape = coefficients.shape[:-1]
    scan_shape = () if positions is None else positions.shape[:-1]
    m, n = len(ensemble_shape), len(scan_shape)

    # per position and thickness, a block holds the pattern, the padded canvas,
    # and a disc with its weighted amplitudes
    nx, ny = probe.gpts
    _, box = _disc_box(pixels)
    per_position = int(
        nx * ny + (nx + 2 * box[0]) * (ny + 2 * box[1]) + 2 * box[0] * box[1]
    )

    def scan_chunks(shape):
        chunks = validate_chunks(
            ensemble_shape + shape + (len(thicknesses), per_position),
            (-1,) * m + ("auto",) * len(shape) + (-1, -1),
            dtype=get_dtype(complex=True),
            device=bloch_waves._device,
        )
        return chunks[m : m + len(shape)]

    if not lazy:
        if positions is None:
            return _cbed_array(amplitudes.array, None, **kwargs)
        flat = positions.reshape(-1, 2)
        (sizes,) = scan_chunks((len(flat),))
        arrays = [
            _cbed_array(amplitudes.array, batch, **kwargs)
            for batch in np.split(flat, np.cumsum(sizes)[:-1])
        ]
        array = get_array_module(arrays[0]).concatenate(arrays, axis=m)
        return array.reshape(ensemble_shape + scan_shape + array.shape[m + 1 :])

    # every block of scan positions takes all plane waves and beams, and makes
    # the patterns of every member of the aberration and aperture ensembles
    ensemble_axes, scan_axes = tuple(range(m)), tuple(range(m, m + n))
    plane_waves, beams, xy, thickness, kx, ky = range(m + n, m + n + 6)

    args: list = [amplitudes.array, (plane_waves, thickness, beams)]
    if positions is None:
        args += [None, None]
    else:
        args += [
            da.from_array(positions, chunks=scan_chunks(scan_shape) + (-1,)),
            scan_axes + (xy,),
        ]

    return da.blockwise(
        _cbed_array,
        ensemble_axes + scan_axes + (thickness, kx, ky),
        *args,
        new_axes=dict(zip(ensemble_axes + (kx, ky), ensemble_shape + probe.gpts)),
        concatenate=True,
        meta=get_array_module(bloch_waves._device).array((), dtype=get_dtype()),
        **kwargs,
    )


def calculate_cbed_patterns(
    bloch_waves: BlochWaves,
    probe: Probe,
    thicknesses: float | Sequence[float] | np.ndarray,
    scan: Optional[Sequence | np.ndarray | BaseScan] = None,
    lazy: bool = True,
    pbar: Optional[bool] = None,
) -> DiffractionPatterns:
    """See :meth:`BlochWaves.calculate_cbed_patterns`."""
    if not isinstance(probe, Probe):
        raise TypeError(f"probe must be a Probe, not {type(probe).__name__}")
    probe.grid.check_is_defined()
    if pbar is None:
        pbar = config.get("diagnostics.task_progress", False)

    energies, energy_ensemble = _energies(probe, bloch_waves)
    tilts = _tilts(probe)

    squeeze_thickness = np.ndim(thicknesses) == 0
    thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))

    def member(tilt, energy):
        member = probe.copy()
        member.tilt = tuple(tilt)
        member.energy = float(energy)
        return member

    if probe.scan_positions.ensemble_shape:
        if scan is not None:
            raise ValueError(
                "the probe has scan positions of its own; give the positions "
                "either to the probe or as scan, not both"
            )
        scan = probe.scan_positions

    scan_axes_metadata: list[AxisMetadata] = []
    positions = None
    if scan is not None:
        scan = validate_scan(scan, member(tilts.reshape(-1, 2)[0], energies[0]))
        positions = scan.get_positions()
        scan_axes_metadata = scan.ensemble_axes_metadata

    def stack(arrays, axis):
        if lazy:
            return da.stack(arrays, axis)
        return get_array_module(arrays[0]).stack(arrays, axis)

    # the probe's axes come in the order tilt, aberrations, aperture, scan, energy
    energy_axis = len(probe.aberrations.ensemble_shape) + len(
        probe.aperture.ensemble_shape
    )
    energy_axis += 0 if positions is None else positions.ndim - 1
    arrays = []
    for tilt in tilts.reshape(-1, 2):
        per_energy = [
            _cbed_member(
                bloch_waves.select_energy(energy),
                member(tilt, energy),
                thicknesses,
                positions,
                lazy,
                pbar,
            )
            for energy in energies
        ]
        arrays.append(
            stack(per_energy, energy_axis) if energy_ensemble else per_energy[0]
        )
    tilt_shape = tilts.shape[:-1]
    if tilt_shape:
        array = stack(arrays, 0)
        array = array.reshape(tilt_shape + array.shape[1:])
    else:
        array = arrays[0]

    # the probe's own scan positions, if any, came in as the scan
    probe_axes = [
        *probe.tilt.ensemble_axes_metadata,
        *probe.aberrations.ensemble_axes_metadata,
        *probe.aperture.ensemble_axes_metadata,
    ]
    if isinstance(probe.energy, BaseDistribution):
        energy_axes = probe.ensemble_axes_metadata[-1:]
    elif energy_ensemble:
        energy_axes = [EnergyAxis(values=tuple(float(e) for e in energies))]
    else:
        energy_axes = []
    ensemble_axes_metadata = [*probe_axes, *scan_axes_metadata, *energy_axes]

    if squeeze_thickness:
        array = array[..., 0, :, :]
    else:
        ensemble_axes_metadata.append(
            ThicknessAxis(label="z", units="Å", values=tuple(thicknesses))
        )

    patterns = DiffractionPatterns(
        array,
        sampling=probe.reciprocal_space_sampling,
        fftshift=True,
        ensemble_axes_metadata=ensemble_axes_metadata,
        metadata={
            **probe.metadata,
            "energy": None if energy_ensemble else float(energies[0]),
            "label": "intensity",
            "units": "arb. unit",
            "sg_max": bloch_waves.sg_max,
            "g_max": bloch_waves.g_max,
        },
    )
    return _squeeze_flagged_axes(patterns)
