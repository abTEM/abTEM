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
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Sequence

import dask.array as da
import numpy as np
from scipy.spatial.transform import Rotation

from abtem.bloch.utils import reciprocal_cell
from abtem.core import config
from abtem.core.axes import AxisMetadata, ThicknessAxis
from abtem.core.backend import asnumpy, get_array_module
from abtem.core.chunks import validate_chunks
from abtem.core.complex import abs2, complex_exponential
from abtem.core.energy import energy2wavelength
from abtem.core.utils import get_dtype
from abtem.measurements import DiffractionPatterns
from abtem.prism.utils import _planewave_shift_coefficients
from abtem.scan import BaseScan, validate_scan
from abtem.waves import Probe, _squeeze_flagged_axes

if TYPE_CHECKING:
    from abtem.bloch.dynamical import BlochWaves


def _validate_probe(probe: Probe, energy: float) -> Probe:
    if not isinstance(probe, Probe):
        raise TypeError(f"probe must be a Probe, not {type(probe).__name__}")

    probe = probe.copy()
    if probe.energy is None:
        probe.energy = energy
    elif not np.isclose(probe.energy, energy):
        raise ValueError(
            f"the probe's energy ({probe.energy:g} eV) differs from the Bloch "
            f"waves' ({energy:g} eV)"
        )

    probe.grid.check_is_defined()

    if probe.ensemble_shape:
        raise ValueError(
            "probes with ensemble axes, such as a defocus series, are not supported"
        )

    if probe.tilt.tilt_x != 0.0 or probe.tilt.tilt_y != 0.0:
        raise ValueError(
            "tilting the probe is not supported; tilt the crystal with "
            "BlochWaves.rotate instead"
        )
    return probe


def _incident_plane_waves(probe: Probe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The probe's plane waves: their pixel indices (fftshifted), wave vectors
    [1/Å] and complex coefficients, normalized to unit total intensity."""
    aperture = asnumpy(probe.aperture._evaluate_kernel(probe))
    aberrations = asnumpy(probe.aberrations._evaluate_kernel(probe))

    ix, iy = np.nonzero(aperture)
    nx, ny = probe.gpts
    kx = np.fft.fftfreq(nx, probe.sampling[0])[ix]
    ky = np.fft.fftfreq(ny, probe.sampling[1])[iy]

    coefficients = (aperture * aberrations)[ix, iy].astype(np.complex128)
    coefficients /= np.sqrt(np.sum(np.abs(coefficients) ** 2))

    pixels = np.stack([(ix + nx // 2) % nx, (iy + ny // 2) % ny], -1)
    return pixels, np.stack([kx, ky], -1), coefficients


def _orientations(bloch_waves: BlochWaves, directions: np.ndarray) -> np.ndarray:
    """Euler angles ('xyz', radians) of the crystal orientations that bring each
    direction, given in the frame of `bloch_waves`, onto +z."""
    # the orientation of `bloch_waves` relative to its structure factor's cell
    base = np.linalg.solve(
        np.array(bloch_waves.structure_factor.cell), np.array(bloch_waves.cell)
    ).T

    # R_y(b) R_x(a) maps d to +z for a = atan2(d_y, d_z), b = -asin(d_x)
    tilts = Rotation.from_euler(
        "xy",
        np.stack(
            [
                np.arctan2(directions[:, 1], directions[:, 2]),
                -np.arcsin(directions[:, 0]),
            ],
            -1,
        ),
    )
    return (tilts * Rotation.from_matrix(base)).as_euler("xyz")


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
    coefficients: np.ndarray,
    groups: np.ndarray,
    group_index: np.ndarray,
    g_z: np.ndarray,
    thicknesses: np.ndarray,
    energy: float,
    gpts: tuple[int, int],
) -> np.ndarray:
    """Place the beams of every plane wave on the detector.

    `amplitudes` has shape (plane waves, thicknesses, beams). Without `positions`,
    the intensities of the plane waves are summed, giving an array of shape
    (thicknesses,) + gpts; with positions of shape (..., 2), their amplitudes are
    summed at each position, giving (...,) + (thicknesses,) + gpts.
    """
    xp = get_array_module(amplitudes)
    complex_dtype = get_dtype(complex=True)
    thicknesses = xp.asarray(thicknesses, dtype=get_dtype())

    # beams sharing a transverse wave vector form one exit plane wave
    exit_phases = complex_exponential(
        2 * np.pi * xp.asarray(g_z, dtype=get_dtype())[None] * thicknesses[:, None]
    )
    one_hot = np.zeros((len(group_index), len(groups)))
    one_hot[np.arange(len(group_index)), group_index] = 1.0
    amplitudes = (amplitudes * exit_phases[None]) @ xp.asarray(
        one_hot, dtype=complex_dtype
    )

    if positions is None:
        weights = xp.asarray(np.abs(coefficients) ** 2, dtype=get_dtype())
        leading_shape: tuple[int, ...] = ()
        canvas_shape: tuple[int, ...] = (len(thicknesses),)
        dtype = get_dtype()
    else:
        # the free-space phase of each plane wave across the thickness, which the
        # crystal-rotated Bloch-wave calculation leaves out
        wavelength = energy2wavelength(energy)
        k2 = np.sum(wave_vectors**2, axis=1)
        k_z = -k2 * wavelength / (np.sqrt(1 - wavelength**2 * k2) + 1)
        carrier = complex_exponential(
            2 * np.pi * xp.asarray(k_z, dtype=get_dtype())[:, None] * thicknesses[None]
        )
        amplitudes = amplitudes * carrier[..., None]

        leading_shape = positions.shape[:-1]
        positions = xp.asarray(positions, dtype=get_dtype()).reshape(-1, 2)
        weights = _planewave_shift_coefficients(
            positions, xp.asarray(wave_vectors, dtype=get_dtype())
        ) * xp.asarray(coefficients, dtype=complex_dtype)
        canvas_shape = (len(positions), len(thicknesses))
        dtype = complex_dtype

    # each group of beams is the illuminated disc, shifted by its offset; add
    # the discs through a padded canvas, dropping those that miss the detector
    lower = pixels.min(0)
    box = tuple(pixels.max(0) - lower + 1)
    nx, ny = gpts
    canvas = xp.zeros(canvas_shape + (nx + 2 * box[0], ny + 2 * box[1]), dtype=dtype)
    disc = xp.zeros(canvas_shape + box, dtype=dtype)
    rows = xp.asarray(pixels[:, 0] - lower[0])
    cols = xp.asarray(pixels[:, 1] - lower[1])
    for i, (ox, oy) in enumerate(groups):
        x0, y0 = lower[0] + ox + box[0], lower[1] + oy + box[1]
        if not (0 <= x0 < nx + box[0] and 0 <= y0 < ny + box[1]):
            continue
        if positions is None:
            values = (abs2(amplitudes[:, :, i]) * weights[:, None]).T
        else:
            values = weights[:, None] * amplitudes[:, :, i].T[None]
        disc[..., rows, cols] = values
        canvas[..., x0 : x0 + box[0], y0 : y0 + box[1]] += disc

    array = canvas[..., box[0] : box[0] + nx, box[1] : box[1] + ny]
    if positions is not None:
        array = abs2(array)
    return array.reshape(leading_shape + array.shape[-3:])


def calculate_cbed_patterns(
    bloch_waves: BlochWaves,
    probe: Probe,
    thicknesses: float | Sequence[float] | np.ndarray,
    scan: Optional[Sequence | np.ndarray | BaseScan] = None,
    lazy: bool = True,
    pbar: Optional[bool] = None,
) -> DiffractionPatterns:
    """See :meth:`BlochWaves.calculate_cbed_patterns`."""
    from abtem.bloch.dynamical import BlochwaveEnsemble

    bloch_waves._require_single_energy("calculate_cbed_patterns")
    if pbar is None:
        pbar = config.get("diagnostics.task_progress", False)

    probe = _validate_probe(probe, bloch_waves.energy)
    wavelength = energy2wavelength(bloch_waves.energy)

    pixels, wave_vectors, coefficients = _incident_plane_waves(probe)
    directions = np.concatenate(
        [
            wave_vectors * wavelength,
            np.sqrt(1 - wavelength**2 * np.sum(wave_vectors**2, 1))[:, None],
        ],
        axis=1,
    )
    ensemble = BlochwaveEnsemble(
        "xyz",
        _orientations(bloch_waves, directions),
        structure_factor=bloch_waves.structure_factor,
        energy=bloch_waves.energy,
        sg_max=bloch_waves.sg_max,
        g_max=bloch_waves.g_max,
        centering=bloch_waves._centering,
        device=bloch_waves._device,
        use_wave_eq=bloch_waves.use_wave_eq,
    )

    squeeze_thickness = np.ndim(thicknesses) == 0
    thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))
    amplitudes = ensemble.calculate_diffraction_patterns(
        thicknesses, return_complex=True, lazy=lazy, pbar=pbar
    )

    offsets, g_z = _beam_offsets(bloch_waves, amplitudes.miller_indices, probe.extent)
    groups, group_index = np.unique(offsets, axis=0, return_inverse=True)
    kwargs = dict(
        pixels=pixels,
        wave_vectors=wave_vectors,
        coefficients=coefficients,
        groups=groups,
        group_index=group_index.ravel(),
        g_z=g_z,
        thicknesses=thicknesses,
        energy=bloch_waves.energy,
        gpts=probe.gpts,
    )

    ensemble_axes_metadata: list[AxisMetadata] = []
    if scan is None:
        positions = None
        scan_shape: tuple[int, ...] = ()
    else:
        scan = validate_scan(scan, probe)
        positions = scan.get_positions()
        scan_shape = positions.shape[:-1]
        ensemble_axes_metadata += scan.ensemble_axes_metadata

    xp = get_array_module(bloch_waves._device)
    shape = scan_shape + (len(thicknesses),) + probe.gpts
    if lazy:
        # every block of scan positions takes all plane waves and beams
        scan_axes = tuple(range(len(scan_shape)))
        plane_waves, beams, xy, thickness, kx, ky = range(
            len(scan_axes), len(shape) + 3
        )
        args: list = [amplitudes.array, (plane_waves, thickness, beams)]
        if positions is not None:
            chunks = validate_chunks(
                shape,
                ("auto",) * len(scan_axes) + (-1, -1, -1),
                dtype=get_dtype(complex=True),
            )
            args += [
                da.from_array(positions, chunks=chunks[: len(scan_axes)] + (-1,)),
                scan_axes + (xy,),
            ]
        else:
            args += [None, None]
        array = da.blockwise(
            _cbed_array,
            scan_axes + (thickness, kx, ky),
            *args,
            new_axes={kx: probe.gpts[0], ky: probe.gpts[1]},
            concatenate=True,
            meta=xp.array((), dtype=get_dtype()),
            **kwargs,
        )
    else:
        array = _cbed_array(amplitudes.array, positions, **kwargs)

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
            "label": "intensity",
            "units": "arb. unit",
            "sg_max": bloch_waves.sg_max,
            "g_max": bloch_waves.g_max,
        },
    )
    return _squeeze_flagged_axes(patterns)
