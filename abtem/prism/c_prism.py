"""Module describing the compressed scattering matrix used in the C-PRISM algorithm."""

from __future__ import annotations

import warnings

import numpy as np

from functools import partial

from abtem.core import config
from abtem.core.axes import AxisMetadata, ScanAxis, UnknownAxis
from abtem.core.backend import get_array_module, validate_device
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import Accelerator
from abtem.core.ensemble import _wrap_with_array
from abtem.core.fft import ifft2
from abtem.core.grid import Grid
from abtem.core.utils import CopyMixin, EqualityMixin, get_dtype, itemset
from abtem.detectors import BaseDetector, validate_detectors
from abtem.measurements import BaseMeasurements
from abtem.multislice import allocate_multislice_measurements
from abtem.prism.s_matrix import (
    BaseSMatrix,
    SMatrix,
    _finalize_lazy_measurements,
    _wrap_measurements,
)
from abtem.scan import BaseScan, GridScan, validate_scan
from abtem.transfer import CTF
from abtem.waves import Probe, Waves


def _dense_wave_vector_indices(
    extent: tuple[float, float],
    gpts: tuple[int, int],
    energy: float,
    semiangle_cutoff: float,
) -> np.ndarray:
    """Integer Fourier-space indices of all plane waves inside the aperture at
    interpolation (1, 1), including the soft edge."""
    probe = Probe._from_ctf(
        extent=extent,
        gpts=gpts,
        ctf=CTF(energy=energy, semiangle_cutoff=semiangle_cutoff),
        energy=energy,
        device="cpu",
    )
    aperture = probe.aperture._evaluate_kernel(probe)

    indices = np.where(aperture > 0.0)

    n = np.fft.fftfreq(aperture.shape[0], d=1 / aperture.shape[0])[indices[0]]
    m = np.fft.fftfreq(aperture.shape[1], d=1 / aperture.shape[1])[indices[1]]
    return np.stack([n, m], axis=-1).astype(int)


def _randomized_svd(
    a, n_components: int, xp, n_oversamples: int = 16, n_iter: int = 2, seed: int = 13
):
    """Randomized truncated SVD (Halko et al.) with QR-stabilized power iterations.

    Works for numpy and cupy arrays.
    """
    n = a.shape[1]
    q = min(n, n_components + n_oversamples)

    random_state = xp.random.RandomState(seed)
    dtype = get_dtype(complex=True)
    projection = (
        random_state.standard_normal((n, q))
        + 1.0j * random_state.standard_normal((n, q))
    ).astype(dtype)

    Q, _ = xp.linalg.qr(a @ projection)
    for _ in range(n_iter):
        Z, _ = xp.linalg.qr(a.conj().T @ Q)
        Q, _ = xp.linalg.qr(a @ Z)

    B = Q.conj().T @ a
    U_B, s, Vh = xp.linalg.svd(B, full_matrices=False)
    return Q @ U_B, s, Vh


class CPRISMArray(BaseSMatrix, CopyMixin, EqualityMixin):
    """
    A compressed scattering matrix defined by its truncated singular value
    decomposition. The phase-removed scattering matrix is factored as
    :math:`T \\approx U \\Sigma V^H`, where the left singular vectors :math:`U` are
    real-space images and the right singular vectors are interpolated to the plane
    waves of the aperture at interpolation (1, 1).

    Parameters
    ----------
    u : np.ndarray
        Left singular vectors of the phase-removed scattering matrix of shape
        (K, gpts_x, gpts_y), where K is the number of retained modes.
    sigma : np.ndarray
        Retained singular values of shape (K,).
    vh_dense : np.ndarray
        Right singular vectors interpolated to the dense plane-wave expansion of
        shape (K, number of dense plane waves).
    dense_indices : np.ndarray
        Integer Fourier-space indices of the dense plane waves of shape (N, 2).
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    extent : two float
        Lateral extent of the scattering matrix [Å].
    interpolation : two int
        Interpolation factor used for the coarse plane-wave expansion.
    window_gpts : two int
        The number of grid points describing the cropping window of the reduced
        wave functions.
    device : str
        The device used for the reduction ('cpu' or 'gpu').
    metadata : dict
        A dictionary defining wave function metadata.
    """

    def __init__(
        self,
        u: np.ndarray,
        sigma: np.ndarray,
        vh_dense: np.ndarray,
        dense_indices: np.ndarray,
        semiangle_cutoff: float,
        energy: float,
        extent: tuple[float, float],
        interpolation: tuple[int, int],
        window_gpts: tuple[int, int],
        position_quantization: int = None,
        device: str = None,
        metadata: dict = None,
    ):
        self._u = u
        self._sigma = sigma
        self._vh_dense = vh_dense
        self._dense_indices = dense_indices

        self._grid = Grid(extent=extent, gpts=u.shape[-2:], lock_gpts=True)
        self._accelerator = Accelerator(energy=energy)

        self._semiangle_cutoff = semiangle_cutoff
        self._interpolation = interpolation
        self._window_gpts = tuple(window_gpts)
        self._position_quantization = position_quantization
        self._device = validate_device(device)
        self._metadata = {} if metadata is None else metadata

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def u(self) -> np.ndarray:
        """Left singular vectors of shape (K, gpts_x, gpts_y)."""
        return self._u

    @property
    def sigma(self) -> np.ndarray:
        """Retained singular values."""
        return self._sigma

    @property
    def vh_dense(self) -> np.ndarray:
        """Right singular vectors at the dense plane-wave expansion."""
        return self._vh_dense

    @property
    def rank(self) -> int:
        """Number of retained modes."""
        return len(self._sigma)

    @property
    def metadata(self) -> dict:
        self._metadata["energy"] = self.energy
        return self._metadata

    @property
    def interpolation(self) -> tuple[int, int]:
        return self._interpolation

    @property
    def semiangle_cutoff(self) -> float:
        return self._semiangle_cutoff

    @property
    def wave_vectors(self) -> np.ndarray:
        """The wave vectors of the dense plane-wave expansion."""
        extent = self.extent
        wave_vectors = self._dense_indices.astype(np.float32)
        wave_vectors[:, 0] /= np.float32(extent[0])
        wave_vectors[:, 1] /= np.float32(extent[1])
        return wave_vectors

    @property
    def window_gpts(self) -> tuple[int, int]:
        return self._window_gpts

    @property
    def window_extent(self) -> tuple[float, float]:
        return (
            self.window_gpts[0] * self.sampling[0],
            self.window_gpts[1] * self.sampling[1],
        )

    def _calculate_ctf_coefficients(self, ctf: CTF):
        xp = get_array_module(self._device)
        wave_vectors = xp.asarray(self.wave_vectors)

        alpha = (
            xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * ctf.wavelength
        )
        phi = xp.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])
        array = ctf._evaluate_from_angular_grid(alpha, phi)
        array = array / xp.sqrt((xp.abs(array) ** 2).sum(axis=-1, keepdims=True))
        return array

    def _coefficient_values(self, coefficients):
        """The dense plane-wave amplitudes of each mode of the expansion."""
        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        values = xp.asarray(self._sigma[:, None] * self._vh_dense, dtype=dtype)
        return values * xp.asarray(coefficients, dtype=dtype)[None]

    def _window_kernel(self, values, fractional_offset):
        """The window kernels :math:`B_k` obtained by reducing the dense plane waves
        for a probe displaced by a fraction of a pixel."""
        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        gpts = self.gpts
        window_gpts = self.window_gpts

        if np.any(np.abs(fractional_offset) > 1e-9):
            offset = xp.asarray(fractional_offset * np.array(self.sampling))
            wave_vectors = xp.asarray(self.wave_vectors)
            values = values * complex_exponential(
                -2.0
                * xp.pi
                * (wave_vectors[:, 0] * offset[0] + wave_vectors[:, 1] * offset[1])
            )[None].astype(dtype)

        indices = xp.asarray(self._dense_indices)
        scattered = xp.zeros((self.rank,) + gpts, dtype=dtype)
        scattered[:, indices[:, 0] % gpts[0], indices[:, 1] % gpts[1]] = values

        # The normalization of the wave functions scales with the number of grid
        # points of the cropping window, such that the intensity of each reduced
        # probe is unity.
        normalization = np.prod(gpts) * np.sqrt(np.prod(gpts) / np.prod(window_gpts))

        kernel = ifft2(scattered, overwrite_x=True) * normalization

        ix = (xp.arange(window_gpts[0]) - window_gpts[0] // 2) % gpts[0]
        iy = (xp.arange(window_gpts[1]) - window_gpts[1] // 2) % gpts[1]
        return kernel[:, ix[:, None], iy[None, :]]

    def _reduce_to_waves(self, u_transposed, snapped_pixels, kernel):
        """Reduce the compressed scattering matrix to wave functions at the given
        snapped pixel positions."""
        xp = get_array_module(self._device)

        gpts = self.gpts
        window_gpts = self.window_gpts

        ix = (
            snapped_pixels[:, 0, None] + xp.arange(window_gpts[0]) - window_gpts[0] // 2
        ) % gpts[0]
        iy = (
            snapped_pixels[:, 1, None] + xp.arange(window_gpts[1]) - window_gpts[1] // 2
        ) % gpts[1]

        flat_indices = (ix[:, :, None] * gpts[1] + iy[:, None, :]).reshape(
            len(snapped_pixels), -1
        )

        kernel_transposed = xp.ascontiguousarray(
            kernel.reshape(self.rank, -1).T
        )

        num_window_gpts = int(np.prod(window_gpts))

        waves = xp.zeros(
            (len(snapped_pixels), num_window_gpts), dtype=get_dtype(complex=True)
        )

        # The gathered rows of the transposed left singular vectors are contiguous
        # in the mode index, making the reduction memory efficient. The positions
        # are chunked to bound the memory of the gathered windows.
        max_batch = max(1, int(64e6 / (num_window_gpts * self.rank * 8)))
        for start in range(0, len(snapped_pixels), max_batch):
            chunk = slice(start, start + max_batch)
            waves[chunk] = (
                u_transposed[flat_indices[chunk]] * kernel_transposed[None]
            ).sum(-1)

        return waves.reshape((len(snapped_pixels),) + window_gpts)

    def _group_by_fractional_offset(self, pixel_positions, decimals: int = 4):
        """Group probe positions by their fractional pixel offset.

        The offsets are rounded to ``10**-decimals`` pixels, which is well below
        the numerical precision of the probe positions.
        """
        xp = get_array_module(pixel_positions)

        snapped = xp.rint(pixel_positions).astype(int)
        fractional = pixel_positions - snapped

        fractional = fractional if xp is np else fractional.get()

        if self._position_quantization:
            fractional = (
                np.round(fractional * self._position_quantization)
                / self._position_quantization
            )

        rounded = np.round(fractional, decimals=decimals)
        rounded += 0.0  # remove negative zero
        unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
        return snapped, unique, inverse

    def _batch_reduce_to_measurements(
        self,
        scan: BaseScan,
        ctf: CTF,
        detectors: list[BaseDetector],
        max_batch_reduction: int,
        pbar: bool = False,
    ) -> tuple[BaseMeasurements | Waves, ...]:
        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(
            dummy_probes,
            detectors,
            extra_ensemble_axes_shape=(),
            extra_ensemble_axes_metadata=[],
        )

        xp = get_array_module(self._device)

        u_transposed = xp.ascontiguousarray(
            xp.asarray(self._u).reshape(self.rank, -1).T
        )

        n_positions = int(np.prod(scan.shape + ctf.ensemble_shape))
        pbar = TqdmWrapper(
            enabled=pbar, total=n_positions, leave=False, desc="reduce"
        )

        sampling = xp.asarray(self.sampling)

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):
            sub_ctf = sub_ctf.item()
            coefficients = self._calculate_ctf_coefficients(sub_ctf)

            # the generated blocks contain a single ensemble member
            coefficients = coefficients.reshape((-1, coefficients.shape[-1]))[0]

            values = self._coefficient_values(coefficients)

            for _, slics, sub_scan in scan.generate_blocks(max_batch_reduction):
                sub_scan = sub_scan.item()

                positions = xp.asarray(sub_scan.get_positions())
                scan_shape = positions.shape[:-1]
                positions = positions.reshape((-1, 2))

                pixel_positions = positions.astype(np.float64) / sampling

                snapped, unique_offsets, inverse = self._group_by_fractional_offset(
                    pixel_positions
                )

                waves_array = xp.zeros(
                    (len(positions),) + self.window_gpts,
                    dtype=get_dtype(complex=True),
                )
                for i, offset in enumerate(unique_offsets):
                    kernel = self._window_kernel(values, offset)
                    mask = xp.asarray(inverse == i)
                    waves_array[mask] = self._reduce_to_waves(
                        u_transposed, snapped[mask], kernel
                    )

                waves_array = waves_array.reshape(
                    (1,) * len(sub_ctf.ensemble_shape)
                    + scan_shape
                    + self.window_gpts
                )

                ensemble_axes_metadata = [
                    UnknownAxis() for _ in range(len(sub_ctf.ensemble_shape))
                ]
                ensemble_axes_metadata += [ScanAxis() for _ in range(len(scan_shape))]

                waves = Waves(
                    waves_array,
                    sampling=tuple(self.sampling),
                    energy=self.energy,
                    ensemble_axes_metadata=ensemble_axes_metadata,
                    metadata=self.metadata,
                )

                indices = ctf_slics + slics

                pbar.update_if_exists(len(positions))

                for detector, measurement in zip(detectors, measurements):
                    measurement.array[indices] = detector.detect(waves).array

        pbar.close_if_exists()

        return tuple(measurements)

    def reduce(
        self,
        scan: BaseScan = None,
        ctf: CTF = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        max_batch_reduction: int | str = "auto",
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, reduces a single
            probe at the center of the potential.
        ctf : CTF, optional
            The probe contrast transfer function. Default is None (aperture is set by
            the plane-wave cutoff).
        detectors : BaseDetector or list of BaseDetector
            The detectors recording the measurements.
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. If 'auto' (default), the
            batch size is automatically chosen based on the abTEM user configuration
            settings "dask.chunk-size" and "dask.chunk-size-gpu".

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurements or Waves
        """
        self.accelerator.check_is_defined()

        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff)

        ctf.grid.match(self.dummy_probes())
        ctf.accelerator.match(self)

        if ctf.semiangle_cutoff == np.inf:
            ctf.semiangle_cutoff = self.semiangle_cutoff

        squeeze = () if isinstance(scan, BaseScan) else (-3,)

        if scan is None:
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = validate_scan(
            scan, Probe._from_ctf(extent=self.extent, ctf=ctf, energy=self.energy)
        )

        detectors = validate_detectors(detectors, self.dummy_probes())

        from abtem.core.chunks import validate_chunks

        shape = (len(scan),) + self.window_gpts
        chunks = (max_batch_reduction, -1, -1)
        max_batch_reduction = validate_chunks(
            shape, chunks, dtype=np.dtype("complex64")
        )[0][0]

        pbar = config.get("diagnostics.task_progress", False)

        measurements = self._batch_reduce_to_measurements(
            scan, ctf, detectors, max_batch_reduction, pbar=pbar
        )

        measurements = [measurement.squeeze(squeeze) for measurement in measurements]
        return _wrap_measurements(measurements)

    def scan(
        self,
        scan: BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF = None,
        max_batch_reduction: int | str = "auto",
    ):
        """
        Reduce the compressed scattering matrix at the positions of a scan.

        See :meth:`.CPRISMArray.reduce`.
        """
        if scan is None:
            scan = GridScan()

        return self.reduce(
            scan=scan,
            detectors=detectors,
            ctf=ctf,
            max_batch_reduction=max_batch_reduction,
        )


class CPRISM(SMatrix):
    """
    The compressed scattering matrix is used for simulating STEM experiments using
    the C-PRISM algorithm.

    C-PRISM builds a PRISM scattering matrix from a coarse plane-wave expansion
    given by the interpolation factors. The rapidly oscillating plane-wave phase is
    factored out of each of the propagated waves, exposing their smooth variation
    with the wave vector. The resulting phase-removed scattering matrix is
    compressed by an adaptive truncated singular value decomposition, and the right
    singular vectors are interpolated back to the full plane-wave expansion of the
    aperture. Every probe is then reduced from the full expansion, avoiding the
    real-space cropping and coarsened aperture sampling errors of PRISM at the same
    interpolation factor.

    The coarse plane-wave expansion is padded to the bounding rectangle of the
    aperture (plus a one-cell buffer), so that the interpolation of the right
    singular vectors is supported on all sides. The padded plane waves are assigned
    zero weight in the reduction.

    Parameters
    ----------
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    potential : Atoms or AbstractPotential, optional
        Atoms or a potential that the scattering matrix represents. If given as
        atoms, a default potential will be created. If nothing is provided the
        scattering matrix will represent a vacuum potential, in which case the
        sampling and extent must be provided.
    gpts : one or two int, optional
        Number of grid points describing the scattering matrix. Provide only if
        potential is not given.
    sampling : one or two float, optional
        Lateral sampling of scattering matrix [Å]. Provide only if potential is not
        given. Will be ignored if 'gpts' is also provided.
    extent : one or two float, optional
        Lateral extent of scattering matrix [Å]. Provide only if potential is not
        given.
    interpolation : one or two int, optional
        Interpolation factor of the coarse plane-wave expansion in the `x` and `y`
        directions (default is 1, ie. no interpolation). If a single value is
        provided, assumed to be the same for both directions. Unlike PRISM, the
        interpolation factor does not affect the size of the cropping window of the
        reduced wave functions, only the number of multislice runs required to build
        the scattering matrix.
    tolerance : float, optional
        Relative singular value threshold of the adaptive truncation. All modes with
        singular values within this factor of the largest singular value are
        retained (default is 1e-3). Decrease for higher accuracy at increased cost
        of the reduction.
    max_rank : int, optional
        Maximum number of retained modes. If None (default), the rank is set
        adaptively by the tolerance.
    window_gpts : one or two int, optional
        The number of grid points describing the cropping window of the reduced
        wave functions. If None (default), the reduced wave functions are not
        cropped. Unlike PRISM, the window is decoupled from the interpolation
        factor; a window a few times larger than the scattered probe may be used to
        speed up the reduction at any interpolation factor.
    downsample : {'cutoff', 'valid'} or float or bool
        Controls whether to downsample the scattering matrix after running the
        multislice algorithm (default is 'cutoff'). See :class:`.SMatrix`.
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu').
        Default is 'cpu'. The default is determined by the user configuration.
    store_on_host : bool, optional
        If True, store the scattering matrix in host (cpu) memory so that the
        necessary memory is transferred as chunks to the device to run calculations
        (default is False).

    Notes
    -----
    The probe positions are decomposed into a whole and a fractional number of
    pixels. The reduction is exact for any probe position, however, a separate
    reduction kernel is calculated for each unique fractional offset in a scan. The
    reduction is fastest when the scan sampling is commensurate with the sampling of
    the scattering matrix.
    """

    def __init__(
        self,
        semiangle_cutoff: float,
        energy: float,
        potential=None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        extent: float | tuple[float, float] = None,
        interpolation: int | tuple[int, int] = 1,
        tolerance: float = 1e-3,
        max_rank: int = None,
        window_gpts: int | tuple[int, int] = None,
        position_quantization: int = None,
        downsample: bool | str = "cutoff",
        device: str = None,
        store_on_host: bool = False,
    ):
        super().__init__(
            semiangle_cutoff=semiangle_cutoff,
            energy=energy,
            potential=potential,
            gpts=gpts,
            sampling=sampling,
            extent=extent,
            interpolation=interpolation,
            downsample=downsample,
            device=device,
            store_on_host=store_on_host,
        )
        self._tolerance = tolerance
        self._max_rank = max_rank
        self._position_quantization = position_quantization

        if window_gpts is not None:
            if np.isscalar(window_gpts):
                window_gpts = (int(window_gpts),) * 2
            else:
                window_gpts = tuple(int(n) for n in window_gpts)

        self._window_gpts = window_gpts

    @classmethod
    def _c_prism(cls, *args, potential_partial, **kwargs):
        potential = potential_partial(*args).item()
        c_prism = cls(potential=potential, **kwargs)
        return _wrap_with_array(c_prism)

    def _from_partitioned_args(self, *args, **kwargs):
        if self.potential is not None:
            potential_partial = self.potential._from_partitioned_args()
            kwargs = self._copy_kwargs(exclude=("potential", "sampling", "extent"))
        else:

            def potential_partial(*args, **kwargs):
                return _wrap_with_array(None, 1)

            kwargs = self._copy_kwargs(exclude=("potential",))

        return partial(self._c_prism, potential_partial=potential_partial, **kwargs)

    @property
    def tolerance(self) -> float:
        """Relative singular value threshold of the adaptive truncation."""
        return self._tolerance

    @property
    def max_rank(self) -> int | None:
        """Maximum number of retained modes."""
        return self._max_rank

    @property
    def position_quantization(self) -> int | None:
        """Quantization of the fractional probe positions in fractions of a pixel."""
        return self._position_quantization

    @property
    def downsampled_gpts(self) -> tuple[int, int]:
        """The gpts of the scattering matrix after downsampling. Unlike PRISM, the
        downsampled gpts are independent of the interpolation factor, hence probe
        positions commensurate with the grid remain commensurate at any
        interpolation."""
        if self.downsample:
            downsampled_gpts = self._gpts_within_angle(self.downsample)
            return tuple(n + (-n) % 4 for n in downsampled_gpts)
        return self.gpts

    @property
    def window_gpts(self) -> tuple[int, int]:
        """The number of grid points describing the cropping window of the reduced
        wave functions. If not given, the reduced wave functions are not cropped."""
        if self._window_gpts is None:
            return self.downsampled_gpts

        return (
            min(self._window_gpts[0], self.downsampled_gpts[0]),
            min(self._window_gpts[1], self.downsampled_gpts[1]),
        )

    @property
    def window_extent(self) -> tuple[float, float]:
        sampling = (
            self.extent[0] / self.downsampled_gpts[0],
            self.extent[1] / self.downsampled_gpts[1],
        )
        return (
            self.window_gpts[0] * sampling[0],
            self.window_gpts[1] * sampling[1],
        )

    def _dense_indices(self) -> np.ndarray:
        return _dense_wave_vector_indices(
            self.extent, self.downsampled_gpts, self.energy, self.semiangle_cutoff
        )

    def _coarse_bounds(self) -> tuple[int, int]:
        dense_indices = self._dense_indices()
        bounds = ()
        for i in range(2):
            n_max = int(np.abs(dense_indices[:, i]).max())
            bounds += (-(n_max // -self.interpolation[i]) + 1,)
        return bounds

    @property
    def wave_vectors(self) -> np.ndarray:
        """The wave vectors of the coarse plane-wave expansion. The expansion is
        padded to the bounding rectangle of the aperture plus a one-cell buffer."""
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        bounds = self._coarse_bounds()

        n = np.arange(-bounds[0], bounds[0] + 1, dtype=np.float32)
        m = np.arange(-bounds[1], bounds[1] + 1, dtype=np.float32)

        w, h = self.extent

        kx = n / w * np.float32(self.interpolation[0])
        ky = m / h * np.float32(self.interpolation[1])

        kx, ky = np.meshgrid(kx, ky, indexing="ij")

        xp = get_array_module(self.device)
        return xp.asarray([kx.ravel(), ky.ravel()]).T

    def _interpolate_vh_dense(self, vh, dense_indices) -> np.ndarray:
        """Trigonometric interpolation of the right singular vectors from the coarse
        rectangle of plane waves to the dense plane-wave expansion."""
        xp = get_array_module(vh)
        dtype = get_dtype(complex=True)

        bounds = self._coarse_bounds()
        shape = (2 * bounds[0] + 1, 2 * bounds[1] + 1)

        vh = vh.reshape((-1,) + shape)

        coefficients = xp.fft.fft2(vh, axes=(-2, -1)) / np.prod(shape)

        kernels = ()
        for i, (bound, length) in enumerate(zip(bounds, shape)):
            frequencies = xp.fft.fftfreq(length, d=1 / length).astype(int)
            dense_coordinate = (
                xp.arange(-bound * self.interpolation[i], bound * self.interpolation[i] + 1)
                / self.interpolation[i]
            )
            kernels += (
                complex_exponential(
                    2.0
                    * xp.pi
                    * (dense_coordinate[:, None] + bound)
                    * frequencies[None]
                    / length
                ).astype(dtype),
            )

        interpolated = xp.tensordot(coefficients, kernels[0], axes=[[-2], [-1]])
        interpolated = xp.tensordot(interpolated, kernels[1], axes=[[-2], [-1]])

        offset = (bounds[0] * self.interpolation[0], bounds[1] * self.interpolation[1])
        return interpolated[
            :, dense_indices[:, 0] + offset[0], dense_indices[:, 1] + offset[1]
        ]

    def _compress(self, array) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Phase removal, adaptive truncated SVD and interpolation of the right
        singular vectors to the dense plane-wave expansion."""
        xp = get_array_module(array)
        dtype = get_dtype(complex=True)

        gpts = array.shape[-2:]
        extent = self.extent

        wave_vectors = self.wave_vectors
        x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
        y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)

        array = array.reshape((-1,) + tuple(gpts))

        normalization = np.prod(self.interpolation).astype(np.float32)
        max_batch = max(1, int(256**3 / np.prod(gpts)))
        for start in range(0, len(array), max_batch):
            chunk = slice(start, start + max_batch)
            phase = complex_exponential(
                -2.0 * xp.pi * wave_vectors[chunk, 0, None, None] * x[:, None]
            ) * complex_exponential(
                -2.0 * xp.pi * wave_vectors[chunk, 1, None, None] * y[None, :]
            )
            array[chunk] *= phase / normalization

        matrix = array.reshape((len(array), -1)).T

        if self._max_rank is not None:
            n_components = min(self._max_rank, len(array))
        else:
            n_components = min(max(192, len(array) // 4), len(array))

        u, sigma, vh = _randomized_svd(matrix, n_components, xp)

        rank = int((sigma >= self._tolerance * sigma[0]).sum())
        rank = max(1, rank)

        if self._max_rank is not None:
            rank = min(rank, self._max_rank)
        elif rank == len(sigma) and rank < len(array):
            warnings.warn(
                "The adaptive rank of the C-PRISM expansion reached the maximum "
                "number of probed modes; the tolerance may not be met. Provide "
                "'max_rank' to increase the number of probed modes."
            )

        u = xp.ascontiguousarray(u[:, :rank].T).reshape((rank,) + tuple(gpts))
        sigma = sigma[:rank]
        vh = vh[:rank]

        dense_indices = self._dense_indices()
        vh_dense = self._interpolate_vh_dense(vh, xp.asarray(dense_indices))

        return u, sigma.astype(get_dtype(complex=False)), vh_dense.astype(dtype), dense_indices

    def build(
        self, lazy: bool = None, max_batch: int | str = "auto", bound=None
    ) -> CPRISMArray:
        """
        Build the coarse scattering matrix and compress it by phase removal and an
        adaptive truncated singular value decomposition.

        The multislice stage may be computed lazily, however, the compression
        requires the scattering matrix in memory, hence the returned
        :class:`.CPRISMArray` is always computed.

        Parameters
        ----------
        lazy : bool, optional
            If True, build the scattering matrix lazily before the compression.
            If not given, defaults to the setting in the user configuration file.
        max_batch : int or str, optional
            The number of expansion plane waves in each run of the multislice
            algorithm.

        Returns
        -------
        c_prism_array : CPRISMArray
            The compressed scattering matrix.
        """
        if np.prod(self.ensemble_shape) > 1:
            raise NotImplementedError(
                "CPRISM.build does not support ensemble potentials; use "
                "CPRISM.reduce or CPRISM.scan, which average over the potential "
                "ensemble."
            )

        s_matrix_array = super().build(lazy=lazy, max_batch=max_batch, bound=bound)

        array = s_matrix_array.array
        if s_matrix_array.is_lazy:
            array = array.compute()

        metadata = dict(s_matrix_array.metadata)

        u, sigma, vh_dense, dense_indices = self._compress(array)

        return CPRISMArray(
            u=u,
            sigma=sigma,
            vh_dense=vh_dense,
            dense_indices=dense_indices,
            semiangle_cutoff=self.semiangle_cutoff,
            energy=self.energy,
            extent=self.extent,
            interpolation=self.interpolation,
            window_gpts=self.window_gpts,
            position_quantization=self._position_quantization,
            device=self.device,
            metadata=metadata,
        )

    def _eager_build_detect(self, scan, ctf, detectors, squeeze: bool):
        extra_ensemble_axes_shape = ()
        extra_ensemble_axes_metadata = []
        for shape, axis_metadata in zip(
            self.ensemble_shape, self.ensemble_axes_metadata
        ):
            extra_ensemble_axes_metadata += [axis_metadata]
            if axis_metadata._ensemble_mean:
                extra_ensemble_axes_shape += (1,)
            else:
                extra_ensemble_axes_shape += (shape,)

        detectors = validate_detectors(detectors)

        if self.ensemble_shape:
            measurements = allocate_multislice_measurements(
                self.dummy_probes(scan, ctf),
                detectors,
                extra_ensemble_axes_shape,
                extra_ensemble_axes_metadata,
            )
        else:
            measurements = None

        num_blocks = 0
        for i, _, c_prism in self.generate_blocks(1):
            c_prism = c_prism.item()
            c_prism_array = c_prism.build(lazy=False)

            new_measurements = c_prism_array.reduce(
                scan=scan, detectors=detectors, ctf=ctf
            )

            new_measurements = (
                [new_measurements]
                if not isinstance(new_measurements, list)
                else list(new_measurements)
            )

            if measurements is None:
                measurements = new_measurements
            else:
                for measurement, new_measurement in zip(measurements, new_measurements):
                    if measurement.axes_metadata[0]._ensemble_mean:
                        measurement.array[:] += new_measurement.array
                    else:
                        measurement.array[i] = new_measurement.array

            num_blocks += 1

        for i, measurement in enumerate(measurements):
            if (
                measurement.axes_metadata
                and measurement.axes_metadata[0]._ensemble_mean
            ):
                if num_blocks > 1:
                    measurement.array[:] /= num_blocks
                if squeeze:
                    measurements[i] = measurement.squeeze((0,))

        return measurements

    @staticmethod
    def _lazy_build_detect(c_prism, scan, ctf, detectors):
        c_prism = c_prism.item()
        measurements = c_prism._eager_build_detect(
            scan=scan, ctf=ctf, detectors=detectors, squeeze=False
        )

        array = np.zeros((1,) + (1,) * len(scan.shape), dtype=object)
        itemset(array, 0, measurements)
        return array

    def reduce(
        self,
        scan: np.ndarray | BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF | dict = None,
        max_batch_multislice: str | int = "auto",
        max_batch_reduction: str | int = "auto",
        lazy: bool = None,
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Run the multislice algorithm, compress the scattering matrix, then reduce it
        to obtain the exit wave functions at given initial probe positions and
        aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, reduces a single
            probe at the center of the potential.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should
            be converted to measurements after running the multislice algorithm.
            See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function used for calculating the expansion
            coefficients in the reduction of the scattering matrix.
        max_batch_multislice : int, optional
            The number of wave functions in each chunk of the Dask array.
            If 'auto' (default), the batch size is automatically chosen based on the
            abTEM user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly.
            If None, this defaults to the value set in the configuration file.

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurements or Waves
        """
        from abtem.array import validate_lazy

        detectors = validate_detectors(detectors, self.dummy_probes())

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        lazy = validate_lazy(lazy)

        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff)
        elif isinstance(ctf, dict):
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff, **ctf)

        ctf.accelerator.match(self)

        scan = validate_scan(
            scan, Probe._from_ctf(extent=self.extent, ctf=ctf, energy=self.energy)
        )

        if not lazy:
            measurements = self._eager_build_detect(scan, ctf, detectors, squeeze=True)
            return _wrap_measurements(measurements)

        from abtem.core.utils import tuple_range

        blocks = self.ensemble_blocks(1)

        chunks = ()
        drop_axis = ()
        if not self.ensemble_shape:
            blocks = blocks[None]
            drop_axis = (0,)
            new_axis = tuple_range(
                offset=0, length=len(scan.shape) + len(ctf.ensemble_shape)
            )
        else:
            chunks += blocks.chunks
            new_axis = tuple_range(
                offset=len(blocks.shape),
                length=len(scan.shape) + len(ctf.ensemble_shape),
            )

        chunks += ctf.ensemble_shape + scan.shape

        arrays = blocks.map_blocks(
            self._lazy_build_detect,
            drop_axis=drop_axis,
            new_axis=new_axis,
            chunks=chunks,
            scan=scan,
            ctf=ctf,
            detectors=detectors,
            meta=np.array((), dtype=object),
        )

        waves = self.dummy_probes(scan=scan)

        extra_axes_metadata = []
        if self.potential is not None:
            extra_axes_metadata = self.potential.ensemble_axes_metadata

        extra_axes_metadata = extra_axes_metadata + ctf.ensemble_axes_metadata

        measurements = _finalize_lazy_measurements(
            arrays, waves, detectors, extra_axes_metadata
        )

        return _wrap_measurements(measurements)

    def scan(
        self,
        scan: np.ndarray | BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF | dict = None,
        max_batch_multislice: str | int = "auto",
        max_batch_reduction: str | int = "auto",
        lazy: bool = None,
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Run the multislice algorithm, compress the scattering matrix, then reduce it
        at the positions of a scan.

        See :meth:`.CPRISM.reduce`; if the scan is not given, scans across the
        entire potential at Nyquist sampling.
        """
        if scan is None:
            scan = GridScan(
                start=(0, 0),
                end=self.extent,
                sampling=self.dummy_probes().aperture.nyquist_sampling,
            )

        if detectors is None:
            from abtem.detectors import FlexibleAnnularDetector

            detectors = FlexibleAnnularDetector()

        return self.reduce(
            scan=scan,
            detectors=detectors,
            ctf=ctf,
            max_batch_multislice=max_batch_multislice,
            max_batch_reduction=max_batch_reduction,
            lazy=lazy,
        )
