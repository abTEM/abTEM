"""Module describing the scattering matrix used in the PRISM algorithm."""
import inspect
import operator
import warnings
from abc import abstractmethod
from functools import partial, reduce
from typing import Union, Tuple, Dict, List

import dask
import dask.array as da
import numpy as np
from ase import Atoms

from abtem.core.array import validate_lazy, HasArray, ComputableList
from abtem.core.axes import OrdinalAxis, AxisMetadata, ScanAxis, UnknownAxis
from abtem.core.backend import get_array_module, cp, validate_device, copy_to_device
from abtem.core.chunks import chunk_ranges, validate_chunks, equal_sized_chunks, Chunks
from abtem.core.complex import abs2, complex_exponential
from abtem.core.energy import Accelerator
from abtem.core.fft import fft2
from abtem.core.grid import Grid, GridUndefinedError
from abtem.core.utils import safe_ceiling_int, expand_dims_to_match
from abtem.measurements import BaseMeasurement
from abtem.detectors import (
    BaseDetector,
    _validate_detectors,
    WavesDetector,
    FlexibleAnnularDetector,
)
from abtem.potentials import BasePotential, _validate_potential
from abtem.waves import Waves, Probe, _finalize_lazy_measurements
from abtem.waves import BaseWaves
from abtem.multislice import (
    allocate_multislice_measurements,
    multislice_and_detect,
)
from abtem.prism.utils import (
    prism_wave_vectors,
    plane_waves,
    wrapped_crop_2d,
    prism_coefficients,
    minimum_crop,
    batch_crop_2d,
    _planewave_shift_coefficients,
)
from abtem.scan import BaseScan, _validate_scan, GridScan
from abtem.transfer import CTF


def _round_gpts_to_multiple_of_interpolation(
    gpts: Tuple[int, int], interpolation: Tuple[int, int]
) -> Tuple[int, int]:
    return tuple(n + (-n) % f for f, n in zip(interpolation, gpts))  # noqa


class BaseSMatrix(BaseWaves):
    """Base class for scattering matrices. Documented in subclasses"""

    @property
    @abstractmethod
    def wave_vectors(self):
        pass

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def base_axes_metadata(self) -> List[AxisMetadata]:
        return [
            OrdinalAxis(
                label="(n, m)",
                values=_pack_wave_vectors(self.wave_vectors),
            )
        ] + super().base_axes_metadata  # noqa

    @property
    def base_shape(self):
        return (len(self),) + super().base_shape


def _validate_interpolation(interpolation: Union[int, Tuple[int, int]]):
    if isinstance(interpolation, int):
        interpolation = (interpolation,) * 2
    elif not len(interpolation) == 2:
        raise ValueError("Interpolation factor must be an integer.")
    return tuple(interpolation)


def _common_kwargs(a, b):
    a_kwargs = inspect.signature(a).parameters.keys()
    b_kwargs = inspect.signature(b).parameters.keys()
    return set(a_kwargs).intersection(b_kwargs)


def _pack_wave_vectors(wave_vectors):
    return tuple(
        (float(wave_vector[0]), float(wave_vector[1])) for wave_vector in wave_vectors
    )


class SMatrixArray(HasArray, BaseSMatrix):
    """
    A scattering matrix defined by a given array of dimension 3, where the first indexes the probe plane waves and the
    latter two are the `y` and `x` scan directions.

    Parameters
    ----------
    array : np.ndarray
        Array defining the scattering matrix.
    wave_vectors : np.ndarray
        Array defining the wave vectors corresponding to each probe plane wave.
    planewave_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Provide only if potential is not given. Will be ignored if 'gpts'
        is also provided.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions (default is 1, ie. no interpolation). If a single value is
        provided, assumed to be the same for both directions.
    cropping_window : tuple of int
    window_offset : tuple of int
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu'). Default is 'cpu'. The default is determined by the user configuration.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
    """

    _base_dims = 3  # S matrices are assumed to have three dimensions

    def __init__(
        self,
        array: np.ndarray,
        wave_vectors: np.ndarray,
        planewave_cutoff: float,
        energy: float = None,
        interpolation: Union[int, Tuple[int, int]] = (1, 1),
        sampling: Union[float, Tuple[float, float]] = None,
        extent: Union[float, Tuple[float, float]] = None,
        # tilt: Tuple[float, float] = (0.0, 0.0),
        cropping_window: Tuple[int, int] = (0, 0),
        window_offset: Tuple[int, int] = (0, 0),
        periodic: Tuple[bool, bool] = (True, True),
        device=None,
        ensemble_axes_metadata: List[AxisMetadata] = None,
        metadata: Dict = None,
    ):

        if len(array.shape) < 2:
            raise RuntimeError("Wave function array should have 2 dimensions or more")

        self._array = array
        self._grid = Grid(
            extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True
        )
        self._accelerator = Accelerator(energy=energy)
        # self._beam_tilt = BeamTilt(tilt=tilt)

        self._ensemble_axes_metadata = (
            [] if ensemble_axes_metadata is None else ensemble_axes_metadata
        )

        self._metadata = {} if metadata is None else metadata

        self._wave_vectors = wave_vectors
        self._planewave_cutoff = planewave_cutoff
        self._cropping_window = tuple(cropping_window)
        self._window_offset = tuple(window_offset)
        self._interpolation = _validate_interpolation(interpolation)
        self._device = device
        self._periodic = periodic
        self._check_axes_metadata()

    @classmethod
    def _pack_kwargs(cls, attrs, kwargs):
        kwargs["wave_vectors"] = _pack_wave_vectors(kwargs["wave_vectors"])
        super()._pack_kwargs(attrs, kwargs)

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = super()._unpack_kwargs(attrs)
        kwargs["wave_vectors"] = np.array(kwargs["wave_vectors"], dtype=np.float32)
        return kwargs

        # kwargs["wave_vectors"] = _pack_wave_vectors(kwargs["wave_vectors"])

    def copy_to_device(self, device: str) -> "SMatrixArray":
        s_matrix = super().copy_to_device(device)
        s_matrix._wave_vectors = copy_to_device(self.array, device)
        return s_matrix

    @staticmethod
    def _packed_wave_vectors(wave_vectors):
        return _pack_wave_vectors(wave_vectors)

    @property
    def tilt(self):
        return 0.0, 0.0

    @property
    def device(self):
        return self._device

    @classmethod
    def from_waves(cls, waves, **kwargs):
        kwargs.update({key: getattr(waves, key) for key in _common_kwargs(cls, Waves)})
        kwargs["ensemble_axes_metadata"] = kwargs["ensemble_axes_metadata"][:-1]
        return cls(**kwargs)

    @property
    def waves(self):
        kwargs = {
            key: getattr(self, key) for key in _common_kwargs(self.__class__, Waves)
        }
        kwargs["ensemble_axes_metadata"] = (
            kwargs["ensemble_axes_metadata"] + self.base_axes_metadata[:-2]
        )
        return Waves(**kwargs)

    def _copy_with_new_waves(self, waves):
        keys = set(
            inspect.signature(self.__class__).parameters.keys()
        ) - _common_kwargs(self.__class__, Waves)
        kwargs = {key: getattr(self, key) for key in keys}
        return self.from_waves(waves, **kwargs)

    @property
    def periodic(self):
        return self._periodic

    @property
    def metadata(self) -> Dict:
        self._metadata["energy"] = self.energy
        return self._metadata

    @property
    def ensemble_axes_metadata(self) -> List[AxisMetadata]:
        return self._ensemble_axes_metadata

    @property
    def window_offset(self) -> Tuple[float, float]:
        return self._window_offset

    @property
    def ensemble_shape(self) -> Tuple[int, int]:
        return self.array.shape[:-3]

    @property
    def interpolation(self) -> Tuple[int, int]:
        return self._interpolation

    def rechunk(self, chunks: Chunks, in_place: bool = True):
        array = self.array.rechunk(chunks)
        if in_place:
            self._array = array
            return self
        else:
            kwargs = self._copy_kwargs(exclude=("array",))
            return self.__class__(array, **kwargs)

    @property
    def planewave_cutoff(self) -> float:
        return self._planewave_cutoff

    @property
    def wave_vectors(self) -> np.ndarray:
        return self._wave_vectors

    @property
    def cropping_window(self) -> Tuple[int, int]:
        return self._cropping_window

    @property
    def window_extent(self):
        return (
            self.cropping_window[0] * self.sampling[0],
            self.cropping_window[1] * self.sampling[1],
        )

    @property
    def antialias_cutoff_gpts(self):
        return self.waves.antialias_cutoff_gpts

    @property
    def interpolated_antialias_cutoff_gpts(self):
        if self.antialias_cutoff_gpts is None:
            return None

        return (
            self.antialias_cutoff_gpts[0] // self.interpolation[0],
            self.antialias_cutoff_gpts[1] // self.interpolation[1],
        )

    def multislice(self, potential: BasePotential = None):
        waves = self.waves.multislice(potential)
        return self._copy_with_new_waves(waves)

    def _reduce_to_waves(
        self,
        array,
        positions,
        position_coefficients,
    ):
        xp = get_array_module(self._device)

        if self.cropping_window != self.gpts:
            pixel_positions = positions / xp.array(self.waves.sampling) - xp.asarray(
                self.window_offset
            )

            crop_corner, size, corners = minimum_crop(
                pixel_positions, self.cropping_window
            )
            array = wrapped_crop_2d(array, crop_corner, size)

            if self._device == "gpu" and isinstance(array, np.ndarray):
                array = cp.asarray(array)

            array = xp.tensordot(position_coefficients, array, axes=[-1, -3])

            if len(self.waves.shape) > 3:
                array = xp.moveaxis(array, -3, 0)

            array = batch_crop_2d(array, corners, self.cropping_window)

        else:
            array = xp.tensordot(position_coefficients, array, axes=[-1, -3])

            if len(self.waves.shape) > 3:
                array = xp.moveaxis(array, -3, 0)

        return array

    def dummy_probes(self, scan: BaseScan = None, ctf: CTF = None):

        if ctf is None:
            ctf = CTF(energy=self.energy, semiangle_cutoff=self.planewave_cutoff)

        probes = Probe._from_ctf(
            extent=self.window_extent,
            gpts=self.cropping_window,
            ctf=ctf,
            energy=self.energy,
            device=self._device,
        )

        if scan is not None:
            probes = probes.insert_transform(scan)

        return probes

    def _calculate_positions_coefficients(self, scan):
        xp = get_array_module(self.wave_vectors)

        if isinstance(scan, GridScan):
            x = xp.asarray(scan._x_coordinates())
            y = xp.asarray(scan._y_coordinates())
            coefficients = complex_exponential(
                -2.0 * xp.pi * x[:, None, None] * self.wave_vectors[None, None, :, 0]
            ) * complex_exponential(
                -2.0 * xp.pi * y[None, :, None] * self.wave_vectors[None, None, :, 1]
            )
        else:
            positions = xp.asarray(scan.get_positions())
            coefficients = complex_exponential(
                -2.0 * xp.pi * positions[..., 0, None] * self.wave_vectors[:, 0][None]
                - 2.0 * xp.pi * positions[..., 1, None] * self.wave_vectors[:, 1][None]
            )

        return coefficients

    def _calculate_ctf_coefficients(self, ctf):
        wave_vectors = self.wave_vectors
        xp = get_array_module(wave_vectors)

        alpha = (
            xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2) * ctf.wavelength
        )
        phi = xp.arctan2(wave_vectors[:, 0], wave_vectors[:, 1])

        return ctf._evaluate_with_alpha_and_phi(alpha, phi)

    def _batch_reduce_to_measurements(
        self,
        scan: BaseScan,
        ctf: CTF,
        detectors: List[BaseDetector],
        max_batch_reduction: int,
    ) -> Tuple[Union[BaseMeasurement, Waves], ...]:

        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(
            dummy_probes,
            detectors,
            extra_ensemble_axes_shape=self.waves.ensemble_shape[:-1],
            extra_ensemble_axes_metadata=self.waves.ensemble_axes_metadata[:-1],
        )

        xp = get_array_module(self._device)

        if self._device == "gpu" and isinstance(self.waves.array, np.ndarray):
            array = cp.asarray(self.waves.array)
        else:
            array = self.waves.array

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):

            ctf_coefficients = self._calculate_ctf_coefficients(ctf)

            for _, slics, sub_scan in scan.generate_blocks(max_batch_reduction):

                positions = xp.asarray(sub_scan.get_positions())

                positions_coefficients = self._calculate_positions_coefficients(
                    sub_scan
                )

                if ctf_coefficients is not None:
                    (
                        expanded_ctf_coefficients,
                        positions_coefficients,
                    ) = expand_dims_to_match(
                        ctf_coefficients,
                        positions_coefficients,
                        match_dims=[(-1,), (-1,)],
                    )
                    coefficients = positions_coefficients * expanded_ctf_coefficients

                ensemble_axes_metadata = [UnknownAxis()] * len(array.shape[:-3]) + [
                    ScanAxis() for _ in range(len(scan.shape))
                ]

                waves_array = self._reduce_to_waves(array, positions, coefficients)

                waves = Waves(
                    waves_array,
                    sampling=self.sampling,
                    energy=self.energy,
                    ensemble_axes_metadata=ensemble_axes_metadata,
                )

                indices = (
                    (slice(None),) * (len(self.waves.shape) - 3) + ctf_slics + slics
                )

                for detector, measurement in measurements.items():
                    measurement.array[indices] = detector.detect(waves).array

        return tuple(measurements.values())

    def _chunk_extents(self):
        return tuple(
            tuple(((cc[0] + o) * d, (cc[1] + o) * d) for cc in c)
            for c, d, o in zip(
                chunk_ranges(self.chunks[-2:]), self.sampling, self.window_offset
            )
        )

    def _window_overlap(self):
        return self.cropping_window[0] // 2, self.cropping_window[1] // 2

    def _overlap_depth(self):
        if self.cropping_window == self.gpts:
            return 0

        window_overlap = self._window_overlap()

        return {
            **{i: 0 for i in range(0, len(self.waves.shape) - 2)},
            **{
                j: window_overlap[i]
                for i, j in enumerate(
                    range(len(self.waves.shape) - 2, len(self.waves.shape))
                )
            },
        }

    def _validate_rechunk_scheme(self, rechunk="auto", shape=None):
        if shape is None:
            shape = self.shape

        if rechunk == "auto":
            num_chunks = max(self.interpolation)
            num_chunks = (
                (num_chunks, 1)
                if num_chunks == self.interpolation[0]
                else (1, num_chunks)
            )

        elif isinstance(rechunk, tuple):
            num_chunks = rechunk

            assert len(rechunk) == 2
        else:
            raise RuntimeError

        if num_chunks[0] != 1 and num_chunks[1] != 1:
            raise NotImplementedError()

        chunks = tuple(
            equal_sized_chunks(n, num_chunks=nsc)
            for n, nsc in zip(shape[-2:], num_chunks)
        )

        if chunks is None:
            chunks = self.chunks[-2:]
        else:
            chunks = validate_chunks(shape[-2:], chunks)

        return self.chunks[:-3] + ((shape[-3],),) + chunks

    def _validate_max_batch_reduction(
        self, scan, max_batch_reduction: Union[int, str] = "auto"
    ):

        shape = (len(scan),) + self.cropping_window
        chunks = (max_batch_reduction, -1, -1)

        return validate_chunks(shape, chunks, dtype=np.dtype("complex64"))[0][0]

    @staticmethod
    def _lazy_reduce(
        array,
        waves_partial,
        ensemble_axes_metadata,
        scan,
        ctf,
        detectors,
        max_batch_reduction,
        kwargs,
    ):

        waves = waves_partial(array, ensemble_axes_metadata=ensemble_axes_metadata)

        s_matrix = SMatrixArray.from_waves(waves, **kwargs)

        measurements = s_matrix._batch_reduce_to_measurements(
            scan, ctf, detectors, max_batch_reduction
        )

        arr = np.zeros((1,) * (len(array.shape) - 1), dtype=object)
        arr.itemset(measurements)
        return arr

    def _rechunk_for_reduction(self, rechunk):
        array = self.array

        chunks = self._validate_rechunk_scheme(rechunk=rechunk)

        pad_amounts = tuple(
            0 if len(c) == 1 else o for o, c in zip(self._window_overlap(), chunks[-2:])
        )

        pad_width = ((0,) * 2,) * len(array.shape[:-2]) + (
            (pad_amounts[0],) * 2,
            (pad_amounts[1],) * 2,
        )

        pad_chunks = array.chunks[:-2] + (
            array.shape[-2] + sum(pad_width[-2]),
            array.shape[-1] + sum(pad_width[-1]),
        )

        array = array.map_blocks(
            np.pad,
            pad_width=pad_width,
            meta=array._meta,
            chunks=pad_chunks,
            mode="wrap",
        )

        padded_chunks = ()
        for c, p in zip(chunks, pad_width):
            c = (p[0],) + c if p[0] else c
            c = c + (p[1],) if p[1] else c
            padded_chunks += (c,)

        array = array.rechunk(padded_chunks)

        kwargs = self._copy_kwargs(exclude=("array", "extent"))

        kwargs["periodic"] = tuple(
            False if pad_amount else periodic
            for periodic, pad_amount in zip(kwargs["periodic"], pad_amounts)
        )

        kwargs["window_offset"] = tuple(
            window_offset - pad_amount
            for window_offset, pad_amount in zip(kwargs["window_offset"], pad_amounts)
        )

        return self.__class__(array, **kwargs)

    def _index_overlap_reduce(self, scan, detectors, ctf, rechunk, max_batch_reduction):

        s_matrix = self._rechunk_for_reduction(rechunk)

        waves_partial = self.waves.from_partitioned_args()
        ensemble_axes_metadata = self.waves.ensemble_axes_metadata
        # new_chunks = (
        #     s_matrix.array.chunks[:-3] + (1,) * len(ctf.ensemble_shape)
        # )

        kwargs = s_matrix._copy_kwargs(exclude=("array", "extent"))
        scan, scan_chunks = scan.sort_into_extents(s_matrix._chunk_extents())

        old_window_offset = kwargs["window_offset"]

        ctf_chunks = tuple((n,) for n in ctf.ensemble_shape)
        chunks = s_matrix.array.chunks[:-3] + ctf_chunks

        shape = tuple(len(c) for c, p in zip(scan_chunks, s_matrix.periodic))
        blocks = np.zeros((1,) * len(s_matrix.array.shape[:-3]) + shape, dtype=object)

        for indices, _, sub_scan in scan.generate_blocks(scan_chunks):

            if len(sub_scan) == 0:
                blocks.itemset(
                    (0,) * len(s_matrix.array.shape[:-3]) + indices,
                    da.zeros(
                        (0,) * len(blocks.shape),
                        dtype=np.complex64,
                    ),
                )
                continue

            slics = (slice(None),) * (len(s_matrix.shape) - 2)
            window_offset = list(old_window_offset)
            for l, k in enumerate(indices):
                if not (k > 0 and k < len(s_matrix.array.chunks[-2 + l]) - 1):
                    if s_matrix.periodic[l]:
                        slics += (slice(None),)
                    else:
                        raise RuntimeError()
                else:
                    slics += (slice(k - 1, k + 2),)

                window_offset[l] += sum(s_matrix.array.chunks[-2 + l][: k - 1])

            new_chunks = chunks + sub_scan.shape

            kwargs["window_offset"] = tuple(window_offset)

            new_block = s_matrix.array.blocks[slics].rechunk(
                s_matrix.array.chunks[:-2] + (-1, -1)
            )

            if len(scan.shape) == 1:
                drop_axis = (len(self.shape) - 3, len(self.shape) - 1)
            elif len(scan.shape) == 2:
                drop_axis = (len(self.shape) - 3,)
            else:
                raise NotImplementedError

            new_block = da.map_blocks(
                self._lazy_reduce,
                new_block,
                scan=sub_scan,
                drop_axis=drop_axis,
                chunks=new_chunks,
                waves_partial=waves_partial,
                kwargs=kwargs,
                ensemble_axes_metadata=ensemble_axes_metadata,
                ctf=ctf,
                detectors=detectors,
                max_batch_reduction=max_batch_reduction,
                meta=np.array((), dtype=np.complex64),
            )

            blocks.itemset((0,) * len(s_matrix.array.shape[:-3]) + indices, new_block)

        array = da.block(blocks.tolist())

        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        # if len(scan.ensemble_shape) != 2:
        #     array = array.reshape(
        #         array.shape[:-2] + (array.shape[-2] * array.shape[-1],)
        #     )

        measurements = _finalize_lazy_measurements(
            array,
            waves=dummy_probes,
            detectors=detectors,
            extra_ensemble_axes_metadata=self.ensemble_axes_metadata,
        )

        measurements = (
            (measurements,) if not isinstance(measurements, tuple) else measurements
        )

        return measurements

    def reduce(
        self,
        scan: BaseScan = None,
        ctf: CTF = None,
        detectors: Union[BaseDetector, List[BaseDetector]] = None,
        max_batch_reduction: Union[int, str] = "auto",
        rechunk: Union[Tuple[int, int], str] = "auto",
    ):

        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : List of Detector objects
            The detectors recording the measurements.
        scan : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function. Default is None (aperture is set by the planewave cutoff).
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. If 'auto' (default), the batch size
            is automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : two int or str, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly partitioned chunks.
            Should be equal to or greater than the interpolation.
        """

        self.accelerator.check_is_defined()

        if ctf is None:
            ctf = CTF(energy=self.waves.energy, semiangle_cutoff=self.planewave_cutoff)

        if ctf.semiangle_cutoff == np.inf:
            ctf.semiangle_cutoff = self.planewave_cutoff

        squeeze_scan = False
        if scan is None:
            squeeze_scan = True
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = _validate_scan(
            scan, Probe._from_ctf(extent=self.extent, ctf=ctf, energy=self.energy)
        )

        detectors = _validate_detectors(detectors)

        max_batch_reduction = self._validate_max_batch_reduction(
            scan, max_batch_reduction
        )

        if self.is_lazy:

            # s_matrix = self.rechunk_planewaves(rechunk_scheme=rechunk_scheme)

            measurements = self._index_overlap_reduce(
                scan, detectors, ctf, rechunk, max_batch_reduction
            )

            # chunks = s_matrix.array.chunks
            #
            # boundary = ["none", "none"] + [
            #     "periodic" if len(c) > 1 else "none" for c in chunks
            # ]
            #
            # scan, scan_chunks = scan.sort_into_extents(self._chunk_extents())
            #
            # chunks = s_matrix.chunks[:-3] + (1,) * len(ctf.ensemble_shape) + (1, 1)
            #
            # kwargs = {
            #     "wave_vectors": self.wave_vectors,
            #     "planewave_cutoff": self.planewave_cutoff,
            #     "device": self.device,
            #     "cropping_window": self.cropping_window,
            #     "interpolation": self.interpolation,
            # }
            #
            # smatrix_array_partial = partial(
            #     self._smatrix_array_partial,
            #     waves_partial=self.waves.from_partitioned_args(),
            #     ensemble_axes_metadata=self.waves.ensemble_axes_metadata,
            #     kwargs=kwargs,
            # )
            #
            # depth = self._overlap_depth()
            # depth[2] = 0
            #
            # # with dask.annotate(priority=-1):
            #
            # array = da.map_overlap(
            #     self._lazy_reduce,
            #     s_matrix.array,
            #     scan=scan,
            #     scan_chunks=scan_chunks,
            #     drop_axis=len(self.shape) - 3,
            #     align_arrays=False,
            #     allow_rechunk=False,
            #     chunks=chunks,
            #     depth=depth,
            #     s_matrix_partial=smatrix_array_partial,
            #     window_overlap=self._window_overlap(),
            #     ctf=ctf,
            #     detectors=detectors,
            #     max_batch_reduction=max_batch_reduction,
            #     trim=False,
            #     boundary=boundary,
            #     meta=np.array((), dtype=np.complex64),
            # )

            #     dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)
            #
            #     if len(scan.ensemble_shape) != 2:
            #         array = array.reshape(
            #             array.shape[:-2] + (array.shape[-2] * array.shape[-1],)
            #         )
            #
            #     ctf_chunks = tuple((n,) for n in ctf.ensemble_shape)
            #
            #     chunks = self.chunks[:-3] + ctf_chunks + scan_chunks
            #
            #     measurements = _finalize_lazy_measurements(
            #         array,
            #         waves=dummy_probes,
            #         detectors=detectors,
            #         extra_ensemble_axes_metadata=self.ensemble_axes_metadata,
            #         chunks=chunks,
            #     )
            #
            #     measurements = (
            #         (measurements,) if not isinstance(measurements, tuple) else measurements
            #     )
            #
        else:
            measurements = self._batch_reduce_to_measurements(
                scan, ctf, detectors, max_batch_reduction
            )

        if squeeze_scan:
            if isinstance(measurements, tuple):
                measurements = tuple(
                    measurement.squeeze((-3,)) for measurement in measurements
                )
            else:
                measurements = measurements.squeeze((-3,))

        return (
            measurements[0] if len(measurements) == 1 else ComputableList(measurements)
        )

    def scan(
        self,
        scan: BaseScan = None,
        detectors: Union[BaseDetector, List[BaseDetector]] = None,
        ctf: CTF = None,
        max_batch_reduction: Union[int, str] = "auto",
        rechunk: Union[Tuple[int, int], str] = "auto",
    ):
        """
        Reduce the SMatrix using coefficients calculated by a BaseScan and a CTF, to obtain the exit wave functions
        at given initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion coefficients in the reduction of the
            SMatrix.
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. If 'auto' (default), the batch size
            is automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : str or tuple of int
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """
        if scan is None:
            scan = GridScan()

        if detectors is None:
            detectors = [FlexibleAnnularDetector()]

        return self.reduce(
            scan=scan,
            ctf=ctf,
            detectors=detectors,
            max_batch_reduction=max_batch_reduction,
            rechunk=rechunk,
        )


class SMatrix(BaseSMatrix):
    """
    The scattering matrix is used for simulating STEM experiments using the PRISM algorithm.

    Parameters
    ----------
    planewave_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    potential : Atoms or AbstractPotential, optional
        Atoms or a potential that the scattering matrix represents. If given as atoms, a default potential will be created.
        If nothing is provided the scattering matrix will represent a vacuum potential, in which case the sampling and extent
        must be provided.
    gpts : one or two int, optional
        Number of grid points describing the wave functions. Provide only if potential is not given.
    sampling : one or two float, optional
        Lateral sampling of wave functions [1 / Å]. Provide only if potential is not given. Will be ignored if 'gpts'
        is also provided.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions (default is 1, ie. no interpolation). If a single value is
        provided, assumed to be the same for both directions.
    normalize : {'probe', 'planewaves'}
        Normalization of the scattering matrix. The default 'probe' is standard S matrix formalism, whereby the sum of
        all waves in the PRISM expansion is equal to 1; 'planewaves' is needed for core-loss calculations.
    downsample : {'cutoff', 'valid'} or float or bool
        Controls whether to downsample the probe wave functions after each run of the multislice algorithm.

            ``cutoff`` :
                Downsample to the antialias cutoff scattering angle (default).

            ``valid`` :
                Downsample to the largest rectangle that fits inside the circle with a radius defined by the antialias cutoff
                scattering angle.

            float :
                Downsample to a specified maximum scattering angle [mrad].
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu'). Default is 'cpu'. The default is determined by the user configuration.
    store_on_host : bool, optional
        If True, store the scattering matrix in host (cpu) memory so that the necessary memory is transferred as chunks to
        the device to run calculations (default is False).
    """

    def __init__(
        self,
        planewave_cutoff: float,
        energy: float,
        potential: Union[Atoms, BasePotential] = None,
        gpts: Union[int, Tuple[int, int]] = None,
        sampling: Union[float, Tuple[float, float]] = None,
        extent: Union[float, Tuple[float, float]] = None,
        interpolation: Union[int, Tuple[int, int]] = 1,
        normalize: str = "probe",
        downsample: Union[bool, str] = "cutoff",
        # tilt: Tuple[float, float] = (0.0, 0.0),
        device: str = None,
        store_on_host: bool = False,
    ):

        if downsample is True:
            downsample = "cutoff"

        self._device = validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self.set_potential(potential)

        self._interpolation = _validate_interpolation(interpolation)
        self._planewave_cutoff = planewave_cutoff
        self._downsample = downsample

        self._accelerator = Accelerator(energy=energy)
        # self._beam_tilt = BeamTilt(tilt=tilt)

        self._normalize = normalize
        self._store_on_host = store_on_host

        assert planewave_cutoff > 0.0

        if not all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
            warnings.warn(
                "The interpolation factor does not exactly divide 'gpts', normalization may not be exactly preserved."
            )

    def set_potential(self, potential):
        self._potential = _validate_potential(potential)

        if self._potential is not None:
            self.grid.match(self._potential)
            self._grid = self._potential.grid
        else:
            try:
                self.grid.check_is_defined()
            except GridUndefinedError:
                raise ValueError("Provide a potential or provide 'extent' and 'gpts'.")

    @property
    def tilt(self):
        return 0.0, 0.0

    def round_gpts_to_interpolation(self) -> "SMatrix":
        """
        Round the gpts of the SMatrix to the closest multiple of the interpolation factor.

        Returns
        -------
        s_matrix_with_rounded_gpts : SMatrix
        """

        rounded = _round_gpts_to_multiple_of_interpolation(
            self.gpts, self.interpolation
        )
        if rounded == self.gpts:
            return self

        self.gpts = rounded
        return self

    @property
    def downsample(self) -> Union[str, bool]:
        return self._downsample

    @property
    def store_on_host(self) -> bool:
        return self._store_on_host

    @property
    def metadata(self):
        return {"energy": self.energy}

    @property
    def shape(self) -> Tuple[int, ...]:
        return (len(self),) + self.gpts

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        if self.potential is not None:
            return self.potential.ensemble_shape
        else:
            return ()

    @property
    def wave_vectors(self) -> List[Tuple[float, float]]:
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
        xp = np if self.store_on_host else get_array_module(self.device)
        wave_vectors = prism_wave_vectors(
            self.planewave_cutoff, self.extent, self.energy, self.interpolation, xp=xp
        )
        return wave_vectors  # _validate_wave_vectors(wave_vectors)

    @property
    def potential(self) -> BasePotential:
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = potential
        self._grid = potential.grid

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def planewave_cutoff(self) -> float:
        """Plane-wave expansion cutoff."""
        return self._planewave_cutoff

    @planewave_cutoff.setter
    def planewave_cutoff(self, value: float):
        self._planewave_cutoff = value

    @property
    def interpolation(self) -> Tuple[int, int]:
        """Interpolation factor."""
        return self._interpolation

    @property
    def interpolated_gpts(self) -> Tuple[int, int]:
        return (
            self.gpts[0] // self.interpolation[0],
            self.gpts[1] // self.interpolation[0],
        )

    def _wave_vector_chunks(self, max_batch):
        if isinstance(max_batch, int):
            max_batch = max_batch * reduce(operator.mul, self.gpts)

        chunks = validate_chunks(
            shape=(len(self),) + self.gpts,
            chunks=("auto", -1, -1),
            limit=max_batch,
            dtype=np.dtype("complex64"),
            device=self.device,
        )
        return chunks

    @property
    def downsampled_gpts(self) -> Tuple[int, int]:
        if self.downsample:
            downsampled_gpts = self._gpts_within_angle(self.downsample)
            return _round_gpts_to_multiple_of_interpolation(
                downsampled_gpts, self.interpolation
            )
        else:
            return self.gpts

    def _build_s_matrix(self, wave_vector_range=slice(None)):

        xp = get_array_module(self.device)

        wave_vectors = xp.asarray(self.wave_vectors, dtype=xp.float32)

        array = plane_waves(wave_vectors[wave_vector_range], self.extent, self.gpts)

        # if all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
        # normalization_constant = (
        #    np.prod(self.gpts)
        #    * xp.sqrt(len(wave_vectors))
        #    / np.prod(self.interpolation)
        # )

        # corner = cropping_window[0] // 2, cropping_window[1] // 2
        # cropped_array = wrapped_crop_2d(array, corner, cropping_window)
        # normalization_constant = xp.sqrt(abs2(fft2(cropped_array.sum(0))).sum())

        cropping_window = (
            self.gpts[0] / self.interpolation[0],
            self.gpts[1] / self.interpolation[1],
        )
        normalization_constant = np.prod(cropping_window) * xp.sqrt(len(wave_vectors))

        array = array / normalization_constant.astype(xp.float32)

        waves = Waves(
            array,
            energy=self.energy,
            extent=self.extent,
            ensemble_axes_metadata=[
                OrdinalAxis(values=wave_vectors[wave_vector_range])
            ],
        )

        if self.potential is not None:
            waves = multislice_and_detect(waves, self.potential, [WavesDetector()])[0]

        if self.downsampled_gpts != self.gpts:
            waves = waves.downsample(
                gpts=self.downsampled_gpts, normalization="intensity"
            )

        if self.store_on_host and self.device == "gpu":
            with cp.cuda.Stream():
                waves._array = cp.asnumpy(waves.array)

        return waves

    @property
    def cropping_window(self):
        # print((self.downsampled_gpts[0] / self.interpolation[0] / 2) * 2)
        return (
            safe_ceiling_int(self.downsampled_gpts[0] / self.interpolation[0]),
            safe_ceiling_int(self.downsampled_gpts[1] / self.interpolation[1]),
        )

    @staticmethod
    def _wrapped_build_s_matrix(*args, s_matrix_partial):
        s_matrix = s_matrix_partial(*tuple(arg.item() for arg in args[:-1]))

        wave_vector_range = slice(*np.squeeze(args[-1]))
        array = s_matrix._build_s_matrix(wave_vector_range).array
        return array

    def _s_matrix_partial(self):
        def s_matrix(*args, potential_partial, **kwargs):
            if potential_partial is not None:
                potential = potential_partial(*args + (np.array([None], dtype=object),))
            else:
                potential = None
            return SMatrix(potential=potential, **kwargs)

        potential_partial = (
            self.potential._from_partitioned_args()
            if self.potential is not None
            else None
        )
        return partial(
            s_matrix,
            potential_partial=potential_partial,
            **self._copy_kwargs(exclude=("potential",))
        )

    def dummy_probes(self, scan=None, ctf=None):
        return self.build(lazy=True).dummy_probes(scan=scan, ctf=ctf)

    def multislice(
        self,
        potential=None,
        lazy: bool = None,
        max_batch: Union[int, str] = "auto",
    ):
        """


        Parameters
        ----------
        potential
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If not given, defaults to the
            setting in the user configuration file.
        max_batch : int or str, optional
            The number of expansion plane waves in each run of the multislice algorithm.

        Returns
        -------

        """
        s_matrix = self.__class__(
            potential=potential, **self._copy_kwargs(exclude=("potential",))
        )
        return s_matrix.build(lazy=lazy, max_batch=max_batch)

    def build(
        self, lazy: bool = None, max_batch: Union[int, str] = "auto"
    ) -> SMatrixArray:
        """
        Build the plane waves of the scattering matrix and propagate them through the potential using the
        multislice algorithm.

        Parameters
        ----------
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly. If not given, defaults to the
            setting in the user configuration file.
        max_batch : int or str, optional
            The number of expansion plane waves in each run of the multislice algorithm.

        Returns
        -------
        s_matrix_array : SMatrixArray
            The built scattering matrix.
        """
        lazy = validate_lazy(lazy)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        downsampled_gpts = self.downsampled_gpts

        wave_vector_blocks = np.array(chunk_ranges(wave_vector_chunks)[0], dtype=int)

        if self.potential is not None:
            if len(self.potential.exit_planes) > 1:
                raise NotImplementedError(
                    "Thickness series not yet implemented for PRISM."
                )

            potential_blocks = self.potential._partition_args()[0], (0,)
            ensemble_axes_metadata = self.potential.ensemble_axes_metadata

            if self.potential.ensemble_shape:
                symbols = tuple(range(4))
            else:
                symbols = tuple(range(1, 4))

            wave_vector_blocks = np.tile(
                wave_vector_blocks[None], (len(potential_blocks[0]), 1, 1)
            )

            wave_vector_blocks = da.from_array(
                wave_vector_blocks, chunks=(1, 1, 2), name=False
            )

            blocks = potential_blocks + (wave_vector_blocks, (0, 1, -1))

            adjust_chunks = {1: wave_vector_chunks[0]}
            new_axes = {2: (downsampled_gpts[0],), 3: (downsampled_gpts[1],)}
        else:
            potential_blocks = ()
            ensemble_axes_metadata = []
            symbols = tuple(range(0, 3))
            adjust_chunks = {0: wave_vector_chunks[0]}
            new_axes = {1: (downsampled_gpts[0],), 2: (downsampled_gpts[1],)}
            wave_vector_blocks = da.from_array(
                wave_vector_blocks, chunks=(1, 2), name=False
            )
            blocks = potential_blocks + (wave_vector_blocks, (0, -1))

        if lazy:
            xp = get_array_module(self.device)

            arr = da.blockwise(
                self._wrapped_build_s_matrix,
                symbols,
                *blocks,
                new_axes=new_axes,
                adjust_chunks=adjust_chunks,
                concatenate=True,
                meta=xp.array((), dtype=np.complex64),
                **{"s_matrix_partial": self._s_matrix_partial()}
            )

            waves = Waves(
                arr,
                energy=self.energy,
                extent=self.extent,
                ensemble_axes_metadata=ensemble_axes_metadata
                + self.base_axes_metadata[:1],
            )
        else:
            waves = self._build_s_matrix()

        return SMatrixArray.from_waves(
            waves,
            wave_vectors=self.wave_vectors,
            interpolation=self.interpolation,
            planewave_cutoff=self.planewave_cutoff,
            cropping_window=self.cropping_window,
            device=self.device,
        )

    def scan(
        self,
        scan: Union[np.ndarray, BaseScan] = None,
        detectors: Union[BaseDetector, List[BaseDetector]] = None,
        ctf: Union[CTF, Dict] = None,
        max_batch_multislice: Union[str, int] = "auto",
        max_batch_reduction: Union[str, int] = "auto",
        rechunk: Union[Tuple[int, int], str] = "auto",
        lazy: bool = None,
    ) -> Union[BaseMeasurement, Waves, List[Union[BaseMeasurement, Waves]]]:

        """
        Run the multislice algorithm, then reduce the SMatrix using coefficients calculated by a BaseScan and a CTF,
        to obtain the exit wave functions at given initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion coefficients in the reduction of the
            SMatrix.
        max_batch_multislice : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. If 'auto' (default), the batch size
            is automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : str or tuple of int
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """

        if scan is None:
            scan = GridScan()

        return self.build(max_batch=max_batch_multislice, lazy=lazy).scan(
            scan=scan,
            detectors=detectors,
            max_batch_reduction=max_batch_reduction,
            ctf=ctf,
            rechunk=rechunk,
        )

    def reduce(
        self,
        scan: Union[np.ndarray, BaseScan] = None,
        detectors: Union[BaseDetector, List[BaseDetector]] = None,
        ctf: Union[CTF, Dict] = None,
        rechunk: str = "auto",
        max_batch_multislice: Union[str, int] = "auto",
        max_batch_reduction: Union[str, int] = "auto",
        lazy: bool = None,
    ):

        """
        Run the multislice algorithm, then reduce the SMatrix using coefficients calculated by a BaseScan and a CTF,
        to obtain the exit wave functions at given initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be converted to measurements after
            running the multislice algorithm. See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion coefficients in the reduction of the
            SMatrix.
        max_batch_multislice : int, optional
            The number of wave functions in each chunk of the Dask array. If 'auto' (default), the batch size is
            automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions better utilize thread
            parallelization, but requires more memory and floating point operations. If 'auto' (default), the batch size
            is automatically chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : str or tuple of int
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        detected_waves : BaseMeasurement or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
        """

        return self.build(max_batch=max_batch_multislice, lazy=lazy).reduce(
            scan=scan,
            detectors=detectors,
            rechunk=rechunk,
            max_batch_reduction=max_batch_reduction,
            ctf=ctf,
        )
