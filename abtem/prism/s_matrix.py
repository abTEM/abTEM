"""Module describing the scattering matrix used in the PRISM algorithm."""

from __future__ import annotations

import inspect
import operator
import warnings
from abc import abstractmethod
from functools import partial, reduce

import dask.array as da
import numpy as np
from ase import Atoms
from dask.graph_manipulation import wait_on

from abtem.array import ArrayObject, ComputableList, validate_lazy
from abtem.core import config
from abtem.core.axes import (
    AxisMetadata,
    OrdinalAxis,
    ScanAxis,
    UnknownAxis,
    WaveVectorAxis,
)
from abtem.core.backend import copy_to_device, cp, get_array_module, validate_device
from abtem.core.chunks import Chunks, chunk_ranges, equal_sized_chunks, validate_chunks
from abtem.core.complex import complex_exponential
from abtem.core.diagnostics import TqdmWrapper
from abtem.core.energy import Accelerator
from abtem.core.ensemble import Ensemble, _wrap_with_array
from abtem.core.fft import fft2, ifft2
from abtem.core.grid import Grid, GridUndefinedError
from abtem.core.utils import (
    CopyMixin,
    EqualityMixin,
    ensure_list,
    expand_dims_to_broadcast,
    get_dtype,
    itemset,
    safe_ceiling_int,
    tuple_range,
)
from abtem.detectors import (
    AnnularDetector,
    BaseDetector,
    FlexibleAnnularDetector,
    PixelatedDetector,
    SegmentedDetector,
    WavesDetector,
    validate_detectors,
)
from abtem.measurements import BaseMeasurements
from abtem.multislice import allocate_multislice_measurements, multislice_and_detect
from abtem.potentials.iam import BasePotential, validate_potential
from abtem.prism.utils import batch_crop_2d, minimum_crop, plane_waves, wrapped_crop_2d
from abtem.scan import BaseScan, GridScan, validate_scan
from abtem.transfer import CTF
from abtem.waves import BaseWaves, Probe, Waves, _antialias_cutoff_gpts


def _extract_measurement(array, index):
    if array.size == 0:
        return array

    array = array.item()[index].array
    return array


def _wrap_measurements(measurements):
    return measurements[0] if len(measurements) == 1 else ComputableList(measurements)


def _finalize_lazy_measurements(
    arrays, waves, detectors, extra_ensemble_axes_metadata=None, chunks=None
):
    if extra_ensemble_axes_metadata is None:
        extra_ensemble_axes_metadata = []

    measurements = []
    for i, detector in enumerate(detectors):
        base_shape = detector._out_base_shape(waves)[0]

        if isinstance(detector, AnnularDetector):
            # TODO
            base_shape = ()

        meta = detector._out_meta(waves)[0]

        new_axis = tuple(range(len(arrays.shape), len(arrays.shape) + len(base_shape)))

        if chunks is None:
            chunks = arrays.chunks

        array = arrays.map_blocks(
            _extract_measurement,
            i,
            chunks=chunks + tuple((n,) for n in base_shape),
            new_axis=new_axis,
            meta=meta,
        )

        ensemble_axes_metadata = detector._out_ensemble_axes_metadata(waves)[0]

        base_axes_metadata = detector._out_base_axes_metadata(waves)[0]

        axes_metadata = ensemble_axes_metadata + base_axes_metadata

        metadata = detector._out_metadata(waves)[0]

        cls = detector._out_type(waves)[0]

        axes_metadata = extra_ensemble_axes_metadata + axes_metadata

        measurement = cls.from_array_and_metadata(
            array, axes_metadata=axes_metadata, metadata=metadata
        )

        if hasattr(measurement, "reduce_ensemble"):
            measurement = measurement.reduce_ensemble()

        measurements.append(measurement)

    return measurements


def _round_gpts_to_multiple_of_interpolation(
    gpts: tuple[int, int], interpolation: tuple[int, int]
) -> tuple[int, int]:
    return tuple(n + (-n) % f for f, n in zip(interpolation, gpts))  # noqa


class BaseSMatrix(BaseWaves):
    """Base class for scattering matrices."""

    _device: str
    ensemble_axes_metadata: list[AxisMetadata]
    ensemble_shape: tuple[int, ...]
    _base_dims = 3

    @property
    def device(self):
        """The device where the S-Matrix is created and reduced."""
        return self._device

    @property
    @abstractmethod
    def interpolation(self):
        """Interpolation factor in the `x` and `y` directions"""
        pass

    @property
    @abstractmethod
    def wave_vectors(self) -> np.ndarray:
        """The wave vectors corresponding to each plane wave."""
        pass

    @property
    @abstractmethod
    def semiangle_cutoff(self) -> float:
        """The radial cutoff of the plane-wave expansion [mrad]."""
        pass

    @property
    @abstractmethod
    def window_extent(self):
        """The cropping window extent of the waves."""
        pass

    @property
    @abstractmethod
    def window_gpts(self):
        """The number of grid points describing the cropping window of the wave
        functions."""
        pass

    def __len__(self) -> int:
        return len(self.wave_vectors)

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        wave_axes_metadata = super().base_axes_metadata
        return [
            WaveVectorAxis(
                label="q",
                values=tuple(tuple(value) for value in self.wave_vectors),
            ),
            wave_axes_metadata[0],
            wave_axes_metadata[1],
        ]

    def dummy_probes(
        self,
        scan: BaseScan = None,
        ctf: CTF = None,
        plane: str = "entrance",
        downsample: bool = True,
        **kwargs,
    ) -> Probe:
        """
        A probe or an ensemble of probes equivalent reducing the SMatrix at a single
        position.

        Parameters
        ----------
        scan : BaseScan
        ctf : CTF
        plane : str

        Returns
        -------
        dummy_probes : Probes
        """

        if ctf is None:
            ctf = CTF(energy=self.energy, semiangle_cutoff=self.semiangle_cutoff)
        elif isinstance(ctf, dict):
            ctf = CTF(energy=self.energy, semiangle_cutoff=self.semiangle_cutoff, **ctf)
        elif isinstance(ctf, CTF):
            ctf = ctf.copy()
        else:
            raise ValueError()

        if plane == "exit":
            defocus = 0.0
            if hasattr(self, "potential"):
                if self.potential is not None:
                    defocus = self.potential.thickness

            elif "accumulated_defocus" in self.metadata:
                defocus = self.metadata["accumulated_defocus"]

            ctf.defocus = ctf.defocus - defocus

        if ctf.semiangle_cutoff is None or ctf.semiangle_cutoff == np.inf:
            ctf.semiangle_cutoff = self.semiangle_cutoff

        default_kwargs = {"device": self.device, "metadata": {**self.metadata}}
        kwargs = {**default_kwargs, **kwargs}

        if downsample:
            window_gpts = self.window_gpts
        else:
            window_gpts = (
                safe_ceiling_int(self.gpts[0] / self.interpolation[0]),
                safe_ceiling_int(self.gpts[1] / self.interpolation[1]),
            )

        probes = Probe._from_ctf(
            extent=self.window_extent,
            gpts=window_gpts,
            ctf=ctf,
            energy=self.energy,
            **kwargs,
        )

        if scan is not None:
            probes.scan_positions = scan

        return probes


def _validate_interpolation(interpolation: int | tuple[int, int]):
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


def _chunked_axis(s_matrix_array):
    window_margin = s_matrix_array._window_margin

    argsort = np.argsort(
        (
            -s_matrix_array.gpts[0] // window_margin[0],
            -s_matrix_array.gpts[1] // window_margin[1],
        )
    )
    return int(argsort[0]), int(argsort[1])


def _chunks_for_multiple_rechunk_reduce(partitions):
    chunks_1 = ()
    chunk_indices_1 = ()
    for i in range(1, len(partitions) - 1, 3):
        chunks_1 += (sum(partitions[i - 1 : i + 2]),)
        chunk_indices_1 += (i - 1,)
    chunks_1 = chunks_1 + (sum(partitions[i + 2 :]),)

    assert sum(chunks_1) == sum(partitions)

    chunks_2 = (sum(partitions[:1]),)
    chunk_indices_2 = ()
    for i in range(2, len(partitions) - 1, 3):
        chunks_2 += (sum(partitions[i - 1 : i + 2]),)
        chunk_indices_2 += (i - 1,)
    chunks_2 = chunks_2 + (sum(partitions[i + 2 :]),)

    assert sum(chunks_2) == sum(partitions)

    chunks_3 = (sum(partitions[:2]),)
    chunk_indices_3 = ()
    for i in range(3, len(partitions) - 1, 3):
        chunks_3 += (sum(partitions[i - 1 : i + 2]),)
        chunk_indices_3 += (i - 1,)
    chunks_3 = chunks_3 + (sum(partitions[i + 2 :]),)

    assert sum(chunks_3) == sum(partitions)
    assert len(chunk_indices_1 + chunk_indices_2 + chunk_indices_3) == (
        len(partitions) - 2
    )

    chunks = (chunks_1, chunks_2, chunks_3)
    chunk_indices = (chunk_indices_1, chunk_indices_2, chunk_indices_3)

    return chunks, chunk_indices


def _lazy_reduce(
    array: np.ndarray,
    waves_partial: partial,
    ensemble_axes_metadata: list[AxisMetadata],
    from_waves_kwargs: dict,
    scan: BaseScan,
    ctf: CTF,
    detectors: list[BaseDetector],
    max_batch_reduction: int,
    pbar: bool = False,
) -> np.ndarray:
    args = (array, ensemble_axes_metadata)
    waves = waves_partial(args).item()

    s_matrix = SMatrixArray._from_waves(waves, **from_waves_kwargs)

    measurements = s_matrix._batch_reduce_to_measurements(
        scan, ctf, detectors, max_batch_reduction, pbar
    )

    n = len(array.shape) - 3 + len(scan.shape) + len(ctf.ensemble_shape)
    arr = np.zeros((1,) * n, dtype=object)
    itemset(arr, 0, measurements)
    return arr


def _map_blocks(array, scans, block_indices, window_offset=(0, 0), **kwargs):
    ctf_chunks = tuple((n,) for n in kwargs["ctf"].ensemble_shape)

    blocks = ()
    for i, scan in zip(block_indices, scans):
        block = array.blocks[(slice(None),) * (len(array.shape) - 2) + i]

        new_chunks = array.chunks[:-3] + ctf_chunks + scan.shape

        kwargs["from_waves_kwargs"]["window_offset"] = (
            window_offset[0] + sum(array.chunks[-2][: i[0]]),
            window_offset[1] + sum(array.chunks[-1][: i[1]]),
        )

        if len(scan.shape) == 1:
            drop_axis = (len(array.shape) - 3, len(array.shape) - 1)
        elif len(scan.shape) == 2:
            drop_axis = (len(array.shape) - 3,)
        else:
            raise NotImplementedError

        drop_axis = (len(array.shape) - 3, len(array.shape) - 2, len(array.shape) - 1)

        new_axis = tuple(
            i
            for i in range(
                len(array.shape) - 3,
                len(array.shape) - 3 + len(scan.shape) + len(ctf_chunks),
            )
        )

        block = da.map_blocks(
            _lazy_reduce,
            block,
            scan=scan,
            drop_axis=drop_axis,
            new_axis=new_axis,
            chunks=new_chunks,
            **kwargs,
            meta=np.array((), dtype=np.complex64),
        )

        if len(scan) == 0:
            block = da.zeros(
                (0,) * len(block.shape),
                dtype=np.complex64,
            )

        blocks += (block,)

    return blocks


def _tuple_from_index_value_pairs(*args):
    temp_list = [None] * (len(args) // 2)

    for arg1, arg2 in zip(args[::2], args[1::2]):
        temp_list[arg1] = arg2

    return tuple(temp_list)


def _multiple_rechunk_reduce(
    s_matrix_array: SMatrixArray,
    scan: BaseScan,
    detectors: list[BaseDetector],
    ctf: CTF,
    max_batch_reduction: int,
    pbar: bool = False,
):
    assert np.all(s_matrix_array.periodic)

    window_margin = s_matrix_array._window_margin
    chunked_axis, nochunks_axis = _chunked_axis(s_matrix_array)

    pad_amounts = _tuple_from_index_value_pairs(
        chunked_axis, (window_margin[chunked_axis],) * 2, nochunks_axis, (0, 0)
    )
    s_matrix_array = s_matrix_array._pad(pad_amounts)

    chunk_size = window_margin[chunked_axis]

    size = s_matrix_array.shape[-2:][chunked_axis] - window_margin[chunked_axis] * 2

    num_chunks = -(size // -chunk_size)

    partitions = _tuple_from_index_value_pairs(
        chunked_axis,
        (chunk_size,) * num_chunks,
        nochunks_axis,
        (s_matrix_array.shape[-2:][nochunks_axis],),
    )

    chunk_extents = tuple(
        tuple(((cc[0]) * d, (cc[1]) * d) for cc in c)
        for c, d in zip(chunk_ranges(partitions), s_matrix_array.sampling)
    )

    scan, scan_chunks = scan._sort_into_extents(chunk_extents)

    scans = [
        (indices, scan.item()) for indices, _, scan in scan.generate_blocks(scan_chunks)
    ]

    partitions = (pad_amounts[chunked_axis][0],) + partitions[chunked_axis]
    partitions = partitions + (
        s_matrix_array.shape[len(s_matrix_array.shape) - 2 + chunked_axis]
        - sum(partitions),
    )

    (
        (chunks_1, chunks_2, chunks_3),
        (
            scan_indices_1,
            scan_indices_2,
            scan_indices_3,
        ),
    ) = _chunks_for_multiple_rechunk_reduce(partitions)

    chunks_1 = (
        s_matrix_array.array.chunks[:-3]
        + (-1,)
        + _tuple_from_index_value_pairs(chunked_axis, chunks_1, nochunks_axis, -1)
    )
    chunks_2 = (
        s_matrix_array.array.chunks[:-3]
        + (-1,)
        + _tuple_from_index_value_pairs(chunked_axis, chunks_2, nochunks_axis, -1)
    )
    chunks_3 = (
        s_matrix_array.array.chunks[:-3]
        + (-1,)
        + _tuple_from_index_value_pairs(chunked_axis, chunks_3, nochunks_axis, -1)
    )

    shape = tuple(len(c) for c in scan_chunks)
    blocks = np.zeros(shape, dtype=object)

    kwargs = {
        "waves_partial": s_matrix_array.waves._from_partitioned_args(),
        "ensemble_axes_metadata": s_matrix_array.waves.ensemble_axes_metadata,
        "from_waves_kwargs": s_matrix_array._copy_kwargs(exclude=("array", "extent")),
        "ctf": ctf,
        "detectors": detectors,
        "max_batch_reduction": max_batch_reduction,
        "pbar": pbar,
    }

    array = s_matrix_array.array.rechunk(chunks_1)

    window_offset = s_matrix_array.window_offset

    block_indices = [
        _tuple_from_index_value_pairs(chunked_axis, i, nochunks_axis, 0)
        for i in range(len(scan_indices_1))
    ]

    new_blocks = _map_blocks(
        array,
        [scans[i][1] for i in scan_indices_1],
        block_indices,
        window_offset=window_offset,
        **kwargs,
    )

    for i, block in zip(scan_indices_1, new_blocks):
        itemset(blocks, scans[i][0], block)

    if s_matrix_array.ensemble_shape:
        fp_arrays = []
        for i in np.ndindex(s_matrix_array.ensemble_shape):
            try:
                fp_new_blocks = tuple(block[i] for block in new_blocks)
                fp_array = wait_on(array[i], *fp_new_blocks)[0]
                fp_arrays.append(fp_array)
            except IndexError:
                fp_arrays.append(array[i])

        array = da.stack(fp_arrays, axis=0)

    array = array.rechunk(chunks_2)

    block_indices = [
        _tuple_from_index_value_pairs(chunked_axis, i, nochunks_axis, 0)
        for i in range(1, len(scan_indices_2) + 1)
    ]

    new_blocks = _map_blocks(
        array,
        [scans[i][1] for i in scan_indices_2],
        block_indices,
        window_offset=window_offset,
        **kwargs,
    )

    for i, block in zip(scan_indices_2, new_blocks):
        itemset(blocks, scans[i][0], block)

    if s_matrix_array.ensemble_shape:
        fp_arrays = []
        for i in np.ndindex(s_matrix_array.ensemble_shape):
            try:
                fp_new_blocks = tuple(block[i] for block in new_blocks)
                fp_array = wait_on(array[i], *fp_new_blocks)[0]
                fp_arrays.append(fp_array)
            except IndexError:
                fp_arrays.append(array[i])

        array = da.stack(fp_arrays, axis=0)

    array = array.rechunk(chunks_3)

    block_indices = [
        _tuple_from_index_value_pairs(chunked_axis, i, nochunks_axis, 0)
        for i in range(1, len(scan_indices_3) + 1)
    ]

    new_blocks = _map_blocks(
        array,
        [scans[i][1] for i in scan_indices_3],
        block_indices,
        window_offset=window_offset,
        **kwargs,
    )

    for i, block in zip(scan_indices_3, new_blocks):
        itemset(blocks, scans[i][0], block)

    array = da.block(blocks.tolist())

    dummy_probes = s_matrix_array.dummy_probes(scan=scan, ctf=ctf)

    measurements = _finalize_lazy_measurements(
        array,
        waves=dummy_probes,
        detectors=detectors,
        extra_ensemble_axes_metadata=s_matrix_array.ensemble_axes_metadata,
    )

    return measurements


def _single_rechunk_reduce(
    s_matrix_array: "SMatrixArray",
    scan: BaseScan,
    detectors: list[BaseDetector],
    ctf: CTF,
    max_batch_reduction: int,
):
    chunked_axis, nochunks_axis = _chunked_axis(s_matrix_array)

    num_chunks = (
        s_matrix_array.gpts[chunked_axis] // s_matrix_array._window_margin[chunked_axis]
    )

    chunks = equal_sized_chunks(
        s_matrix_array.shape[-2:][chunked_axis], num_chunks=num_chunks
    )

    assert np.all(np.array(chunks) > s_matrix_array._window_margin[chunked_axis])

    chunks = (
        s_matrix_array.array.chunks[:-3]
        + (-1,)
        + _tuple_from_index_value_pairs(chunked_axis, chunks, nochunks_axis, -1)
    )

    array = s_matrix_array._array.rechunk(chunks)

    assert all(s_matrix_array.periodic)

    # chunk_extents = tuple(
    #     tuple(((cc[0]) * d, (cc[1]) * d) for cc in c)
    #     for c, d in zip(chunk_ranges(array.chunks[-2:]), s_matrix_array.sampling)
    # )
    chunk_extents_x = tuple(
        ((cc[0]) * s_matrix_array.sampling[0], (cc[1]) * s_matrix_array.sampling[0])
        for cc in array.chunks[-2]
    )
    chunk_extents_y = tuple(
        ((cc[0]) * s_matrix_array.sampling[1], (cc[1]) * s_matrix_array.sampling[1])
        for cc in array.chunks[-1]
    )

    chunk_extents = (chunk_extents_x, chunk_extents_y)
    scan, scan_chunks = scan._sort_into_extents(chunk_extents)

    ctf_chunks = tuple((n,) for n in ctf.ensemble_shape)
    chunks = array.chunks[:-3] + ctf_chunks

    shape = tuple(len(c) for c, p in zip(scan_chunks, s_matrix_array.periodic))
    blocks = np.zeros((1,) * len(array.shape[:-3]) + shape, dtype=object)

    kwargs = {
        "waves_partial": s_matrix_array.waves._from_partitioned_args(),
        "ensemble_axes_metadata": s_matrix_array.waves.ensemble_axes_metadata,
        "from_waves_kwargs": s_matrix_array._copy_kwargs(exclude=("array", "extent")),
        "ctf": ctf,
        "detectors": detectors,
        "max_batch_reduction": max_batch_reduction,
    }

    for indices, _, sub_scan in scan.generate_blocks(scan_chunks):
        sub_scan = sub_scan.item()

        if len(sub_scan) == 0:
            itemset(
                blocks,
                (0,) * len(array.shape[:-3]) + indices,
                da.zeros(
                    (0,) * len(blocks.shape),
                    dtype=np.complex64,
                ),
            )
            continue

        slics = (slice(None),) * (len(array.shape) - 2)
        window_offset = ()
        for i, k in enumerate(indices):
            if len(array.chunks[-2:][i]) > 1:
                slics += ([k - 1, k, (k + 1) % len(array.chunks[-2:][i])],)
                window_offset += (
                    sum(array.chunks[-2:][i][:k]) - array.chunks[-2:][i][k - 1],
                )

            else:
                slics += (slice(None),)
                window_offset += (0,)

        new_block = array.blocks[slics]
        new_block = new_block.rechunk(array.chunks[:-2] + (-1, -1))
        new_chunks = chunks + sub_scan.shape

        kwargs["from_waves_kwargs"]["window_offset"] = tuple(window_offset)

        if len(scan.shape) == 1:
            drop_axis = (len(array.shape) - 3, len(array.shape) - 1)
        elif len(scan.shape) == 2:
            drop_axis = (len(array.shape) - 3,)
        else:
            raise NotImplementedError

        new_block = da.map_blocks(
            _lazy_reduce,
            new_block,
            scan=sub_scan,
            drop_axis=drop_axis,
            chunks=new_chunks,
            **kwargs,
            meta=np.array((), dtype=np.complex64),
        )

        itemset(blocks, (0,) * len(array.shape[:-3]) + indices, new_block)

    array = da.block(blocks.tolist())

    dummy_probes = s_matrix_array.dummy_probes(scan=scan, ctf=ctf)

    measurements = _finalize_lazy_measurements(
        array,
        waves=dummy_probes,
        detectors=detectors,
        extra_ensemble_axes_metadata=s_matrix_array.ensemble_axes_metadata,
    )

    return measurements


def _no_chunks_reduce(
    s_matrix_array: "SMatrixArray",
    scan: BaseScan,
    detectors: list[BaseDetector],
    ctf: CTF,
    max_batch_reduction: int = 1,
    pbar: bool = False,
):
    kwargs = {
        "waves_partial": s_matrix_array.waves._from_partitioned_args(),
        "ensemble_axes_metadata": s_matrix_array.waves.ensemble_axes_metadata,
        "from_waves_kwargs": s_matrix_array._copy_kwargs(exclude=("array", "extent")),
        "ctf": ctf,
        "detectors": detectors,
        "max_batch_reduction": max_batch_reduction,
        "pbar": pbar,
    }

    array = s_matrix_array.array

    ctf_chunks = tuple((n,) for n in ctf.ensemble_shape)

    chunks = array.chunks[:-3] + ctf_chunks + scan.shape

    drop_axis = (len(array.shape) - 3, len(array.shape) - 2, len(array.shape) - 1)

    new_axis = tuple(
        i
        for i in range(
            len(array.shape) - 3,
            len(array.shape) - 3 + len(scan.shape) + len(ctf_chunks),
        )
    )

    array = da.map_blocks(
        _lazy_reduce,
        array,
        scan=scan,
        drop_axis=drop_axis,
        new_axis=new_axis,
        chunks=chunks,
        **kwargs,
        meta=np.array((), dtype=np.complex64),
    )

    dummy_probes = s_matrix_array.dummy_probes(scan=scan, ctf=ctf)

    measurements = _finalize_lazy_measurements(
        array,
        waves=dummy_probes,
        detectors=detectors,
        extra_ensemble_axes_metadata=s_matrix_array.ensemble_axes_metadata,
    )
    return measurements


class SMatrixArray(BaseSMatrix, ArrayObject):
    """
    A scattering matrix defined by a given array of dimension 3, where the first indexes
    the probe plane waves and the latter two are the `y` and `x` scan directions.

    Parameters
    ----------
    array : np.ndarray
        Array defining the scattering matrix. Must be 3D or higher, dimensions before
        the last three dimensions should represent ensemble dimensions, the next
        dimension indexes the plane waves and the last two dimensions represent the
        spatial extent of the plane waves.
    wave_vectors : np.ndarray
        Array defining the wave vectors corresponding to each plane wave.
        Must have shape Nx2, where N is equal to the number of plane waves.
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    sampling : one or two float, optional
        Lateral sampling of wave functions [Å]. Provide only if potential is not given.
        Will be ignored if 'gpts' is also provided.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions
        (default is 1, ie. no interpolation). If a single value is provided, assumed to
        be the same for both directions.
    window_gpts : tuple of int
        The number of grid points describing the cropping window of the wave functions.
    window_offset : tuple of int
        The number of grid points from the origin the cropping windows of the wave
        functions is displaced.
    periodic: tuple of bool
        Specifies whether the SMatrix should be assumed to be periodic along the x and
        y-axis.
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu').
        Default is 'cpu'. The default is determined by the user configuration.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with
        the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the
        metadata of measurements derived from the waves.
    """

    def __init__(
        self,
        array: np.ndarray,
        wave_vectors: np.ndarray,
        semiangle_cutoff: float,
        energy: float = None,
        interpolation: int | tuple[int, int] = (1, 1),
        sampling: float | tuple[float, float] = None,
        extent: float | tuple[float, float] = None,
        window_gpts: tuple[int, int] = (0, 0),
        window_offset: tuple[int, int] = (0, 0),
        periodic: tuple[bool, bool] = (True, True),
        device: str = None,
        ensemble_axes_metadata: list[AxisMetadata] = None,
        metadata: dict = None,
    ):
        self._grid = Grid(
            extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True
        )
        self._accelerator = Accelerator(energy=energy)
        self._wave_vectors = wave_vectors

        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

        self._semiangle_cutoff = semiangle_cutoff
        self._window_gpts = tuple(window_gpts)
        self._window_offset = tuple(window_offset)
        self._interpolation = _validate_interpolation(interpolation)
        self._device = device
        self._periodic = periodic

    @classmethod
    def _pack_kwargs(cls, kwargs):
        kwargs["wave_vectors"] = _pack_wave_vectors(kwargs["wave_vectors"])
        return super()._pack_kwargs(kwargs)

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = super()._unpack_kwargs(attrs)
        kwargs["wave_vectors"] = np.array(kwargs["wave_vectors"], dtype=np.float32)
        return kwargs

        # kwargs["wave_vectors"] = _pack_wave_vectors(kwargs["wave_vectors"])

    def copy_to_device(self, device: str) -> "SMatrixArray":
        """Copy SMatrixArray to specified device."""
        s_matrix = super().copy_to_device(device)
        s_matrix._wave_vectors = copy_to_device(self._wave_vectors, device)
        return s_matrix

    @staticmethod
    def _packed_wave_vectors(wave_vectors):
        return _pack_wave_vectors(wave_vectors)

    def from_array_and_metadata(array, axes_metadata, metadata):
        raise NotImplementedError

    @property
    def device(self):
        """The device on which the SMatrixArray is reduced."""
        return self._device

    @property
    def storage_device(self):
        """The device on which the SMatrixArray is stored."""
        return super().device

    @classmethod
    def _from_waves(cls, waves: Waves, **kwargs):
        common_kwargs = _common_kwargs(cls, Waves)
        kwargs.update({key: getattr(waves, key) for key in common_kwargs})
        kwargs["ensemble_axes_metadata"] = kwargs["ensemble_axes_metadata"][:-1]

        return cls(**kwargs)

    @property
    def waves(self) -> Waves:
        """The wave vectors describing each plane wave."""
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
        return self._from_waves(waves, **kwargs)

    @property
    def periodic(self) -> tuple[bool, bool]:
        """If True the SMatrix is assumed to be periodic along corresponding axis."""
        return self._periodic

    @property
    def metadata(self) -> dict:
        self._metadata["energy"] = self.energy
        return self._metadata

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        """Axis metadata for each ensemble axis."""
        return self._ensemble_axes_metadata

    @property
    def ensemble_shape(self) -> tuple[int, int]:
        return self.array.shape[:-3]

    @property
    def interpolation(self) -> tuple[int, int]:
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
    def semiangle_cutoff(self) -> float:
        """The cutoff semiangle of the plane wave expansion."""
        return self._semiangle_cutoff

    @property
    def wave_vectors(self) -> np.ndarray:
        return self._wave_vectors

    @property
    def window_gpts(self) -> tuple[int, int]:
        return self._window_gpts

    @property
    def window_extent(self) -> tuple[float, float]:
        return (
            self.window_gpts[0] * self.sampling[0],
            self.window_gpts[1] * self.sampling[1],
        )

    @property
    def window_offset(self) -> tuple[float, float]:
        """The number of grid points from the origin the cropping windows of the wave
        functions is displaced."""
        return self._window_offset

    def multislice(self, potential: BasePotential = None) -> "SMatrixArray":
        """


        Parameters
        ----------
        potential :

        Returns
        -------

        """
        waves = self.waves.multislice(potential)
        return self._copy_with_new_waves(waves)

    def _reduce_to_waves(
        self,
        array,
        positions,
        position_coefficients,
    ):
        xp = get_array_module(self._device)

        if self._device == "gpu" and isinstance(array, np.ndarray):
            array = xp.asarray(array)

        position_coefficients = xp.array(
            position_coefficients, dtype=get_dtype(complex=True)
        )

        if self.window_gpts != self.gpts:
            pixel_positions = positions / xp.array(self.waves.sampling) - xp.asarray(
                self.window_offset
            )

            crop_corner, size, corners = minimum_crop(pixel_positions, self.window_gpts)

            array = wrapped_crop_2d(array, crop_corner, size)

            array = xp.tensordot(position_coefficients, array, axes=[-1, -3])

            if len(self.waves.shape) > 3:
                array = xp.moveaxis(array, -3, 0)

            array = batch_crop_2d(array, corners, self.window_gpts)

        else:
            array = xp.tensordot(position_coefficients, array, axes=[-1, -3])

            if len(self.waves.shape) > 3:
                array = xp.moveaxis(array, -3, 0)

        return array

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
        phi = xp.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])
        array = ctf._evaluate_from_angular_grid(alpha, phi)
        # the coefficients may be complex when the ctf includes aberrations, hence
        # the normalization must use the absolute square
        array = array / xp.sqrt((xp.abs(array) ** 2).sum(axis=-1, keepdims=True))
        return array

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
            extra_ensemble_axes_shape=self.waves.ensemble_shape[:-1],
            extra_ensemble_axes_metadata=self.waves.ensemble_axes_metadata[:-1],
        )

        xp = get_array_module(self._device)

        if self._device == "gpu" and isinstance(self.waves.array, np.ndarray):
            array = cp.asarray(self.waves.array)
        else:
            array = self.waves.array

        n_positions = int(np.prod(scan.shape + ctf.ensemble_shape))

        pbar = TqdmWrapper(enabled=pbar, total=n_positions, leave=False, desc="reduce")

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):
            sub_ctf = sub_ctf.item()
            ctf_coefficients = self._calculate_ctf_coefficients(sub_ctf)

            for _, slics, sub_scan in scan.generate_blocks(max_batch_reduction):
                sub_scan = sub_scan.item()
                positions = xp.asarray(sub_scan.get_positions())

                positions_coefficients = self._calculate_positions_coefficients(
                    sub_scan
                )

                if ctf_coefficients is not None:
                    (
                        expanded_ctf_coefficients,
                        positions_coefficients,
                    ) = expand_dims_to_broadcast(
                        ctf_coefficients,
                        positions_coefficients,
                        match_dims=[(-1,), (-1,)],
                    )
                    coefficients = positions_coefficients * expanded_ctf_coefficients
                else:
                    coefficients = positions_coefficients

                ensemble_shape = len(array.shape[:-3]) + len(sub_ctf.ensemble_shape)

                ensemble_axes_metadata = []
                ensemble_axes_metadata.extend(
                    [UnknownAxis() for _ in range(ensemble_shape)]
                )
                ensemble_axes_metadata.extend(
                    [ScanAxis() for _ in range(len(scan.shape))]
                )

                waves_array = self._reduce_to_waves(array, positions, coefficients)

                waves = Waves(
                    waves_array,
                    sampling=self.sampling,
                    energy=self.energy,
                    ensemble_axes_metadata=ensemble_axes_metadata,
                    metadata=self.metadata,
                )

                indices = (
                    (slice(None),) * (len(self.waves.shape) - 3) + ctf_slics + slics
                )

                pbar.update_if_exists(len(sub_scan))

                for detector, measurement in zip(detectors, measurements):
                    measurement.array[indices] = detector.detect(waves).array

        pbar.close_if_exists()

        return tuple(measurements)

    @property
    def _window_margin(self):
        return -(self.window_gpts[0] // -2), -(self.window_gpts[1] // -2)

    def _pad(self, pad_width):
        array = self.array

        pad_width = ((0,) * 2,) * len(array.shape[:-2]) + tuple(pad_width)

        pad_amounts = sum(pad_width[-2]), sum(pad_width[-1])

        pad_chunks = array.chunks[:-2] + (
            array.shape[-2] + pad_amounts[-2],
            array.shape[-1] + pad_amounts[-1],
        )

        array = array.map_blocks(
            np.pad,
            pad_width=pad_width,
            meta=array._meta,
            chunks=pad_chunks,
            mode="wrap",
        )

        kwargs = self._copy_kwargs(exclude=("array", "extent"))

        kwargs["periodic"] = tuple(
            False if pad_amount else periodic
            for periodic, pad_amount in zip(kwargs["periodic"], pad_amounts)
        )

        kwargs["window_offset"] = tuple(
            window_offset - pad_amount[0]
            for window_offset, pad_amount in zip(
                kwargs["window_offset"], pad_width[-2:]
            )
        )

        return self.__class__(array, **kwargs)

    def _chunks_for_reduction(self):
        chunks = (
            -(self.gpts[0] // -(self.interpolation[0] * 2)),
            -(self.gpts[1] // -(self.interpolation[1] * 2)),
        )

        num_chunks = self.gpts[0] // chunks[0], self.gpts[1] // chunks[1]

        if num_chunks[1] > num_chunks[0]:
            num_chunks = (1, num_chunks[1])
        else:
            num_chunks = (num_chunks[0], 1)

        chunks = tuple(
            equal_sized_chunks(n, num_chunks=nsc)
            for n, nsc in zip(self.shape[-2:], num_chunks)
        )

        if chunks is None:
            chunks = self.array.chunks[-2:]
        else:
            chunks = validate_chunks(self.shape[-2:], chunks)

        return chunks

    def _validate_max_batch_reduction(
        self, scan, max_batch_reduction: int | str = "auto"
    ):
        shape = (len(scan),) + self.window_gpts
        chunks = (max_batch_reduction, -1, -1)

        return validate_chunks(shape, chunks, dtype=np.dtype("complex64"))[0][0]

    def _validate_reduction_scheme(self, reduction_scheme):
        if self.interpolation == (1, 1) and reduction_scheme == "no-chunks":
            raise NotImplementedError

        if reduction_scheme == "auto" and max(self.interpolation) <= 2:
            return "no-chunks"
        elif reduction_scheme == "auto":
            return "multiple-rechunk"

        return reduction_scheme

    def reduce(
        self,
        scan: BaseScan = None,
        ctf: CTF = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        max_batch_reduction: int | str = "auto",
        reduction_scheme: str = "auto",
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Scan the probe across the potential and record a measurement for each detector.

        Parameters
        ----------
        detectors : list of Detector objects
            The detectors recording the measurements.
        scan : Scan object
            Scan defining the positions of the probe wave functions.
        ctf: CTF object, optional
            The probe contrast transfer function.
            Default is None (aperture is set by the planewave cutoff).
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions
            better utilize thread parallelization, but requires more memory and floating
            point operations. If 'auto' (default), the batch size is automatically
            chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : two int or str, optional
            Partitioning of the scan. The scattering matrix will be reduced in similarly
            partitioned chunks. Should be equal to or greater than the interpolation.
        """

        self.accelerator.check_is_defined()

        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff)

        ctf.grid.match(self.dummy_probes())
        ctf.accelerator.match(self)

        if ctf.semiangle_cutoff == np.inf:
            ctf.semiangle_cutoff = self.semiangle_cutoff

        if not isinstance(scan, BaseScan):
            squeeze = (-3,)
        else:
            squeeze = ()

        if scan is None:
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = validate_scan(
            scan, Probe._from_ctf(extent=self.extent, ctf=ctf, energy=self.energy)
        )
        detectors = detectors = validate_detectors(
            detectors, self.dummy_probes(downsample=False)
        )

        max_batch_reduction = self._validate_max_batch_reduction(
            scan, max_batch_reduction
        )

        reduction_scheme = self._validate_reduction_scheme(reduction_scheme)

        pbar = config.get("diagnostics.task_progress", False)

        if self.is_lazy:
            measurements = _no_chunks_reduce(
                self, scan, detectors, ctf, max_batch_reduction, pbar=pbar
            )
            # if reduction_scheme == "multiple-rechunk":
            #     measurements = _multiple_rechunk_reduce(
            #         self, scan, detectors, ctf, max_batch_reduction, pbar=pbar
            #     )
            # elif reduction_scheme == "single-rechunk":
            #     raise NotImplementedError
            #     measurements = _single_rechunk_reduce(
            #         self, scan, detectors, ctf, max_batch_reduction
            #     )
            # elif reduction_scheme == "no-chunks":
            # else:
            #     raise ValueError()
        else:
            measurements = self._batch_reduce_to_measurements(
                scan, ctf, detectors, max_batch_reduction, pbar=pbar
            )

        measurements = [measurement.squeeze(squeeze) for measurement in measurements]
        out = _wrap_measurements(measurements)
        return out

    def scan(
        self,
        scan: BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF = None,
        max_batch_reduction: int | str = "auto",
        rechunk: tuple[int, int] | str = "auto",
    ):
        """
        Reduce the SMatrix using coefficients calculated by a BaseScan and a CTF, to
        obtain the exit wave functions at given initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire
            potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be
            converted to measurements after running the multislice algorithm.
            See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion
            coefficients in the reduction of the SMatrix.
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions
            better utilize thread parallelization, but requires more memory and floating
            point operations. If 'auto' (default), the batch size is automatically
            chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        rechunk : str or tuple of int, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from
            chunking along the expansion axis to chunking over the spatial axes.
            If given as a tuple of int of length the SMatrix is rechunked to have those
            chunks. If 'auto' (default) the chunks are taken to be identical to the
            interpolation factor.

        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential
            (if no detector(s) given).
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
            reduction_scheme=rechunk,
        )


# the coarse plane-wave expansion of the upsampled (C-PRISM) scattering matrix is
# built on a disk around the aperture: every beam within this normalized radius
# (unity at the aperture edge) of the aperture, or one coarse cell, whichever is
# larger. The far corners of the bounding rectangle are dropped, which reduces the
# number of multislice runs at large interpolation factors without affecting the
# accuracy of the reduction.
_COARSE_SUPPORT_MARGIN = 0.0
# (plateau, edge) of the drift-space apodization of the smooth interpolant used
# by the interpolated branch of the reduction, or None (default) for the plain
# (box) trigonometric interpolant. A window helps thin specimens, where the
# drifted signal occupies the plateau and only aliased junk is attenuated, but
# on thick specimens the signal fills the whole period and the roll-off eats
# it (measured: Si t=54 A improved 5x, Ge t=198 A regressed 3x), hence the box
# is the default; see _interpolate_beam_functions
_SMOOTH_APODIZATION = None


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


class CompressedSMatrixArray(BaseSMatrix, CopyMixin, EqualityMixin):
    """
    A compressed scattering matrix defined by its truncated singular value
    decomposition, returned by :meth:`.SMatrix.build` when ``upsample=True`` (the
    C-PRISM algorithm). The coarse phase-removed scattering matrix is interpolated
    to the plane waves of the aperture at interpolation (1, 1) and factored as
    :math:`T \\approx U \\Sigma V^H`, where the left singular vectors :math:`U` are
    real-space images and the right singular vectors hold the plane-wave
    coefficients of each mode.

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
    position_quantization : int, optional
        If given, the fractional part of the probe positions is quantized to this
        number of fractions of a pixel. The default is None, ie. the positions are
        not quantized.
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
        max_batch_expansion: int | str = "auto",
        blend_angle: float = None,
        device: str = None,
        metadata: dict = None,
        singular_values: np.ndarray = None,
        smooth_values: np.ndarray = None,
    ):
        self._u = u
        self._sigma = sigma
        self._vh_dense = vh_dense
        self._dense_indices = dense_indices
        self._singular_values = singular_values
        self._smooth_values = smooth_values
        self._max_batch_expansion = max_batch_expansion
        self._blend_angle = blend_angle

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
    def singular_values(self) -> np.ndarray:
        """The full singular-value spectrum of the interpolated operator,
        including the modes truncated by ``tolerance`` and ``max_rank``. Useful
        for choosing the tolerance; the retained part is :attr:`sigma`."""
        if self._singular_values is None:
            return self._sigma
        return self._singular_values

    @property
    def max_batch_expansion(self) -> int | str:
        """Number of plane waves expanded at a time by the full-window
        reduction; 'auto' materializes the full expansion."""
        return self._max_batch_expansion

    @property
    def blend_angle(self) -> float | None:
        """Scattering angle [mrad] above which the reduction follows the
        plane-wave (PRISM) reduction of the built beams (None: no blending)."""
        return self._blend_angle

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

    def _coefficient_values(self, coefficients, smooth: bool = False):
        """The dense plane-wave amplitudes of each mode of the expansion.

        ``smooth=True`` selects the amplitudes of the apodized interpolant when
        they are stored — the interpolated branch of the reduction — while the
        exact (box) amplitudes make the plane-wave branch the PRISM algorithm.
        """
        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        if smooth and self._smooth_values is not None:
            values = xp.asarray(self._smooth_values, dtype=dtype)
        else:
            values = xp.asarray(self._sigma[:, None] * self._vh_dense, dtype=dtype)
        return values * xp.asarray(coefficients, dtype=dtype)[None]

    def _lattice_coefficients(self, coefficients):
        """CTF coefficients restricted to the built (coarse lattice) plane waves.

        The trigonometric interpolation is interpolatory — the reconstruction at
        the lattice plane waves equals the built scattering matrix — hence
        reducing with the restricted (and renormalized) coefficients yields the
        wave functions of the PRISM algorithm from the same compressed factors.
        """
        xp = get_array_module(self._device)

        indices = np.asarray(self._dense_indices)
        mask = (indices[:, 0] % self._interpolation[0] == 0) & (
            indices[:, 1] % self._interpolation[1] == 0
        )

        restricted = coefficients * xp.asarray(mask)
        return restricted / xp.sqrt(
            (xp.abs(restricted) ** 2).sum(axis=-1, keepdims=True)
        )

    def _blend_weight(self, blend_angle, window: tuple[int, int], taper=None):
        """Radial Fourier-space weight switching from the interpolated (C-PRISM)
        wave below ``blend_angle`` to the plane-wave (PRISM) wave above it, with
        a smooth cosine taper.

        ``blend_angle='aperture'`` uses the amplitude of the probe-forming
        aperture as the weight instead: the interpolated wave inside the
        bright-field disk, the plane-wave reduction outside it, with the soft
        aperture edge as the transition.
        """
        xp = get_array_module(self._device)

        if isinstance(blend_angle, str):
            if blend_angle != "aperture":
                raise ValueError(
                    f"blend_angle must be a number, 'aperture' or None; "
                    f"got {blend_angle!r}"
                )
            probe = self.dummy_probes()
            probe.grid.check_is_defined()
            kernel = np.abs(np.asarray(probe.aperture._evaluate_kernel(probe)))
            if kernel.shape != tuple(window):
                raise RuntimeError(
                    "aperture weight does not match the reduction window"
                )
            kernel /= kernel.max()
            return xp.asarray(kernel, dtype=get_dtype(complex=False))

        wavelength = self.wavelength
        qx = np.fft.fftfreq(window[0], d=self.sampling[0])
        qy = np.fft.fftfreq(window[1], d=self.sampling[1])
        angle = (
            np.sqrt(qx[:, None] ** 2 + qy[None, :] ** 2) * wavelength * 1e3
        )  # mrad

        # the blend angle is an upper bound on the validity of the
        # interpolation, hence the taper ends AT it rather than straddling it
        if taper is None:
            taper = max(4.0, 0.2 * blend_angle)

        if taper <= 0.0:
            # a sharp cut, used when the blend angle has been snapped to a
            # detector boundary: a taper would reach back into the band below
            # and mix the branches inside it
            weight = (angle < blend_angle).astype(np.float64)
        else:
            weight = 0.5 * (
                1.0 + np.cos(np.pi * (angle - blend_angle + taper) / taper)
            )
            weight[angle <= blend_angle - taper] = 1.0
            weight[angle >= blend_angle] = 0.0
        return xp.asarray(weight, dtype=get_dtype(complex=False))

    @staticmethod
    def _snapped_blend_angle(blend_angle, detectors):
        """The blend angle lowered to a detector collection boundary.

        Above the blend angle the reduction is the plane-wave branch alone,
        which is the PRISM algorithm exactly; below it the interpolated branch
        takes over. A detector whose collection range straddles the blend angle
        therefore mixes the two, and is the only way the blended reduction can
        come out worse than PRISM on a band. Snapping the angle down to the
        nearest boundary leaves every band wholly on one side: the bands above
        are PRISM, the bands below are the interpolated reduction.
        """
        if blend_angle is None or isinstance(blend_angle, str):
            return blend_angle

        if detectors is None:
            return blend_angle
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]

        bounds = set()
        for detector in detectors:
            for name in ("inner", "outer"):
                value = getattr(detector, name, None)
                if value is not None and np.isfinite(value):
                    bounds.add(float(value))

        below = [value for value in bounds if 0.0 < value <= blend_angle]
        return max(below) if below else blend_angle

    @staticmethod
    def _blend_branches(blend_angle, blend_component):
        """Which of the two blended reductions the result actually needs.

        Selecting a component keeps a single branch, so the other one — and the
        window kernel it would be reduced with — is never evaluated.
        """
        return (
            blend_component != "high",
            blend_angle is not None and blend_component != "low",
        )

    def _blend_wave_batches(self, interpolated, plane_wave, weight, component=None):
        """Combine the two reductions in Fourier space with the radial weight.

        ``component='low'`` returns the interpolated branch alone weighted by
        ``sqrt(weight)``, ``component='high'`` the plane-wave branch alone
        weighted by ``sqrt(1 - weight)``: detecting the two and summing the
        measurements blends the intensities instead of the amplitudes, which
        permits a different reduction window per branch. The branch a component
        discards may be given as ``None`` (see :meth:`_blend_branches`).
        """
        if component == "low":
            interpolated = fft2(interpolated, overwrite_x=True)
            interpolated *= np.sqrt(weight)[None]
            return ifft2(interpolated, overwrite_x=True)
        if component == "high":
            plane_wave = fft2(plane_wave, overwrite_x=True)
            plane_wave *= np.sqrt(1.0 - weight)[None]
            return ifft2(plane_wave, overwrite_x=True)
        interpolated = fft2(interpolated, overwrite_x=True)
        plane_wave = fft2(plane_wave, overwrite_x=True)
        interpolated *= weight[None]
        interpolated += plane_wave * (1.0 - weight)[None]
        return ifft2(interpolated, overwrite_x=True)

    def _window_kernel(self, values, fractional_offset, center: bool = True):
        """The window kernels :math:`B_k` obtained by reducing the dense plane waves
        for a probe displaced by a fraction of a pixel.

        With ``center=True`` (default) the kernel is cropped to the reduction
        window with the probe at its center; with ``center=False`` the kernel is
        returned on the full grid indexed by the displacement from the probe
        (used by the full-window mode reduction, which keeps the absolute frame).
        """
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

        kernel = ifft2(scattered, overwrite_x=True)
        kernel *= get_dtype(complex=False)(normalization)

        if not center:
            return kernel

        ix = (xp.arange(window_gpts[0]) - window_gpts[0] // 2) % gpts[0]
        iy = (xp.arange(window_gpts[1]) - window_gpts[1] // 2) % gpts[1]
        return kernel[:, ix[:, None], iy[None, :]]

    # memory budget of one gathered block in the batched mode contractions;
    # the modes are chunked so several probe positions fit in every batch even
    # at large ranks, keeping the device operations big and few
    _REDUCE_BATCH_BYTES = 2**30
    _REDUCE_MODE_CHUNK = 64
    # the lattice reduction re-gathers a halo of window / step scan rows per
    # row block, hence the device rows blocks are several times larger
    _REDUCE_GPU_ROW_BLOCK_FACTOR = 8
    # ... but never larger than the device can hold: detecting a block also
    # holds its Fourier transform, the transform work area and the detected
    # intensity, so the peak is a few times the block itself
    _REDUCE_GPU_MEMORY_FRACTION = 0.35

    def _reduce_memory_budget(self):
        """Bytes that one reduced block of wave functions may occupy.

        Unbounded on the host; on the device a fraction of the memory that is
        actually free, counting the blocks the memory pool holds but is not
        using.
        """
        if self._device != "gpu" or cp is None:
            return np.inf

        free = cp.cuda.Device().mem_info[0] + cp.get_default_memory_pool().free_bytes()
        return int(free * self._REDUCE_GPU_MEMORY_FRACTION)

    def _contract_modes_batched(self, fields, flat_indices, kernel, gather_kernel):
        """Contract the modes for a batch of probe positions: for every position
        ``p`` and window pixel ``w``, ``sum_k gathered[p, w, k] * fixed[w, k]``,
        where the gathered operand is indexed by ``flat_indices[p, w]``.

        ``gather_kernel`` selects which operand is gathered per position: the
        windows of the left singular vectors (windowed reduction) or the
        displaced kernel (full-window reduction). The contraction is chunked
        over the modes first — so the gathered blocks stay within the memory
        budget at several positions per batch — and accumulated.
        """
        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        num_positions = flat_indices.shape[0]
        out_shape = flat_indices.shape[1:]
        num_pixels = int(np.prod(out_shape))
        num_modes = kernel.shape[-1]

        flat_indices = flat_indices.reshape(num_positions, num_pixels)
        fields = fields.reshape(-1, num_modes)
        kernel = kernel.reshape(-1, num_modes)

        waves = xp.zeros((num_positions, num_pixels), dtype=dtype)

        mode_chunk = min(num_modes, self._REDUCE_MODE_CHUNK)
        max_batch = max(
            1, int(self._REDUCE_BATCH_BYTES // max(num_pixels * mode_chunk * 8, 1))
        )

        for k_start in range(0, num_modes, mode_chunk):
            k_stop = min(k_start + mode_chunk, num_modes)
            gathered_source = xp.ascontiguousarray(
                (kernel if gather_kernel else fields)[:, k_start:k_stop]
            )
            fixed = xp.ascontiguousarray(
                (fields if gather_kernel else kernel)[:, k_start:k_stop]
            )
            for start in range(0, num_positions, max_batch):
                stop = min(start + max_batch, num_positions)
                gathered = gathered_source[flat_indices[start:stop]]
                waves[start:stop] += xp.einsum(
                    "pwk,wk->pw", gathered, fixed, optimize=True
                )

        return waves.reshape((num_positions,) + out_shape)

    def _lattice_geometry(self, scan, warn: bool = False):
        """Describe a scan as a lattice of the pixel grid, or return None.

        The fast reduction below requires the probe positions to form a regular
        grid whose step is a whole number of pixels, so that a window offset
        splits into a whole-step and a sub-step part. The scan may cover any
        part of the grid and its origin may fall between pixels — a common
        fractional offset is applied to the reduction kernel instead.

        Returns ``(origin, step, shape, offset)``: the whole-pixel origin, the
        integer pixel step, the scan shape, and the common fractional offset.
        """
        if not isinstance(scan, GridScan):
            return None

        positions = np.asarray(scan.get_positions())
        if positions.ndim != 3 or min(positions.shape[:2]) < 2:
            return None

        pixels = positions / np.array(self.sampling)
        origin = pixels[0, 0]
        steps = (pixels[1, 0] - pixels[0, 0], pixels[0, 1] - pixels[0, 0])

        if abs(steps[0][1]) > 1e-6 or abs(steps[1][0]) > 1e-6:
            return None

        step = (steps[0][0], steps[1][1])
        shape = positions.shape[:2]

        # the scan must be the exact lattice implied by its first row and column
        if not (
            np.allclose(pixels[:, 0, 0], origin[0] + np.arange(shape[0]) * step[0])
            and np.allclose(pixels[0, :, 1], origin[1] + np.arange(shape[1]) * step[1])
        ):
            return None

        if any(abs(value - round(value)) > 1e-6 for value in step):
            if warn:
                suggestion = tuple(
                    min(
                        (d for d in range(1, gpts + 1) if gpts % d == 0),
                        key=lambda d: abs(d - n),
                    )
                    for n, gpts in zip(shape, self.gpts)
                )
                warnings.warn(
                    "The scan step is not a whole number of pixels "
                    f"({step[0]:.3f}, {step[1]:.3f}), so the reduction of the "
                    "compressed scattering matrix falls back to its general "
                    "(much slower) implementation. Choose a scan whose step "
                    "divides the grid — for this scattering matrix "
                    f"{self.gpts} — for example gpts={suggestion}.",
                    stacklevel=3,
                )
            return None

        step = tuple(int(round(value)) for value in step)
        if min(step) < 1:
            return None

        # a common fractional origin is exact: it is applied as a sub-pixel
        # phase ramp on the reduction kernel
        whole = tuple(int(np.rint(value)) for value in origin)
        offset = np.array(origin) - np.array(whole)

        # a half-pixel offset rounds inconsistently along the scan (ties round
        # to even), which would centre the cropping windows of neighbouring
        # positions one pixel apart; leave those scans to the general path
        if np.any(np.abs(np.abs(offset) - 0.5) < 1e-6):
            return None

        return whole, step, shape, offset

    def _lattice_waves_block(
        self, u_modes, kernel, origin, step, scan_shape, x_start, x_stop
    ):
        """Reduce a block of scan rows by the lattice decomposition of the
        window offsets.

        The reduction ``psi[p, j] = sum_k U[r_p + j - c, k] B[j, k]`` gathers a
        window of the modes per probe position, which reuses no data: every
        gathered element is consumed by a single multiply, so it runs at a few
        percent of the achievable rate. When the probe positions lie on a
        lattice of the pixel grid with step ``s``, the window offset splits as
        ``j - c = a * s + b``, and the probe position enters only through the
        whole-step part::

            psi[p, a s + b] = sum_k U[(p + a) s + b, k] B[a s + b, k]

        For each sub-step offset ``b`` the modes are sliced with stride ``s``
        (a view, not a gather) and every ``(position + a)`` pair is evaluated
        by a single matrix product; the probe-dependent shift is then applied
        by extracting ``G[p + a, a]``. The matrix products reuse both operands
        across the whole block, which is what the gather formulation cannot do.
        """
        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        gpts = self.gpts
        window = tuple(kernel.shape[:2])
        num_modes = kernel.shape[-1]
        step_x, step_y = step
        num_y = scan_shape[1]
        num_block = x_stop - x_start

        offsets = []
        for length, center, stride in zip(
            window, (window[0] // 2, window[1] // 2), step
        ):
            displacement = np.arange(length) - center
            sub_step = displacement % stride
            offsets.append((sub_step, (displacement - sub_step) // stride))
        (sub_x, whole_x), (sub_y, whole_y) = offsets

        def axis_indices(value, steps, start, num, axis):
            """Grid indices of the strided slice and the extraction offsets.

            When the scan lattice tiles the periodic axis the window offsets
            wrap onto the same points, otherwise the slice is extended by the
            halo of points the window reaches beyond the scan.
            """
            if num * step[axis] == gpts[axis] and start == 0:
                index = (origin[axis] + value + np.arange(num) * step[axis]) % gpts[axis]
                return xp.asarray(index), xp.asarray(
                    (np.arange(num)[:, None] + steps[None, :]) % num
                )
            first = int(steps.min())
            length = num + int(steps.max()) - first
            index = (
                origin[axis] + value + (start + first + np.arange(length)) * step[axis]
            ) % gpts[axis]
            return xp.asarray(index), xp.asarray(
                np.arange(num)[:, None] + (steps - first)[None, :]
            )

        waves = xp.zeros((num_block, num_y) + window, dtype=dtype)

        for value_x in range(step_x):
            select_x = np.flatnonzero(sub_x == value_x)
            if not len(select_x):
                continue
            index_x, row_index = axis_indices(
                value_x, whole_x[select_x], x_start, num_block, 0
            )

            for value_y in range(step_y):
                select_y = np.flatnonzero(sub_y == value_y)
                if not len(select_y):
                    continue
                index_y, column_index = axis_indices(
                    value_y, whole_y[select_y], 0, num_y, 1
                )

                modes = u_modes[index_x][:, index_y]
                block_kernel = kernel[xp.asarray(select_x)][:, xp.asarray(select_y)]

                product = (
                    modes.reshape(-1, num_modes)
                    @ block_kernel.reshape(-1, num_modes).T
                ).reshape(
                    len(index_x), len(index_y), len(select_x), len(select_y)
                )

                waves[
                    :,
                    :,
                    xp.asarray(select_x)[:, None],
                    xp.asarray(select_y)[None, :],
                ] = product[
                    row_index[:, None, :, None],
                    column_index[None, :, None, :],
                    xp.arange(len(select_x))[None, None, :, None],
                    xp.arange(len(select_y))[None, None, None, :],
                ]

        return waves.reshape((num_block * num_y,) + window)

    def _lattice_batch_reduce_to_measurements(
        self, scan, ctf, detectors, lattice, pbar: bool = False,
        blend_angle: float = None,
        blend_component: str = None,
        blend_taper: float = None,
    ):
        """Windowed reduction of a lattice scan (see :meth:`_lattice_waves_block`)."""
        origin, step, scan_shape, offset = lattice

        measurements = allocate_multislice_measurements(
            self.dummy_probes(scan=scan, ctf=ctf),
            detectors,
            extra_ensemble_axes_shape=(),
            extra_ensemble_axes_metadata=[],
        )

        xp = get_array_module(self._device)
        u_modes = xp.ascontiguousarray(xp.asarray(self._u).transpose(1, 2, 0))
        window = self.window_gpts

        # block the scan rows so the reduced wave functions of one block stay
        # within a fixed budget; larger blocks amortize the halo of window /
        # step rows that neighbouring blocks re-gather and re-multiply, hence
        # the device budget is set several times higher than the host one
        budget = self._REDUCE_BATCH_BYTES * (
            self._REDUCE_GPU_ROW_BLOCK_FACTOR if self._device == "gpu" else 2
        )
        budget = min(budget, self._reduce_memory_budget())
        row_bytes = scan_shape[1] * int(np.prod(window)) * 8
        num_rows = max(1, int(budget // max(row_bytes, 1)))
        num_rows = min(num_rows, scan_shape[0])
        detect_rows = max(1, int(self._REDUCE_BATCH_BYTES // max(row_bytes, 1)))
        detect_rows = min(detect_rows, num_rows)

        keep_interpolated, keep_plane_wave = self._blend_branches(
            blend_angle, blend_component
        )

        pbar = TqdmWrapper(
            enabled=pbar,
            total=int(np.prod(scan.shape + ctf.ensemble_shape)),
            leave=False,
            desc="reduce",
        )

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):
            sub_ctf = sub_ctf.item()
            coefficients = self._calculate_ctf_coefficients(sub_ctf)
            coefficients = coefficients.reshape((-1, coefficients.shape[-1]))[0]
            kernel = plane_wave_kernel = None
            if keep_interpolated:
                kernel = xp.ascontiguousarray(
                    self._window_kernel(
                        self._coefficient_values(coefficients, smooth=True), offset
                    ).transpose(1, 2, 0)
                )
            if keep_plane_wave:
                plane_wave_scale = get_dtype(complex=False)(
                    np.sqrt(np.prod(self.gpts) / np.prod(self.window_gpts))
                )
                plane_wave_kernel = xp.ascontiguousarray(
                    self._window_kernel(
                        plane_wave_scale
                        * self._coefficient_values(
                            self._lattice_coefficients(coefficients)
                        ),
                        offset,
                    ).transpose(1, 2, 0)
                )
            if blend_angle is not None:
                blend_weight = self._blend_weight(
                    blend_angle, self.window_gpts, taper=blend_taper
                )

            for x_start in range(0, scan_shape[0], num_rows):
                x_stop = min(x_start + num_rows, scan_shape[0])

                interpolated = (
                    self._lattice_waves_block(
                        u_modes, kernel, origin, step, scan_shape, x_start, x_stop
                    )
                    if keep_interpolated
                    else None
                )
                plane_wave = (
                    self._lattice_waves_block(
                        u_modes, plane_wave_kernel, origin, step, scan_shape,
                        x_start, x_stop,
                    )
                    if keep_plane_wave
                    else None
                )

                # the row block is sized for the matrix products; the blend and
                # the detectors transform what it produces, which needs the
                # transform, its work area and the detected intensity live at
                # once, so they walk the block in plain-budget chunks
                for start in range(x_start, x_stop, detect_rows):
                    stop = min(start + detect_rows, x_stop)
                    rows = slice(
                        (start - x_start) * scan_shape[1],
                        (stop - x_start) * scan_shape[1],
                    )

                    waves_array = interpolated[rows] if keep_interpolated else None
                    if blend_angle is not None:
                        waves_array = self._blend_wave_batches(
                            waves_array,
                            plane_wave[rows] if keep_plane_wave else None,
                            blend_weight,
                            component=blend_component,
                        )
                    waves_array = waves_array.reshape(
                        (1,) * len(sub_ctf.ensemble_shape)
                        + (stop - start, scan_shape[1])
                        + window
                    )

                    ensemble_axes_metadata = [
                        UnknownAxis() for _ in range(len(sub_ctf.ensemble_shape))
                    ] + [ScanAxis(), ScanAxis()]

                    waves = Waves(
                        waves_array,
                        sampling=tuple(self.sampling),
                        energy=self.energy,
                        ensemble_axes_metadata=ensemble_axes_metadata,
                        metadata=self.metadata,
                    )

                    indices = ctf_slics + (slice(start, stop),)

                    pbar.update_if_exists((stop - start) * scan_shape[1])

                    for detector, measurement in zip(detectors, measurements):
                        measurement.array[indices] = detector.detect(waves).array

                del interpolated, plane_wave

        pbar.close_if_exists()

        return tuple(measurements)

    def _reduce_to_waves_batched(self, u_windows, snapped_pixels, kernel):
        """Vectorized equivalent of :meth:`_reduce_to_waves`: the windows of the
        left singular vectors are gathered for batches of probe positions and
        contracted with the kernel in large batched einsums.

        The per-position loop of :meth:`_reduce_to_waves` evaluates thousands of
        small kernels, which is launch-overhead bound on the GPU.
        """
        xp = get_array_module(self._device)

        gpts = self.gpts
        window_gpts = self.window_gpts

        corners = (
            snapped_pixels
            - xp.asarray((window_gpts[0] // 2, window_gpts[1] // 2))[None]
        ) % xp.asarray(gpts)[None]

        window_x = (corners[:, 0, None] + xp.arange(window_gpts[0])[None]) % gpts[0]
        window_y = (corners[:, 1, None] + xp.arange(window_gpts[1])[None]) % gpts[1]
        flat_indices = (
            window_x[:, :, None] * gpts[1] + window_y[:, None, :]
        ).astype(np.int32)

        return self._contract_modes_batched(
            u_windows, flat_indices, kernel, gather_kernel=False
        )

    def _reduce_to_waves_absolute(self, u_full, snapped_pixels, kernel):
        """Full-window reduction in the absolute frame: contract the modes with
        the kernel displaced to each probe position.

        ``kernel`` is the uncentered kernel on the full grid (mode axis last),
        indexed by the displacement from the probe; the result matches the
        reduction of the expanded scattering matrix to floating point precision.
        """
        xp = get_array_module(self._device)

        gpts = self.gpts

        shift_x = (xp.arange(gpts[0])[None] - snapped_pixels[:, 0, None]) % gpts[0]
        shift_y = (xp.arange(gpts[1])[None] - snapped_pixels[:, 1, None]) % gpts[1]
        flat_indices = (
            shift_x[:, :, None] * gpts[1] + shift_y[:, None, :]
        ).astype(np.int32)

        return self._contract_modes_batched(
            u_full, flat_indices, kernel, gather_kernel=True
        )

    def _reduce_to_waves(self, u_windows, snapped_pixels, kernel):
        """Reduce the compressed scattering matrix to wave functions at the given
        snapped pixel positions.

        Parameters
        ----------
        u_windows : array
            Left singular vectors with the mode axis last, of shape
            (gpts_x, gpts_y, K).
        snapped_pixels : array of int
            Whole-pixel probe positions of shape (n, 2).
        kernel : array
            Reduction kernel with the mode axis last, of shape
            (window_gpts_x, window_gpts_y, K).
        """
        xp = get_array_module(self._device)

        if xp is not np:
            # per-position loops are launch-overhead bound on the GPU
            return self._reduce_to_waves_batched(u_windows, snapped_pixels, kernel)

        gpts = self.gpts
        window_gpts = self.window_gpts

        corners = (
            snapped_pixels
            - xp.asarray((window_gpts[0] // 2, window_gpts[1] // 2))[None]
        ) % xp.asarray(gpts)[None]
        corners = corners if xp is np else corners.get()

        waves = xp.zeros(
            (len(snapped_pixels),) + window_gpts, dtype=get_dtype(complex=True)
        )

        # Each window is at most four contiguous blocks of the scattering matrix
        # (due to the periodic wrap-around), hence the contraction over the modes
        # is evaluated on views without gathering; the mode axis is contiguous in
        # both operands.
        def reduce_position(n):
            cx, cy = int(corners[n, 0]), int(corners[n, 1])
            x_split = min(gpts[0] - cx, window_gpts[0])
            y_split = min(gpts[1] - cy, window_gpts[1])
            for wx0, wx1, sx in ((0, x_split, cx), (x_split, window_gpts[0], 0)):
                if wx0 == wx1:
                    continue
                for wy0, wy1, sy in ((0, y_split, cy), (y_split, window_gpts[1], 0)):
                    if wy0 == wy1:
                        continue
                    waves[n, wx0:wx1, wy0:wy1] = xp.einsum(
                        "ijk,ijk->ij",
                        u_windows[sx : sx + wx1 - wx0, sy : sy + wy1 - wy0],
                        kernel[wx0:wx1, wy0:wy1],
                    )

        def reduce_chunk(chunk):
            for n in range(chunk.start, min(chunk.stop, len(snapped_pixels))):
                reduce_position(n)

        num_threads = int(config.get("fftw.threads", 1)) if xp is np else 1

        if num_threads > 1 and len(snapped_pixels) > 1:
            # the contraction of each position releases the GIL for its large
            # array operations; the positions write to disjoint slices
            from concurrent.futures import ThreadPoolExecutor

            max_batch = -(len(snapped_pixels) // -(num_threads * 4))
            chunks = [
                slice(start, start + max_batch)
                for start in range(0, len(snapped_pixels), max_batch)
            ]
            with ThreadPoolExecutor(num_threads) as executor:
                list(executor.map(reduce_chunk, chunks))
        else:
            reduce_chunk(slice(0, len(snapped_pixels)))

        return waves

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

    def _expanded_slab(self, start: int, stop: int, xp, dtype) -> np.ndarray:
        """The plane waves ``[start, stop)`` of the scattering matrix expanded
        to interpolation (1, 1): one matrix product over the modes followed by
        the reattachment of the plane-wave phases."""
        gpts = self.gpts
        extent = self.extent

        if self._smooth_values is not None:
            values = xp.asarray(self._smooth_values[:, start:stop], dtype=dtype)
        else:
            values = xp.asarray(
                self._sigma[:, None] * self._vh_dense[:, start:stop], dtype=dtype
            )
        u = xp.asarray(self._u).reshape(self.rank, -1)
        slab = (values.T @ u).reshape((-1,) + tuple(gpts))

        wave_vectors = xp.asarray(self.wave_vectors[start:stop])
        x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
        y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)
        slab *= complex_exponential(
            2.0 * xp.pi * wave_vectors[:, 0, None, None] * x[:, None]
        ) * complex_exponential(
            2.0 * xp.pi * wave_vectors[:, 1, None, None] * y[None, :]
        )
        return slab

    def _expanded_s_matrix_array(self) -> SMatrixArray:
        """Expand the compressed factorization to the interpolated scattering
        matrix at interpolation (1, 1).

        The expanded matrix has the same memory footprint as a PRISM scattering
        matrix at interpolation (1, 1); provide `max_batch_expansion` to stream
        the expansion instead, or `window_gpts` to reduce from the compressed
        modes.
        """
        if getattr(self, "_s_matrix_array", None) is not None:
            return self._s_matrix_array

        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        gpts = self.gpts

        n_dense = len(self.wave_vectors)
        array = xp.empty((n_dense,) + tuple(gpts), dtype=dtype)

        max_batch = max(1, int(256**3 / np.prod(gpts)))
        for start in range(0, n_dense, max_batch):
            stop = min(start + max_batch, n_dense)
            array[start:stop] = self._expanded_slab(start, stop, xp, dtype)

        self._s_matrix_array = SMatrixArray(
            array,
            wave_vectors=np.asarray(self.wave_vectors, dtype=np.float64),
            semiangle_cutoff=self.semiangle_cutoff,
            energy=self.energy,
            interpolation=(1, 1),
            sampling=tuple(self.sampling),
            window_gpts=tuple(gpts),
            window_offset=(0, 0),
            periodic=(True, True),
            device=self._device,
            ensemble_axes_metadata=[],
            metadata=dict(self.metadata),
        )
        return self._s_matrix_array

    def _batch_reduce_to_measurements(
        self,
        scan: BaseScan,
        ctf: CTF,
        detectors: list[BaseDetector],
        max_batch_reduction: int,
        pbar: bool = False,
        absolute: bool = False,
        blend_angle: float = None,
        blend_component: str = None,
        blend_taper: float = None,
    ) -> tuple[BaseMeasurements | Waves, ...]:
        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(
            dummy_probes,
            detectors,
            extra_ensemble_axes_shape=(),
            extra_ensemble_axes_metadata=[],
        )

        xp = get_array_module(self._device)

        u_windows = xp.ascontiguousarray(xp.asarray(self._u).transpose(1, 2, 0))

        n_positions = int(np.prod(scan.shape + ctf.ensemble_shape))
        pbar = TqdmWrapper(enabled=pbar, total=n_positions, leave=False, desc="reduce")

        sampling = xp.asarray(self.sampling)

        keep_interpolated, keep_plane_wave = self._blend_branches(
            blend_angle, blend_component
        )

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):
            sub_ctf = sub_ctf.item()
            coefficients = self._calculate_ctf_coefficients(sub_ctf)

            # the generated blocks contain a single ensemble member
            coefficients = coefficients.reshape((-1, coefficients.shape[-1]))[0]

            values = self._coefficient_values(coefficients, smooth=True)
            if keep_plane_wave:
                # the plane-wave reduction spreads a unit probe over
                # prod(interpolation) periodized copies; a window holds
                # window / gpts of them, hence the amplitude is rescaled so the
                # in-window intensity matches the interpolated branch (exact
                # when the window is a multiple of the period gpts /
                # interpolation, unity for the full window)
                plane_wave_scale = get_dtype(complex=False)(
                    np.sqrt(np.prod(self.gpts) / np.prod(self.window_gpts))
                )
                plane_wave_values = plane_wave_scale * self._coefficient_values(
                    self._lattice_coefficients(coefficients)
                )
            if blend_angle is not None:
                blend_weight = self._blend_weight(
                    blend_angle, self.window_gpts, taper=blend_taper
                )

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
                    mask = xp.asarray(inverse == i)
                    reduce_to_waves = (
                        self._reduce_to_waves_absolute
                        if absolute
                        else self._reduce_to_waves
                    )

                    def branch(branch_values):
                        kernel = xp.ascontiguousarray(
                            self._window_kernel(
                                branch_values, offset, center=not absolute
                            ).transpose(1, 2, 0)
                        )
                        return reduce_to_waves(u_windows, snapped[mask], kernel)

                    new_waves = branch(values) if keep_interpolated else None
                    if blend_angle is not None:
                        new_waves = self._blend_wave_batches(
                            new_waves,
                            branch(plane_wave_values) if keep_plane_wave else None,
                            blend_weight,
                            component=blend_component,
                        )
                    waves_array[mask] = new_waves

                waves_array = waves_array.reshape(
                    (1,) * len(sub_ctf.ensemble_shape) + scan_shape + self.window_gpts
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

    def _streamed_batch_reduce_to_measurements(
        self,
        scan: BaseScan,
        ctf: CTF,
        detectors: list[BaseDetector],
        max_batch_reduction: int,
        max_batch_expansion: int,
        pbar: bool = False,
    ) -> tuple[BaseMeasurements | Waves, ...]:
        """Full-window reduction streaming the interpolation-(1, 1) expansion.

        Equivalent to ``self._expanded_s_matrix_array().reduce(...)``, but the
        expanded scattering matrix is never materialized: the plane waves are
        expanded in slabs of ``max_batch_expansion`` and contracted with the
        reduction coefficients on the fly. Peak memory is one slab plus one
        batch of reduced wave functions, instead of the full ``n x gpts``
        expanded matrix. The expansion is repeated for every batch of probe
        positions, hence the relative overhead is ``rank / batch size`` matrix
        product work; the caller enlarges the reduction batches accordingly.

        The coefficients must match :meth:`SMatrixArray.reduce` exactly (same
        position phases and the same globally normalized CTF coefficients), so
        the streamed and expanded reductions agree to floating point precision.
        """
        dummy_probes = self.dummy_probes(scan=scan, ctf=ctf)

        measurements = allocate_multislice_measurements(
            dummy_probes,
            detectors,
            extra_ensemble_axes_shape=(),
            extra_ensemble_axes_metadata=[],
        )

        xp = get_array_module(self._device)
        dtype = get_dtype(complex=True)

        wave_vectors = xp.asarray(self.wave_vectors)
        n_dense = len(wave_vectors)

        n_positions = int(np.prod(scan.shape + ctf.ensemble_shape))
        pbar = TqdmWrapper(enabled=pbar, total=n_positions, leave=False, desc="reduce")

        for _, ctf_slics, sub_ctf in ctf.generate_blocks(1):
            sub_ctf = sub_ctf.item()

            # must match SMatrixArray._calculate_ctf_coefficients: normalized
            # by the absolute square over the full plane-wave expansion
            alpha = (
                xp.sqrt(wave_vectors[:, 0] ** 2 + wave_vectors[:, 1] ** 2)
                * sub_ctf.wavelength
            )
            phi = xp.arctan2(wave_vectors[:, 1], wave_vectors[:, 0])
            ctf_coefficients = sub_ctf._evaluate_from_angular_grid(alpha, phi)
            ctf_coefficients = ctf_coefficients / xp.sqrt(
                (xp.abs(ctf_coefficients) ** 2).sum(axis=-1, keepdims=True)
            )

            for _, slics, sub_scan in scan.generate_blocks(max_batch_reduction):
                sub_scan = sub_scan.item()

                # must match SMatrixArray._calculate_positions_coefficients
                if isinstance(sub_scan, GridScan):
                    x = xp.asarray(sub_scan._x_coordinates())
                    y = xp.asarray(sub_scan._y_coordinates())
                    positions_coefficients = complex_exponential(
                        -2.0 * xp.pi * x[:, None, None] * wave_vectors[None, None, :, 0]
                    ) * complex_exponential(
                        -2.0 * xp.pi * y[None, :, None] * wave_vectors[None, None, :, 1]
                    )
                else:
                    positions = xp.asarray(sub_scan.get_positions())
                    positions_coefficients = complex_exponential(
                        -2.0 * xp.pi * positions[..., 0, None] * wave_vectors[:, 0][None]
                        - 2.0 * xp.pi * positions[..., 1, None] * wave_vectors[:, 1][None]
                    )

                (
                    expanded_ctf_coefficients,
                    positions_coefficients,
                ) = expand_dims_to_broadcast(
                    ctf_coefficients,
                    positions_coefficients,
                    match_dims=[(-1,), (-1,)],
                )
                coefficients = xp.asarray(
                    positions_coefficients * expanded_ctf_coefficients, dtype=dtype
                )

                waves_array = xp.zeros(
                    coefficients.shape[:-1] + tuple(self.gpts), dtype=dtype
                )
                for start in range(0, n_dense, max_batch_expansion):
                    stop = min(start + max_batch_expansion, n_dense)
                    slab = self._expanded_slab(start, stop, xp, dtype)
                    waves_array += xp.tensordot(
                        coefficients[..., start:stop], slab, axes=[-1, -3]
                    )

                ensemble_axes_metadata = [
                    UnknownAxis() for _ in range(len(sub_ctf.ensemble_shape))
                ]
                ensemble_axes_metadata += [
                    ScanAxis() for _ in range(len(sub_scan.shape))
                ]

                waves = Waves(
                    waves_array,
                    sampling=tuple(self.sampling),
                    energy=self.energy,
                    ensemble_axes_metadata=ensemble_axes_metadata,
                    metadata=self.metadata,
                )

                indices = ctf_slics + slics

                pbar.update_if_exists(int(np.prod(sub_scan.shape)))

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
        max_batch_expansion: int | str = None,
        method: str = "auto",
        blend_angle: float = None,
        blend_window_gpts: int | tuple[int, int] | str = None,
        blend_taper: float = None,
        _blend_component: str = None,
        _blend_taper: float = None,
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
        max_batch_expansion : int or str, optional
            The number of plane waves expanded at a time when the reduced wave
            functions are not cropped. If 'auto', the full plane-wave expansion is
            materialized once (fastest, but with the memory footprint of a PRISM
            scattering matrix at interpolation 1); an integer streams the expansion
            instead, bounding the memory at one batch of plane waves plus one batch
            of reduced wave functions, at the cost of repeating the expansion for
            every batch of probe positions. If not given (default), the value set
            on the :class:`.SMatrix` is used. Only used with ``method='expand'``.
        method : {'auto', 'expand', 'modes'}, optional
            How the full-window reduction is evaluated. ``'expand'`` expands the
            compressed factorization to the interpolation-(1, 1) scattering matrix
            and reduces it with one large matrix product per batch of probe
            positions — high arithmetic intensity, fastest on the CPU, but with a
            cost proportional to the number of dense plane waves. ``'modes'``
            contracts the retained modes directly against a probe-displaced
            reduction kernel — a cost proportional to the number of modes (usually
            far fewer than the plane waves), fastest on the GPU where the
            contraction saturates memory bandwidth. Both produce identical wave
            functions to floating point precision. ``'auto'`` (default) selects
            'modes' on the GPU and 'expand' on the CPU, unless streaming was
            requested through ``max_batch_expansion``. Ignored when the reduced
            wave functions are cropped (``window_gpts``), which always contracts
            the modes.

        blend_angle : float, optional
            Above this scattering angle [mrad] the reduced wave functions follow
            the plane-wave (PRISM) reduction of the built beams, below it the
            interpolated (C-PRISM) reduction, with a smooth taper between them.
            The interpolation is band limited and aliases the contributions of
            electrons displaced beyond half its period, which harms high-angle
            detectors; the plane-wave reduction of the same built beams does not,
            hence blending bounds the high-angle error by that of the PRISM
            algorithm while keeping the interpolated accuracy at low angles. If
            not given, the value set on the :class:`.SMatrix` is used ('auto'
            derives it from the aliasing limit of the interpolation); a
            non-positive value disables blending.

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurements or Waves
        """
        self.accelerator.check_is_defined()

        explicit_blend = blend_angle is not None
        if blend_angle is None:
            blend_angle = self._blend_angle
        if (
            blend_angle is not None
            and not isinstance(blend_angle, str)
            and blend_angle <= 0
        ):
            blend_angle = None

        if (
            blend_window_gpts is not None
            and blend_angle is not None
            and _blend_component is None
        ):
            return self._composite_blend_reduce(
                scan=scan,
                ctf=ctf,
                detectors=detectors,
                max_batch_reduction=max_batch_reduction,
                method=method,
                blend_angle=blend_angle,
                blend_window_gpts=blend_window_gpts,
                blend_taper=0.0 if blend_taper is None else blend_taper,
            )

        if blend_angle is not None and _blend_component is None:
            cut = (
                float(self._semiangle_cutoff)
                if isinstance(blend_angle, str)
                else blend_angle
            )
            if detectors is not None:
                routed = self._routed_reduce(
                    scan, ctf, detectors, cut, max_batch_reduction, method,
                    blend_taper=0.0 if blend_taper is None else blend_taper,
                )
                if routed is not None:
                    return routed
            if not explicit_blend:
                # the default blend acts only through the routing: when the
                # detectors are not routable (a band straddles the cut, or the
                # output is wave functions or a full diffraction pattern) the
                # reduction is the plain interpolated one, and blending must be
                # requested explicitly
                blend_angle = None

        if _blend_taper is None and blend_taper is not None:
            _blend_taper = blend_taper

        if method not in ("auto", "expand", "modes"):
            raise ValueError(
                f"method must be 'auto', 'expand' or 'modes'; got {method!r}"
            )

        if max_batch_expansion is None:
            max_batch_expansion = self._max_batch_expansion

        full_window = tuple(self.window_gpts) == tuple(self.gpts)

        if not full_window:
            if max_batch_expansion != "auto":
                raise ValueError(
                    "max_batch_expansion applies to the reduction of the expanded "
                    "scattering matrix; it cannot be combined with window_gpts."
                )
            if method == "expand":
                raise ValueError(
                    "method='expand' applies to the full-window reduction; with "
                    "window_gpts the modes are always contracted directly."
                )
        else:
            if method == "auto":
                if max_batch_expansion != "auto":
                    method = "expand"
                elif self._device == "gpu":
                    method = "modes"
                else:
                    method = "expand"
            if method == "modes" and max_batch_expansion != "auto":
                raise ValueError(
                    "max_batch_expansion streams the expanded scattering matrix; "
                    "it cannot be combined with method='modes'."
                )

        if blend_angle is not None and full_window and method == "expand":
            # the blend combines two kernel reductions, hence needs the mode path
            method = "modes"

        if full_window and method == "expand" and max_batch_expansion == "auto":
            return self._expanded_s_matrix_array().reduce(
                scan=scan,
                ctf=ctf,
                detectors=detectors,
                max_batch_reduction=max_batch_reduction,
            )

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
        validated_max_batch_reduction = validate_chunks(
            shape, chunks, dtype=np.dtype("complex64")
        )[0][0]

        pbar = config.get("diagnostics.task_progress", False)

        if full_window and method == "expand":
            if max_batch_reduction == "auto":
                # the expansion is repeated for every batch of probe positions
                # with a relative matrix product overhead of rank / batch size;
                # enlarge the automatic batches so the overhead stays small
                validated_max_batch_reduction = min(
                    max(validated_max_batch_reduction, 4 * self.rank), len(scan)
                )
            measurements = self._streamed_batch_reduce_to_measurements(
                scan,
                ctf,
                detectors,
                validated_max_batch_reduction,
                int(max_batch_expansion),
                pbar=pbar,
            )
        else:
            lattice = (
                None if full_window else self._lattice_geometry(scan, warn=True)
            )
            if lattice is not None:
                measurements = self._lattice_batch_reduce_to_measurements(
                    scan, ctf, detectors, lattice, pbar=pbar,
                    blend_angle=blend_angle,
                    blend_component=_blend_component,
                    blend_taper=_blend_taper,
                )
            else:
                measurements = self._batch_reduce_to_measurements(
                    scan,
                    ctf,
                    detectors,
                    validated_max_batch_reduction,
                    pbar=pbar,
                    absolute=full_window,
                    blend_angle=blend_angle,
                    blend_component=_blend_component,
                    blend_taper=_blend_taper,
                )

        measurements = [measurement.squeeze(squeeze) for measurement in measurements]
        return _wrap_measurements(measurements)

    def _routing_sides(self, cut, detectors, taper: float = 0.0):
        """Which branch each detector reads from, or None when not routable.

        A detector collecting only below the blend angle reads the interpolated
        reduction, one collecting only above it the plane-wave (PRISM)
        reduction; nothing is mixed, so no Fourier weighting is needed. With a
        taper, a band overlapping the taper zone ``[cut - taper, cut]`` reads
        the tapered combination of the two intensities, which makes the
        underlying angular density continuous across the cut. A detector
        straddling the cut without a taper, or one whose collection range is
        not an angular band, is not routable.
        """
        sides = []
        for detector in detectors:
            if isinstance(detector, (AnnularDetector, SegmentedDetector)):
                outer = detector.outer
                inner = detector.inner
                if outer is not None and outer <= cut - taper:
                    sides.append("low")
                elif inner is not None and inner >= cut:
                    sides.append("high")
                elif taper > 0.0 and inner is not None and outer is not None:
                    sides.append("taper")
                elif outer is not None and outer <= cut:
                    sides.append("low")
                else:
                    return None
            elif isinstance(detector, PixelatedDetector):
                max_angle = detector.max_angle
                if isinstance(max_angle, (int, float)) and max_angle <= cut:
                    sides.append("low")
                else:
                    return None
            else:
                return None
        return sides

    def _routed_reduce(
        self, scan, ctf, detectors, blend_angle, max_batch_reduction, method,
        blend_taper: float = 0.0,
    ):
        """Route each detector to the branch its band lies in, or return None.

        The blend angle is snapped to a detector boundary, so that every
        detector band lies wholly below or above it: the bands below read the
        interpolated (C-PRISM) reduction on this array's window, the bands
        above the plane-wave reduction on one interpolation period — the
        window and the algorithm of PRISM. This is the composite blend without
        any Fourier weighting, possible whenever no band straddles the cut.
        """
        single = not isinstance(detectors, (list, tuple))
        detectors = [detectors] if single else list(detectors)

        cut = self._snapped_blend_angle(blend_angle, detectors)
        sides = self._routing_sides(cut, detectors, taper=blend_taper)
        if sides is None:
            return None

        def measure(component, subset):
            if component == "taper":
                measurements = self._composite_blend_reduce(
                    scan=scan,
                    ctf=ctf,
                    detectors=subset,
                    max_batch_reduction=max_batch_reduction,
                    method=method,
                    blend_angle=cut,
                    blend_window_gpts="period",
                    blend_taper=blend_taper,
                    snap=False,
                )
            elif component == "low":
                measurements = self.reduce(
                    scan=scan,
                    ctf=ctf,
                    detectors=subset,
                    max_batch_reduction=max_batch_reduction,
                    method=method,
                    blend_angle=0.0,
                )
            else:
                period = tuple(
                    min(-(-g // i), g)
                    for g, i in zip(self.gpts, self._interpolation)
                )
                measurements = self._with_window(period).reduce(
                    scan=scan,
                    ctf=ctf,
                    detectors=subset,
                    max_batch_reduction=max_batch_reduction,
                    blend_angle=cut,
                    _blend_component="high",
                    _blend_taper=0.0,
                )
            if not isinstance(measurements, (list, tuple)):
                measurements = [measurements]
            return list(measurements)

        ordered = [None] * len(detectors)
        for component in ("low", "high", "taper"):
            subset = [d for d, side in zip(detectors, sides) if side == component]
            if not subset:
                continue
            for index, measurement in zip(
                (i for i, side in enumerate(sides) if side == component),
                measure(component, subset),
            ):
                ordered[index] = measurement

        return ordered[0] if single else _wrap_measurements(ordered)

    def _with_window(self, window_gpts):
        """A view of this compressed scattering matrix with another reduction
        window; the factors are shared, not copied."""
        return self.__class__(
            u=self._u,
            sigma=self._sigma,
            vh_dense=self._vh_dense,
            dense_indices=self._dense_indices,
            semiangle_cutoff=self._semiangle_cutoff,
            energy=self.energy,
            extent=self.extent,
            interpolation=self._interpolation,
            window_gpts=window_gpts,
            position_quantization=self._position_quantization,
            blend_angle=self._blend_angle,
            device=self._device,
            metadata=self._metadata,
            singular_values=self._singular_values,
            smooth_values=self._smooth_values,
        )

    def _composite_blend_reduce(
        self,
        scan,
        ctf,
        detectors,
        max_batch_reduction,
        method,
        blend_angle,
        blend_window_gpts,
        blend_taper: float = 0.0,
        snap: bool = True,
    ):
        """Blend the intensities of two reductions with different windows.

        The interpolated branch is reduced on this array's own (typically full)
        window, weighted by ``sqrt(weight)`` in Fourier space; the plane-wave
        branch is reduced on ``blend_window_gpts`` — ``'period'`` selects one
        period ``gpts / interpolation`` of its periodized wave functions, the
        window the PRISM algorithm itself uses, which restores the local
        high-angle signal that the full-grid periodized field averages over its
        copies — weighted by ``sqrt(1 - weight)``. The detected intensities add,
        hence the detectors must be intensity-valued (not :class:`WavesDetector`)
        and produce window-independent shapes (annular and radial detectors; the
        diffraction patterns of the two branches have different samplings).
        """
        if isinstance(blend_window_gpts, str):
            if blend_window_gpts != "period":
                raise ValueError(
                    "blend_window_gpts must be an int, a pair of ints or "
                    f"'period'; got {blend_window_gpts!r}"
                )
            window = tuple(
                min(-(-g // i), g)
                for g, i in zip(self.gpts, self._interpolation)
            )
        elif np.isscalar(blend_window_gpts):
            window = (int(blend_window_gpts),) * 2
        else:
            window = tuple(int(n) for n in blend_window_gpts)

        # snap to a collection boundary, so that no detector band straddles
        # the blend without a taper: every band is either the plane-wave
        # (PRISM) reduction exactly, or the interpolated one, or (inside the
        # taper zone) a convex combination of the two intensities
        if snap:
            blend_angle = self._snapped_blend_angle(blend_angle, detectors)

        low = self.reduce(
            scan=scan,
            ctf=ctf,
            detectors=detectors,
            max_batch_reduction=max_batch_reduction,
            method=method,
            blend_angle=blend_angle,
            _blend_component="low",
            _blend_taper=blend_taper,
        )

        if self._device == "gpu" and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()

        high_array = self.__class__(
            u=self._u,
            sigma=self._sigma,
            vh_dense=self._vh_dense,
            dense_indices=self._dense_indices,
            semiangle_cutoff=self._semiangle_cutoff,
            energy=self.energy,
            extent=self.extent,
            interpolation=self._interpolation,
            window_gpts=window,
            position_quantization=self._position_quantization,
            blend_angle=self._blend_angle,
            device=self._device,
            metadata=self._metadata,
            singular_values=self._singular_values,
            smooth_values=self._smooth_values,
        )
        high = high_array.reduce(
            scan=scan,
            ctf=ctf,
            detectors=detectors,
            max_batch_reduction=max_batch_reduction,
            blend_angle=blend_angle,
            _blend_component="high",
            _blend_taper=blend_taper,
        )

        low_list, high_list = ensure_list(low), ensure_list(high)
        for low_measurement, high_measurement in zip(low_list, high_list):
            if np.iscomplexobj(low_measurement.array) or (
                low_measurement.array.shape != high_measurement.array.shape
            ):
                raise NotImplementedError(
                    "blend_window_gpts adds the detected intensities of the "
                    "two branches, hence it requires intensity-valued "
                    "detectors whose measurements do not depend on the "
                    "reduction window (for example annular detectors)."
                )
            low_measurement.array[:] += high_measurement.array

        return low if not isinstance(low, list) else _wrap_measurements(low_list)

    def scan(
        self,
        scan: BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF = None,
        max_batch_reduction: int | str = "auto",
        max_batch_expansion: int | str = None,
        method: str = "auto",
        blend_angle: float = None,
        blend_window_gpts: int | tuple[int, int] | str = None,
        blend_taper: float = None,
    ):
        """
        Reduce the compressed scattering matrix at the positions of a scan.

        See :meth:`.CompressedSMatrixArray.reduce`.
        """
        if scan is None:
            scan = GridScan()

        return self.reduce(
            scan=scan,
            detectors=detectors,
            ctf=ctf,
            max_batch_reduction=max_batch_reduction,
            max_batch_expansion=max_batch_expansion,
            method=method,
            blend_angle=blend_angle,
            blend_window_gpts=blend_window_gpts,
            blend_taper=blend_taper,
        )


class SMatrix(BaseSMatrix, Ensemble, CopyMixin, EqualityMixin):
    """
    The scattering matrix is used for simulating STEM experiments using the PRISM
    algorithm.

    Parameters
    ----------
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    potential : Atoms or AbstractPotential, optional
        Atoms or a potential that the scattering matrix represents. If given as atoms,
        a default potential will be created. If nothing is provided the scattering
        matrix will represent a vacuum potential, in which case the sampling and extent
        must be provided.
    gpts : one or two int, optional
        Number of grid points describing the scattering matrix. Provide only if
        potential is not given.
    sampling : one or two float, optional
        Lateral sampling of scattering matrix [Å]. Provide only if potential is not
        given. Will be ignored if 'gpts' is also provided.
    extent : one or two float, optional
        Lateral extent of scattering matrix [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions (default is 1, ie. no
        interpolation). If a single value is provided, assumed to be the same for both
        directions.
    upsample : bool, optional
        If True, interpolate the plane-wave expansion built at the given
        interpolation factor back to the full plane-wave expansion of the aperture
        and compress it by an exact adaptive truncated singular value decomposition
        (the C-PRISM algorithm); :meth:`.SMatrix.build` then returns a
        :class:`.CompressedSMatrixArray`. Every probe is reduced from the full
        expansion, avoiding the real-space cropping and coarsened aperture sampling
        errors of PRISM at the same interpolation factor: the interpolation factor
        only affects the number of multislice runs required to build the scattering
        matrix. At an interpolation factor of 1 the expansion is already complete
        and this option has no effect. Default is False.
    tolerance : float, optional
        Relative singular value threshold of the adaptive truncation performed when
        ``upsample=True``. All modes with singular values within this factor of the
        largest singular value of the interpolated scattering matrix are retained
        (default is 1e-3). Decrease for higher accuracy at increased cost of the
        reduction. Ignored when ``upsample=False``.
    max_rank : int, optional
        Maximum number of modes retained by the compression when ``upsample=True``.
        If None (default), the rank is set adaptively by the tolerance.
    blend_angle : float or str, optional
        Scattering angle [mrad] above which the reduction of the compressed
        scattering matrix follows the plane-wave (PRISM) reduction of the built
        beams, below which the interpolated (C-PRISM) reduction. Acts through
        the detector routing of the reduction: the angle is snapped down to a
        detector collection boundary and each detector reads the branch its
        band lies in, guaranteeing the dark-field bands match the PRISM
        algorithm. 'auto' (default with ``upsample=True``) derives the angle
        from the aliasing limit of the interpolation,
        ``extent / (2 * interpolation * thickness)``; a number fixes it;
        0 disables blending. Only used when ``upsample=True``.
    window_gpts : one or two int or 'full', optional
        The number of grid points describing the cropping window of the wave
        functions reduced from the compressed scattering matrix. Only used when
        ``upsample=True``. If None (default), the window is inferred from the
        specimen and the probe (the probe tails plus the beam spreading over
        the thickness), falling back to the full grid when there is no
        potential; 'full' disables cropping. Unlike the PRISM cropping window,
        this window is decoupled from the interpolation factor.
    position_quantization : int, optional
        If given, the fractional part of the probe positions is quantized to this
        number of fractions of a pixel, limiting the number of reduction kernels
        calculated by the windowed compressed reduction for scans that are
        incommensurate with the grid of the scattering matrix. The maximum position
        error is half a quantization step. Only used when ``upsample=True``. The
        default is None, ie. the positions are not quantized.
    max_batch_expansion : int, optional
        The number of plane waves expanded at a time by the reduction of the
        compressed scattering matrix. By default the full plane-wave expansion is
        materialized once (fastest, but with the memory footprint of a PRISM
        scattering matrix at interpolation 1); providing a batch size streams the
        expansion instead, bounding the memory at one batch of plane waves plus
        one batch of reduced wave functions, at the cost of repeating the
        expansion for every batch of probe positions. Only used when
        ``upsample=True`` and the reduced wave functions are not cropped.
    downsample : {'cutoff', 'valid'} or float or bool
        Controls whether to downsample the scattering matrix after running the
        multislice algorithm.

            ``cutoff`` :
                Downsample to the antialias cutoff scattering angle (default).

            ``valid`` :
                Downsample to the largest rectangle that fits inside the circle with a
                radius defined by the antialias cutoff scattering angle.

            float :
                Downsample to a specified maximum scattering angle [mrad].
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu').
        Default is 'cpu'. The default is determined by the user configuration.
    store_on_host : bool, optional
        If True, store the scattering matrix in host (cpu) memory so that the necessary
        memory is transferred as chunks to the device to run calculations
        (default is False).
    """

    def __init__(
        self,
        semiangle_cutoff: float,
        energy: float,
        potential: Atoms | BasePotential = None,
        gpts: int | tuple[int, int] = None,
        sampling: float | tuple[float, float] = None,
        extent: float | tuple[float, float] = None,
        interpolation: int | tuple[int, int] = 1,
        upsample: bool = False,
        tolerance: float = 1e-3,
        blend_angle: float | str = None,
        max_rank: int = None,
        window_gpts: int | tuple[int, int] = None,
        position_quantization: int = None,
        max_batch_expansion: int | str = "auto",
        downsample: bool | str = "cutoff",
        # tilt: Tuple[float, float] = (0.0, 0.0),
        device: str = None,
        store_on_host: bool = False,
    ):
        if downsample is True:
            downsample = "cutoff"

        self._device = validate_device(device)
        self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        if potential is None:
            try:
                self.grid.check_is_defined()
            except GridUndefinedError:
                raise ValueError("Provide a potential or provide 'extent' and 'gpts'.")
        else:
            potential = validate_potential(potential)
            self.grid.match(potential)
            self._grid = potential.grid

        self._potential = potential
        self._interpolation = _validate_interpolation(interpolation)
        self._semiangle_cutoff = semiangle_cutoff
        self._downsample = downsample

        if not upsample:
            invalid = tuple(
                name
                for name, value in (
                    ("max_rank", max_rank),
                    ("blend_angle", blend_angle),
                    ("position_quantization", position_quantization),
                    (
                        "max_batch_expansion",
                        None if max_batch_expansion == "auto" else max_batch_expansion,
                    ),
                )
                if value is not None
            )
            if invalid:
                raise ValueError(
                    f"{' and '.join(invalid)} require(s) upsample=True."
                )

        self._upsample = bool(upsample)
        self._tolerance = tolerance
        # the blend against the plane-wave (PRISM) reduction is on by default:
        # it acts through the detector routing of the reduction, which uses the
        # interpolated wave functions below the blend angle and the plane-wave
        # reduction above it; pass 0 to disable
        if upsample and blend_angle is None:
            blend_angle = "auto"
        self._blend_angle = blend_angle
        self._max_rank = max_rank
        self._position_quantization = position_quantization
        self._max_batch_expansion = max_batch_expansion

        self._accelerator = Accelerator(energy=energy)
        # self._beam_tilt = BeamTilt(tilt=tilt)

        # window_gpts is only used by the compressed (upsampled) reduction; it is
        # not stored otherwise, so that copies of an SMatrix (which pass the
        # derived PRISM cropping window back through this argument) round-trip
        if not upsample:
            window_gpts = None
        elif isinstance(window_gpts, str):
            if window_gpts != "full":
                raise ValueError(
                    f"window_gpts must be an int, a pair of ints, 'full' or "
                    f"None (automatic); got {window_gpts!r}"
                )
        elif window_gpts is not None:
            if np.isscalar(window_gpts):
                window_gpts = (int(window_gpts),) * 2
            else:
                window_gpts = tuple(int(n) for n in window_gpts)
            if window_gpts == tuple(self.downsampled_gpts):
                # a window covering the full grid is no window; copies pass the
                # derived full-grid window back through this argument
                window_gpts = "full"
            elif max_batch_expansion != "auto":
                raise ValueError(
                    "max_batch_expansion applies to the reduction of the expanded "
                    "scattering matrix; it cannot be combined with window_gpts."
                )

        self._window_gpts = window_gpts

        self._store_on_host = store_on_host

        assert semiangle_cutoff > 0.0

        if not self._upsample and not all(
            n % f == 0 for f, n in zip(self.interpolation, self.gpts)
        ):
            warnings.warn(
                "The interpolation factor does not exactly divide 'gpts', "
                "normalization may not be exactly preserved."
            )

    @property
    def base_shape(self) -> tuple[int, int, int]:
        """Shape of the base axes of the SMatrix."""
        return len(self), self.gpts[0], self.gpts[1]

    @property
    def tilt(self):
        """The small-angle tilt of applied to the Fresnel propagator [mrad]."""
        return 0.0, 0.0

    def round_gpts_to_interpolation(self) -> SMatrix:
        """
        Round the gpts of the SMatrix to the closest multiple of the interpolation
        factor.

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
    def downsample(self) -> str | bool:
        """How to downsample the scattering matrix after running the multislice
        algorithm."""
        return self._downsample

    @property
    def store_on_host(self) -> bool:
        """Store the SMatrix in host memory. The reduction may still be calculated on
        the device."""
        return self._store_on_host

    @property
    def metadata(self):
        return {"energy": self.energy}

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the SMatrix."""
        return self.ensemble_shape + (len(self),) + self.gpts

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        """Shape of the SMatrix ensemble axes."""
        if self.potential is None:
            return ()
        else:
            return self.potential.ensemble_shape

    @property
    def ensemble_axes_metadata(self):
        """Axis metadata for each ensemble axis."""
        if self.potential is None:
            return []
        else:
            return self.potential.ensemble_axes_metadata

    @property
    def wave_vectors(self) -> np.ndarray:
        """The wave vectors of the plane-wave expansion. When upsampling, the coarse
        expansion spans a disk around the aperture (see :meth:`_coarse_mask`),
        padding it by a support margin so that the interpolation of the compressed
        modes is supported on all sides."""
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()

        if self._upsample_enabled:
            bounds = self._coarse_bounds()

            n = np.arange(-bounds[0], bounds[0] + 1, dtype=np.float32)
            m = np.arange(-bounds[1], bounds[1] + 1, dtype=np.float32)

            w, h = self.extent

            kx = n / w * np.float32(self.interpolation[0])
            ky = m / h * np.float32(self.interpolation[1])

            kx, ky = np.meshgrid(kx, ky, indexing="ij")

            mask = self._coarse_mask()
            xp = get_array_module(self.device)
            return xp.asarray([kx.ravel()[mask], ky.ravel()[mask]]).T

        dummy_probes = self.dummy_probes(device="cpu")

        aperture = dummy_probes.aperture._evaluate_kernel(dummy_probes)

        indices = np.where(aperture > 0.0)

        n = np.fft.fftfreq(aperture.shape[0], d=1 / aperture.shape[0])[indices[0]]
        m = np.fft.fftfreq(aperture.shape[1], d=1 / aperture.shape[1])[indices[1]]

        w, h = self.extent

        kx = n / w * np.float32(self.interpolation[0])
        ky = m / h * np.float32(self.interpolation[1])

        xp = get_array_module(self.device)
        return xp.asarray([kx, ky]).T

    @property
    def potential(self) -> BasePotential:
        """The potential described by the SMatrix."""
        return self._potential

    @potential.setter
    def potential(self, potential: BasePotential):
        self._potential = potential
        self._grid = potential.grid

    @property
    def semiangle_cutoff(self) -> float:
        """Plane-wave expansion cutoff."""
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        self._semiangle_cutoff = value

    @property
    def interpolation(self) -> tuple[int, int]:
        return self._interpolation

    @property
    def upsample(self) -> bool:
        """Interpolate the coarse plane-wave expansion back to the full expansion of
        the aperture and compress it (the C-PRISM algorithm)."""
        return self._upsample

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
    def max_batch_expansion(self) -> int | str:
        """Number of plane waves expanded at a time by the reduction of the
        compressed scattering matrix; 'auto' materializes the full expansion."""
        return self._max_batch_expansion

    @property
    def blend_angle(self) -> float | str | None:
        """Scattering angle [mrad] above which the reduction follows the
        plane-wave (PRISM) reduction of the built beams; 'auto' derives it from
        the aliasing limit of the interpolation, None disables blending."""
        return self._blend_angle

    def _resolved_blend_angle(self) -> float | None:
        """The blend angle in mrad, resolving 'auto' from the aliasing limit.

        An electron scattered to an angle theta drifts theta * t laterally over
        the thickness t, and the band-limited interpolation aliases once the
        drift exceeds half its period extent / interpolation. The interpolated
        reduction is therefore trusted up to::

            theta_max = min_i extent_i / (2 * interpolation_i * t)
        """
        if self._blend_angle is None:
            return None
        if not isinstance(self._blend_angle, str):
            return float(self._blend_angle)
        if self._blend_angle == "aperture":
            return "aperture"
        if self._blend_angle != "auto":
            raise ValueError(
                f"blend_angle must be a number, 'auto', 'aperture' or None; "
                f"got {self._blend_angle!r}"
            )
        thickness = self.potential.thickness if self.potential is not None else 0.0
        if thickness <= 0.0:
            return None
        angle = 1e3 * min(
            extent / (2.0 * interpolation * thickness)
            for extent, interpolation in zip(self.extent, self.interpolation)
        )
        if angle < self.semiangle_cutoff:
            # blending below the bright-field disk imports the periodized ghost
            # probes of the plane-wave reduction; clamp to the aperture edge and
            # warn that the interpolation is aliased inside the disk itself
            warnings.warn(
                "The interpolation of the compressed scattering matrix is "
                f"aliased above {angle:.1f} mrad, inside the bright-field disk "
                f"({self.semiangle_cutoff:.1f} mrad): the interpolation factor "
                "is too large for this thickness and the accuracy will be "
                "degraded at every angle. The blend is clamped to the aperture "
                "edge."
            )
            return "aperture"

        # theta_max bounds where the interpolation is *valid*, which is the
        # right default when the goal is accuracy against multislice: below it
        # the extra beams carry real information and usually beat the sparse
        # plane-wave sampling. It does not, however, promise the interpolated
        # reduction beats PRISM on every band below it. Pass
        # ``blend_angle='aperture'`` to confine the interpolation to the
        # bright-field disk, which trades some low-angle accuracy for
        # PRISM-or-better on every dark-field band of any specimen.
        return angle
        return angle

    @property
    def _upsample_enabled(self) -> bool:
        """The compression applies only when the coarse expansion is incomplete; at
        an interpolation factor of (1, 1) the scattering matrix is identical to the
        PRISM scattering matrix."""
        return self._upsample and self.interpolation != (1, 1)

    def _wave_vector_chunks(self, max_batch):
        if isinstance(max_batch, int):
            max_batch = max_batch * reduce(operator.mul, self.gpts)

        chunks = validate_chunks(
            shape=(len(self),) + self.gpts,
            chunks=("auto", -1, -1),
            max_elements=max_batch,
            dtype=np.dtype("complex64"),
            device=self.device,
        )
        return chunks

    @property
    def downsampled_gpts(self) -> tuple[int, int]:
        """The gpts of the SMatrix after downsampling. When upsampling, the
        downsampled gpts are independent of the interpolation factor, hence probe
        positions commensurate with the grid remain commensurate at any
        interpolation."""
        if self.downsample:
            downsampled_gpts = self._gpts_within_angle(self.downsample)
            if self._upsample:
                return tuple(n + (-n) % 4 for n in downsampled_gpts)
            rounded = _round_gpts_to_multiple_of_interpolation(
                downsampled_gpts, self.interpolation
            )
            return rounded
        else:
            return self.gpts

    # empirical extent of the reduced wave functions, calibrated against
    # multislice on thick cells: the exit wave spreads by roughly twice
    # thickness x aperture through multiple scattering, and the aperture-limited
    # probe carries tails of several Airy lobes
    _WINDOW_SPREAD_FACTOR = 2.0
    _WINDOW_TAIL_LOBES = 6.0

    def _auto_window_gpts(self):
        """The cropping window inferred from the specimen and the probe, or None
        (the full grid) when there is no potential to infer it from."""
        if self._potential is None:
            return None

        thickness = self._potential.thickness
        alpha = self._semiangle_cutoff * 1e-3
        if alpha <= 0.0 or thickness is None:
            return None

        half_extent = (
            self._WINDOW_SPREAD_FACTOR * thickness * alpha
            + self._WINDOW_TAIL_LOBES * self.wavelength / alpha
        )

        window = ()
        for extent, gpts, interpolation in zip(
            self.extent, self.downsampled_gpts, self.interpolation
        ):
            n = int(np.ceil(2.0 * half_extent / (extent / gpts) / 16.0)) * 16
            n = max(n, safe_ceiling_int(gpts / interpolation))
            window += (min(n, gpts),)
        return window

    @property
    def window_gpts(self):
        """The number of grid points describing the cropping window of the reduced
        wave functions."""
        if self._upsample:
            if self._window_gpts == "full":
                return self.downsampled_gpts
            if self._window_gpts is None:
                window = self._auto_window_gpts()
                return self.downsampled_gpts if window is None else window

            return (
                min(self._window_gpts[0], self.downsampled_gpts[0]),
                min(self._window_gpts[1], self.downsampled_gpts[1]),
            )

        return (
            safe_ceiling_int(self.downsampled_gpts[0] / self.interpolation[0]),
            safe_ceiling_int(self.downsampled_gpts[1] / self.interpolation[1]),
        )

    @property
    def window_extent(self):
        sampling = (
            self.extent[0] / self.downsampled_gpts[0],
            self.extent[1] / self.downsampled_gpts[1],
        )

        return (
            self.window_gpts[0] * sampling[0],
            self.window_gpts[1] * sampling[1],
        )

    def multislice(
        self,
        potential=None,
        lazy: bool = None,
        max_batch: int | str = "auto",
    ):
        """


        Parameters
        ----------
        potential
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly.
            If not given, defaults to the setting in the user configuration file.
        max_batch : int or str, optional
            The number of expansion plane waves in each run of the multislice algorithm.

        Returns
        -------

        """
        s_matrix = self.__class__(
            potential=potential, **self._copy_kwargs(exclude=("potential",))
        )
        return s_matrix.build(lazy=lazy, max_batch=max_batch)

    @property
    def _default_ensemble_chunks(self):
        return self.potential._default_ensemble_chunks

    def _partition_args(self, chunks=(1,), lazy: bool = True):
        if self.potential is not None:
            return self.potential._partition_args(chunks, lazy=lazy)
        else:
            array = np.empty((), dtype=object)
            if lazy:
                array = da.from_array(array, chunks=1)
            return (array,)

    @staticmethod
    def _s_matrix(*args, potential_partial, **kwargs):
        potential = potential_partial(*args).item()
        s_matrix = SMatrix(potential=potential, **kwargs)
        return _wrap_with_array(s_matrix)

    def _from_partitioned_args(self, *args, **kwargs):
        if self.potential is not None:
            potential_partial = self.potential._from_partitioned_args()
            kwargs = self._copy_kwargs(exclude=("potential", "sampling", "extent"))
        else:

            def potential_partial(*args, **kwargs):
                return _wrap_with_array(None, 1)

            # potential_partial = lambda *args, **kwargs: _wrap_with_array(None, 1)
            kwargs = self._copy_kwargs(exclude=("potential",))

        return partial(self._s_matrix, potential_partial=potential_partial, **kwargs)

    @staticmethod
    def _wave_vector_blocks(wave_vector_chunks, lazy: bool = True):
        wave_vector_blocks = chunk_ranges(wave_vector_chunks)[0]

        array = np.zeros(len(wave_vector_blocks), dtype=object)
        for i, wave_vector_block in enumerate(wave_vector_blocks):
            itemset(array, i, wave_vector_block)

        if lazy:
            array = da.from_array(array, chunks=1)
        return array

    @staticmethod
    def _build_s_matrix(s_matrix, wave_vector_range=slice(None), pbar: bool = False):
        if isinstance(s_matrix, np.ndarray):
            s_matrix = s_matrix.item()

        if isinstance(wave_vector_range, np.ndarray):
            wave_vector_range = slice(*wave_vector_range.item())

        xp = get_array_module(s_matrix.device)

        wave_vectors = xp.asarray(s_matrix.wave_vectors, dtype=xp.float32)

        array = plane_waves(
            wave_vectors[wave_vector_range], s_matrix.extent, s_matrix.gpts
        )

        array *= np.prod(s_matrix.interpolation) / np.prod(array.shape[-2:])

        waves = Waves(
            array,
            energy=s_matrix.energy,
            extent=s_matrix.extent,
            ensemble_axes_metadata=[
                OrdinalAxis(values=wave_vectors[wave_vector_range])
            ],
        )

        if s_matrix.potential is not None:
            waves = multislice_and_detect(
                waves, s_matrix.potential, [WavesDetector()], pbar=pbar
            )[0]

        if s_matrix.downsampled_gpts != s_matrix.gpts:
            waves.metadata["adjusted_antialias_cutoff_gpts"] = (
                waves.antialias_cutoff_gpts
            )

            waves = waves.downsample(
                gpts=s_matrix.downsampled_gpts,
                normalization="intensity",
            )

        if s_matrix.store_on_host and s_matrix.device == "gpu":
            waves = waves.to_cpu()

        return waves.array

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

    def _coarse_mask(self) -> np.ndarray:
        """Boolean mask over the raveled coarse bounding rectangle selecting the
        beams that are built (run through the multislice algorithm).

        Only the beams within a support disk around the aperture are kept: those
        with a normalized radius (unity at the aperture edge) within one coarse
        cell, or a fixed margin, of the aperture. The far corners of the bounding
        rectangle are dropped. At a coarse interpolation the grid under-samples
        the scattering matrix at those corners, and including them makes the
        trigonometric interpolation overfit; dropping them both reduces the
        number of multislice runs and improves the interpolation, most strongly
        at large interpolation factors where the corners dominate the rectangle.
        """
        bounds = self._coarse_bounds()
        dense_indices = self._dense_indices()
        n_max = [max(1, int(np.abs(dense_indices[:, i]).max())) for i in range(2)]

        n = np.arange(-bounds[0], bounds[0] + 1)
        m = np.arange(-bounds[1], bounds[1] + 1)

        radius_x = (n[:, None] * self.interpolation[0]) / n_max[0]
        radius_y = (m[None, :] * self.interpolation[1]) / n_max[1]
        radius = np.sqrt(radius_x**2 + radius_y**2)

        cell = max(self.interpolation[0] / n_max[0], self.interpolation[1] / n_max[1])
        keep_radius = 1.0 + max(cell, _COARSE_SUPPORT_MARGIN)
        return (radius <= keep_radius).ravel()

    def _coarse_fill_indices(self) -> np.ndarray:
        """For every position of the coarse bounding rectangle, the index (into
        the built, disk-masked beams) of the nearest built beam.

        The dropped corners of the rectangle are filled by the nearest built beam
        rather than by zeros, so that the trigonometric interpolation extends the
        scattering matrix smoothly into the corners instead of dropping a
        discontinuity there. In particular a rank-one (vacuum) scattering matrix,
        constant over the built beams, is filled to a constant and reconstructed
        exactly.
        """
        from scipy import ndimage

        bounds = self._coarse_bounds()
        shape = (2 * bounds[0] + 1, 2 * bounds[1] + 1)
        kept = self._coarse_mask().reshape(shape)

        nearest = ndimage.distance_transform_edt(
            ~kept, return_distances=False, return_indices=True
        )
        nearest_flat = (nearest[0] * shape[1] + nearest[1]).ravel()

        # map a flat rectangle index to its position in the built (kept) order
        rectangle_to_kept = np.full(int(np.prod(shape)), -1, dtype=int)
        rectangle_to_kept[np.flatnonzero(kept.ravel())] = np.arange(int(kept.sum()))
        return rectangle_to_kept[nearest_flat]

    def _interpolate_beam_functions(
        self, functions, dense_indices, apodization=None
    ) -> np.ndarray:
        """Trigonometric interpolation of functions of the coarse plane waves
        (given as rows over the coarse rectangle) to the dense plane-wave
        expansion.

        The trigonometric interpolant is band limited: it does not alias the
        interpolation error to displaced copies of the probe. A local (spline)
        interpolant leaks such attenuated ghost probes displaced by
        extent/interpolation, which an annular detector integrates as a large
        error even when the interpolant is more accurate in the mean-square
        sense.
        """
        xp = get_array_module(functions)
        dtype = get_dtype(complex=True)

        bounds = self._coarse_bounds()
        shape = (2 * bounds[0] + 1, 2 * bounds[1] + 1)

        # the built beams span a disk, not the full rectangle; extend them into
        # the rectangle expected by the (rectangular) fft by filling the dropped
        # corners with the nearest built beam (smooth, and exact for a rank-one
        # scattering matrix) rather than with zeros
        num_rectangle = int(np.prod(shape))
        if functions.shape[-1] != num_rectangle:
            fill_indices = xp.asarray(self._coarse_fill_indices())
            functions = functions[..., fill_indices]

        functions = functions.reshape((-1,) + shape)

        coefficients = xp.fft.fft2(functions, axes=(-2, -1))
        coefficients /= get_dtype(complex=False)(np.prod(shape))

        # The fft coefficients live in drift space: coefficient m carries the
        # content of electrons displaced laterally by m / length periods of
        # extent / interpolation. The plain interpolant weights them with a
        # sharp box over one period; an apodization rolls the weight off
        # smoothly toward the period boundary, where the content is aliased
        # (drift beyond half a period folds back), trading its broadband
        # ringing for a smooth attenuation. None keeps the box.
        if apodization is not None:
            plateau, edge = apodization
            for axis, length, interpolation in zip(
                (-2, -1), shape, self.interpolation
            ):
                if interpolation == 1:
                    # nothing is aliased along an uninterpolated axis
                    continue
                xi = np.abs(np.fft.fftfreq(length))
                window = np.ones(length)
                roll = xi > plateau / 2.0
                window[roll] = 0.5 * (
                    1.0
                    + np.cos(
                        np.pi * (xi[roll] - plateau / 2.0) / (0.5 - plateau / 2.0)
                    )
                )
                window = edge + (1.0 - edge) * window
                orientation = (
                    (slice(None), None) if axis == -2 else (None, slice(None))
                )
                coefficients *= xp.asarray(
                    window[orientation], dtype=get_dtype(complex=False)
                )

        kernels = ()
        for i, (bound, length) in enumerate(zip(bounds, shape)):
            frequencies = xp.fft.fftfreq(length, d=1 / length).astype(int)
            dense_coordinate = (
                xp.arange(
                    -bound * self.interpolation[i], bound * self.interpolation[i] + 1
                )
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

    def _defocus_phase(self, wave_vectors) -> np.ndarray:
        """The propagation (defocus) phase :math:`\\exp(-i \\pi \\lambda t |k|^2)`
        accumulated by each plane wave propagating through the cell of thickness
        :math:`t`.

        Factoring this phase out of the scattering matrix before the
        interpolation (and adding it back on the dense plane waves) references the
        beams to the entrance surface, where the probe is focused. It flattens the
        quadratic phase variation of the phase-removed scattering matrix across
        the aperture, which the coarse interpolation would otherwise have to
        capture, improving the accuracy at no additional cost. In vacuum, or for a
        zero-thickness potential, the phase is unity and the compression is
        unchanged.
        """
        thickness = self.potential.thickness if self.potential is not None else 0.0
        wavelength = self.wavelength
        squared_wave_vectors = wave_vectors[..., 0] ** 2 + wave_vectors[..., 1] ** 2
        return complex_exponential(
            -np.pi * wavelength * thickness * squared_wave_vectors
        )

    def _compress(self, array) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Phase removal, interpolation to the dense plane-wave expansion and
        exact truncated SVD of the interpolated operator.

        The plane-wave tilt phase and the propagation (defocus) phase are factored
        out of each beam before the interpolation, then added back on the dense
        plane waves (see :meth:`_defocus_phase`). The phase-removed scattering
        matrix is factored as :math:`T = L Q` with orthonormal :math:`Q`; the
        interpolation acts on the small beam-side factor :math:`L`, hence the
        singular value decomposition of the interpolated operator is obtained
        exactly without ever forming it.
        """
        xp = get_array_module(array)
        dtype = get_dtype(complex=True)

        gpts = array.shape[-2:]
        extent = self.extent

        wave_vectors = self.wave_vectors
        x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
        y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)

        array = array.reshape((-1,) + tuple(gpts))

        # the conjugate of the propagation phase flattens the quadratic phase
        # variation of the phase-removed matrix; it is added back below
        defocus_phase = self._defocus_phase(wave_vectors).conj()

        normalization = np.prod(self.interpolation).astype(np.float32)
        max_batch = max(1, int(256**3 / np.prod(gpts)))
        for start in range(0, len(array), max_batch):
            chunk = slice(start, start + max_batch)
            phase = complex_exponential(
                -2.0 * xp.pi * wave_vectors[chunk, 0, None, None] * x[:, None]
            ) * complex_exponential(
                -2.0 * xp.pi * wave_vectors[chunk, 1, None, None] * y[None, :]
            )
            array[chunk] *= phase * defocus_phase[chunk, None, None] / normalization

        matrix = array.reshape((len(array), -1))

        # T = L Q with the rows of Q orthonormal. The orthonormal factor is
        # obtained from the Gram matrix G = T T^H rather than a QR
        # factorization: the tall QR is far slower than the two matrix products
        # this costs (measured 5-14x) and materializes Q, a full extra copy of
        # the scattering matrix, whereas here Q is contracted on the fly. The
        # Gram matrix is accumulated in double precision so that its
        # eigenvalues resolve the singular values down to the smallest
        # supported tolerance, and the eigenvalues near the round-off floor
        # (the coarse expansion is rank deficient) are dropped.
        gram = xp.zeros((len(matrix), len(matrix)), dtype=np.complex128)
        pixel_batch = max(1, int(256**3 / max(len(matrix), 1)))
        for start in range(0, matrix.shape[1], pixel_batch):
            chunk = matrix[:, start : start + pixel_batch].astype(np.complex128)
            gram += chunk @ chunk.T.conj()

        eigenvalues, eigenvectors = xp.linalg.eigh(gram)
        eigenvalues = xp.clip(eigenvalues[::-1], 0.0, None)
        eigenvectors = eigenvectors[:, ::-1]

        # a round-off floor only: the tolerance must not truncate here, or the
        # row space of the built beams is already incomplete before the
        # interpolation and the plane-wave branch stops being exact
        keep = max(1, int((eigenvalues > eigenvalues[0] * 1e-14).sum()))
        singular_values = xp.sqrt(eigenvalues[:keep])

        # L = V diag(s) and Q = diag(1 / s) V^H T, with T = L Q exact on the
        # retained subspace
        beam_factor = (eigenvectors[:, :keep] * singular_values[None]).astype(dtype)

        dense_indices = self._dense_indices()

        # the interpolated operator T_dense = (P L) Q shares the pixel-side
        # factor Q, hence its exact SVD follows from the small matrix P L,
        # obtained by interpolating the columns of L over the coarse plane waves
        projected = self._interpolate_beam_functions(
            xp.ascontiguousarray(beam_factor.T), xp.asarray(dense_indices)
        ).T
        projected = xp.ascontiguousarray(projected.astype(dtype))

        # The retained subspace is chosen in two parts rather than by singular
        # value alone. The plane-wave (PRISM) branch of the reduction uses only
        # the rows of the dense expansion that coincide with built beams, so
        # their row space is retained WHOLE; the leading directions of what is
        # left over are then retained by tolerance. This makes the plane-wave
        # branch exact at any tolerance — the blended reduction is bounded by
        # PRISM on every band it covers — while the tolerance still controls the
        # cost of the interpolated part.
        moment = projected.conj().T @ projected

        def leading(hermitian, threshold):
            values, vectors = xp.linalg.eigh(hermitian)
            values = xp.clip(values[::-1], 0.0, None)
            count = int((xp.sqrt(values) >= threshold).sum())
            return vectors[:, ::-1][:, :count].T.conj(), xp.sqrt(values)

        _, spectrum = leading(moment, 0.0)
        largest = float(spectrum[0]) if len(spectrum) else 0.0

        lattice = (dense_indices[:, 0] % self._interpolation[0] == 0) & (
            dense_indices[:, 1] % self._interpolation[1] == 0
        )
        # the floor sits above the single-precision noise of the coarse matrix
        # (~1e-7 relative) and far below any usable tolerance, so the row space
        # is captured whole without admitting round-off directions
        lattice_rows = projected[xp.asarray(lattice)]
        lattice_basis, _ = leading(
            lattice_rows.conj().T @ lattice_rows, largest * 1e-6
        )

        # what the built beams do not already span, by tolerance
        residual = xp.eye(moment.shape[0], dtype=dtype)
        residual = residual - lattice_basis.conj().T @ lattice_basis
        extra_basis, _ = leading(
            residual @ moment @ residual, self._tolerance * largest
        )

        w = xp.concatenate([lattice_basis, extra_basis], axis=0)

        # order the retained modes by how much of the expansion they carry, so
        # that a rank cap keeps the largest and the reported spectrum descends
        amplitudes = xp.ascontiguousarray((projected @ w.conj().T).T)
        order = xp.argsort(xp.linalg.norm(amplitudes, axis=1))[::-1]
        if self._max_rank is not None:
            order = order[: max(1, self._max_rank)]
        w, amplitudes = w[order], xp.ascontiguousarray(amplitudes[order])
        rank = max(1, len(w))

        # U = W Q = (W diag(1 / s) V^H) T without ever materializing Q
        project = (
            w * (1.0 / singular_values)[None].astype(dtype)
        ) @ eigenvectors[:, :keep].T.conj().astype(dtype)
        u = (project @ matrix).reshape((rank,) + tuple(gpts))

        # the dense plane-wave amplitudes of each retained mode
        sigma = xp.linalg.norm(amplitudes, axis=1)
        vh_dense = amplitudes / xp.clip(sigma, 1e-30, None)[:, None]
        singular_values = spectrum

        # the same modes carry a second set of dense amplitudes from the
        # apodized (smooth) interpolant, used by the interpolated branch of the
        # reduction; the exact (box) amplitudes above are what make the
        # plane-wave branch the PRISM algorithm
        smooth_values = None
        if _SMOOTH_APODIZATION is not None and max(self.interpolation) > 1:
            projected_smooth = self._interpolate_beam_functions(
                xp.ascontiguousarray(beam_factor.T),
                xp.asarray(dense_indices),
                apodization=_SMOOTH_APODIZATION,
            ).T
            smooth_values = xp.ascontiguousarray(
                (projected_smooth.astype(dtype) @ w.conj().T).T
            )

        # add the propagation phase back on the dense plane waves
        dense_wave_vectors = xp.asarray(dense_indices, dtype=np.float32)
        dense_wave_vectors = dense_wave_vectors / xp.asarray(extent, dtype=np.float32)
        defocus_phase = self._defocus_phase(dense_wave_vectors)[None].astype(dtype)
        vh_dense = vh_dense * defocus_phase
        if smooth_values is not None:
            smooth_values = smooth_values * defocus_phase

        return (
            u,
            sigma[:rank].astype(get_dtype(complex=False)),
            vh_dense.astype(dtype),
            dense_indices,
            singular_values.astype(get_dtype(complex=False)),
            None if smooth_values is None else smooth_values.astype(dtype),
        )

    def build(
        self, lazy: bool = None, max_batch: int | str = "auto", bound: bool = None
    ) -> SMatrixArray | CompressedSMatrixArray:
        """
        Build the plane waves of the scattering matrix and propagate them through the
        potential using the multislice algorithm.

        When ``upsample=True``, the scattering matrix is subsequently compressed by
        phase removal, interpolation to the full plane-wave expansion and an
        adaptive truncated singular value decomposition, and a
        :class:`.CompressedSMatrixArray` is returned. The multislice stage may be
        computed lazily, however, the compression requires the scattering matrix in
        memory, hence the returned :class:`.CompressedSMatrixArray` is always
        computed. At an interpolation factor of (1, 1) the plane-wave expansion is
        complete, hence the compression provides no benefit and the uncompressed
        :class:`.SMatrixArray` is returned; the reduction is then identical to the
        PRISM algorithm.

        Parameters
        ----------
        lazy : bool, optional
            If True, create the wave functions lazily, otherwise, calculate instantly.
            If not given, defaults to the setting in the user configuration file.
        max_batch : int or str, optional
            The number of expansion plane waves in each run of the multislice algorithm.

        Returns
        -------
        s_matrix_array : SMatrixArray or CompressedSMatrixArray
            The built scattering matrix.
        """
        if self._upsample_enabled and np.prod(self.ensemble_shape) > 1:
            raise NotImplementedError(
                "SMatrix.build does not support ensemble potentials with "
                "upsample=True; use SMatrix.reduce or SMatrix.scan, which average "
                "over the potential ensemble."
            )

        lazy = validate_lazy(lazy)

        downsampled_gpts = self.downsampled_gpts

        s_matrix_blocks = self.ensemble_blocks(1)

        xp = get_array_module(self.device)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        if lazy:
            wave_vector_blocks = self._wave_vector_blocks(
                wave_vector_chunks, lazy=False
            )

            if not hasattr(s_matrix_blocks, "len"):
                s_matrix_blocks = s_matrix_blocks[None]

            if self.potential is not None and self.potential.ensemble_shape:
                s_matrix_blocks = s_matrix_blocks[0]

            wave_vector_blocks = np.tile(
                wave_vector_blocks[None], (len(s_matrix_blocks), 1)
            )

            wave_vector_blocks = da.from_array(wave_vector_blocks, chunks=1)

            from dask.graph_manipulation import bind

            if bound is not None:
                wave_vector_blocks = bind(wave_vector_blocks, bound)

            adjust_chunks = {
                1: wave_vector_chunks[0],
                2: (downsampled_gpts[0],),
                3: (downsampled_gpts[1],),
            }

            symbols = (0, 1, 2, 3)
            if self.potential is None or not self.potential.ensemble_shape:
                symbols = symbols[1:]

            pbar = config.get("diagnostics.task_progress", False)

            array = da.blockwise(
                self._build_s_matrix,
                symbols,
                s_matrix_blocks,
                (0,),
                wave_vector_blocks[..., None, None],
                (0, 1, 2, 3),
                concatenate=True,
                adjust_chunks=adjust_chunks,
                pbar=pbar,
                meta=xp.array((), dtype=get_dtype(complex=True)),
            )

        else:
            wave_vector_blocks = self._wave_vector_blocks(
                wave_vector_chunks, lazy=False
            )

            if self.store_on_host:
                array = np.zeros(
                    self.ensemble_shape + (len(self),) + self.downsampled_gpts,
                    dtype=np.complex64,
                )
            else:
                array = xp.zeros(
                    self.ensemble_shape + (len(self),) + self.downsampled_gpts,
                    dtype=np.complex64,
                )

            pbar = config.get("diagnostics.task_progress", False)

            for i, _, s_matrix in self.generate_blocks(1):
                s_matrix = s_matrix.item()
                for start, stop in wave_vector_blocks:
                    items = (slice(start, stop),)
                    if self.ensemble_shape:
                        items = i + items

                    new_array = self._build_s_matrix(
                        s_matrix, slice(start, stop), pbar=pbar
                    )

                    if self.store_on_host:
                        new_array = xp.asnumpy(new_array)

                    array[items] = new_array

        waves = Waves(
            array,
            energy=self.energy,
            extent=self.extent,
            ensemble_axes_metadata=self.ensemble_axes_metadata
            + self.base_axes_metadata[:1],
        )

        if self.downsampled_gpts != self.gpts:
            waves.metadata["adjusted_antialias_cutoff_gpts"] = _antialias_cutoff_gpts(
                self.window_gpts, self.sampling
            )

        s_matrix_array = SMatrixArray._from_waves(
            waves,
            wave_vectors=self.wave_vectors,
            interpolation=self.interpolation,
            semiangle_cutoff=self.semiangle_cutoff,
            window_gpts=self.window_gpts,
            device=self.device,
        )

        if not self._upsample_enabled:
            return s_matrix_array

        compress_array = s_matrix_array.array
        if s_matrix_array.is_lazy:
            compress_array = compress_array.compute()

        metadata = dict(s_matrix_array.metadata)

        pbar = config.get("diagnostics.task_progress", False)
        if pbar:
            print(
                f"compressing scattering matrix: {len(self)} plane waves "
                f"interpolated to {len(self._dense_indices())}",
                flush=True,
            )

        u, sigma, vh_dense, dense_indices, singular_values, smooth_values = (
            self._compress(compress_array)
        )

        if pbar:
            print(f"kept {len(sigma)} modes at tolerance {self._tolerance:g}")

        return CompressedSMatrixArray(
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
            max_batch_expansion=self._max_batch_expansion,
            blend_angle=self._resolved_blend_angle(),
            device=self.device,
            metadata=metadata,
            singular_values=singular_values,
            smooth_values=smooth_values,
        )

    def scan(
        self,
        scan: np.ndarray | BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF | dict = None,
        max_batch_multislice: str | int = "auto",
        max_batch_reduction: str | int = "auto",
        reduction_scheme: str = "auto",
        disable_s_matrix_chunks: bool = "auto",
        lazy: bool = None,
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Run the multislice algorithm, then reduce the SMatrix using coefficients
        calculated by a BaseScan and a CTF, to obtain the exit wave functions at given
        initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire
            potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be
            converted to measurements after running the multislice algorithm.
            See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion
            coefficients in the reduction of the SMatrix.
        max_batch_multislice : int, optional
            The number of wave functions in each chunk of the Dask array.
            If 'auto' (default), the batch size is automatically chosen based on the
            abTEM user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions
            better utilize thread parallelization, but requires more memory and floating
            point operations. If 'auto' (default), the batch size is automatically
            chosen based on the abtem user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        reduction_scheme : str or tuple of int, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from
            chunking along the expansion axis to chunking over the spatial axes.
            If given as a tuple of int of length the SMatrix is rechunked to have those
            chunks. If 'auto' (default) the chunks are taken to be identical to the
            interpolation factor.
        disable_s_matrix_chunks : bool, optional
            If True, each S-Matrix is kept as a single chunk, thus lowering the
            communication overhead, but providing fewer opportunities for
            parallelization.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly.
            If None, this defaults to the value set in the configuration file.

        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s)
            given).
        """

        if scan is None:
            scan = GridScan(
                start=(0, 0),
                end=self.extent,
                sampling=self.dummy_probes().aperture.nyquist_sampling,
            )

        if detectors is None:
            detectors = FlexibleAnnularDetector()
        return self.reduce(
            scan=scan,
            detectors=detectors,
            max_batch_reduction=max_batch_reduction,
            max_batch_multislice=max_batch_multislice,
            ctf=ctf,
            reduction_scheme=reduction_scheme,
            disable_s_matrix_chunks=disable_s_matrix_chunks,
            lazy=lazy,
        )

    def _eager_build_s_matrix_detect(self, scan, ctf, detectors, squeeze):
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

        if self.potential is not None and len(self.potential.exit_planes) > 1:
            extra_ensemble_axes_shape = extra_ensemble_axes_shape + (
                len(self.potential.exit_planes),
            )
            extra_ensemble_axes_metadata = extra_ensemble_axes_metadata + [
                self.potential.base_axes_metadata[0]
            ]

        detectors = validate_detectors(detectors)

        if self.ensemble_shape:
            if self._upsample_enabled:
                # building would compute the compression of a single ensemble
                # member eagerly (and raises for ensemble potentials); the
                # builder's dummy probes carry the same grid and metadata
                dummy_probes = self.dummy_probes(scan, ctf)
            else:
                dummy_probes = self.build(lazy=True).dummy_probes(scan, ctf)

            measurements = allocate_multislice_measurements(
                dummy_probes,
                detectors,
                extra_ensemble_axes_shape,
                extra_ensemble_axes_metadata,
            )
        else:
            measurements = None

        num_blocks = 0
        for i, _, s_matrix in self.generate_blocks(1):
            s_matrix = s_matrix.item()
            s_matrix_array = s_matrix.build(lazy=False)

            new_measurements = s_matrix_array.reduce(
                scan=scan, detectors=detectors, ctf=ctf
            )

            new_measurements = ensure_list(new_measurements)

            if measurements is None:
                measurements = new_measurements
            else:
                for measurement, new_measurement in zip(measurements, new_measurements):
                    if measurement.axes_metadata[0]._ensemble_mean:
                        measurement.array[:] += new_measurement.array
                    else:
                        measurement.array[i] = new_measurement.array

            num_blocks += 1

        # measurements = list(measurements.values())

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
    def _lazy_build_s_matrix_detect(s_matrix, scan, ctf, detectors):
        s_matrix = s_matrix.item()
        measurements = s_matrix._eager_build_s_matrix_detect(
            scan=scan, ctf=ctf, detectors=detectors, squeeze=False
        )
        # measurements = ensure_list(measurements)

        array = np.zeros((1,) + (1,) * len(scan.shape), dtype=object)
        itemset(array, 0, measurements)

        return array

    def reduce(
        self,
        scan: np.ndarray | BaseScan = None,
        detectors: BaseDetector | list[BaseDetector] = None,
        ctf: CTF | dict = None,
        reduction_scheme: str = "auto",
        max_batch_multislice: str | int = "auto",
        max_batch_reduction: str | int = "auto",
        disable_s_matrix_chunks: bool = "auto",
        lazy: bool = None,
    ) -> BaseMeasurements | Waves | list[BaseMeasurements | Waves]:
        """
        Run the multislice algorithm, then reduce the SMatrix using coefficients
        calculated by a BaseScan and a CTF, to obtain the exit wave functions at given
        initial probe positions and aberrations.

        Parameters
        ----------
        scan : BaseScan
            Positions of the probe wave functions. If not given, scans across the entire
            potential at Nyquist sampling.
        detectors : BaseDetector, list of BaseDetector, optional
            A detector or a list of detectors defining how the wave functions should be
            converted to measurements after running the multislice algorithm.
            See abtem.measurements.detect for a list of implemented detectors.
        ctf : CTF
            Contrast transfer function from used for calculating the expansion
            coefficients in the reduction of the SMatrix.
        max_batch_multislice : int, optional
            The number of wave functions in each chunk of the Dask array.
            If 'auto' (default), the batch size is automatically chosen based on the
            abTEM user configuration settings "dask.chunk-size" and
            "dask.chunk-size-gpu".
        max_batch_reduction : int or str, optional
            Number of positions per reduction operation. A large number of positions
            better utilize thread parallelization, but requires more memory and floating
            point operations. If 'auto' (default), the batch size
            is automatically chosen based on the abtem user configuration settings
            "dask.chunk-size" and "dask.chunk-size-gpu".
        reduction_scheme : str, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from
            chunking along the expansion axis to chunking over the spatial axes.
            If given as a tuple of int of length the SMatrix is rechunked to have those
            chunks. If 'auto' (default) the chunks are taken to be identical to the
            interpolation factor.
        disable_s_matrix_chunks : bool, optional
            If True, each S-Matrix is kept as a single chunk, thus lowering the
            communication overhead, but providing fewer opportunities for
            parallelization.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly.
            If None, this defaults to the value set in the configuration file.

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurements or list of
        Waves
            The detected measurement (if detector(s) given).
        """

        if self._upsample_enabled:
            # the compressed scattering matrix spans the full downsampled grid
            detectors = validate_detectors(detectors, self.dummy_probes())
        else:
            detectors = validate_detectors(
                detectors, self.dummy_probes(downsample=False)
            )

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        lazy = validate_lazy(lazy)

        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff)
        elif isinstance(ctf, dict):
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff, **ctf)

        if self._upsample_enabled:
            # the compression requires the scattering matrix in memory, hence each
            # member of the potential ensemble is built and reduced as one task
            disable_s_matrix_chunks = True
        elif self.device == "gpu" and disable_s_matrix_chunks == "auto":
            disable_s_matrix_chunks = True
        elif disable_s_matrix_chunks == "auto":
            disable_s_matrix_chunks = False

        if not lazy:
            scan = validate_scan(scan, self)

            measurements = self._eager_build_s_matrix_detect(
                scan, ctf, detectors, squeeze=True
            )
            return _wrap_measurements(measurements)

        if disable_s_matrix_chunks:
            scan = validate_scan(scan, self)

            blocks = self.ensemble_blocks(1)

            chunks = ()
            drop_axis = ()
            if not self.ensemble_shape:
                blocks = blocks[None]  # expand 0-d to 1-d so drop_axis=(0,) is valid
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
                self._lazy_build_s_matrix_detect,
                drop_axis=drop_axis,
                new_axis=new_axis,
                chunks=chunks,
                scan=scan,
                ctf=ctf,
                detectors=detectors,
                meta=np.array((), dtype=object),
            )

            if self._upsample_enabled:
                # building would compute the compression eagerly; the builder's
                # dummy probes carry the same grid and metadata
                waves = self.dummy_probes(scan=scan)
            else:
                waves = self.build(lazy=True).dummy_probes(scan=scan)

            extra_axes_metadata = []
            if self.potential is not None:
                extra_axes_metadata = self.potential.ensemble_axes_metadata

            extra_axes_metadata = extra_axes_metadata + ctf.ensemble_axes_metadata

            measurements = _finalize_lazy_measurements(
                arrays, waves, detectors, extra_axes_metadata
            )

            return _wrap_measurements(measurements)

        s_matrix_array = self.build(max_batch=max_batch_multislice, lazy=lazy)
        return s_matrix_array.reduce(
            scan=scan,
            detectors=detectors,
            reduction_scheme=reduction_scheme,
            max_batch_reduction=max_batch_reduction,
            ctf=ctf,
        )
