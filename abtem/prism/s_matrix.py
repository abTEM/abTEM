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

from abtem.array import ArrayObject, ComputableList, _validate_lazy
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
    WavesDetector,
    validate_detectors,
)
from abtem.measurements import BaseMeasurements
from abtem.multislice import allocate_multislice_measurements, multislice_and_detect
from abtem.potentials.iam import BasePotential, _validate_potential
from abtem.prism.utils import batch_crop_2d, minimum_crop, plane_waves, wrapped_crop_2d
from abtem.scan import BaseScan, GridScan, _validate_scan
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
        """The number of grid points describing the cropping window of the wave functions."""
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
        A probe or an ensemble of probes equivalent reducing the SMatrix at a single position.

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
            probes._positions = scan

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
    print(pad_amounts)
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
    print(partitions, s_matrix_array.sampling, s_matrix_array.shape)

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
    A scattering matrix defined by a given array of dimension 3, where the first indexes the probe plane waves and the
    latter two are the `y` and `x` scan directions.

    Parameters
    ----------
    array : np.ndarray
        Array defining the scattering matrix. Must be 3D or higher, dimensions before the last three dimensions should
        represent ensemble dimensions, the next dimension indexes the plane waves and the last two dimensions
        represent the spatial extent of the plane waves.
    wave_vectors : np.ndarray
        Array defining the wave vectors corresponding to each plane wave. Must have shape Nx2, where N is equal to the
        number of plane waves.
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    sampling : one or two float, optional
        Lateral sampling of wave functions [Å]. Provide only if potential is not given. Will be ignored if 'gpts'
        is also provided.
    extent : one or two float, optional
        Lateral extent of wave functions [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions (default is 1, ie. no interpolation). If a single value is
        provided, assumed to be the same for both directions.
    window_gpts : tuple of int
        The number of grid points describing the cropping window of the wave functions.
    window_offset : tuple of int
        The number of grid points from the origin the cropping windows of the wave functions is displaced.
    periodic: tuple of bool
        Specifies whether the SMatrix should be assumed to be periodic along the x and y-axis.
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu'). Default is 'cpu'. The default is
        determined by the user configuration.
    ensemble_axes_metadata : list of AxesMetadata
        Axis metadata for each ensemble axis. The axis metadata must be compatible with the shape of the array.
    metadata : dict
        A dictionary defining wave function metadata. All items will be added to the metadata of measurements derived
        from the waves.
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
        """The number of grid points from the origin the cropping windows of the wave functions is displaced."""
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

        position_coefficients = xp.array(position_coefficients, dtype=xp.complex64)

        # return xp.zeros(position_coefficients.shape[:2] + self.window_gpts, dtype=xp.complex64)

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
        array = array / xp.sqrt((array**2).sum(axis=-1, keepdims=True))
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
            squeeze_scan = True
            scan = self.extent[0] / 2, self.extent[1] / 2

        scan = _validate_scan(
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
            if reduction_scheme == "multiple-rechunk":
                measurements = _multiple_rechunk_reduce(
                    self, scan, detectors, ctf, max_batch_reduction, pbar=pbar
                )
            elif reduction_scheme == "single-rechunk":
                raise NotImplementedError
                measurements = _single_rechunk_reduce(
                    self, scan, detectors, ctf, max_batch_reduction
                )
            elif reduction_scheme == "no-chunks":
                measurements = _no_chunks_reduce(
                    self, scan, detectors, ctf, max_batch_reduction, pbar=pbar
                )

            else:
                raise ValueError()
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
        rechunk : str or tuple of int, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.

        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
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


class SMatrix(BaseSMatrix, Ensemble, CopyMixin, EqualityMixin):
    """
    The scattering matrix is used for simulating STEM experiments using the PRISM algorithm.

    Parameters
    ----------
    semiangle_cutoff : float
        The radial cutoff of the plane-wave expansion [mrad].
    energy : float
        Electron energy [eV].
    potential : Atoms or AbstractPotential, optional
        Atoms or a potential that the scattering matrix represents. If given as atoms, a default potential will be
        created. If nothing is provided the scattering matrix will represent a vacuum potential, in which case the
        sampling and extent must be provided.
    gpts : one or two int, optional
        Number of grid points describing the scattering matrix. Provide only if potential is not given.
    sampling : one or two float, optional
        Lateral sampling of scattering matrix [Å]. Provide only if potential is not given. Will be ignored if 'gpts'
        is also provided.
    extent : one or two float, optional
        Lateral extent of scattering matrix [Å]. Provide only if potential is not given.
    interpolation : one or two int, optional
        Interpolation factor in the `x` and `y` directions (default is 1, ie. no interpolation). If a single value is
        provided, assumed to be the same for both directions.
    downsample : {'cutoff', 'valid'} or float or bool
        Controls whether to downsample the scattering matrix after running the multislice algorithm.

            ``cutoff`` :
                Downsample to the antialias cutoff scattering angle (default).

            ``valid`` :
                Downsample to the largest rectangle that fits inside the circle with a radius defined by the antialias
                cutoff scattering angle.

            float :
                Downsample to a specified maximum scattering angle [mrad].
    device : str, optional
        The calculations will be carried out on this device ('cpu' or 'gpu'). Default is 'cpu'. The default is
        determined by the user configuration.
    store_on_host : bool, optional
        If True, store the scattering matrix in host (cpu) memory so that the necessary memory is transferred as chunks
        to the device to run calculations (default is False).
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
            potential = _validate_potential(potential)
            self.grid.match(potential)
            self._grid = potential.grid

        self._potential = potential
        self._interpolation = _validate_interpolation(interpolation)
        self._semiangle_cutoff = semiangle_cutoff
        self._downsample = downsample

        self._accelerator = Accelerator(energy=energy)
        # self._beam_tilt = BeamTilt(tilt=tilt)

        self._store_on_host = store_on_host

        assert semiangle_cutoff > 0.0

        if not all(n % f == 0 for f, n in zip(self.interpolation, self.gpts)):
            warnings.warn(
                "The interpolation factor does not exactly divide 'gpts', normalization may not be exactly preserved."
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
    def downsample(self) -> str | bool:
        """How to downsample the scattering matrix after running the multislice algorithm."""
        return self._downsample

    @property
    def store_on_host(self) -> bool:
        """Store the SMatrix in host memory. The reduction may still be calculated on the device."""
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
        self.grid.check_is_defined()
        self.accelerator.check_is_defined()
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
        """The gpts of the SMatrix after downsampling."""
        if self.downsample:
            downsampled_gpts = self._gpts_within_angle(self.downsample)
            rounded = _round_gpts_to_multiple_of_interpolation(
                downsampled_gpts, self.interpolation
            )
            return rounded
        else:
            return self.gpts

    @property
    def window_gpts(self):
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

    # @staticmethod
    # def _wrapped_build_s_matrix(*args, s_matrix_partial):
    #     s_matrix = s_matrix_partial(*tuple(arg.item() for arg in args[:-1]))
    #
    #     wave_vector_range = slice(*np.squeeze(args[-1]))
    #     array = s_matrix._build_s_matrix(wave_vector_range).array
    #     return array
    #
    # def _s_matrix_partial(self):
    #     def s_matrix(*args, potential_partial, **kwargs):
    #         if potential_partial is not None:
    #             potential = potential_partial(*args + (np.array([None], dtype=object),))
    #         else:
    #             potential = None
    #         return SMatrix(potential=potential, **kwargs)
    #
    #     potential_partial = (
    #         self.potential._from_partitioned_args()
    #         if self.potential is not None
    #         else None
    #     )
    #     return partial(
    #         s_matrix,
    #         potential_partial=potential_partial,
    #         **self._copy_kwargs(exclude=("potential",)),
    #     )

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

    @property
    def _default_ensemble_chunks(self):
        return self.potential._default_ensemble_chunks

    def _partition_args(self, chunks=(1,), lazy: bool = True):
        if self.potential is not None:
            return self.potential._partition_args(chunks, lazy=lazy)
        else:
            array = np.empty((1,), dtype=object)
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
            potential_partial = lambda *args, **kwargs: _wrap_with_array(None, 1)
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

    def build(
        self, lazy: bool = None, max_batch: int | str = "auto", bound: bool = None
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
        lazy = _validate_lazy(lazy)

        downsampled_gpts = self.downsampled_gpts

        s_matrix_blocks = self.ensemble_blocks(1)
        xp = get_array_module(self.device)

        wave_vector_chunks = self._wave_vector_chunks(max_batch)

        if lazy:
            wave_vector_blocks = self._wave_vector_blocks(
                wave_vector_chunks, lazy=False
            )

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

            for i, _, s_matrix in self.generate_blocks(1):
                s_matrix = s_matrix.item()
                for start, stop in wave_vector_blocks:
                    items = (slice(start, stop),)
                    if self.ensemble_shape:
                        items = i + items

                    new_array = self._build_s_matrix(s_matrix, slice(start, stop))

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

        return s_matrix_array

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
        reduction_scheme : str or tuple of int, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.
        disable_s_matrix_chunks : bool, optional
            If True, each S-Matrix is kept as a single chunk, thus lowering the communication overhead, but providing
            fewer opportunities for parallelization.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        detected_waves : BaseMeasurements or list of BaseMeasurement
            The detected measurement (if detector(s) given).
        exit_waves : Waves
            Wave functions at the exit plane(s) of the potential (if no detector(s) given).
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
            extra_ensemble_axes_shape += (shape,)

            if axis_metadata._ensemble_mean:
                extra_ensemble_axes_shape = (1,) + extra_ensemble_axes_shape[1:]

        if self.potential is not None and len(self.potential.exit_planes) > 1:
            extra_ensemble_axes_shape = extra_ensemble_axes_shape + (
                len(self.potential.exit_planes),
            )
            extra_ensemble_axes_metadata = extra_ensemble_axes_metadata + [
                self.potential.base_axes_metadata[0]
            ]

        detectors = validate_detectors(detectors)

        if self.ensemble_shape:
            measurements = allocate_multislice_measurements(
                self.build(lazy=True).dummy_probes(scan, ctf),
                detectors,
                extra_ensemble_axes_shape,
                extra_ensemble_axes_metadata,
            )
        else:
            measurements = None

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

        # measurements = list(measurements.values())

        for i, measurement in enumerate(measurements):
            if (
                hasattr(measurement.axes_metadata[0], "_ensemble_mean")
                and measurement.axes_metadata[0]._ensemble_mean
            ) and squeeze:
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
        reduction_scheme : str, optional
            Parallel reduction of the SMatrix requires rechunking the Dask array from chunking along the expansion axis
            to chunking over the spatial axes. If given as a tuple of int of length the SMatrix is rechunked to have
            those chunks. If 'auto' (default) the chunks are taken to be identical to the interpolation factor.
        disable_s_matrix_chunks : bool, optional
            If True, each S-Matrix is kept as a single chunk, thus lowering the communication overhead, but providing
            fewer opportunities for parallelization.
        lazy : bool, optional
            If True, create the measurements lazily, otherwise, calculate instantly. If None, this defaults to the value
            set in the configuration file.

        Returns
        -------
        measurements : BaseMeasurements or Waves or list of BaseMeasurements or list of Waves
            The detected measurement (if detector(s) given).
        """

        detectors = validate_detectors(detectors, self.dummy_probes(downsample=False))

        if scan is None:
            scan = (self.extent[0] / 2, self.extent[1] / 2)

        lazy = _validate_lazy(lazy)

        if ctf is None:
            ctf = CTF(semiangle_cutoff=self.semiangle_cutoff)

        if self.device == "gpu" and disable_s_matrix_chunks == "auto":
            disable_s_matrix_chunks = True
        elif disable_s_matrix_chunks == "auto":
            disable_s_matrix_chunks = False

        if not lazy:
            scan = _validate_scan(scan, self)

            measurements = self._eager_build_s_matrix_detect(
                scan, ctf, detectors, squeeze=True
            )
            return _wrap_measurements(measurements)

        if disable_s_matrix_chunks:
            scan = _validate_scan(scan, self)

            blocks = self.ensemble_blocks(1)

            chunks = ()
            drop_axis = ()
            if not self.ensemble_shape:
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
