import itertools

import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
from hypothesis import given
from utils import gpu

from abtem import AnnularDetector, PixelatedDetector, PlaneWave
from abtem.core.chunks import chunk_ranges, validate_chunks
from abtem.core.ensemble import _wrap_with_array, unpack_blockwise_args
from abtem.core.utils import itemset
from abtem.inelastic.phonons import BaseFrozenPhonons, FrozenPhonons, FrozenPhononsAxis
from abtem.scan import CustomScan


def to_numpy(array):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if hasattr(array, "get"):  # CuPy array
        return np.asarray(array.get())
    return np.asarray(array)


# @reproduce_failure('6.56.3', b'AXicY2BAAoxwhkUDA2HASppyFBtwMYEAAJNaAXw=')
@given(data=st.data())
@pytest.mark.parametrize("lazy", [True], ids=["not_lazy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("ensemble_mean", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        # abtem_st.plane_wave,
        # abtem_st.s_matrix,
    ],
)
def test_multislice_with_frozen_phonons(
    data, waves_builder, device, ensemble_mean, lazy
):
    waves = data.draw(waves_builder(device=device))
    frozen_phonons = data.draw(abtem_st.frozen_phonons())
    exit_waves = waves.multislice(frozen_phonons)

    assert exit_waves.shape[0] == len(frozen_phonons)
    assert len(exit_waves.shape) == len(waves.shape) + 1


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "not_lazy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("ensemble_mean", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
    ],
)
def test_multislice_detect_with_frozen_phonons(
    data, waves_builder, device, ensemble_mean, lazy
):
    waves = data.draw(waves_builder(device=device, allow_distribution=False))
    frozen_phonons = data.draw(abtem_st.frozen_phonons(ensemble_mean=ensemble_mean))

    detector = PixelatedDetector(max_angle=None)
    exit_waves = waves.multislice(frozen_phonons, detectors=detector, lazy=lazy)

    if ensemble_mean:
        assert exit_waves.shape == waves.shape
        assert len(exit_waves.shape) == len(waves.shape)
    else:
        assert exit_waves.shape[0] == len(frozen_phonons)
        assert len(exit_waves.shape) == len(waves.shape) + 1


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "not_lazy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("ensemble_mean", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.s_matrix,
    ],
)
def test_s_matrix_multislice_detect_with_frozen_phonons(
    data, waves_builder, device, ensemble_mean, lazy
):
    waves = data.draw(waves_builder(device=device, allow_distribution=False))
    frozen_phonons = data.draw(abtem_st.frozen_phonons(ensemble_mean=ensemble_mean))

    detector = PixelatedDetector(max_angle=None)

    exit_waves = waves.multislice(frozen_phonons).reduce(detectors=detector)

    if ensemble_mean:
        assert len(exit_waves.shape) == len(waves.shape) - 1
    else:
        assert exit_waves.shape[0] == len(frozen_phonons)
        assert len(exit_waves.shape) == len(waves.shape)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "not_lazy"])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
    ],
)
def test_multislice_thickness_series(data, waves_builder, device, lazy):
    waves = data.draw(waves_builder(device=device, allow_distribution=False))
    potential = data.draw(abtem_st.potential(exit_planes=True, ensemble_mean=False))
    exit_waves = waves.multislice(potential, lazy=lazy)

    if len(potential.exit_planes) > 1:
        assert exit_waves.shape[1] == len(potential.exit_planes)
    assert exit_waves.shape[0] == potential.num_configurations
    assert exit_waves.gpts == potential.gpts


@pytest.mark.slow
@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("frozen_phonons", [True, False])
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
        abtem_st.annular_detector,
    ],
)
@pytest.mark.parametrize(
    "scan",
    [abtem_st.grid_scan, abtem_st.line_scan, abtem_st.custom_scan],
)
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
    ],
)
def test_probe_scan(data, waves_builder, detector, scan, device, frozen_phonons, lazy):
    probe = data.draw(waves_builder(allow_distribution=False))
    detector = data.draw(detector())
    scan = data.draw(scan())

    potential = data.draw(
        abtem_st.potential(no_frozen_phonons=not frozen_phonons, ensemble_mean=False)
    )
    # scan.match_probe(probe)
    probe.grid.match(potential)

    measurement_shape = detector._out_shape(probe)[0]

    measurement = probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)

    # if isinstance(scan, CustomScan) and scan.shape == (1,):
    #    expected_shape = potential.ensemble_shape + measurement_shape
    # else:
    expected_shape = potential.ensemble_shape + scan.ensemble_shape + measurement_shape

    # print(potential.ensemble_shape, scan.ensemble_shape, measurement_shape)
    # print(measurement.shape)
    # try:
    assert measurement.shape == expected_shape
    #     )
    # except:
    #     print(frozen_phonons)
    #     print(potential.ensemble_shape, scan.ensemble_shape, measurement_shape)
    #     print(measurement.shape)
    #     raise

    assert measurement.dtype == detector._out_dtype(probe)[0]
    assert type(measurement) == detector._out_type(probe.build(scan))[0]

    if not isinstance(detector, AnnularDetector):
        assert (
            measurement.base_axes_metadata
            == detector._out_base_axes_metadata(probe.build(scan))[0]
        )


# # @given(data=st.data(),
# #        gpts=core_st.gpts(min_value=32, max_value=64),
# #        planewave_cutoff=st.floats(5, 10),
# #        energy=st.floats(100e3, 200e3))
# # @pytest.mark.parametrize('lazy', [True, False])
# # @pytest.mark.parametrize('device', ['cpu', gpu])
# # @pytest.mark.parametrize('detector', list(all_detectors.keys()))
# # @pytest.mark.parametrize('downsample', ['cutoff', False])
# # @pytest.mark.parametrize('interpolation', [1, 2, 3])
# # @pytest.mark.parametrize('frozen_phonons', [True, False])
# # def test_s_matrix_scan_and_detect(data,
# #                                   gpts,
# #                                   planewave_cutoff,
# #                                   energy,
# #                                   detector,
# #                                   lazy,
# #                                   device,
# #                                   downsample,
# #                                   interpolation,
# #                                   frozen_phonons):
# #     if frozen_phonons:
# #         atoms = data.draw(atoms_st.random_atoms(min_side_length=5, max_side_length=10))
# #     else:
# #         atoms = data.draw(atoms_st.random_frozen_phonons(min_side_length=5, max_side_length=10))
# #
# #     s_matrix = SMatrix(potential=atoms,
# #                        gpts=gpts,
# #                        planewave_cutoff=planewave_cutoff,
# #                        interpolation=interpolation,
# #                        energy=energy,
# #                        device=device)
# #
# #     if downsample:
# #         probe = s_matrix.build(stop=0, lazy=True).downsample(max_angle=downsample).comparable_probe()
# #     else:
# #         probe = s_matrix.build(stop=0, lazy=True).comparable_probe()
# #
# #     detectors = [
# #         data.draw(all_detectors[detector](max_angle=np.floor(min(probe.cutoff_angles)), allow_detect_every=False))]
# #
# #     scan = GridScan()
# #     measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy, downsample=downsample)
# #     measurements.compute()
# #
# #     assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan=scan)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("ensemble_mean", [True, False])
def test_frozen_phonon_lazy_vs_eager(device, ensemble_mean):
    atoms = Atoms("Si", positions=[(0, 0, 1)], cell=(5, 5, 2), pbc=True)
    frozen_phonons = FrozenPhonons(
        atoms, num_configs=2, sigmas=0.1, ensemble_mean=ensemble_mean, seed=42
    )

    waves = PlaneWave(energy=100e3, gpts=(32, 32), device=device)
    detector = PixelatedDetector(max_angle=None)

    result_lazy = waves.multislice(frozen_phonons, detectors=detector, lazy=True)
    result_lazy = result_lazy.compute()

    result_eager = waves.multislice(frozen_phonons, detectors=detector, lazy=False)

    np.testing.assert_allclose(
        result_eager.array, result_lazy.array, rtol=1e-5, atol=1e-7
    )


class _TwoAxisFrozenPhonons(BaseFrozenPhonons):
    """Test-only frozen phonons with two independent, non-trivial ensemble
    axes. No built-in ensemble class combines two such axes, but this is
    needed to exercise `Waves.multislice`'s eager potential-ensemble
    iteration (`_generate_potential_configurations` in multislice.py), which
    previously mishandled indices whenever a potential ensemble had more than
    one non-trivial axis.
    """

    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory
        atomic_numbers, cell = self._validate_atomic_numbers_and_cell(
            trajectory.ravel()[0], None, None
        )
        super().__init__(atomic_numbers=atomic_numbers, cell=cell, ensemble_mean=False)

    @property
    def ensemble_shape(self):
        return self._trajectory.shape

    @property
    def ensemble_axes_metadata(self):
        return [
            FrozenPhononsAxis(_ensemble_mean=False) for _ in self._trajectory.shape
        ]

    @property
    def _default_ensemble_chunks(self):
        return (1,) * len(self.ensemble_shape)

    @property
    def atoms(self):
        return self._trajectory.ravel()[0]

    def __len__(self):
        return int(np.prod(self.ensemble_shape))

    @property
    def num_configs(self):
        return len(self)

    def randomize(self, atoms):
        return atoms

    def _partition_args(self, chunks=None, lazy=True):
        if chunks is None:
            chunks = self._default_ensemble_chunks
        chunks = validate_chunks(self.ensemble_shape, chunks)

        array = np.zeros(tuple(len(c) for c in chunks), dtype=object)
        for block_index, ranges in zip(
            np.ndindex(array.shape), itertools.product(*chunk_ranges(chunks))
        ):
            slices = tuple(slice(start, stop) for start, stop in ranges)
            sub = _TwoAxisFrozenPhonons(self._trajectory[slices])
            itemset(array, block_index, _wrap_with_array(sub, ndims=len(chunks)))

        if lazy:
            array = da.from_array(array, chunks=1)

        return (array,)

    @staticmethod
    def _from_partition_args_func(*args):
        args = unpack_blockwise_args(args)
        ensemble = args[0]
        return _wrap_with_array(ensemble, len(ensemble.ensemble_shape))

    def _from_partitioned_args(self):
        return self._from_partition_args_func


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_multislice_two_axis_ensemble_eager_vs_lazy_vs_reference(device):
    """Regression test: `Waves.multislice(potential, lazy=False)` must match
    `lazy=True` (and an independent single-member reference) when the
    potential's ensemble has two non-trivial axes at once.
    """
    rng = np.random.default_rng(0)
    base_atoms = Atoms("Si", positions=[(0, 0, 1)], cell=(5, 5, 2), pbc=True)

    shape = (2, 2)
    trajectory = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        displaced = base_atoms.copy()
        displaced.positions += rng.normal(scale=0.05, size=displaced.positions.shape)
        itemset(trajectory, index, displaced)

    ensemble = _TwoAxisFrozenPhonons(trajectory)

    waves = PlaneWave(energy=100e3, gpts=(32, 32), device=device)

    result_eager = waves.multislice(ensemble, lazy=False)
    result_lazy = waves.multislice(ensemble, lazy=True).compute()

    for index in np.ndindex(shape):
        reference = PlaneWave(energy=100e3, gpts=(32, 32), device=device).multislice(
            trajectory[index], lazy=False
        )

        reference_array = to_numpy(reference.array)
        np.testing.assert_allclose(
            to_numpy(result_eager.array[index]), reference_array, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            to_numpy(result_lazy.array[index]), reference_array, rtol=1e-5, atol=1e-7
        )
