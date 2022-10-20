import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume, reproduce_failure

import strategies as abtem_st
from abtem import GridScan, WavesDetector
from abtem.core.backend import cp
from utils import gpu, assert_array_matches_device


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = s_matrix.reduce(lazy=lazy).diffraction_patterns(
        max_angle=None
    )
    probe_diffraction_patterns = probe.build(lazy=lazy).diffraction_patterns(
        max_angle=None
    )

    assert np.allclose(
        s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe_with_interpolation(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device, max_interpolation=3))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = s_matrix.reduce(lazy=lazy).diffraction_patterns(
        max_angle=None
    )
    probe_diffraction_patterns = probe.build(lazy=lazy).diffraction_patterns(
        max_angle=None
    )

    assert np.allclose(
        s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe_with_multislice(data, lazy, device):

    potential = data.draw(abtem_st.potential(device=device))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = s_matrix.reduce(lazy=lazy).diffraction_patterns(
        None
    )
    probe_diffraction_patterns = probe.multislice(
        potential=potential, lazy=lazy
    ).diffraction_patterns(None)

    assert np.allclose(
        s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("interpolation", [True, False])
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
def test_prism_scan(data, interpolation, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))

    max_interpolation = 3 if interpolation else 1

    s_matrix = data.draw(
        abtem_st.s_matrix(
            potential=potential, max_interpolation=max_interpolation, device=device
        )
    )
    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    scan.match_probe(probe)
    measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)

    measurement_shape = detector.measurement_shape(probe)
    assert (
        measurement.shape
        == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    )
    assert measurement.dtype == detector.measurement_dtype

    measurement = measurement.compute()
    assert (
        measurement.shape
        == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    )
    assert measurement.dtype == detector.measurement_dtype


# @reproduce_failure('6.54.6', b'AXicY2DABxgJMXFqRFZu0YAkwwoACyYAxA==')
@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
# @pytest.mark.parametrize('interpolation', [True, False])
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
def test_s_matrix_matches_probe_no_interpolation(data, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))
    detector = data.draw(detector())
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )

    s_matrix_measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    )

    assert s_matrix_measurement == probe_measurement


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("interpolation", [True, False])
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
def test_prism_scan(data, interpolation, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))

    max_interpolation = 3 if interpolation else 1

    s_matrix = data.draw(
        abtem_st.s_matrix(
            potential=potential, max_interpolation=max_interpolation, device=device
        )
    )
    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )

    measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)

    scan.match_probe(probe)
    measurement_shape = detector.measurement_shape(probe)
    assert (
        measurement.shape
        == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    )
    assert measurement.dtype == detector.measurement_dtype

    measurement = measurement.compute()
    assert (
        measurement.shape
        == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    )
    assert measurement.dtype == detector.measurement_dtype


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.skipif(cp is None, reason="no gpu")
def test_s_matrix_store_on_host(data, lazy):
    potential = data.draw(abtem_st.potential(device="gpu", ensemble_mean=False))
    s_matrix = data.draw(abtem_st.s_matrix(potential=potential))
    s_matrix = s_matrix.build().compute()
    assert_array_matches_device(s_matrix.array, "cpu")


# @given(data=st.data())
# @pytest.mark.parametrize('lazy', [True, False])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# @pytest.mark.parametrize('detector', [
#     abtem_st.segmented_detector,
#     abtem_st.flexible_annular_detector,
#     abtem_st.pixelated_detector,
#     abtem_st.waves_detector,
#     abtem_st.annular_detector
# ])
# def test_prism_scan_match_probe_scan(data, detector, lazy, device):
#     potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))
#     s_matrix = data.draw(abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device))
#     detector = data.draw(detector())
#
#     s_matrix = s_matrix.round_gpts_to_interpolation()
#     probe = s_matrix.dummy_probes()
#
#     scan = GridScan()
#     scan.match_probe(probe)
#
#     prism_measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)
#     probe_measurement = probe.scan(potential=potential, scan=scan, detectors=detector, lazy=lazy)
#
#     assert prism_measurement == probe_measurement

# @given(atoms=abtem_st.atoms(min_side_length=5, max_side_length=10),
#        gpts=abtem_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(10, 15),
#        interpolation=st.integers(min_value=1, max_value=4),
#        energy=st.floats(100e3, 200e3),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu'])
# def test_prism_interpolation(data, atoms, gpts, planewave_cutoff, energy, lazy, device, interpolation):
#     detectors = data.draw(abtem_st.detectors(max_detectors=1))
#
#     potential = Potential(atoms, gpts=gpts, device=device).build(lazy=lazy)
#     # scan = GridScan(start=(0, 0), end=potential.extent)
#     #
#     # probe = Probe(semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
#     # probe.grid.match(potential)
#
#     # tiled_potential = potential.tile((interpolation,) * 2)
#     s_matrix = SMatrix(potential=potential, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device, downsample=False)
#
#     # diffraction_pattern = probe.build(lazy=False).diffraction_patterns()
#     prism_diffraction_pattern = s_matrix.build(stop=0, lazy=False).reduce().diffraction_patterns()
#
#     # xp = get_array_module(device)
#     # assume(xp.abs(diffraction_pattern.array - prism_diffraction_pattern.array).max() < 1e-6)
#     # assume_valid_probe_and_detectors(probe, detectors)
#     #
#     # measurement = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
#     # prism_measurement = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy)
#     #
#     # assert np.allclose(measurement.array, prism_measurement.array, atol=1e-6)
#
# #
#

#
#
#
# @given(atoms=abtem_st.atoms(min_side_length=5, max_side_length=10),
#        gpts=abtem_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(5, 15),
#        energy=st.floats(100e3, 200e3),
#        interpolation=st.integers(min_value=2, max_value=4),
#        distribute_scan=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# @pytest.mark.parametrize('store_on_host', [False, True])
# def test_distribute_scan(data, atoms, gpts, planewave_cutoff, energy, lazy, distribute_scan, interpolation,
#                          device, store_on_host):
#     detectors = data.draw(abtem_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device, store_on_host=True)
#
#     probe = s_matrix.build(stop=0, lazy=True).comparable_probe()
#
#     assert isinstance(s_matrix.build().compute().array, np.ndarray)
#
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     scan = GridScan()
#     measurements = s_matrix.scan(scan=scan, detectors=detectors, distribute_scan=distribute_scan, lazy=lazy,
#                                  downsample=False)
#     measurements.compute()
#
#     assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)
