import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume

import strategies as abtem_st
from abtem import RealSpaceLineProfiles, Images
from abtem.measurements.detectors import AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from abtem.waves.core import Probe
from utils import gpu


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
@pytest.mark.parametrize('detector', [
    abtem_st.segmented_detector,
    abtem_st.flexible_annular_detector,
    abtem_st.pixelated_detector,
    abtem_st.waves_detector
])
def test_detect(data, detector, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    detector = data.draw(detector())

    assume(all(waves._gpts_within_angle(min(detector.angular_limits(waves)))))
    assume(min(waves.cutoff_angles) > 1.)
    measurement = detector.detect(waves)

    assert measurement.ensemble_shape == waves.ensemble_shape
    assert measurement.dtype == detector.measurement_dtype
    assert measurement.base_shape == detector.measurement_shape(waves)
    assert type(measurement) == detector.measurement_type(waves)
    print(measurement.base_axes_metadata[0] == detector.measurement_axes_metadata(waves)[0])
    assert measurement.base_axes_metadata == detector.measurement_axes_metadata(waves)


    if detector.to_cpu:
        assert measurement.device == 'cpu'


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_annular_detector(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device, min_scan_dims=1))
    detector = data.draw(abtem_st.annular_detector())

    assume(waves.num_scan_axes > 0)
    assume(waves.num_scan_axes < 3)
    assume(all(waves._gpts_within_angle(min(detector.angular_limits(waves)))))
    assume(min(waves.cutoff_angles) > 1.)

    measurement = detector.detect(waves)

    shape = tuple(n for i, n in enumerate(waves.ensemble_shape) if i not in waves.scan_axes[-2:])

    assert measurement.ensemble_shape == shape
    assert measurement.dtype == detector.measurement_dtype
    assert measurement.base_shape == waves.scan_shape[-2:]

    if len(waves.scan_axes) == 1:
        assert type(measurement) == RealSpaceLineProfiles
    elif len(waves.scan_axes) > 1:
        assert type(measurement) == Images

    if detector.to_cpu:
        assert measurement.device == 'cpu'


@given(data=st.data())
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize('device', [gpu, 'cpu'])
def test_integrate_consistent(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device, min_scan_dims=1))

    assume(min(waves.cutoff_angles) > 10.)

    extent = np.floor(data.draw(st.floats(min_value=1., max_value=np.floor(min(waves.cutoff_angles)) - 1.)))
    inner = np.floor(data.draw(st.floats(min_value=0., max_value=min(waves.cutoff_angles) - extent)))
    outer = inner + extent

    annular_measurement = AnnularDetector(inner=inner, outer=outer).detect(waves)
    flexible_measurement = FlexibleAnnularDetector().detect(waves)
    pixelated_measurement = PixelatedDetector(max_angle='cutoff').detect(waves)

    assert annular_measurement == flexible_measurement.integrate_radial(inner, outer)
    assert annular_measurement == pixelated_measurement.integrate_radial(inner, outer)


@given(gpts=st.integers(min_value=64, max_value=128),
       extent=st.floats(min_value=5, max_value=10))
@pytest.mark.parametrize('device', ['cpu', gpu])
def test_interpolate_diffraction_patterns(gpts, extent, device):
    probe1 = Probe(energy=100e3, semiangle_cutoff=30, extent=(extent * 2, extent), gpts=(gpts * 2, gpts), device=device)
    probe2 = Probe(energy=100e3, semiangle_cutoff=30, extent=extent, gpts=gpts, device=device)

    measurement1 = probe1.build(lazy=False).diffraction_patterns(max_angle=None).interpolate('uniform')
    measurement2 = probe2.build(lazy=False).diffraction_patterns(max_angle=None)

    assert np.allclose(measurement1.array, measurement2.array)
