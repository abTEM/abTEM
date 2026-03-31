import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import assert_array_matches_device, gpu

from abtem import GridScan, WavesDetector
from abtem.core.backend import cp

@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_prism_matches_probe(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )
    probe_diffraction_patterns = (
        probe.build(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )

    s_matrix_diffraction_patterns.compute()
    probe_diffraction_patterns.compute()
    assert np.allclose(
        s_matrix_diffraction_patterns.compute().array,
        probe_diffraction_patterns.compute().array,
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe_with_interpolation(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device, max_interpolation=3))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )
    probe_diffraction_patterns = (
        probe.build(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
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

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(None).to_cpu().compute()
    )
    probe_diffraction_patterns = (
        probe.multislice(potential=potential, lazy=lazy)
        .diffraction_patterns(None)
        .to_cpu()
        .compute()
    )

    if np.all(probe_diffraction_patterns.array < 1e-7):
        pass
    else:
        assert np.allclose(
            s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
        )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True])
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

    s_matrix_measurement = s_matrix.scan(
        scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()

    assert s_matrix_measurement == probe_measurement


@pytest.mark.slow
@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize(
    "downsample", [True, False], ids=["downsample", "no_downsample"]
)
@pytest.mark.parametrize("device", [gpu, "cpu"], ids=["gpu", "cpu"])
@pytest.mark.parametrize(
    "interpolation", [False, True], ids=["no_interpolation", "interpolation"]
)
@pytest.mark.parametrize(
    "frozen_phonons", [True, False], ids=["frozen_phonons", "no_frozen_phonons"]
)
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
def test_prism_scan(
    data, interpolation, detector, downsample, lazy, frozen_phonons, device
):
    potential = data.draw(
        abtem_st.potential(
            device=device, no_frozen_phonons=not frozen_phonons, ensemble_mean=False
        )
    )
    ctf = data.draw(abtem_st.ctf(partial_coherence=False))

    max_interpolation = 3 if interpolation else 1

    s_matrix = data.draw(
        abtem_st.s_matrix(
            potential=potential,
            max_interpolation=max_interpolation,
            device=device,
            downsample=downsample,
        )
    )

    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.build(lazy=True).dummy_probes()

    scan = GridScan()
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )
    scan.match_probe(probe)
    measurement_shape = detector._out_shape(probe)[0]

    measurement = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=lazy)

    assert (
        measurement.shape
        == potential.ensemble_shape
        + ctf.ensemble_shape
        + scan.ensemble_shape
        + measurement_shape
    )
    assert measurement.dtype == detector._out_dtype(probe)[0]

    # measurement = measurement.compute()
    # assert (
    #     measurement.shape
    #     == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    # )
    # assert measurement.dtype == detector._out_dtype(probe)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.skipif(cp is None, reason="no gpu")
def test_s_matrix_store_on_host(data, lazy):
    potential = data.draw(abtem_st.potential(device="gpu", ensemble_mean=False))
    s_matrix = data.draw(abtem_st.s_matrix(potential=potential))
    s_matrix = s_matrix.build().compute()
    assert_array_matches_device(s_matrix.array, "cpu")


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
def test_prism_scan_match_probe_scan(data, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )
    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    scan.match_probe(probe)

    prism_measurement = s_matrix.scan(
        scan=scan, detectors=detector, lazy=lazy
    ).compute()
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    ).compute()

    # assert prism_measurement.shape == probe_measurement.shape
    # assert prism_measurement.to_cpu() == probe_measurement.to_cpu()
