import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given
from hypothesis import strategies as st

from abtem.core.axes import PositionsAxis
from abtem.detectors import AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from abtem.scan import CustomScan, LineScan
from abtem.waves import Probe


@given(
    position=st.tuples(
        abtem_st.sensible_floats(min_value=-100, max_value=100),
        abtem_st.sensible_floats(min_value=-100, max_value=100),
    ),
    extent=abtem_st.sensible_floats(min_value=0.1, max_value=100),
    angle=abtem_st.sensible_floats(min_value=0, max_value=360),
)
def test_linescan_at_position(position, extent, angle):
    linescan = LineScan.at_position(position, extent=extent, angle=angle)
    vector = np.array(linescan.end) - np.array(linescan.start)

    assert np.allclose(extent, np.linalg.norm(vector))
    # assert np.allclose(angle, np.rad2deg(np.arctan2(vector[1], vector[0])) % 360, atol=1.)
    assert np.allclose(
        position, (np.array(linescan.start) + np.array(linescan.end)) / 2
    )


# --- CustomScan tests ---


@given(data=st.data())
def test_custom_scan_shape(data):
    """CustomScan.shape should be (n,) for n positions."""
    scan = data.draw(abtem_st.custom_scan())
    n_positions = len(scan.positions)
    assert scan.shape == (n_positions,)


@given(data=st.data())
def test_custom_scan_positions_are_2d(data):
    """CustomScan positions should always have shape (n, 2)."""
    scan = data.draw(abtem_st.custom_scan())
    assert scan.positions.ndim == 2
    assert scan.positions.shape[1] == 2


def test_custom_scan_positions_stored_correctly():
    """CustomScan stores xy positions exactly as provided."""
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scan = CustomScan(positions)
    assert scan.shape == (3,)
    assert np.allclose(scan.positions, positions)


def test_custom_scan_probe_build_shape():
    """Probe.build with a CustomScan should produce waves with a PositionsAxis of the
    correct length in the ensemble."""
    positions = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
    scan = CustomScan(positions)
    probe = Probe(energy=100e3, semiangle_cutoff=30, extent=5, gpts=64)
    waves = probe.build(scan, lazy=False)
    assert waves.ensemble_shape == (4,)
    assert isinstance(waves.ensemble_axes_metadata[0], PositionsAxis)


@pytest.mark.parametrize("n_positions", [1, 3, 10])
def test_custom_scan_annular_detector_shape(n_positions):
    """Regression test for gh-235: AnnularDetector should produce a (n,) measurement
    when detecting waves built from a CustomScan with n positions."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(0.5, 4.5, size=(n_positions, 2))
    scan = CustomScan(positions)
    probe = Probe(energy=100e3, semiangle_cutoff=30, extent=5, gpts=64)
    waves = probe.build(scan, lazy=False)
    detector = AnnularDetector(inner=5, outer=20)
    measurement = detector.detect(waves)
    assert measurement.shape == (n_positions,)


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    [
        (AnnularDetector, {"inner": 5, "outer": 20}),
        (FlexibleAnnularDetector, {}),
        (PixelatedDetector, {"max_angle": "cutoff"}),
    ],
)
def test_custom_scan_detector_ensemble_shape(detector_cls, kwargs):
    """Waves built from a CustomScan should be detectable by all common detector types
    and produce measurements whose ensemble shape matches the scan shape."""
    positions = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
    scan = CustomScan(positions)
    probe = Probe(energy=100e3, semiangle_cutoff=30, extent=5, gpts=64)
    waves = probe.build(scan, lazy=False)
    detector = detector_cls(**kwargs)
    measurement = detector.detect(waves)
    # The scan positions axis must appear somewhere in the measurement shape
    assert measurement.ensemble_shape == (3,)


# def test_source_offset():
#     distribution = GaussianDistribution(4, num_samples=4, dimension=2)
#
#     s = SourceOffset(distribution)
#
#     blocks = s._ensemble_blockwise(1).compute()
#
#     for i in np.ndindex(blocks.shape):
#         blocks[i] = blocks[i].values
#
#     assert np.allclose(concatenate_blocks(blocks), s.get_positions())
