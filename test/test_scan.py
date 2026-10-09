import numpy as np
import pytest
import strategies as abtem_st
from ase.build import bulk
from hypothesis import given
from hypothesis import strategies as st
from utils import devices

from abtem.core.axes import PositionsAxis
from abtem.detectors import AnnularDetector, FlexibleAnnularDetector, PixelatedDetector
from abtem.potentials.iam import Potential
from abtem.scan import CustomScan, GridScan, LineScan
from abtem.waves import Probe


def _probe():
    return Probe(energy=100e3, semiangle_cutoff=30, extent=5, gpts=64)


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
    # The line points along angle (degrees from the x-axis, counter-clockwise).
    # Compare unit vectors rather than angles, so 0 and 360 deg (and any
    # angle near the arctan2 branch cut) are equivalent.
    expected_direction = (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)))
    assert np.allclose(vector / np.linalg.norm(vector), expected_direction)
    # LineScan.angle is documented in degrees.
    reported = np.deg2rad(linescan.angle)
    assert np.allclose((np.cos(reported), np.sin(reported)), expected_direction)
    assert np.allclose(
        position, (np.array(linescan.start) + np.array(linescan.end)) / 2
    )


@pytest.mark.parametrize("angle", [0.0, 30.0, 135.0, 250.0])
def test_linescan_add_to_plot_width_is_centred_on_line(angle):
    # interpolate_line averages over +-width/2 about the line, so the drawn
    # band must be centred on it: its centre is the line's midpoint and its
    # corners lie width/2 to either side of start and end.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    position, extent, width = (3.0, 2.0), 4.0, 1.0
    linescan = LineScan.at_position(center=position, extent=extent, angle=angle)
    fig, ax = plt.subplots()
    try:
        rect = linescan.add_to_plot(ax, width=width)
        corners = rect.get_patch_transform().transform(
            [(0, 0), (1, 0), (1, 1), (0, 1)]
        )
    finally:
        plt.close(fig)

    direction = np.array(linescan.direction)
    perpendicular = np.array([-direction[1], direction[0]]) * width / 2
    start, end = np.array(linescan.start), np.array(linescan.end)
    expected = [
        start - perpendicular,
        end - perpendicular,
        end + perpendicular,
        start + perpendicular,
    ]
    assert np.allclose(corners, expected)
    assert np.allclose(corners.mean(axis=0), position)


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
    probe = _probe()
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
    probe = _probe()
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
    probe = _probe()
    waves = probe.build(scan, lazy=False)
    detector = detector_cls(**kwargs)
    measurement = detector.detect(waves)
    # The scan positions axis must appear somewhere in the measurement shape
    assert measurement.ensemble_shape == (3,)


# --- GridScan tests ---


@pytest.mark.parametrize("endpoint", [True, False])
def test_single_point_grid_scan_partition_round_trip(endpoint):
    """A GridScan with gpts=(1, 1) has a degenerate extent, but a well-defined
    position. Partitioning it into blocks and rebuilding the scan from those blocks
    must not raise and must give back the same position."""
    scan = GridScan(
        start=(1.0, 1.0), end=(1.0 + 4.05, 1.0 + 4.05), gpts=(1, 1), endpoint=endpoint
    )

    blocks = scan._partition_args(chunks=(1, 1), lazy=False)
    args = tuple(block[0] for block in blocks)
    block_scan = scan._from_partitioned_args()(*args).item()

    assert block_scan.shape == (1, 1)
    assert np.allclose(block_scan.get_positions(), scan.get_positions())
    assert np.allclose(block_scan.get_positions().ravel(), (1.0, 1.0))


@devices
@pytest.mark.parametrize("endpoint", [True, False])
def test_single_point_grid_scan_lazy(device, endpoint):
    """Regression test: building the wave functions of a one-point GridScan lazily used
    to raise "scan extent must be positive" from the partitioning path. The lazy result
    must match the eager one and the equivalent CustomScan."""
    potential = Potential(
        bulk("Al", "fcc", a=4.05, cubic=True), gpts=64, slice_thickness=1.0
    )
    scan = GridScan(
        start=(1.0, 1.0), end=(1.0 + 4.05, 1.0 + 4.05), gpts=(1, 1), endpoint=endpoint
    )

    def multislice(scan, lazy):
        probe = Probe(energy=100e3, semiangle_cutoff=15.0, device=device)
        probe.grid.match(potential)
        waves = probe.build(scan=scan, lazy=lazy).multislice(potential).compute()
        return waves.to_cpu()

    lazy_waves = multislice(scan, lazy=True)
    eager_waves = multislice(scan, lazy=False)
    custom_waves = multislice(CustomScan([[1.0, 1.0]]), lazy=False)

    assert lazy_waves.shape == (1, 1) + potential.gpts
    assert np.allclose(lazy_waves.array, eager_waves.array)
    assert np.allclose(lazy_waves.array.reshape(custom_waves.shape), custom_waves.array)


def test_grid_scan_zero_extent_rejected():
    """A degenerate extent is only allowed for a single-point scan."""
    with pytest.raises(ValueError, match="scan extent must be positive"):
        GridScan(start=(1.0, 1.0), end=(1.0, 1.0), gpts=(4, 4))


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


@pytest.mark.parametrize(
    "extent, sampling, gpts, gpts_without_endpoint",
    [(10.0, 0.3, 35, 34), (5.0, 0.5, 11, 10), (10.8, 0.3, 37, 36)],
)
def test_line_scan_sampling_is_at_most_the_requested_sampling(
    extent, sampling, gpts, gpts_without_endpoint
):
    scan = LineScan(start=(0, 0), end=(extent, 0), sampling=sampling)
    grid_scan = GridScan(
        start=(0, 0), end=(extent, extent), sampling=sampling, endpoint=True
    )
    assert scan.gpts == gpts == grid_scan.gpts[0]
    assert scan.sampling <= sampling * (1 + 1e-12)

    scan = LineScan(start=(0, 0), end=(extent, 0), sampling=sampling, endpoint=False)
    grid_scan = GridScan(
        start=(0, 0), end=(extent, extent), sampling=sampling, endpoint=False
    )
    assert scan.gpts == gpts_without_endpoint == grid_scan.gpts[0]
    assert scan.sampling <= sampling * (1 + 1e-12)


def test_line_scan_of_an_extent_far_below_the_sampling_has_one_interval():
    scan = LineScan(start=(0, 0), end=(1e-9, 0), sampling=1.0, endpoint=False)
    assert scan.gpts == 1
    scan = LineScan(start=(0, 0), end=(1e-9, 0), sampling=1.0)
    assert scan.gpts == 2
    assert scan.sampling == pytest.approx(1e-9)
