import numpy as np
import pytest
import strategies as abtem_st
from ase import Atoms
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


def test_custom_scan_stores_its_positions():
    """CustomScan stores the xy positions as given, one row per position; a single
    position may be given as one (x, y) pair."""
    cases = [
        ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ((1.0, 2.0), [[1.0, 2.0]]),
    ]
    for positions, expected in cases:
        scan = CustomScan(positions)
        assert scan.shape == (len(expected),), positions
        np.testing.assert_array_equal(scan.positions, expected)


def test_custom_scan_probe_build_shape():
    """Probe.build with a CustomScan should produce waves with a PositionsAxis of the
    correct length in the ensemble."""
    positions = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]])
    scan = CustomScan(positions)
    probe = _probe()
    waves = probe.build(scan, lazy=False)
    assert waves.ensemble_shape == (4,)
    assert isinstance(waves.ensemble_axes_metadata[0], PositionsAxis)


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    [
        (AnnularDetector, {"inner": 5, "outer": 20}),
        (FlexibleAnnularDetector, {}),
        (PixelatedDetector, {"max_angle": "cutoff"}),
    ],
)
def test_custom_scan_detector_ensemble_shape(detector_cls, kwargs):
    """Waves built from a CustomScan of n positions, n = 1 included (gh-235), give
    measurements of ensemble shape (n,), each that of its position alone."""
    probe = _probe()
    detector = detector_cls(**kwargs)
    positions = np.random.default_rng(42).uniform(0.5, 4.5, size=(3, 2))
    alone = [
        detector.detect(probe.build(CustomScan(p), lazy=False)).array[0]
        for p in positions
    ]
    for n in (1, 3):
        measurement = detector.detect(
            probe.build(CustomScan(positions[:n]), lazy=False)
        )
        assert measurement.ensemble_shape == (n,)
        for i in range(n):
            np.testing.assert_array_equal(
                measurement.array[i], alone[i], err_msg=f"n={n}, position {i}"
            )


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
    """A zero extent along every axis has one position, so more than one point
    along an axis is refused."""
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


def test_line_scan_of_an_extent_far_below_the_sampling_has_one_position():
    scan = LineScan(start=(0, 0), end=(1e-9, 0), sampling=1.0, endpoint=False)
    assert scan.gpts == 1
    scan = LineScan(start=(0, 0), end=(1e-9, 0), sampling=1.0)
    assert scan.gpts == 1
    assert scan.sampling == pytest.approx(1e-9)


# Both endpoints give one position (test_scans_of_a_zero_extent_have_one_position);
# each is scanned once, one eager and one lazy.
@pytest.mark.parametrize("endpoint, lazy", [(True, False), (False, True)])
def test_zero_length_line_scan_measures_its_one_position(endpoint, lazy):
    atoms = Atoms("C", positions=[(1.5, 1.5, 1.0)], cell=(5, 5, 2), pbc=True)
    potential = Potential(atoms, gpts=64, slice_thickness=2.0)
    probe = _probe()
    detector = AnnularDetector(inner=5, outer=20)

    scan = LineScan(start=(1, 1), end=(1, 1), sampling=0.5, endpoint=endpoint)
    assert scan.get_positions().tolist() == [[1.0, 1.0]]
    measurement = probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)
    expected = probe.scan(
        potential, scan=CustomScan([(1, 1)]), detectors=detector, lazy=lazy
    )
    if lazy:
        measurement, expected = measurement.compute(), expected.compute()

    assert measurement.shape == (1,)
    assert np.isfinite(measurement.array).all()
    np.testing.assert_allclose(measurement.array, expected.array, rtol=1e-6)


def test_scans_of_a_zero_extent_have_one_position():
    # as LineScan and Grid do; GridScan refused it unless gpts was given as 1
    scan = GridScan(start=(1, 2), end=(1, 2), sampling=0.3)
    assert scan.gpts == (1, 1)
    np.testing.assert_array_equal(scan.get_positions().reshape(-1, 2), [[1, 2]])
    assert LineScan(start=(1, 1), end=(1, 1), sampling=0.5).gpts == 1
    assert LineScan(start=(1, 1), end=(1, 1), sampling=0.5, endpoint=False).gpts == 1
    scan = GridScan(start=(0, 0), end=(2, 0), sampling=0.5, endpoint=True)
    assert scan.gpts == (5, 1)
    scan = GridScan(start=(0, 0), end=(2, 0), sampling=0.5, endpoint=False)
    assert scan.gpts == (4, 1)


def test_a_grid_scan_of_a_single_point_without_gpts_has_one_position_once_matched():
    # gpts are undefined until a sampling is, as for any scan without gpts or
    # sampling, and then one position along each axis, as for LineScan
    scan = GridScan(start=(1, 2), end=(1, 2))
    assert scan.gpts is None
    line = LineScan(start=(1, 2), end=(1, 2))
    assert line.gpts is None

    scan.match_probe(_probe())
    line.match_probe(_probe())
    assert scan.gpts == (1, 1) and line.gpts == 1
    np.testing.assert_array_equal(scan.get_positions().reshape(-1, 2), [[1, 2]])


@pytest.mark.filterwarnings("error")
def test_a_zero_length_line_scan_points_along_x():
    import matplotlib.pyplot as plt

    scan = LineScan(start=(1, 1), end=(1, 1), sampling=0.3)
    np.testing.assert_array_equal(scan.direction, [1.0, 0.0])
    assert scan.angle == 0.0

    fig, ax = plt.subplots()
    try:
        band = scan.add_to_plot(ax, width=1.0)
        np.testing.assert_allclose(band.get_xy(), (1.0, 0.5))
        assert (band.get_width(), band.get_height(), band.angle) == (0.0, 1.0, 0.0)
    finally:
        plt.close(fig)

    scan.add_margin(1.0)
    assert (scan.start, scan.end) == ((0.0, 1.0), (2.0, 1.0))
    # sampled at no more than the requested sampling
    assert scan.gpts > 1 and scan.sampling <= 0.3


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_zero_length_line_scan_keeps_its_requested_sampling(endpoint):
    # its positions coincide, so they have no spacing of their own; 0.0 divided
    # every later recalculation of gpts
    scan = LineScan(start=(1, 1), end=(1, 1), sampling=0.3, endpoint=endpoint)
    assert (scan.gpts, scan.sampling) == (1, 0.3)
    scan = LineScan(start=(1, 1), end=(1, 1), gpts=3, endpoint=endpoint)
    assert scan.sampling == 0.0


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_zero_length_line_scan_of_gpts_keeps_them_when_one_end_moves(endpoint):
    # its sampling is 0, which cannot give the number of positions of a longer line
    for end_to_move in ("end", "start"):
        scan = LineScan(start=(1, 1), end=(1, 1), gpts=3, endpoint=endpoint)
        if end_to_move == "end":
            scan.end = (3.0, 1.0)
        else:
            scan.start = (-1.0, 1.0)
        assert scan.gpts == 3
        assert scan.sampling == pytest.approx(2.0 / 2 if endpoint else 2.0 / 3)
        assert scan.get_positions().shape == (3, 2)


@pytest.mark.parametrize("endpoint", [True, False])
def test_a_zero_length_line_scan_of_gpts_can_have_a_margin(endpoint):
    # the first end to move derives a sampling from `gpts`, which the second keeps
    # as for any other line
    scan = LineScan(start=(1, 1), end=(1, 1), gpts=3, endpoint=endpoint).add_margin(1.0)
    assert (scan.start, scan.end) == ((0.0, 1.0), (2.0, 1.0))
    sampling = 1.0 / 2 if endpoint else 1.0 / 3
    assert scan.sampling == pytest.approx(sampling)
    assert scan.gpts == (5 if endpoint else 6)
    positions = scan.get_positions()
    assert positions.shape == (scan.gpts, 2)
    np.testing.assert_allclose(positions[0], (0.0, 1.0))
