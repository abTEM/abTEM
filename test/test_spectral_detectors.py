"""Tests for SpectralSlitDetector and SpectralAnnularDetector."""

import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import asnumpy
from abtem.detectors import (
    SpectralAnnularDetector,
    SpectralSlitDetector,
    _slit_detector_mask,
)
from abtem.measurements import (
    DiffractionPatterns,
    MomentumResolvedSpectrum,
    momentum_resolved_spectrum,
)


def _make_dp(n_energies=3, gpts=64, sampling=1.0, energy=300e3, lazy=False):
    """Create a simple DiffractionPatterns with an EnergyLossAxis for testing.

    lazy=True chunks one element per energy value along the ensemble axis,
    matching the fragmented chunking EnergyResolvedAtomsEnsemble produces by
    default (or that phonon_loss_diffraction_patterns' thermal-weighting
    concatenate leaves behind), rather than one convenient whole-array chunk.
    """
    from abtem.core.axes import EnergyLossAxis, OrdinalAxis

    rng = np.random.default_rng(42)
    e_values = np.array([0.02, 0.05, 0.10])  # eV
    array = rng.random((n_energies, gpts, gpts)).astype(np.float32)
    if lazy:
        import dask.array as da

        array = da.from_array(array, chunks=(1, gpts, gpts))

    energy_axis = EnergyLossAxis(values=e_values, units="eV")
    metadata = {"energy": energy, "label": "intensity", "units": "arb. unit"}

    return DiffractionPatterns(
        array,
        ensemble_axes_metadata=[energy_axis],
        metadata=metadata,
        sampling=sampling,
        fftshift=True,
    )


# ---- SpectralSlitDetector: geometry vs corners mode -------------------------


def test_slit_geometry_and_corners_same_mask():
    """corners=(0, Q, -W/2, W/2) must produce the same physical mask as
    geometry mode width=W, q_min=0, q_max=Q, angle=0."""
    Q, W = 20.0, 4.0

    geom = SpectralSlitDetector(width=W, q_min=0.0, q_max=Q, angle=0.0)
    corn = SpectralSlitDetector(corners=(0.0, Q, -W / 2, W / 2))

    assert geom._corners == pytest.approx(corn._corners, abs=1e-10)


def test_slit_geometry_and_corners_same_spectrum():
    """corners and geometry mode must return identical momentum_resolved_spectrum."""
    Q, W = 15.0, 3.0
    dp = _make_dp(gpts=64, sampling=0.5)

    det_geom = SpectralSlitDetector(width=W, q_min=0.0, q_max=Q, angle=0.0)
    det_corn = SpectralSlitDetector(corners=(0.0, Q, -W / 2, W / 2))

    spec_geom = momentum_resolved_spectrum(dp, det_geom)
    spec_corn = momentum_resolved_spectrum(dp, det_corn)

    np.testing.assert_allclose(
        spec_geom.array, spec_corn.array, rtol=1e-5,
        err_msg="geometry and corners modes give different spectra"
    )
    np.testing.assert_allclose(
        spec_geom._q_values, spec_corn._q_values,
        err_msg="geometry and corners modes give different q_values"
    )


def test_slit_q_axis_starts_at_zero():
    """With q_min=0 (default), the first q-value of the spectrum must be 0."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(0.0, abs=1e-6)


def test_slit_q_min_nonzero_excludes_origin():
    """With q_min > 0, q=0 must not appear in the q-axis."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_min=5.0, q_max=20.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(5.0, abs=1e-6)
    assert spec._q_values[-1] == pytest.approx(20.0, abs=1e-6)


# ---- SpectralAnnularDetector ------------------------------------------------


def test_annular_q_axis_starts_at_zero():
    """With q_min=0 (default), the first q-value must be 0."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralAnnularDetector(outer=2.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(0.0, abs=1e-6)


def test_annular_q_min_nonzero():
    """With q_min > 0, the first q-value must equal q_min."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralAnnularDetector(outer=2.0, q_min=4.0, q_max=16.0)
    spec = momentum_resolved_spectrum(dp, det)
    assert spec._q_values[0] == pytest.approx(4.0, abs=1e-6)


# ---- Consistent q-range between slit and annular ----------------------------


def test_slit_and_annular_same_q_range():
    """SpectralSlitDetector(width=2*r, q_max=Q) and
    SpectralAnnularDetector(outer=r, q_max=Q) must span the same q range,
    both starting at q=0 and ending at q=Q.
    """
    r, Q = 2.0, 20.0
    dp = _make_dp(gpts=64, sampling=0.5)

    det_slit = SpectralSlitDetector(width=2 * r, q_max=Q)
    det_ann = SpectralAnnularDetector(outer=r, q_max=Q)

    spec_slit = momentum_resolved_spectrum(dp, det_slit)
    spec_ann = momentum_resolved_spectrum(dp, det_ann)

    # Both start at q=0
    assert spec_slit._q_values[0] == pytest.approx(spec_ann._q_values[0], abs=1e-6)
    # Both reach q_max as the last point
    assert spec_slit._q_values[-1] == pytest.approx(Q, abs=1e-6)
    assert spec_ann._q_values[-1] == pytest.approx(Q, abs=1e-6)


# ---- q_sampling reduces the number of q-points ------------------------------


def test_annular_q_sampling_reduces_points():
    """q_sampling > outer should produce fewer q-points than the default."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det_default = SpectralAnnularDetector(outer=1.0, q_max=20.0)
    det_coarse = SpectralAnnularDetector(outer=1.0, q_max=20.0, q_sampling=4.0)

    spec_default = momentum_resolved_spectrum(dp, det_default)
    spec_coarse = momentum_resolved_spectrum(dp, det_coarse)

    assert len(spec_coarse._q_values) < len(spec_default._q_values)
    # Both reach the same endpoints
    assert spec_coarse._q_values[0] == pytest.approx(0.0, abs=1e-6)
    assert spec_coarse._q_values[-1] == pytest.approx(20.0, abs=1e-6)


def test_slit_q_sampling_reduces_points():
    """q_sampling should bin the line profile, producing fewer q-points."""
    dp = _make_dp(gpts=256, sampling=0.05)
    det_default = SpectralSlitDetector(width=3.0, q_max=6.0)
    det_coarse = SpectralSlitDetector(width=3.0, q_max=6.0, q_sampling=2.0)

    spec_default = momentum_resolved_spectrum(dp, det_default)
    spec_coarse = momentum_resolved_spectrum(dp, det_coarse)

    assert len(spec_coarse._q_values) < len(spec_default._q_values)


def test_annular_q_sampling_none_matches_default():
    """q_sampling=None should be identical to omitting it (default=outer)."""
    dp = _make_dp(gpts=64, sampling=0.5)
    det_a = SpectralAnnularDetector(outer=2.0, q_max=20.0)
    det_b = SpectralAnnularDetector(outer=2.0, q_max=20.0, q_sampling=None)

    spec_a = momentum_resolved_spectrum(dp, det_a)
    spec_b = momentum_resolved_spectrum(dp, det_b)

    np.testing.assert_allclose(spec_a.array, spec_b.array)
    assert spec_a._q_values == spec_b._q_values


# ---- show() should not raise ------------------------------------------------


def test_spectrum_show_no_warning():
    """show() should not emit a pcolormesh monotonicity warning."""
    import warnings
    dp = _make_dp(gpts=64, sampling=0.5)
    det = SpectralSlitDetector(width=3.0, q_max=15.0)
    spec = momentum_resolved_spectrum(dp, det)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        import matplotlib
        matplotlib.use("Agg")
        spec.show()


# ---- rotated-slit mask correctness -------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 90.0, 137.0])
def test_slit_mask_area_matches_true_rectangle_at_any_angle(angle):
    """The detector mask must integrate the true rotated rectangle, not its
    axis-aligned bounding box (which is larger for any non-90-degree-multiple
    angle and only coincides with the true area at angle=0/90/180/270)."""
    gpts = (512, 512)
    sampling = (0.2, 0.2)
    extent, width = 40.0, 4.0

    det = SpectralSlitDetector(width=width, q_min=0.0, q_max=extent, angle=angle)
    mask = _slit_detector_mask(
        gpts,
        sampling,
        origin=det.offset,
        angle=det.angle,
        q_min=det.q_min,
        q_max=det.q_max,
        width=det.width,
        fftshift=True,
    )

    pixel_area = sampling[0] * sampling[1]
    true_area = extent * width
    aabb_area = (
        (det._corners[1] - det._corners[0]) * (det._corners[3] - det._corners[2])
    )

    mask_area = mask.sum() * pixel_area

    # Should match the true rectangle area (allowing for pixel discretization),
    # not the (generally much larger) axis-aligned bounding box.
    assert mask_area == pytest.approx(true_area, rel=0.1)
    if angle % 90 != 0:
        assert aabb_area > true_area * 1.5  # sanity: AABB is genuinely bigger here
        assert mask_area < aabb_area * 0.5  # would fail against the old AABB bug


def test_slit_corners_mode_rejects_conflicting_geometry_params():
    """corners is documented as incompatible with offset/angle/q_min/q_max/width;
    all five must actually be validated, not just q_max/width."""
    corners = (-10.0, 10.0, -2.0, 2.0)

    SpectralSlitDetector(corners=corners)  # baseline: corners alone is fine

    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, angle=45.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, q_min=5.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, offset=(1.0, 0.0))
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, q_max=10.0)
    with pytest.raises(ValueError, match="not both"):
        SpectralSlitDetector(corners=corners, width=5.0)


# ---- MomentumResolvedSpectrum.show() explode ordering -----------------------


def _make_multiaxis_spectrum(seed=0):
    """A MomentumResolvedSpectrum with 3 distinct-size ensemble axes, so a
    positional (rather than axis-identity) index mix-up is detectable."""
    rng = np.random.default_rng(seed)
    q_values = np.linspace(0, 10, 5)
    e_values = np.linspace(-0.1, 0.1, 6)
    array = rng.random((2, 3, 4, 5, 6))
    ensemble_axes = [
        OrdinalAxis(label="axis0", values=(0, 1)),
        OrdinalAxis(label="axis1", values=(0, 1, 2)),
        OrdinalAxis(label="axis2", values=(0, 1, 2, 3)),
    ]
    return MomentumResolvedSpectrum(
        array,
        q_values=q_values,
        e_values=e_values,
        ensemble_axes_metadata=ensemble_axes,
    ), array


def _panel_arrays(spec_array, ensemble_shape, explode_axes):
    """Ground-truth panel data for the given explode order, computed directly
    from the backing array rather than through show()."""
    import itertools

    n_ensemble = len(ensemble_shape)
    grid_sizes = [ensemble_shape[a] for a in explode_axes]
    indices = list(itertools.product(*[range(s) for s in grid_sizes]))
    panels = []
    for idx in indices:
        axis_to_value = dict(zip(explode_axes, idx))
        full = tuple(axis_to_value.get(d, 0) for d in range(n_ensemble))
        panels.append(spec_array[full])
    return panels


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
# (2, 0) is out of order, (0, 1, 2) explodes every axis; (0, 2) and (2, 1, 0)
# add no order mix-up these do not show.
@pytest.mark.parametrize("explode_axes", [(2, 0), (0, 1, 2)])
def test_show_explode_sequence_any_order(explode_axes):
    """explode as a sequence of axis indices must plot each panel's own data
    regardless of the order the axes are listed in -- not the axis positioned
    at the same index as it appears in the (ascending) array shape."""
    import matplotlib

    matplotlib.use("Agg")
    spec, array = _make_multiaxis_spectrum()
    ensemble_shape = (2, 3, 4)

    fig, axes = spec.show(explode=list(explode_axes))
    expected_panels = _panel_arrays(array, ensemble_shape, explode_axes)

    axes_flat = axes.flatten()
    n = len(expected_panels)
    for k, expected in enumerate(expected_panels):
        plotted = axes_flat[k].collections[0].get_array().reshape(
            expected.shape[1], expected.shape[0]
        )
        np.testing.assert_allclose(plotted, expected.T)
    for k in range(n, len(axes_flat)):
        assert not axes_flat[k].get_visible()


@pytest.mark.filterwarnings("ignore:This figure includes Axes")
def test_show_explode_true_matches_full_range():
    """explode=True must be equivalent to exploding every ensemble axis."""
    import matplotlib

    matplotlib.use("Agg")
    spec, array = _make_multiaxis_spectrum()
    ensemble_shape = (2, 3, 4)

    fig_true, axes_true = spec.show(explode=True)
    fig_all, axes_all = spec.show(explode=[0, 1, 2])
    assert axes_true.shape == axes_all.shape


@pytest.mark.parametrize("n_panels, ncols", [(4, 4), (6, 4)])
def test_show_explode_shares_axes_and_places_colorbar_beside_panels(
    n_panels, ncols
):
    """An exploded grid must not repeat the y-axis label/tick labels on every
    panel (only the leftmost column of each row), and the colorbar must land
    in its own strip beside the panels rather than overlapping the last one
    (regression: tight_layout() called after fig.colorbar() previously
    fought over the colorbar axes' position -- see the "This figure includes
    Axes that are not compatible with tight_layout" warning that used to
    fire here)."""
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(0)
    q_values = np.linspace(0, 10, 5)
    e_values = np.linspace(-0.1, 0.1, 6)
    array = rng.random((n_panels, 5, 6))
    spec = MomentumResolvedSpectrum(
        array,
        q_values=q_values,
        e_values=e_values,
        ensemble_axes_metadata=[
            OrdinalAxis(label="axis0", values=tuple(range(n_panels)))
        ],
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, axes = spec.show(explode=True)

    axes_flat = axes.flatten()[:n_panels]
    nrows = len(axes.flatten()) // ncols

    for k, a in enumerate(axes_flat):
        row, col = divmod(k, ncols)
        is_left_column = col == 0
        assert bool(a.yaxis.get_label().get_text()) == is_left_column

    # The colorbar axes (last Axes added to the figure that isn't one of the
    # panels) must not overlap any panel's bounding box.
    fig.canvas.draw()
    panel_bboxes = [a.get_window_extent() for a in axes_flat]
    cbar_axes = [a for a in fig.axes if a not in list(axes.flatten())]
    assert len(cbar_axes) == 1
    cbar_bbox = cbar_axes[0].get_window_extent()
    for bbox in panel_bboxes:
        assert not bbox.overlaps(cbar_bbox)

    # The colorbar should sit close to the rightmost panel, not stranded with
    # a large empty gap.
    rightmost_panel_bbox = panel_bboxes[ncols - 1 if ncols <= n_panels else n_panels - 1]
    gap = cbar_bbox.x0 - rightmost_panel_bbox.x1
    assert gap < 0.25 * rightmost_panel_bbox.width


def test_show_warns_on_ensemble_collapse():
    """A non-exploded show() on an object with ensemble axes must warn that
    only member (0, ..., 0) is shown."""
    import matplotlib

    matplotlib.use("Agg")
    spec, _ = _make_multiaxis_spectrum()
    with pytest.warns(UserWarning, match="showing member"):
        spec.show()


# ---- MomentumResolvedSpectrum.crop() -----------------------------------------


def _make_simple_spectrum(seed=0):
    rng = np.random.default_rng(seed)
    q_values = np.linspace(0, 10, 6)
    e_values = np.linspace(-0.1, 0.1, 5)
    array = rng.random((len(q_values), len(e_values)))
    return (
        MomentumResolvedSpectrum(array, q_values=q_values, e_values=e_values),
        array,
    )


def test_crop_selects_correct_sub_range():
    spec, array = _make_simple_spectrum()

    cropped = spec.crop(q_range=(2, 8), e_range=(-0.06, 0.06))

    assert cropped.q_values == tuple(spec.q_values[1:5])
    assert cropped.e_values == tuple(spec.e_values[1:4])
    np.testing.assert_allclose(cropped.array, array[1:5, 1:4])


def test_crop_none_keeps_full_range():
    spec, array = _make_simple_spectrum()
    cropped = spec.crop()
    assert cropped.q_values == spec.q_values
    assert cropped.e_values == spec.e_values
    np.testing.assert_allclose(cropped.array, array)


def test_crop_only_q_range():
    spec, array = _make_simple_spectrum()
    cropped = spec.crop(q_range=(2, 8))
    assert cropped.q_values == tuple(spec.q_values[1:5])
    assert cropped.e_values == spec.e_values
    np.testing.assert_allclose(cropped.array, array[1:5, :])


def test_crop_preserves_ensemble_axes():
    spec, array = _make_multiaxis_spectrum()
    cropped = spec.crop(q_range=(0, 5))
    assert cropped.ensemble_shape == spec.ensemble_shape
    assert cropped.ensemble_axes_metadata == spec.ensemble_axes_metadata
    np.testing.assert_allclose(cropped.array, array[..., :3, :])


def test_crop_empty_range_raises():
    spec, _ = _make_simple_spectrum()
    with pytest.raises(ValueError, match="selects no data"):
        spec.crop(q_range=(100, 200))


def test_crop_reversed_range_raises():
    spec, _ = _make_simple_spectrum()
    with pytest.raises(ValueError, match="min must not exceed max"):
        spec.crop(q_range=(8, 2))


def test_cropped_spectrum_shows_without_error():
    import matplotlib

    matplotlib.use("Agg")
    spec, _ = _make_simple_spectrum()
    cropped = spec.crop(q_range=(2, 8), e_range=(-0.06, 0.06))
    fig, ax = cropped.show()
    assert ax.get_xlim()[1] < spec.q_values[-1]


# ---- MomentumResolvedSpectrum.show(logscale=...) -----------------------------


def test_show_logscale_uses_lognorm_and_colours_nonpositive_values():
    """Log scale uses a LogNorm, alone and exploded. Phonon-loss TDS intensities
    can include zero (or, from numerical noise, tiny negative) values; LogNorm
    masks values <= 0 rather than raising, and a masked pixel is drawn with the
    colormap's "bad" colour -- which defaults to fully transparent, showing the
    figure's white background through it and looking exactly like missing data.
    These pixels are real, valid, just-below-the-log-floor intensity (routine for
    TDS: incoherent - coherent is often exactly zero), so they must be coloured
    as the bottom of the colour scale instead of left blank."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    matplotlib.use("Agg")
    spec, array = _make_simple_spectrum()
    array[0, 0] = 0.0

    fig, ax = spec.show(logscale=True)
    norm = ax.collections[0].norm
    assert isinstance(norm, LogNorm)
    assert norm.vmin is None or norm.vmin > 0
    cmap = ax.collections[0].cmap
    assert tuple(cmap.get_bad()) == tuple(cmap(0.0))
    assert cmap.get_bad()[-1] == 1.0  # fully opaque, not the default transparent

    # Must not have mutated the globally registered colormap.
    assert tuple(plt.get_cmap("viridis").get_bad()) == (0.0, 0.0, 0.0, 0.0)

    spec, _ = _make_multiaxis_spectrum()
    fig, axes = spec.show(explode=True, logscale=True)
    assert isinstance(axes.flatten()[0].collections[0].norm, LogNorm)
    plt.close("all")


def test_show_logscale_and_power_raises():
    spec, _ = _make_simple_spectrum()
    with pytest.raises(ValueError, match="logscale"):
        spec.show(logscale=True, power=2.0)


# ---- momentum_resolved_spectrum with a lazy (dask-backed) input -------------


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda: SpectralSlitDetector(width=3.0, q_sampling=1.0, q_max=15.0),
        lambda: SpectralAnnularDetector(outer=1.5, q_sampling=1.0, q_max=15.0),
    ],
    ids=["slit", "annular"],
)
def test_momentum_resolved_spectrum_lazy_matches_eager(make_detector):
    """A lazy DiffractionPatterns (fragmented, one chunk per energy value --
    the chunking EnergyResolvedAtomsEnsemble and phonon_loss_diffraction_patterns'
    thermal weighting produce) must give the same result as the eager input,
    and must actually stay correct once computed (not just declare a correct
    shape while the underlying dask graph silently computes something else --
    see the tensordot negative-axis regression this guards)."""
    dp_eager = _make_dp(gpts=48, sampling=0.5, lazy=False)
    dp_lazy = _make_dp(gpts=48, sampling=0.5, lazy=True)
    detector = make_detector()

    spec_eager = momentum_resolved_spectrum(dp_eager, detector)
    spec_lazy = momentum_resolved_spectrum(dp_lazy, detector)

    assert hasattr(spec_lazy.array, "compute"), "result should stay lazy"
    computed = spec_lazy.array.compute()
    assert computed.shape == spec_eager.array.shape
    np.testing.assert_allclose(computed, spec_eager.array, rtol=1e-4)


# ---- MomentumResolvedSpectrum.show() with GPU-resident (cupy-like) data -----


class _FakeCupyArray:
    """Minimal stand-in for a cupy.ndarray: blocks implicit np.asarray()
    conversion (as real cupy arrays do) but supports .get()."""

    def __init__(self, arr):
        self._arr = arr

    def __array__(self, *args, **kwargs):
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. Please use "
            "`.get()` to construct a NumPy array explicitly."
        )

    def get(self):
        return self._arr

    def __getitem__(self, idx):
        return _FakeCupyArray(self._arr[idx]) if idx else self

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim


def test_show_handles_gpu_resident_array():
    """show() must not call np.asarray() directly on GPU-resident (cupy-like)
    data -- cupy deliberately raises TypeError on that, requiring .get()."""
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    q_values = np.linspace(0, 10, 5)
    e_values = np.linspace(-0.1, 0.1, 6)
    array = _FakeCupyArray(rng.random((5, 6)).astype(np.float32))

    spec = MomentumResolvedSpectrum(array, q_values=q_values, e_values=e_values)
    fig, ax = spec.show()  # must not raise


# ---- On an accelerator -------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "detector",
    [
        SpectralAnnularDetector(outer=2.0, q_max=15.0),
        SpectralSlitDetector(width=2.0, q_min=0.0, q_max=15.0, angle=0.0),
    ],
    ids=["annular", "slit"],
)
@pytest.mark.parametrize("device", [gpu])
def test_momentum_resolved_spectrum_matches_cpu(device, detector, lazy):
    """Diffraction patterns kept on the device (a detector with to_cpu=False)
    are contracted against detector masks there, so the contraction has to
    cope with the two differing in dtype, as NumPy's does."""
    dp = _make_dp(gpts=64, sampling=0.5, lazy=lazy)

    reference = momentum_resolved_spectrum(dp, detector).compute()
    result = momentum_resolved_spectrum(dp.copy_to_device(device), detector).compute()

    np.testing.assert_allclose(
        asnumpy(result.array), asnumpy(reference.array), rtol=1e-5, atol=1e-6
    )


# ---- On a skewed (non-orthogonal) grid ----------------------------------------

_SKEW_A = 20.0  # Å: 1.85 mrad per pixel at 100 keV
# Rows are the in-plane lattice vectors. A hexagonal cell is simulated on a skewed
# grid, whose pixel (h1, h2) sits at h1 b1 + h2 b2; the others on orthogonal grids
# of the given extent.
_SKEW_CELLS = {
    "hexagonal": np.array([[_SKEW_A, 0.0], [-_SKEW_A / 2, _SKEW_A * 3**0.5 / 2]]),
    "rectangular": (_SKEW_A, _SKEW_A * 3**0.5 / 2),
    "square": (_SKEW_A, _SKEW_A),
}
_SKEW_ANGLE = 20.0  # degrees: along neither a grid axis nor a b_i
_SKEW_SIGMA = 2.5  # mrad
_SKEW_OUTER = 5 * _SKEW_SIGMA  # mrad
_SKEW_Q = (20.0, 45.0)  # mrad
_SKEW_WEIGHTS = (1.0, 2.0)
_SKEW_DIRECTION = np.array(
    [np.cos(np.deg2rad(_SKEW_ANGLE)), np.sin(np.deg2rad(_SKEW_ANGLE))]
)


def _gaussians_dp(cell_name):
    """Diffraction patterns (fftshift layout, one energy loss) holding two
    normalised Gaussian angular densities of weights `_SKEW_WEIGHTS`, centred at
    `_SKEW_Q` along `_SKEW_ANGLE`. A pixel holds the density times its area, as a
    pattern of diffuse scattering does, so a region covering a Gaussian collects
    its weight on any grid. The pixel positions are worked out here from the cell,
    independently of abTEM's grids."""
    from abtem.core.axes import EnergyLossAxis
    from abtem.core.energy import energy2wavelength

    cell = _SKEW_CELLS[cell_name]
    energy = 100e3
    to_mrad = energy2wavelength(energy) * 1e3
    gpts = (80, 80)
    if np.ndim(cell) == 1:
        extent, cell = cell, None
        reciprocal = np.diag(1 / np.array(extent))
    else:
        extent = tuple(np.linalg.norm(cell, axis=1))
        reciprocal = np.linalg.inv(cell).T  # rows b1, b2
    h1 = np.arange(gpts[0]) - gpts[0] // 2
    h2 = np.arange(gpts[1]) - gpts[1] // 2
    g = h1[:, None, None] * reciprocal[0] + h2[None, :, None] * reciprocal[1]
    g = g * to_mrad
    pixel_area = abs(np.linalg.det(reciprocal)) * to_mrad**2

    array = np.zeros(gpts)
    for q, weight in zip(_SKEW_Q, _SKEW_WEIGHTS):
        r2 = np.sum((g - q * _SKEW_DIRECTION) ** 2, axis=-1)
        density = np.exp(-r2 / (2 * _SKEW_SIGMA**2)) / (2 * np.pi * _SKEW_SIGMA**2)
        array += weight * density * pixel_area

    metadata = {"energy": energy}
    if cell is not None:
        metadata["cell"] = tuple(map(tuple, cell))
    return DiffractionPatterns(
        array[None],
        sampling=(1 / extent[0], 1 / extent[1]),
        fftshift=True,
        ensemble_axes_metadata=[EnergyLossAxis(values=[0.05], units="eV")],
        metadata=metadata,
    )


def _collected(name, dp):
    """What a region measures on `dp`: the spectra one value per Gaussian, the
    rest the outer Gaussian only."""
    offset = tuple(_SKEW_Q[1] * _SKEW_DIRECTION)
    if name == "spectral_annular":
        detector = SpectralAnnularDetector(
            outer=_SKEW_OUTER,
            q_min=_SKEW_Q[0],
            q_max=_SKEW_Q[1],
            q_sampling=_SKEW_Q[1] - _SKEW_Q[0],
            angle=_SKEW_ANGLE,
        )
        return asnumpy(momentum_resolved_spectrum(dp, detector).array).ravel()
    if name == "spectral_slit":
        detector = SpectralSlitDetector(
            width=2 * _SKEW_OUTER,
            q_min=_SKEW_Q[0] - _SKEW_OUTER,
            q_max=_SKEW_Q[1] + _SKEW_OUTER,
            angle=_SKEW_ANGLE,
        )
        spectrum = momentum_resolved_spectrum(dp, detector)
        values = asnumpy(spectrum.array).ravel()
        first = spectrum._q_values < np.mean(_SKEW_Q)
        return np.array([values[first].sum(), values[~first].sum()])
    if name == "integrate_radial":
        return asnumpy(dp.integrate_radial(0.0, _SKEW_OUTER, offset=offset).array)
    if name == "polar_binning":
        binned = dp.polar_binning(1, 1, 0.0, _SKEW_OUTER, offset=offset)
        return asnumpy(binned.array)
    if name == "annular_region":
        region = abtem.AnnularDetector(0.0, _SKEW_OUTER, offset=offset)
    else:
        region = SpectralSlitDetector(
            width=2 * _SKEW_OUTER,
            q_min=_SKEW_Q[1] - _SKEW_OUTER,
            q_max=_SKEW_Q[1] + _SKEW_OUTER,
            angle=_SKEW_ANGLE,
        )
    mask = region.get_detector_region(dp, fftshift=True).array
    return np.sum(dp.array * mask, axis=(-2, -1))


# A region is placed and sized in physical scattering angles, whatever the grid.
# Measured error relative to the weights on the hexagonal (skewed) grid, before
# the fix: 0.32 and 1.0 for the spectral disks (the second disk was shifted along
# b1 and b2, missing its Gaussian), 1.0 for the offset integration, polar bin,
# annular and slit regions; after it at most 6e-6, against rtol 1e-4. The
# rectangular grid is within 6.2e-6 before and after. What remains is the
# Gaussian tails past the region edge and the rounding of a region centre to a
# pixel.
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "name", ["spectral_annular", "integrate_radial", "polar_binning"]
)
@pytest.mark.parametrize("cell_name", ["hexagonal", "rectangular"])
def test_offset_regions_collect_the_density_they_cover_on_any_grid(
    cell_name, name, device
):
    dp = _gaussians_dp(cell_name).copy_to_device(device)
    expected = _SKEW_WEIGHTS if name.startswith("spectral") else _SKEW_WEIGHTS[1:]
    np.testing.assert_allclose(np.ravel(_collected(name, dp)), expected, rtol=1e-4)


# get_detector_region builds the mask on the host.
@pytest.mark.parametrize("name", ["annular_region", "slit_region"])
@pytest.mark.parametrize("cell_name", ["hexagonal", "rectangular"])
def test_offset_detector_regions_cover_the_density_on_any_grid(cell_name, name):
    np.testing.assert_allclose(
        np.ravel(_collected(name, _gaussians_dp(cell_name))),
        _SKEW_WEIGHTS[1:],
        rtol=1e-4,
    )


# The slit spectrum converts the interpolated average across the slit to a sum by
# counting one sample per square pixel of the smaller sampling, so on a skewed
# grid it must give what a square grid of the same pixel spacing gives (both
# 0.92 of the weights under that convention). Measured: 1.0 off
# before the fix (the line was mapped through the real-space cell, landing off
# the Gaussians), 7.4e-4 after, against rtol 3e-3; the remaining difference is the
# linear interpolation on the two grids.
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_slit_spectrum_on_a_skewed_grid_matches_a_square_grid(device):
    skewed = _collected(
        "spectral_slit", _gaussians_dp("hexagonal").copy_to_device(device)
    )
    square = _collected("spectral_slit", _gaussians_dp("square").copy_to_device(device))
    np.testing.assert_allclose(skewed, square, rtol=3e-3)
