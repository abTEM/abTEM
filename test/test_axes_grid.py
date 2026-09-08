"""Tests for AxesGrid layout: colorbar span and inter-panel tick spacing.

Both bugs were found while auditing exploded-grid plots of energy-ensemble
diffraction patterns, but are general AxesGrid issues reproducible on any
multi-panel measurement (e.g. a plain defocus series), not specific to
energy ensembles.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import abtem


@pytest.fixture
def two_by_two_diffraction_patterns():
    """A 2x2 exploded grid (defocus x spherical aberration), cheap to build."""
    abtem.config.set({"diagnostics.progress_bar": False})
    probe = abtem.Probe(
        gpts=32,
        extent=10,
        energy=100e3,
        semiangle_cutoff=20,
        defocus=[0, 50],
        C30=[0, 1e6],
    )
    return probe.build().compute().diffraction_patterns(max_angle=40)


def test_shared_colorbar_spans_every_row(two_by_two_diffraction_patterns):
    """Regression test: _connect_cbar_sync used to anchor the shared colorbar
    (cbar_mode="single") to a single reference panel (self._axes.ravel()[0])
    regardless of grid shape, so on a multi-row grid it only matched the
    height of one row instead of spanning the whole grid.
    """
    dp = two_by_two_diffraction_patterns
    plt.close("all")
    visualization = dp.show(explode=True, cbar=True, common_color_scale=True)
    axes = visualization.axes
    fig = visualization.get_figure()
    fig.canvas.draw()

    all_axes_bboxes = [ax.get_position(original=False) for ax in axes._axes.ravel()]
    grid_y0 = min(bbox.y0 for bbox in all_axes_bboxes)
    grid_y1 = max(bbox.y1 for bbox in all_axes_bboxes)

    cbar_bbox = axes._caxes.ravel()[0].get_position(original=False)
    assert cbar_bbox.y0 == pytest.approx(grid_y0, abs=1e-6)
    assert cbar_bbox.y1 == pytest.approx(grid_y1, abs=1e-6)


def _assert_each_colorbar_matches_its_own_panel(visualization):
    axes = visualization.axes
    fig = visualization.get_figure()
    fig.canvas.draw()

    for col in range(axes._ncols):
        for row in range(axes._nrows):
            own_ax_bbox = axes._axes[col, row].get_position(original=False)
            for cax in axes._caxes[col, row].ravel():
                if cax is None or cax == 0:
                    continue
                cbar_bbox = cax.get_position(original=False)
                assert cbar_bbox.y0 == pytest.approx(own_ax_bbox.y0, abs=1e-6)
                assert cbar_bbox.y1 == pytest.approx(own_ax_bbox.y1, abs=1e-6)
                # Horizontal (column) position: _connect_cbar_sync only ever
                # corrects the y-extent (see below), so a colorbar assigned
                # to the wrong column by _set_caxes_locators' (nx, ny) loop
                # ordering was invisible to a y-only check -- it would still
                # land in the right row, just horizontally sitting next to
                # a *different* panel (or, for the diagonal cells of a
                # square grid, coincidentally the right one).
                assert cbar_bbox.x0 >= own_ax_bbox.x1 - 1e-6
                assert cbar_bbox.x0 == pytest.approx(
                    own_ax_bbox.x1, abs=own_ax_bbox.width
                )


def test_each_colorbar_matches_its_own_panel(two_by_two_diffraction_patterns):
    """Regression test: with cbar_mode="each", _connect_cbar_sync anchored
    every colorbar to self._axes.ravel()[0] regardless of which panel it
    actually belonged to, so every row but the reference row's got a
    colorbar positioned at the wrong (reference row's) height -- silently
    overlapping that row's own correctly-placed colorbar rather than being
    visibly absent.

    Also covers a second, independent bug found later: _set_caxes_locators'
    (ny, nx) loop order didn't match `self._caxes`' (ncols, nrows, ncbars)
    memory layout (unlike the working _set_axes_locators, which uses (nx,
    ny)), so each colorbar's *horizontal* position was silently assigned to
    the wrong column -- e.g. two panels' colorbars swapped columns on a 2x2
    grid. _connect_cbar_sync's own per-panel y-sync (the fix for the first
    bug) papers over this in the vertical direction, which is why only
    checking y0/y1 here missed it; see the x0 assertions below.
    """
    dp = two_by_two_diffraction_patterns
    plt.close("all")
    visualization = dp.show(explode=True, cbar=True)  # cbar_mode="each"
    _assert_each_colorbar_matches_its_own_panel(visualization)


def test_each_colorbar_matches_its_own_panel_non_square_grid():
    """Same as above, but on a 2x3 (non-square) grid, where a column-index
    bug can't hide behind a square grid's coincidentally-correct diagonal.
    """
    abtem.config.set({"diagnostics.progress_bar": False})
    probe = abtem.Probe(
        gpts=32,
        extent=10,
        energy=100e3,
        semiangle_cutoff=20,
        defocus=[0, 50],
        C30=[0, 1e6, 2e6],
    )
    dp = probe.build().compute().diffraction_patterns(max_angle=40)
    plt.close("all")
    visualization = dp.show(explode=True, cbar=True)
    assert visualization.axes.shape == (2, 3)
    _assert_each_colorbar_matches_its_own_panel(visualization)


def test_abutting_panels_have_nonzero_gap(two_by_two_diffraction_patterns):
    """Regression test: with a single shared colorbar removing the usual
    per-panel gap, adjacent panels used to abut with zero space between
    them, so each panel's boundary-facing tick label (e.g. "100" and its
    neighbour's "-100") visually collided. AxesGrid's "padding" spacer
    must be wide enough to fit a several-character tick label.
    """
    dp = two_by_two_diffraction_patterns
    plt.close("all")
    visualization = dp.show(
        explode=True, units="mrad", cbar=True, common_color_scale=True
    )
    axes = visualization.axes
    fig = visualization.get_figure()
    fig.canvas.draw()

    bboxes = axes._axes.ravel()
    positions = [ax.get_position(original=False) for ax in bboxes]

    # Any two panels at the same row (same y-range) that are horizontally
    # adjacent must have a visible gap between them.
    fig_width_inches = fig.get_size_inches()[0]
    min_gap_inches = 0.15  # comfortably less than the fixed 0.25" padding

    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            if i >= j:
                continue
            same_row = abs(pos_i.y0 - pos_j.y0) < 1e-6
            if not same_row:
                continue
            gap = (
                (pos_j.x0 - pos_i.x1) if pos_j.x0 >= pos_i.x1 else (pos_i.x0 - pos_j.x1)
            )
            assert gap * fig_width_inches > min_gap_inches
