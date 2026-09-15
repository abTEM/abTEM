import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given

import abtem


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
    ],
)
def test_detect(data, detector, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    detector = data.draw(detector())
    assume(all(waves._gpts_within_angle(min(detector.angular_limits(waves)))))
    assume(min(waves.cutoff_angles) > 1.0)

    # measurement = detector.detect(waves).compute()

    # assert measurement.ensemble_shape == waves.ensemble_shape
    # assert measurement.dtype == detector._out_dtype(waves)
    # assert measurement.base_shape == detector._out_base_shape(waves)
    # assert type(measurement) == detector._out_type(waves)
    # assert measurement.base_axes_metadata == detector._out_base_axes_metadata(waves)

    # if detector.to_cpu:
    #    assert measurement.device == "cpu"


# @given(data=st.data())
# @pytest.mark.parametrize("lazy", [True, False])
# @pytest.mark.parametrize("device", ["cpu", gpu])
# def test_annular_detector(data, lazy, device):
#     waves = data.draw(abtem_st.waves(lazy=lazy, device=device, min_scan_dims=1))
#     detector = data.draw(abtem_st.annular_detector())
#
#     assume(len(_scan_shape(waves)) > 0)
#     assume(len(_scan_shape(waves)) < 3)
#     assume(all(waves._gpts_within_angle(min(detector.angular_limits(waves)))))
#     assume(min(waves.cutoff_angles) > 1.0)
#     assume(detector.angular_limits(waves)[1] < min(waves.cutoff_angles))
#
#     measurement = detector.detect(waves)
#
#     scan_axes = _scan_axes(waves)
#
#     shape = tuple(
#         n for i, n in enumerate(waves.ensemble_shape) if i not in scan_axes[-2:]
#     )
#
#     assert measurement.ensemble_shape == shape
#     assert measurement.dtype == detector._out_dtype(waves)
#     assert measurement.base_shape == _scan_shape(waves)
#
#     if len(scan_axes) == 1:
#         assert type(measurement) == RealSpaceLineProfiles
#     elif len(scan_axes) > 1:
#         assert type(measurement) == Images
#
#     if detector.to_cpu:
#         assert measurement.device == "cpu"
#

# @given(data=st.data())
# @pytest.mark.parametrize("lazy", [True, False])
# @pytest.mark.parametrize("device", ["cpu", gpu])
# def test_integrate_consistent(data, lazy, device):
#     waves = data.draw(abtem_st.waves(lazy=lazy, device=device, min_scan_dims=1))
#
#     assume(min(waves.cutoff_angles) > 10.0)
#
#     min_extent = max(waves.angular_sampling)
#     max_extent = np.floor(min(waves.cutoff_angles)) - 1.0
#
#     assume(min_extent < max_extent)
#
#     extent = np.floor(
#         data.draw(
#             st.floats(
#                 min_value=min_extent,
#                 max_value=max_extent,
#             )
#         )
#     )
#     inner = np.floor(
#         data.draw(st.floats(min_value=0.0, max_value=min(waves.cutoff_angles) - extent))
#     )
#     outer = inner + extent
#
#     assume(
#         AnnularDetector(inner=inner, outer=outer).get_detector_region(waves).array.sum()
#         > 0
#     )
#
#     annular_measurement = AnnularDetector(inner=inner, outer=outer).detect(waves)
#     flexible_measurement = FlexibleAnnularDetector(
#         step_size=1, outer=np.floor(min(waves.cutoff_angles))
#     ).detect(waves)
#     pixelated_measurement = PixelatedDetector(max_angle="cutoff").detect(waves)
#
#     assert annular_measurement == flexible_measurement.integrate_radial(inner, outer)
#     assert annular_measurement == pixelated_measurement.integrate_radial(inner, outer)
#
#
# @given(
#     gpts=st.integers(min_value=64, max_value=128),
#     extent=st.floats(min_value=5, max_value=10),
# )
# @pytest.mark.parametrize("device", [gpu, "cpu"])
# def test_interpolate_diffraction_patterns(gpts, extent, device):
#     probe1 = Probe(
#         energy=100e3,
#         semiangle_cutoff=30,
#         extent=(extent * 2, extent),
#         gpts=(gpts * 2, gpts),
#         device=device,
#         soft=False,
#     )
#     probe2 = Probe(
#         energy=100e3,
#         semiangle_cutoff=30,
#         extent=extent,
#         gpts=gpts,
#         device=device,
#         soft=False,
#     )
#
#     measurement1 = (
#         probe1.build(lazy=False)
#         .diffraction_patterns(max_angle=None)
#         .interpolate("uniform")
#         .to_cpu()
#     )
#
#     measurement2 = (
#         probe2.build(lazy=False).diffraction_patterns(max_angle=None).to_cpu()
#     )
#
#     assert np.allclose(measurement1.array, measurement2.array)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "extent,gpts",
    [
        ((5.0, 6.0), (50, 60)),  # rectangular cell
        ((5.0, 5.0), (50, 50)),  # square cell
    ],
)
def test_pixelated_detector_resample_uniform(lazy, extent, gpts):
    """Test that PixelatedDetector with resample='uniform' works for rectangular
    and square cells, both lazy and eager. Regression test for GitHub issue #165.

    The bug caused a shape mismatch (ValueError) during compute() because the
    pre-allocated measurement array shape did not match the actual interpolated
    diffraction pattern shape for non-square grids.
    """
    from ase import Atoms

    atoms = Atoms("Si", positions=[(0, 0, 0)], cell=[extent[0], extent[1], 4.0], pbc=True)
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=2.0)
    probe = abtem.Probe(
        energy=100e3,
        semiangle_cutoff=10,
        extent=potential.extent,
        gpts=potential.gpts,
    )
    detector = abtem.PixelatedDetector(max_angle=30, resample="uniform")

    waves = probe.build(lazy=lazy)
    measurement = waves.multislice(potential, detectors=detector)

    if lazy:
        measurement = measurement.compute()

    # Verify the predicted shape matches the actual array shape.
    expected_shape = detector._out_base_shape(probe.build(lazy=False))
    assert measurement.base_shape == expected_shape[0]

    # Verify sampling is uniform (equal in both dimensions).
    sampling = measurement.sampling
    assert np.isclose(sampling[0], sampling[1]), (
        f"Sampling should be uniform but got {sampling}"
    )


@pytest.mark.parametrize(
    "detector_cls, kwargs",
    [
        (abtem.AnnularDetector, dict(inner=0, outer=20)),
        (abtem.FlexibleAnnularDetector, dict()),
        (
            abtem.SegmentedDetector,
            dict(inner=0, outer=20, nbins_radial=2, nbins_azimuthal=4),
        ),
        (abtem.PixelatedDetector, dict()),
    ],
)
def test_measurement_detectors_default_to_cpu(detector_cls, kwargs):
    """Every detector producing a measurement returns it on the host by
    default.

    SegmentedDetector used to default to ``to_cpu=False`` while its own
    docstring (and every sibling detector) said True, so on a GPU run its
    measurements came back as CuPy arrays while the others were NumPy --
    an inconsistency that only surfaced on a CuPy workstation.
    """
    assert detector_cls(**kwargs).to_cpu is True


def test_waves_detector_keeps_data_on_device_by_default():
    """WavesDetector is deliberately the exception: it returns the (large)
    wave functions themselves and is the implicit detector when none is
    given, so it must not force a device-to-host copy of every exit wave.
    """
    from abtem.detectors import WavesDetector

    assert WavesDetector().to_cpu is False


def _finite_crystallite_exit_waves():
    """A small, non-periodic crystallite disk in a vacuum box, illuminated by
    a plane wave -- the 3DED-style setup PixelatedDetector/WavesDetector's
    margin/window_func are for: real-space exit waves with hard edges at the
    box boundary."""
    from ase.build import bulk

    from abtem.atoms import cut_disk

    bulk_atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    disk = cut_disk(bulk_atoms, box=(16.0, 16.0, 16.0))
    potential = abtem.Potential(disk, gpts=96, slice_thickness=2.0)
    waves = abtem.PlaneWave(energy=200e3, extent=potential.extent, gpts=potential.gpts)
    return waves.build(lazy=False).multislice(potential, detectors=abtem.WavesDetector())


@pytest.mark.parametrize(
    "detector_cls, kwargs",
    [
        (abtem.PixelatedDetector, dict(max_angle="full")),
        (abtem.WavesDetector, dict()),
    ],
)
def test_margin_window_defaults_reproduce_unmodified_detector(detector_cls, kwargs):
    """margin=0.0 and window_func=None (the defaults) must reproduce exactly
    what the detector returned before these parameters existed."""
    exit_waves = _finite_crystallite_exit_waves()

    plain = detector_cls(**kwargs).detect(exit_waves)
    explicit_defaults = detector_cls(**kwargs, margin=0.0, window_func=None).detect(
        exit_waves
    )

    np.testing.assert_array_equal(plain.array, explicit_defaults.array)


@pytest.mark.parametrize(
    "detector_cls, kwargs",
    [
        (abtem.PixelatedDetector, dict(max_angle="full")),
        (abtem.WavesDetector, dict()),
    ],
)
def test_margin_crops_output_shape(detector_cls, kwargs):
    """Non-zero margin must shrink the output shape, and _out_base_shape must
    predict the actual shape -- the failure mode a mismatch here would cause
    is a ValueError from a pre-allocated array that doesn't match during
    multislice, not just a wrong-looking result."""
    exit_waves = _finite_crystallite_exit_waves()

    plain = detector_cls(**kwargs).detect(exit_waves)
    cropped_detector = detector_cls(**kwargs, margin=1.0)
    cropped = cropped_detector.detect(exit_waves)

    expected_shape = cropped_detector._out_base_shape(exit_waves)[0]
    assert cropped.base_shape == expected_shape
    assert expected_shape[0] < plain.base_shape[0]
    assert expected_shape[1] < plain.base_shape[1]


def test_pixelated_detector_window_reduces_diffuse_background():
    """Windowing a finite, non-periodic crystallite's exit wave before
    detecting should measurably suppress the diffuse background between
    Bragg spots -- the truncation-rod streaks a hard-edged box produces --
    relative to detecting without a window. Compared before the automatic
    renormalization (which restores overall intensity scale, including in
    this background region) so the comparison isolates the physical
    windowing effect the renormalization is not meant to undo."""
    from abtem.detectors import _window_power_gain

    exit_waves = _finite_crystallite_exit_waves()

    plain = abtem.PixelatedDetector(max_angle="full").detect(exit_waves).array
    windowed = (
        abtem.PixelatedDetector(max_angle="full", window_func="hann")
        .detect(exit_waves)
        .array
    )
    windowed_raw = windowed / _window_power_gain("hann", exit_waves.base_shape)

    ny, nx = plain.shape[-2:]
    cy, cx = ny // 2, nx // 2
    mask = np.ones((ny, nx), dtype=bool)
    mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = False

    assert windowed_raw[..., mask].mean() < plain[..., mask].mean()


def test_pixelated_detector_window_matches_manual_renormalization():
    """Pins down the exact renormalization convention: PixelatedDetector's
    windowed output must equal manually windowing the waves, taking the
    diffraction pattern, and rescaling by 1 / mean(taper**2) -- the window's
    power gain, generalizing the (8/3)**2 constant the 3DED project's own
    analysis tools apply by hand for a Hann window."""
    from abtem.detectors import _window_power_gain

    exit_waves = _finite_crystallite_exit_waves()

    detected = abtem.PixelatedDetector(
        max_angle="full", window_func="hann", margin=0.5
    ).detect(exit_waves)

    cropped = exit_waves.crop(
        extent=(
            exit_waves.extent[0] - 2 * 0.5,
            exit_waves.extent[1] - 2 * 0.5,
        ),
        centered=True,
    )
    manual = exit_waves.window(window="hann", margin=0.5).diffraction_patterns(
        max_angle="full", parity="same"
    )
    renorm = _window_power_gain("hann", cropped.base_shape)

    np.testing.assert_allclose(detected.array, manual.array * renorm, rtol=1e-6)


def test_windowed_pixelated_detector_alias_matches_pixelated_detector():
    """WindowedPixelatedDetector exists only so code written against py3DED's
    own WindowedPixelatedDetector (same margin/window_func signature) runs
    unmodified against abTEM directly -- it must produce identical output to
    the equivalent PixelatedDetector call."""
    from abtem.detectors import WindowedPixelatedDetector

    exit_waves = _finite_crystallite_exit_waves()

    kwargs = dict(max_angle="full", margin=0.5, window_func="hann")
    reference = abtem.PixelatedDetector(**kwargs).detect(exit_waves)
    alias = WindowedPixelatedDetector(**kwargs).detect(exit_waves)

    np.testing.assert_array_equal(reference.array, alias.array)
