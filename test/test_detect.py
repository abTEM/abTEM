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
def test_radial_sensitivity(lazy):
    """A radially varying detector sensitivity weights the diffraction pattern by
    ``w(alpha)`` before integration. Verify against a manually computed ground truth and
    that uniform sensitivity reproduces the unweighted detector."""
    from abtem.detectors import RadialSensitivity

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, extent=20, gpts=256)
    waves = probe.build(lazy=lazy)

    def value(measurement):
        if measurement.is_lazy:
            measurement = measurement.compute()
        return np.asarray(measurement.array)

    plain = value(abtem.AnnularDetector(inner=0, outer=20).detect(waves))
    unit = value(
        abtem.AnnularDetector(
            inner=0, outer=20, sensitivity=lambda a: np.ones_like(a)
        ).detect(waves)
    )
    assert np.allclose(plain, unit)

    linear = value(
        abtem.AnnularDetector(
            inner=0, outer=20, sensitivity=lambda a: a / 20.0
        ).detect(waves)
    )
    # Down-weighting with w(alpha) <= 1 must reduce the signal.
    assert linear < plain

    # Manual ground truth from the raw diffraction pattern.
    dp = value(
        probe.build(lazy=False).diffraction_patterns(
            max_angle="full", parity="same", fftshift=False
        )
    )
    sampling = probe.build(lazy=False).angular_sampling
    freqs = [np.fft.fftfreq(256, 1 / (s * 256)) for s in sampling]
    kx, ky = np.meshgrid(*freqs, indexing="ij")
    alpha = np.sqrt(kx**2 + ky**2)
    mask = alpha < 20
    manual = (dp * mask * (alpha / 20.0)).sum()
    assert np.isclose(manual, linear, rtol=1e-4)

    # The (angles, values) lookup form matches the equivalent callable.
    lookup = value(
        abtem.AnnularDetector(
            inner=0, outer=20, sensitivity=([0, 10, 20], [0.0, 0.5, 1.0])
        ).detect(waves)
    )
    assert np.isclose(lookup, linear, rtol=1e-4)

    # The FlexibleAnnularDetector weights per-pixel before binning, so integrating its
    # polar bins reproduces the sensitivity-weighted annular detector.
    flexible = abtem.FlexibleAnnularDetector(
        step_size=0.5, inner=0, outer=20, sensitivity=lambda a: a / 20.0
    ).detect(waves)
    flexible_img = value(flexible.integrate_radial(0, 20))
    assert np.isclose(flexible_img, linear, rtol=2e-2)

    # The sensitivity survives a copy (and thus ensemble partitioning).
    detector = abtem.AnnularDetector(
        inner=0, outer=20, sensitivity=([0, 10, 20], [0.0, 0.5, 1.0])
    )
    assert detector.copy().sensitivity == detector.sensitivity

    # AnnularDetector.get_detector_region is the singular efficiency map [%]: a binary
    # 0/100 mask without a sensitivity, a sensitivity-weighted map with one.
    reference = probe.build(lazy=False)
    binary = abtem.AnnularDetector(inner=0, outer=40).get_detector_region(reference)
    assert set(np.unique(binary.array)) <= {0.0, 100.0}
    assert binary.metadata["label"] == "Detector efficiency"
    assert binary.metadata["units"] == "%"
    weighted = abtem.AnnularDetector(
        inner=0, outer=40, sensitivity=lambda a: a / 40.0
    ).get_detector_region(reference)
    assert weighted.array.max() <= 100.0 + 1e-4
    assert weighted.array.sum() < binary.array.sum()

    # FlexibleAnnularDetector exposes the plural get_detector_regions; with a sensitivity
    # the polar bins hold the efficiency [%] at each radial bin center, reflecting the
    # sensitivity profile (here linearly increasing from 0 to 100 %).
    flex = abtem.FlexibleAnnularDetector(
        step_size=1.0, inner=0, outer=40, sensitivity=lambda a: a / 40.0
    )
    regions = flex.get_detector_regions(reference)
    assert regions.metadata["label"] == "Detector efficiency"
    assert regions.metadata["units"] == "%"
    centers = 0 + (np.arange(regions.shape[-2]) + 0.5) * 1.0
    assert np.allclose(regions.array[:, 0], centers / 40.0 * 100.0)
    # The plain (unweighted) regions are integer labels with the original metadata.
    plain_regions = abtem.FlexibleAnnularDetector(
        step_size=1.0, inner=0, outer=40
    ).get_detector_regions(reference)
    assert plain_regions.metadata["label"] == "detector regions"

    # The plural get_detector_regions is not available on the single-region annular
    # detector; it guides the user to the singular get_detector_region.
    with pytest.raises(NotImplementedError, match="get_detector_region"):
        abtem.AnnularDetector(inner=0, outer=40).get_detector_regions(reference)

    # Conversely, the singular get_detector_region is not available on multi-region
    # detectors; it guides the user to the plural get_detector_regions.
    with pytest.raises(NotImplementedError, match="get_detector_regions"):
        abtem.FlexibleAnnularDetector(inner=0, outer=40).get_detector_region(reference)
    with pytest.raises(NotImplementedError, match="get_detector_regions"):
        abtem.SegmentedDetector(
            nbins_radial=2, nbins_azimuthal=4, inner=0, outer=40
        ).get_detector_region(reference)

    # A measured (angles, values) curve whose angles fall outside [inner, outer] is
    # rejected with a descriptive error.
    with pytest.raises(ValueError, match="outside the detector integration range"):
        abtem.AnnularDetector(
            inner=20, outer=60, sensitivity=([0, 30, 80], [0.0, 1.0, 0.5])
        ).detect(reference)

    # A callable producing negative weights over the range is rejected.
    with pytest.raises(ValueError, match="negative weights"):
        abtem.AnnularDetector(
            inner=0, outer=40, sensitivity=lambda a: a - 20.0
        ).detect(reference)

    # Sensitivities greater than 1 are unphysical and rejected, both as a measured
    # curve (at construction) and as a callable (when the range is known).
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        RadialSensitivity(([0, 10, 20], [0.0, 0.5, 1.5]))
    with pytest.raises(ValueError, match="greater than 1"):
        abtem.AnnularDetector(
            inner=0, outer=100, sensitivity=lambda a: a / 50.0
        ).detect(reference)

    # A non-increasing measured curve is rejected.
    with pytest.raises(ValueError, match="strictly increasing"):
        RadialSensitivity(([0, 50, 40], [1.0, 0.5, 0.2]))

    # show() of an annular detector with a sensitivity renders a continuous efficiency
    # gradient (many distinct values), not a single flat region color.
    eff = abtem.AnnularDetector(
        inner=40, outer=100, sensitivity=lambda a: a / 100.0
    ).get_detector_region(reference)
    inside = eff.array[eff.array > 0]
    assert np.unique(np.round(inside, 2)).size > 50
    assert inside.max() <= 100.0 + 1e-4


def test_detector_show_crops_to_outer_angle():
    """show() crops the view to a small margin above the outer angle instead of the full
    (often much larger) sampling-limited angle, for a tighter default visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The waves are sampled to a much higher angle than the detector's outer limit.
    probe = abtem.Probe(sampling=0.05, extent=20, energy=80e3, semiangle_cutoff=20)
    assert min(probe.build(lazy=False).cutoff_angles) > 200

    for detector in (
        abtem.AnnularDetector(inner=40, outer=100),
        abtem.AnnularDetector(inner=40, outer=100, sensitivity=lambda a: a / 100.0),
        abtem.FlexibleAnnularDetector(step_size=5, inner=40, outer=100),
        abtem.SegmentedDetector(nbins_radial=2, nbins_azimuthal=4, inner=40, outer=100),
    ):
        detector.show(probe)
        xlim = plt.gcf().axes[0].get_xlim()
        # Cropped to ~outer * 1.2 = 120 mrad, well below the ~278 mrad sampling limit.
        assert max(abs(x) for x in xlim) < 160
        plt.close("all")


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
