import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from ase.build import bulk
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


@pytest.mark.parametrize("lazy", [False, True])
def test_annular_detector_offset(lazy):
    waves = abtem.Probe(
        energy=100e3, semiangle_cutoff=20, extent=10, gpts=128, defocus=100
    ).build(lazy=lazy)

    diffraction_patterns = waves.diffraction_patterns(
        max_angle="full", parity="same", fftshift=False
    )

    results = []
    for offset in [(0.0, 0.0), (8.0, 8.0)]:
        detected = abtem.AnnularDetector(inner=5, outer=20, offset=offset).detect(waves)
        expected = diffraction_patterns.integrate_radial(5, 20, offset=offset)
        assert np.allclose(detected.compute().array, expected.compute().array)
        results.append(float(detected.compute().array))

    assert not np.isclose(results[0], results[1])


@pytest.mark.parametrize("lazy", [False, True])
def test_annular_detector_line_scan_with_ensemble_axis(lazy):
    """An AnnularDetector over a LineScan (one scan axis) keeps every other
    ensemble axis, here a defocus series.

    It used to drop the last two entries of the ensemble shape regardless of
    how many scan axes there were, so with a single scan axis it also
    discarded the defocus axis, and the declared shape no longer matched the
    data.
    """
    atoms = bulk("Si", cubic=True)
    potential = abtem.Potential(
        atoms, gpts=64, slice_thickness=2, projection="infinite"
    )
    defocus = [0.0, 50.0]
    scan = abtem.LineScan(start=(0, 0), end=(2, 0), gpts=3)
    detector = abtem.AnnularDetector(inner=30, outer=80)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, defocus=defocus)
    measurement = probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)

    assert measurement.ensemble_shape == (2,)
    assert measurement.base_shape == (3,)
    measurement = measurement.compute()
    assert measurement.shape == (2, 3)

    for i, value in enumerate(defocus):
        single = abtem.Probe(energy=100e3, semiangle_cutoff=20, defocus=value).scan(
            potential, scan=scan, detectors=detector
        )
        assert np.allclose(measurement.array[i], single.array)


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
    measurements came back as CuPy arrays while the others were NumPy.
    """
    assert detector_cls(**kwargs).to_cpu is True


@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("gpts", [(64, 65), (65, 64)])
def test_annular_detector_region_is_centred_on_offset(fftshift, gpts):
    # An annulus inner <= |alpha - offset| < outer is symmetric under reflection
    # through `offset`; with `offset` on a grid point and the ring well inside
    # the grid, the reflected grid coincides with itself, so the region's center
    # of mass is exactly `offset`, in whichever storage order it was requested.
    waves = abtem.Waves(
        np.zeros(gpts, dtype=np.complex64), energy=100e3, extent=(32.0, 32.5)
    )
    ax, ay = waves.angular_sampling
    offset = (3 * ax, -2 * ay)
    # Radii off-grid (x.3 pixels) so no pixel sits on a boundary; the ring
    # reaches <= 13.3 px from the centre, inside the >= 20 px half-width of
    # the region within the antialias cutoff.
    detector = abtem.AnnularDetector(inner=2.3 * ax, outer=10.3 * ax, offset=offset)

    region = detector.get_detector_region(waves, fftshift=fftshift)

    com = complex(region.center_of_mass(units="mrad").array)
    # float32 angular coordinates: ~1e-7 relative; a one-pixel error is
    # >= 1 / 3.6 of |offset|.
    assert com == pytest.approx(offset[0] + 1.0j * offset[1], rel=1e-5)


def test_waves_detector_keeps_data_on_device_by_default():
    """WavesDetector is deliberately the exception: it returns the (large)
    wave functions themselves and is the implicit detector when none is
    given, so it must not force a device-to-host copy of every exit wave.
    """
    from abtem.detectors import WavesDetector

    assert WavesDetector().to_cpu is False


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=10, outer=40),
        abtem.FlexibleAnnularDetector(outer=40),
        abtem.SegmentedDetector(inner=10, outer=40, nbins_radial=2, nbins_azimuthal=4),
    ],
    ids=["annular", "flexible_annular", "segmented"],
)
def test_radial_detector_show_without_waves_defaults_units(detector):
    """show() from energy, gpts and sampling alone draws in mrad by default."""
    import matplotlib.pyplot as plt

    def drawn(**kwargs):
        detector.show(energy=100e3, gpts=64, sampling=0.05, **kwargs)
        image = np.ma.filled(plt.gcf().axes[0].images[0].get_array(), np.nan)
        plt.close("all")
        return image

    np.testing.assert_array_equal(drawn(), drawn(units="mrad"))


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=30, outer=100, offset=(5.0, -3.0)),
        abtem.FlexibleAnnularDetector(step_size=10, inner=10, outer=80),
        abtem.PixelatedDetector(max_angle=60, resample="uniform"),
        abtem.WavesDetector(),
        abtem.WavesDetector(gpts=(16, 16)),
    ],
    ids=lambda detector: type(detector).__name__,
)
def test_lazy_detect_matches_eager_detect(detector):
    """A lazy block rebuilds the detector from its constructor arguments, which
    must give back the detector itself."""
    rng = np.random.default_rng(0)
    array = rng.normal(size=(3, 32, 32)) + 1j * rng.normal(size=(3, 32, 32))
    waves = abtem.Waves(
        array.astype(np.complex64),
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[abtem.core.axes.UnknownAxis()],
    )

    eager = detector.detect(waves)
    lazy = detector.detect(waves.ensure_lazy()).compute(progress_bar=False)

    np.testing.assert_array_equal(lazy.array, eager.array)
