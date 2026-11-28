import ase
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from ase.build import bulk
from hypothesis import assume, given
from utils import devices

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


@pytest.mark.parametrize("fftshift", [True, False])
def test_annular_detector_region_is_in_percent(fftshift):
    # The region is labelled as a detector efficiency in %: a fully sensitive pixel
    # is 100, not 1, and the others are 0.
    waves = abtem.Waves(
        np.zeros((64, 65), dtype=np.complex64), energy=100e3, extent=(32.0, 32.5)
    )

    region = abtem.AnnularDetector(inner=5, outer=15).get_detector_region(
        waves, fftshift=fftshift
    )

    assert region.metadata["units"] == "%"
    assert np.issubdtype(region.array.dtype, np.floating)
    np.testing.assert_array_equal(np.unique(region.array), [0.0, 100.0])


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
        abtem.AnnularDetector(inner=10, outer=40),
        abtem.FlexibleAnnularDetector(outer=40),
        abtem.SegmentedDetector(inner=10, outer=40, nbins_radial=2, nbins_azimuthal=4),
    ],
    ids=["annular", "flexible_annular", "segmented"],
)
def test_radial_detector_show_can_be_drawn(detector):
    """show() of a radial detector renders, not only builds the figure: the
    one-region AnnularDetector made a one-colour colormap that matplotlib refused
    when drawing it ("data mapping points must start with x=0 and end with x=1")."""
    import matplotlib.pyplot as plt

    try:
        detector.show(energy=100e3, gpts=64, sampling=0.05)
        plt.gcf().canvas.draw()
    finally:
        plt.close("all")


@pytest.mark.parametrize("num_colors", [1, 2, 5, 10, 13])
def test_discrete_cmap_gives_the_first_colours_of_its_base_map(num_colors):
    """Region i of a detector plot (values -0.5 to num_colors - 0.5) has colour i
    of the base colormap, for one region as for several."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from abtem.visualize.visualizations import discrete_cmap

    base = "tab10" if num_colors <= 10 else "tab20"
    cmap = discrete_cmap(num_colors=num_colors, base_cmap=base)
    norm = Normalize(vmin=-0.5, vmax=num_colors - 0.5)
    np.testing.assert_allclose(
        cmap(norm(np.arange(num_colors))),
        plt.get_cmap(base)(range(num_colors)),
    )


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


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=30, outer=100),
        abtem.FlexibleAnnularDetector(step_size=10, inner=10, outer=80),
        abtem.SegmentedDetector(inner=10, outer=80, nbins_radial=2, nbins_azimuthal=4),
        abtem.PixelatedDetector(max_angle=None),
        abtem.PixelatedDetector(max_angle=60),
        abtem.WavesDetector(),
    ],
    ids=lambda detector: type(detector).__name__
    + str(getattr(detector, "max_angle", "")),
)
@pytest.mark.parametrize(
    "chunks",
    [
        (3, 32, 32),
        (3, 16, 32),
        (3, 32, 11),
        (1, (20, 12), (8, 8, 16)),
    ],
    ids=["unsplit", "split_first_axis", "split_second_axis", "split_both"],
)
@devices
def test_lazy_detect_of_waves_split_along_the_grid_matches_eager(
    device, chunks, detector
):
    """A detector sees the whole grid of lazy waves that are split into several
    dask chunks along the grid."""
    import dask.array as da

    from abtem.core.axes import UnknownAxis
    from abtem.core.backend import asnumpy, copy_to_device

    rng = np.random.default_rng(0)
    array = rng.normal(size=(3, 32, 32)) + 1j * rng.normal(size=(3, 32, 32))
    array = array.astype(np.complex64)
    kwargs = dict(energy=100e3, sampling=0.1, ensemble_axes_metadata=[UnknownAxis()])
    eager = abtem.Waves(copy_to_device(array, device), **kwargs)
    lazy = abtem.Waves(
        copy_to_device(da.from_array(array, chunks=chunks), device), **kwargs
    )

    with abtem.config.set({"fft": "numpy"}):
        expected = detector.detect(eager)
        result = detector.detect(lazy)
        assert result.is_lazy
        computed = result.copy().compute(progress_bar=False)

    expected_array = asnumpy(expected.array)
    np.testing.assert_allclose(
        asnumpy(computed.array),
        expected_array,
        rtol=0,
        atol=1e-5 * np.abs(expected_array).max(),
    )


@pytest.fixture
def float64_numpy_fft():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


def _real_space_setup(sampling, energy=60e3, scan_gpts=(3, 4), cells=(2, 1, 1)):
    import ase.build

    atoms = ase.build.mx2("WSe2", vacuum=2) * cells
    potential = abtem.Potential(atoms, sampling=sampling, slice_thickness=2)
    probe = abtem.Probe(energy=energy, semiangle_cutoff=20)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=(0.5, 1), fractional=True, potential=potential, gpts=scan_gpts
    )
    return potential, probe, scan


@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize("resample", [False, 0.1])
@pytest.mark.parametrize("entry_point", ["detect", "detect_lazy", "scan", "scan_lazy"])
def test_real_space_pixelated_detector_declares_what_it_returns(entry_point, resample):
    """PixelatedDetector(reciprocal_space=False) declares the shape and sampling
    of the intensity image it returns, |psi|^2 on the waves' grid or on the grid
    `resample` gives, through every entry point."""
    potential, probe, scan = _real_space_setup(sampling=0.05)
    waves = probe.scan(
        potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
    )
    detector = abtem.PixelatedDetector(reciprocal_space=False, resample=resample)

    expected = np.abs(waves.array) ** 2
    expected_sampling = waves.sampling
    if resample:
        gpts = tuple(int(np.ceil(e / resample)) for e in waves.extent)
        expected_sampling = tuple(e / n for e, n in zip(waves.extent, gpts))
        expected = waves.intensity().interpolate(sampling=resample).array
        assert expected.shape[-2:] == gpts

    if entry_point == "detect":
        result = detector.detect(waves)
    elif entry_point == "detect_lazy":
        result = detector.detect(waves.copy().ensure_lazy())
    else:
        result = probe.scan(
            potential, scan=scan, detectors=detector, lazy=entry_point == "scan_lazy"
        )

    assert result.shape == expected.shape
    result = result.compute(scheduler="synchronous") if result.is_lazy else result
    assert result.shape == expected.shape
    np.testing.assert_allclose(
        (result.axes_metadata[-2].sampling, result.axes_metadata[-1].sampling),
        expected_sampling,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.array, expected, rtol=0, atol=1e-12 * expected.max()
    )


@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize("lazy", [False, True])
def test_real_space_pixelated_detector_multislice(lazy):
    potential, probe, _ = _real_space_setup(sampling=0.05)
    waves = probe.multislice(potential, detectors=abtem.WavesDetector(), lazy=False)
    expected = np.abs(waves.array) ** 2

    result = probe.multislice(
        potential,
        detectors=abtem.PixelatedDetector(reciprocal_space=False),
        lazy=lazy,
    )
    result = result.compute(scheduler="synchronous") if lazy else result

    assert result.shape == expected.shape == (128, 221)
    np.testing.assert_allclose(
        result.array, expected, rtol=0, atol=1e-12 * expected.max()
    )


def test_real_space_pixelated_detector_refuses_uniform_resampling():
    # "uniform" equalises the angular sampling of diffraction patterns
    with pytest.raises(ValueError, match="diffraction patterns only"):
        abtem.PixelatedDetector(reciprocal_space=False, resample="uniform")


@pytest.mark.parametrize("max_angle", [0, 0.0, -1, -0.5, np.int64(0), np.float32(-3)])
def test_pixelated_detector_refuses_a_non_positive_max_angle(max_angle):
    # a non-positive angle selects no pattern: it is refused at construction
    with pytest.raises(ValueError, match="max_angle must be positive"):
        abtem.PixelatedDetector(max_angle=max_angle)


@pytest.mark.parametrize("max_angle", [None, "valid", "cutoff", "full", 30, 0.5])
def test_pixelated_detector_accepts_a_positive_max_angle_or_a_name(max_angle):
    assert abtem.PixelatedDetector(max_angle=max_angle).max_angle == max_angle


def _waves_detector_setup():
    import ase.build

    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    potential = abtem.Potential(atoms, sampling=0.1, slice_thickness=2)
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        (0, 0), (0.5, 1), fractional=True, potential=potential, gpts=(3, 4)
    )
    return potential, probe, scan


def _run_waves_detector(entry, potential, probe, scan, detector, lazy):
    if entry == "detect":
        waves = probe.scan(
            potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
        )
        if lazy:
            waves = waves.copy().ensure_lazy()
        return detector.detect(waves)
    if entry == "scan":
        return probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)
    if entry == "multislice":
        return probe.multislice(potential, detectors=detector, lazy=lazy)
    if entry == "prism":
        s_matrix = abtem.SMatrix(potential=potential, energy=60e3, semiangle_cutoff=20)
        return s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)
    raise ValueError(entry)


# (32, 48) crops, (80, 128) pads the (64, 111) exit-wave grid; both differ along
# x and y and from the (3, 4) scan, so a swapped or misplaced axis changes the shape
@pytest.mark.parametrize("gpts", [(32, 48), (80, 128)])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("entry", ["detect", "scan", "multislice", "prism"])
def test_waves_detector_gpts_matches_downsample(entry, lazy, gpts):
    """`WavesDetector(gpts)` gives what `Waves.downsample(gpts)` gives on the
    full-grid waves: `gpts` points over the unchanged extent."""
    with abtem.config.set({"precision": "float64", "fft": "numpy", "device": "cpu"}):
        potential, probe, scan = _waves_detector_setup()
        full = _run_waves_detector(
            entry, potential, probe, scan, abtem.WavesDetector(), lazy
        )
        full = full.compute() if lazy else full
        expected = full.downsample(gpts=gpts)

        result = _run_waves_detector(
            entry, potential, probe, scan, abtem.WavesDetector(gpts=gpts), lazy
        )
        declared_shape = result.shape
        result = result.compute() if lazy else result

    assert declared_shape == result.shape == expected.shape
    assert np.allclose(result.sampling, expected.sampling, rtol=1e-12, atol=0)
    assert np.allclose(result.extent, full.extent, rtol=1e-12, atol=0)
    assert result.antialias_cutoff_gpts == expected.antialias_cutoff_gpts
    scale = np.abs(expected.array).max()
    np.testing.assert_allclose(result.array, expected.array, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("gpts", [None, ()])
@pytest.mark.parametrize("lazy", [False, True])
def test_waves_detector_without_gpts_returns_scanned_waves(lazy, gpts):
    """No `gpts`, as `None` or an empty tuple, leaves ensemble waves as they are."""
    with abtem.config.set({"precision": "float64", "fft": "numpy", "device": "cpu"}):
        potential, probe, scan = _waves_detector_setup()
        waves = probe.scan(
            potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
        )
        source = waves.copy().ensure_lazy() if lazy else waves
        result = abtem.WavesDetector(gpts=gpts).detect(source)
        result = result.compute() if lazy else result

    assert result.shape == waves.shape
    assert np.array_equal(result.sampling, waves.sampling)
    assert np.array_equal(result.array, waves.array)


@devices
@pytest.mark.parametrize("pair", [tuple, list])
def test_pixelated_detector_downsamples_to_two_samplings(device, pair):
    # both samplings are coarser than the waves' (0.1 and 0.077 1 / Å)
    probe = abtem.Probe(
        energy=100e3,
        semiangle_cutoff=20,
        extent=(10, 13),
        gpts=(64, 80),
        device=device,
    )
    waves = probe.build()
    detector = abtem.PixelatedDetector(resample=pair((0.137, 0.15)))

    detected = detector.detect(waves)

    assert detected.sampling == (0.137, 0.15)
    # the pre-allocated output shape is the computed one
    assert detected.base_shape == detector._out_base_shape(waves)[0] == (22, 20)


def _resample_entries(device):
    # the waves are sampled at (0.1, 0.0769) 1 / Å, unequal along the two axes;
    # a PRISM reduction at interpolation 2 at (0.2, 0.154) 1 / Å
    atoms = ase.Atoms("C", positions=[(5, 6.5, 1)], cell=(10, 13, 2))
    potential = abtem.Potential(atoms, gpts=(64, 80), device=device)
    phonon_potential = abtem.Potential(
        abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.1, seed=1),
        gpts=(64, 80),
        device=device,
    )
    # the step is a whole number of pixels of the PRISM grids, (11, 7) of (44, 56)
    scan = abtem.GridScan(start=(0, 0), end=(5, 6.5), gpts=(2, 4))

    def probe():
        return abtem.Probe(
            energy=100e3,
            semiangle_cutoff=20,
            extent=(10, 13),
            gpts=(64, 80),
            device=device,
        )

    def s_matrix(**kwargs):
        return abtem.SMatrix(
            potential=potential,
            energy=100e3,
            semiangle_cutoff=20,
            device=device,
            **kwargs,
        )

    return {
        "scan": lambda d: probe().scan(potential, scan=scan, detectors=d, lazy=False),
        "lazy scan": lambda d: probe().scan(
            potential, scan=scan, detectors=d, lazy=True
        ),
        "detect": lambda d: d.detect(
            probe().multislice(potential, scan=scan, lazy=False)
        ),
        "lazy detect": lambda d: d.detect(
            probe().multislice(potential, scan=scan, lazy=True)
        ),
        "lazy frozen-phonon scan": lambda d: probe().scan(
            phonon_potential, scan=scan, detectors=d, lazy=True
        ),
        "prism scan": lambda d: s_matrix(interpolation=2).scan(
            scan=scan, detectors=d, lazy=True
        ),
    }


_RESAMPLE_ENTRIES = (
    "scan",
    "lazy scan",
    "detect",
    "lazy detect",
    "lazy frozen-phonon scan",
    "prism scan",
)


@devices
@pytest.mark.parametrize("entry", _RESAMPLE_ENTRIES)
# None keeps the whole grid
@pytest.mark.parametrize("max_angle", ["valid", 60.0, None])
# finer along both axes, along x only, along y only, uniform, coarser
@pytest.mark.parametrize(
    "resample", [0.05, (0.09, 0.2), (0.2, 0.07), "uniform", (0.12, 0.08)]
)
def test_pixelated_detector_resample_equals_interpolate(
    device, entry, max_angle, resample
):
    """A resampling detector gives the patterns of the same run without
    resampling, interpolated with DiffractionPatterns.interpolate, for samplings
    coarser and finer than the waves'."""
    from abtem.core.backend import asnumpy

    run = _resample_entries(device)[entry]
    expected = run(abtem.PixelatedDetector(max_angle=max_angle))
    expected = expected.compute().interpolate(sampling=resample)

    result = run(abtem.PixelatedDetector(max_angle=max_angle, resample=resample))
    declared = result.shape
    result = result.compute()

    assert declared == result.shape == expected.shape
    assert result.sampling == expected.sampling
    expected = asnumpy(expected.array)
    np.testing.assert_allclose(
        asnumpy(result.array), expected, rtol=0, atol=1e-5 * expected.max()
    )


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_real_space_pixelated_detector_resamples_to_a_sampling_that_divides_the_extent(
    lazy,
):
    # 10.8 / 0.3 is 36.00000000000001 in floats, so the detector declared 37 pixels
    # where the images had 36.
    from ase import Atoms

    atoms = Atoms("C", positions=[(5.4, 5.4, 1.0)], cell=(10.8, 10.8, 2.0), pbc=True)
    potential = abtem.Potential(atoms, gpts=36, slice_thickness=2.0)
    detector = abtem.PixelatedDetector(reciprocal_space=False, resample=0.3)

    result = abtem.PlaneWave(energy=100e3).multislice(
        potential, detectors=detector, lazy=lazy
    )
    result = result.compute() if lazy else result

    assert result.shape == (36, 36)
    assert result.base_axes_metadata[0].sampling == pytest.approx(0.3)
