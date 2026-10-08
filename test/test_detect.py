import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import gpu, to_host_array

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


@pytest.mark.parametrize("fftshift", [True, False])
@pytest.mark.parametrize("gpts", [(64, 65), (65, 64)])
def test_annular_detector_region_is_centred_on_offset(fftshift, gpts):
    # An annulus inner <= |alpha - offset| < outer is symmetric under reflection
    # through `offset`; with `offset` on a grid point and the ring well inside
    # the grid, the reflected grid coincides with itself, so the region's center
    # of mass is exactly `offset`, in whichever storage order it was requested.
    from abtem.measurements import DiffractionPatterns

    base = DiffractionPatterns(
        np.zeros(gpts, dtype=np.float32),
        sampling=(0.02, 0.025),
        fftshift=True,
        metadata={"energy": 100e3},
    )
    ax, ay = base.angular_sampling
    offset = (3 * ax, -2 * ay)
    # Radii off-grid (x.3 pixels) so no pixel sits on a boundary; the ring
    # reaches <= 13.3 px from the centre, inside the >= 32 px half-width.
    detector = abtem.AnnularDetector(inner=2.3 * ax, outer=10.3 * ax, offset=offset)

    region = detector.get_detector_region(base, fftshift=fftshift)

    com = complex(region.center_of_mass(units="mrad").array)
    # float32 angular coordinates: ~1e-7 relative; a one-pixel error is
    # >= 1 / 3.6 of |offset|.
    assert com == pytest.approx(offset[0] + 1.0j * offset[1], rel=1e-5)


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=30, outer=100, offset=(5.0, -3.0)),
        abtem.FlexibleAnnularDetector(step_size=10, inner=10, outer=80),
        abtem.PixelatedDetector(max_angle=60, resample="uniform"),
        abtem.WavesDetector(),
        abtem.SpectralSlitDetector(width=20, q_max=60, angle=30.0, offset=(5.0, 0.0)),
        abtem.SpectralSlitDetector(corners=(-10.0, 50.0, -5.0, 5.0)),
        abtem.SpectralAnnularDetector(outer=10, q_max=40, angle=45.0),
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
        abtem.SpectralSlitDetector(
            width=20, q_min=5.0, q_max=60, angle=30.0, offset=(5.0, -3.0)
        ),
        abtem.SpectralSlitDetector(corners=(-10.0, 50.0, -5.0, 5.0)),
    ],
    ids=["slit-parameters", "corners"],
)
def test_a_spectral_slit_detector_without_the_attributes_of_an_older_version(detector):
    """An older version did not record the form the geometry was given in
    (`_from_corners`) nor the given `q_max`. Such a detector is rebuilt from its slit
    parameters, which describe the same rectangle."""
    rng = np.random.default_rng(0)
    array = rng.normal(size=(3, 32, 32)) + 1j * rng.normal(size=(3, 32, 32))
    waves = abtem.Waves(
        array.astype(np.complex64),
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[abtem.core.axes.UnknownAxis()],
    )

    detector = detector.copy()
    mask = detector._get_detector_region_array(waves)
    eager = detector.detect(waves)

    del detector._from_corners, detector._q_max
    rebuilt = type(detector)(**detector._copy_kwargs())

    np.testing.assert_allclose(rebuilt.corners, detector.corners, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(rebuilt._get_detector_region_array(waves), mask)
    np.testing.assert_array_equal(rebuilt.detect(waves).array, eager.array)
    lazy = rebuilt.detect(waves.ensure_lazy()).compute(progress_bar=False)
    np.testing.assert_array_equal(lazy.array, eager.array)


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=10, outer=40),
        abtem.FlexibleAnnularDetector(outer=40),
        abtem.SegmentedDetector(
            inner=10, outer=40, nbins_radial=2, nbins_azimuthal=4
        ),
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


@pytest.fixture
def float64_numpy_fft():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


@pytest.fixture
def on_device(device):
    with abtem.config.set({"device": device}):
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


@pytest.mark.float64
@pytest.mark.usefixtures("float64_numpy_fft", "on_device")
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("resample", [False, 0.1])
@pytest.mark.parametrize("entry_point", ["detect", "detect_lazy", "scan", "scan_lazy"])
def test_real_space_pixelated_detector_declares_what_it_returns(
    entry_point, resample, device
):
    """PixelatedDetector(reciprocal_space=False) declares the shape and sampling
    of the intensity image it returns, |psi|^2 on the waves' grid or on the grid
    `resample` gives, through every entry point."""
    potential, probe, scan = _real_space_setup(sampling=0.05)
    waves = probe.scan(
        potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
    )
    detector = abtem.PixelatedDetector(reciprocal_space=False, resample=resample)

    expected = np.abs(to_host_array(waves)) ** 2
    expected_sampling = waves.sampling
    if resample:
        gpts = tuple(int(np.ceil(e / resample)) for e in waves.extent)
        expected_sampling = tuple(e / n for e, n in zip(waves.extent, gpts))
        expected = to_host_array(waves.intensity().interpolate(sampling=resample))
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
        to_host_array(result), expected, rtol=0, atol=1e-12 * expected.max()
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


@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize("lazy", [False, True])
def test_real_space_pixelated_detector_multi_energy(lazy):
    energies = [50e3, 60e3, 70e3, 80e3]
    potential, probe, scan = _real_space_setup(
        sampling=0.1, energy=energies, scan_gpts=(2, 3), cells=(1, 1, 1)
    )
    detector = abtem.PixelatedDetector(reciprocal_space=False)

    result = probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)
    result = result.compute(scheduler="synchronous") if lazy else result

    energy_axis = [type(a).__name__ for a in result.axes_metadata].index("EnergyAxis")
    assert result.shape[energy_axis] == len(energies)
    for i, energy in enumerate(energies):
        single = abtem.Probe(energy=energy, semiangle_cutoff=20).scan(
            potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
        )
        expected = np.abs(single.array) ** 2
        np.testing.assert_allclose(
            np.take(result.array, i, axis=energy_axis),
            expected,
            rtol=0,
            atol=1e-12 * expected.max(),
        )


def _segmented(outer):
    return abtem.SegmentedDetector(
        nbins_radial=3, nbins_azimuthal=4, inner=20, outer=outer, rotation=0.3
    )


@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize(
    "entry_point",
    [
        "detect",
        pytest.param(
            "detect_lazy",
            marks=pytest.mark.skipif(
                not hasattr(abtem.SegmentedDetector, "offset"),
                reason="lazy detect needs SegmentedDetector.offset",
            ),
        ),
        "scan",
        "scan_lazy",
        "multislice",
        "multislice_lazy",
    ],
)
def test_segmented_detector_without_outer_matches_an_explicit_cutoff(entry_point):
    """Without an outer angle the detector integrates up to the antialias cutoff
    angle of the detected waves."""
    potential, probe, scan = _real_space_setup(
        sampling=0.1, scan_gpts=(3, 5), cells=(1, 1, 1)
    )
    lazy = entry_point.endswith("_lazy")

    def detected(outer):
        detector = _segmented(outer)
        if entry_point.startswith("detect"):
            waves = probe.scan(
                potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
            )
            result = detector.detect(waves.ensure_lazy() if lazy else waves)
        elif entry_point.startswith("scan"):
            result = probe.scan(potential, scan=scan, detectors=detector, lazy=lazy)
        else:
            result = probe.multislice(potential, detectors=detector, lazy=lazy)
        return result.compute(scheduler="synchronous") if result.is_lazy else result

    waves = probe.multislice(
        potential, detectors=abtem.WavesDetector(), lazy=False
    )
    outer = min(waves.cutoff_angles)

    result = detected(None)
    expected = detected(outer)

    np.testing.assert_array_equal(result.array, expected.array)
    assert result.axes_metadata[-2].sampling == pytest.approx((outer - 20) / 3)


@pytest.mark.usefixtures("float64_numpy_fft")
def test_segmented_detector_without_outer_follows_the_detected_waves():
    """A detector without an outer angle is sized from the waves it detects, not
    from waves an earlier call on the same object saw."""
    potential, probe, scan = _real_space_setup(
        sampling=0.1, scan_gpts=(3, 5), cells=(1, 1, 1)
    )
    detector = _segmented(None)
    probe.scan(potential, scan=scan, detectors=detector, lazy=True)

    probe_100 = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe_100.grid.match(potential)
    waves = probe_100.scan(
        potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
    )
    outer = min(waves.cutoff_angles)

    result = detector.detect(waves)
    expected = _segmented(outer).detect(waves)

    np.testing.assert_array_equal(result.array, expected.array)
    assert result.axes_metadata[-2].sampling == pytest.approx((outer - 20) / 3)


@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize("lazy", [False, True])
def test_segmented_detector_without_outer_refuses_multi_energy(lazy):
    potential, _, scan = _real_space_setup(
        sampling=0.1, scan_gpts=(3, 5), cells=(1, 1, 1)
    )
    probe = abtem.Probe(energy=[50e3, 60e3, 70e3], semiangle_cutoff=20)
    probe.grid.match(potential)

    with pytest.raises(RuntimeError, match="cannot auto-size"):
        probe.scan(potential, scan=scan, detectors=_segmented(None), lazy=lazy)


def test_real_space_pixelated_detector_refuses_uniform_resampling():
    # "uniform" equalises the angular sampling of diffraction patterns
    with pytest.raises(ValueError, match="diffraction patterns only"):
        abtem.PixelatedDetector(reciprocal_space=False, resample="uniform")


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
@pytest.mark.float64
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("gpts", [(32, 48), (80, 128)])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("entry", ["detect", "scan", "multislice", "prism"])
def test_waves_detector_gpts_matches_downsample(entry, lazy, gpts, device):
    """`WavesDetector(gpts)` gives what `Waves.downsample(gpts)` gives on the
    full-grid waves: `gpts` points over the unchanged extent."""
    with abtem.config.set({"precision": "float64", "fft": "numpy", "device": device}):
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
    expected_array = to_host_array(expected)
    scale = np.abs(expected_array).max()
    np.testing.assert_allclose(
        to_host_array(result), expected_array, rtol=0, atol=1e-12 * scale
    )


@pytest.mark.float64
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("gpts", [None, ()])
@pytest.mark.parametrize("lazy", [False, True])
def test_waves_detector_without_gpts_returns_scanned_waves(lazy, gpts, device):
    """No `gpts`, as `None` or an empty tuple, leaves ensemble waves as they are."""
    with abtem.config.set({"precision": "float64", "fft": "numpy", "device": device}):
        potential, probe, scan = _waves_detector_setup()
        waves = probe.scan(
            potential, scan=scan, detectors=abtem.WavesDetector(), lazy=False
        )
        source = waves.copy().ensure_lazy() if lazy else waves
        result = abtem.WavesDetector(gpts=gpts).detect(source)
        result = result.compute() if lazy else result

    assert result.shape == waves.shape
    assert np.array_equal(result.sampling, waves.sampling)
    assert np.array_equal(to_host_array(result), to_host_array(waves))
