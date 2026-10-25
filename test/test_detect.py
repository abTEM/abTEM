import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import devices, gpu, lazy_params, to_host_array

import abtem
from abtem.detectors import _gpts_and_sampling_from_obj


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
    ],
)
def test_detect(data, detector, lazy):
    waves = data.draw(abtem_st.waves(lazy=lazy, device="cpu"))
    detector = data.draw(detector())
    assume(all(waves._gpts_within_angle(min(detector.angular_limits(waves)))))
    assume(min(waves.cutoff_angles) > 1.0)

    measurement = detector.detect(waves).compute()

    # The _out_* methods describe each output of a multi-output detector.
    assert measurement.ensemble_shape == waves.ensemble_shape
    assert measurement.dtype == detector._out_dtype(waves)[0]
    assert measurement.base_shape == detector._out_base_shape(waves)[0]
    assert type(measurement) is detector._out_type(waves)[0]
    assert measurement.base_axes_metadata == detector._out_base_axes_metadata(waves)[0]

    if detector.to_cpu:
        assert measurement.device == "cpu"


def test_radial_sensitivity():
    """A radially varying detector sensitivity weights the diffraction pattern by
    ``w(alpha)`` before integration. Verify against a manually computed ground truth and
    that uniform sensitivity reproduces the unweighted detector."""
    from abtem.detectors import RadialSensitivity

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, extent=20, gpts=256)

    def value(measurement):
        if measurement.is_lazy:
            measurement = measurement.compute()
        return np.asarray(measurement.array)

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

    # Detection with a sensitivity goes through detect's eager and lazy
    # (map_blocks) paths; the checks after this loop do not detect lazy waves.
    for lazy in (False, True):
        waves = probe.build(lazy=lazy)

        plain = value(abtem.AnnularDetector(inner=0, outer=20).detect(waves))
        unit = value(
            abtem.AnnularDetector(
                inner=0, outer=20, sensitivity=lambda a: np.ones_like(a)
            ).detect(waves)
        )
        assert np.allclose(plain, unit), f"lazy={lazy}"

        linear = value(
            abtem.AnnularDetector(
                inner=0, outer=20, sensitivity=lambda a: a / 20.0
            ).detect(waves)
        )
        # Down-weighting with w(alpha) <= 1 must reduce the signal.
        assert linear < plain, f"lazy={lazy}"
        assert np.isclose(manual, linear, rtol=1e-4), f"lazy={lazy}"

        # The (angles, values) lookup form matches the equivalent callable.
        lookup = value(
            abtem.AnnularDetector(
                inner=0, outer=20, sensitivity=([0, 10, 20], [0.0, 0.5, 1.0])
            ).detect(waves)
        )
        assert np.isclose(lookup, linear, rtol=1e-4), f"lazy={lazy}"

        # The FlexibleAnnularDetector weights per-pixel before binning, so
        # integrating its polar bins reproduces the sensitivity-weighted annular
        # detector.
        flexible = abtem.FlexibleAnnularDetector(
            step_size=0.5, inner=0, outer=20, sensitivity=lambda a: a / 20.0
        ).detect(waves)
        flexible_img = value(flexible.integrate_radial(0, 20))
        assert np.isclose(flexible_img, linear, rtol=2e-2), f"lazy={lazy}"

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

    # The plural get_detector_regions keeps working on the annular detector, as it did
    # before the sensitivity was added.
    annular_regions = abtem.AnnularDetector(inner=0, outer=40).get_detector_regions(
        reference
    )
    assert annular_regions.metadata["label"] == "detector regions"

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


def _sensitivity_probe(device):
    return abtem.Probe(
        energy=100e3, semiangle_cutoff=20, extent=20, gpts=256, device=device
    )


def _hand_region(gpts, sampling, inner, outer, sensitivity, shift=(0, 0)):
    """The sensitivity-weighted annular region on a grid in unshifted FFT layout, by
    hand in float64. The detector centre is displaced by ``shift`` pixels (the
    detectors round an offset to whole pixels)."""
    freqs = [np.fft.fftfreq(n, 1 / (s * n)) for n, s in zip(gpts, sampling)]
    kx, ky = np.meshgrid(*freqs, indexing="ij")
    alpha = np.sqrt(
        (kx - shift[0] * sampling[0]) ** 2 + (ky - shift[1] * sampling[1]) ** 2
    )
    return np.where((alpha >= inner) & (alpha < outer), sensitivity(alpha), 0.0)


def _hand_weighted_signal(waves, inner, outer, sensitivity, shift=(0, 0)):
    """The sensitivity-weighted annular signal of ``waves``, summed by hand in float64
    from the raw diffraction pattern."""
    dp = to_host_array(
        waves.diffraction_patterns(max_angle="full", parity="same", fftshift=False)
    ).astype(np.float64)
    region = _hand_region(
        dp.shape[-2:], waves.angular_sampling, inner, outer, sensitivity, shift
    )
    return float((dp * region).sum())


@devices
@lazy_params
@pytest.mark.parametrize("offset", [(0.0, 0.0), (5.0, -3.0)])
def test_radial_sensitivity_agrees_between_detectors(device, lazy, offset):
    # The sensitivity is a function of the angle from the detector centre, offset
    # included, for every radial detector. Reference: a float64 sum by hand.
    inner, outer = 20.0, 80.0

    def sensitivity(alpha):
        return (100.0 - alpha) / 80.0

    waves = _sensitivity_probe(device).build(lazy=False)
    shift = tuple(int(round(o / s)) for o, s in zip(offset, waves.angular_sampling))
    expected = _hand_weighted_signal(waves, inner, outer, sensitivity, shift)

    # The same signal with the weights centred on the optical axis, which an offset
    # detector must not use.
    axis_centred = _hand_weighted_signal(waves, inner, outer, sensitivity, (0, 0))
    if offset != (0.0, 0.0):
        assert not np.isclose(expected, axis_centred, rtol=1e-2)

    waves = _sensitivity_probe(device).build(lazy=lazy)
    annular = abtem.AnnularDetector(
        inner=inner, outer=outer, offset=offset, sensitivity=sensitivity
    )
    segmented = abtem.SegmentedDetector(
        nbins_radial=3,
        nbins_azimuthal=4,
        inner=inner,
        outer=outer,
        offset=offset,
        sensitivity=sensitivity,
    )

    assert np.isclose(to_host_array(annular.detect(waves)).sum(), expected, rtol=2e-4)
    assert np.isclose(to_host_array(segmented.detect(waves)).sum(), expected, rtol=2e-4)

    if offset == (0.0, 0.0):
        flexible = abtem.FlexibleAnnularDetector(
            step_size=5.0, inner=inner, outer=outer, sensitivity=sensitivity
        )
        assert np.isclose(
            to_host_array(flexible.detect(waves)).sum(), expected, rtol=2e-4
        )


@devices
@pytest.mark.parametrize("offset", [(0.0, 0.0), (5.0, -3.0)])
def test_radial_sensitivity_detector_region_follows_the_offset(device, offset):
    # AnnularDetector.get_detector_region is the efficiency [%] the detector applies,
    # centred on the (whole-pixel) offset detector centre.
    inner, outer = 20.0, 80.0

    def sensitivity(alpha):
        return (100.0 - alpha) / 80.0

    waves = _sensitivity_probe(device).build(lazy=False)
    shift = tuple(int(round(o / s)) for o, s in zip(offset, waves.angular_sampling))
    gpts, sampling, _, _ = _gpts_and_sampling_from_obj(waves)
    region = _hand_region(gpts, sampling, inner, outer, sensitivity, shift)

    detector = abtem.AnnularDetector(
        inner=inner, outer=outer, offset=offset, sensitivity=sensitivity
    )
    shown = to_host_array(detector.get_detector_region(waves, fftshift=False))
    assert np.allclose(shown, region * 100.0, atol=1e-3)


@pytest.mark.parametrize("order", [("float64", "float32"), ("float32", "float64")])
def test_radial_sensitivity_follows_the_configured_precision(order):
    # The weight map is cached per grid; the cache must not hand a float64 map to
    # float32 patterns (or the reverse), which would widen the measurement.
    from abtem.measurements import _radial_weight_map

    def sensitivity(alpha):
        return alpha / 40.0

    expected_dtype = {"float32": np.float32, "float64": np.float64}
    detectors = (
        abtem.AnnularDetector(inner=0, outer=20, sensitivity=sensitivity),
        abtem.FlexibleAnnularDetector(
            step_size=1.0, inner=0, outer=20, sensitivity=sensitivity
        ),
        abtem.SegmentedDetector(
            nbins_radial=2,
            nbins_azimuthal=4,
            inner=0,
            outer=20,
            sensitivity=sensitivity,
        ),
    )

    for precision in order:
        with abtem.config.set({"precision": precision}):
            waves = _sensitivity_probe("cpu").build(lazy=False)
            for detector in detectors:
                measurement = detector.detect(waves)
                assert measurement.array.dtype == expected_dtype[precision], (
                    type(detector).__name__,
                    precision,
                )

            weights = _radial_weight_map(
                (64, 64), [1.0, 1.0], abtem.RadialSensitivity(sensitivity)
            )
            assert weights.dtype == expected_dtype[precision]
            # The map is shared between callers, so it is read-only.
            assert not weights.flags.writeable


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
        abtem.SegmentedDetector(
            nbins_radial=2,
            nbins_azimuthal=4,
            inner=40,
            outer=100,
            offset=(10.0, 0.0),
            sensitivity=lambda a: a / 150.0,
        ),
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


def test_measurement_detectors_default_to_cpu():
    """Every detector producing a measurement returns it on the host by
    default.

    SegmentedDetector used to default to ``to_cpu=False`` while its own
    docstring (and every sibling detector) said True, so on a GPU run its
    measurements came back as CuPy arrays while the others were NumPy --
    an inconsistency that only surfaced on a CuPy workstation.

    WavesDetector is deliberately the exception: it returns the (large) wave
    functions themselves and is the implicit detector when none is given, so it
    must not force a device-to-host copy of every exit wave.
    """
    from abtem.detectors import WavesDetector

    defaults = [
        (abtem.AnnularDetector(inner=0, outer=20), True),
        (abtem.FlexibleAnnularDetector(), True),
        (
            abtem.SegmentedDetector(
                inner=0, outer=20, nbins_radial=2, nbins_azimuthal=4
            ),
            True,
        ),
        (abtem.PixelatedDetector(), True),
        (WavesDetector(), False),
    ]
    for detector, to_cpu in defaults:
        assert detector.to_cpu is to_cpu, type(detector).__name__


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


@pytest.fixture(scope="module")
def exit_waves():
    """The eager exit waves of `_real_space_setup(sampling)` through an entry point
    ("scan", "multislice" or "prism") with a WavesDetector, computed once per
    worker; each call hands out a copy."""
    cache = {}

    def exit_waves(entry, sampling, device):
        key = (entry, sampling, device)
        if key not in cache:
            with abtem.config.set(
                {"precision": "float64", "fft": "numpy", "device": device}
            ):
                potential, probe, scan = _real_space_setup(sampling)
                cache[key] = _run_waves_detector(
                    entry, potential, probe, scan, abtem.WavesDetector(), False
                )
        return cache[key].copy()

    return exit_waves


# The lazy detect takes the block path of the lazy scan, and the eager scan
# detects eager waves as the eager detect does.
@pytest.mark.float64
@pytest.mark.usefixtures("float64_numpy_fft", "on_device")
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("entry_point", ["detect", "scan_lazy"])
def test_real_space_pixelated_detector_declares_what_it_returns(
    entry_point, device, exit_waves
):
    """PixelatedDetector(reciprocal_space=False) declares the shape and sampling
    of the intensity image it returns, |psi|^2 on the waves' grid or on the grid
    `resample` gives, through every entry point."""
    potential, probe, scan = _real_space_setup(sampling=0.05)
    waves = exit_waves("scan", 0.05, device)

    for resample in (False, 0.1):
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
        else:
            result = probe.scan(potential, scan=scan, detectors=detector, lazy=True)

        message = f"resample={resample}"
        assert result.shape == expected.shape, message
        result = result.compute(scheduler="synchronous") if result.is_lazy else result
        assert result.shape == expected.shape, message
        np.testing.assert_allclose(
            (result.axes_metadata[-2].sampling, result.axes_metadata[-1].sampling),
            expected_sampling,
            rtol=1e-12,
            err_msg=message,
        )
        np.testing.assert_allclose(
            to_host_array(result),
            expected,
            rtol=0,
            atol=1e-12 * expected.max(),
            err_msg=message,
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


# The lazy detect takes the block path of the lazy scan; the eager scan and the
# lazy multislice add no path of their own.
@pytest.mark.usefixtures("float64_numpy_fft")
@pytest.mark.parametrize("entry_point", ["detect", "scan_lazy", "multislice"])
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


def _run_waves_detector(entry, potential, probe, scan, detector, lazy):
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
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("entry", ["detect", "scan", "multislice", "prism"])
def test_waves_detector_gpts_matches_downsample(entry, lazy, device, exit_waves):
    """`WavesDetector(gpts)` gives what `Waves.downsample(gpts)` gives on the
    full-grid waves: `gpts` points over the unchanged extent."""
    # "detect" detects the exit waves of the scan
    full = exit_waves("scan" if entry == "detect" else entry, 0.1, device)
    for gpts in [(32, 48), (80, 128)]:
        with abtem.config.set(
            {"precision": "float64", "fft": "numpy", "device": device}
        ):
            expected = full.downsample(gpts=gpts)
            detector = abtem.WavesDetector(gpts=gpts)
            if entry == "detect":
                result = detector.detect(full.copy().ensure_lazy() if lazy else full)
            else:
                potential, probe, scan = _real_space_setup(sampling=0.1)
                result = _run_waves_detector(
                    entry, potential, probe, scan, detector, lazy
                )
            declared_shape = result.shape
            result = result.compute() if lazy else result

        message = f"gpts={gpts}"
        assert declared_shape == result.shape == expected.shape, message
        assert np.allclose(result.sampling, expected.sampling, rtol=1e-12, atol=0)
        assert np.allclose(result.extent, full.extent, rtol=1e-12, atol=0)
        assert result.antialias_cutoff_gpts == expected.antialias_cutoff_gpts
        expected_array = to_host_array(expected)
        scale = np.abs(expected_array).max()
        np.testing.assert_allclose(
            to_host_array(result),
            expected_array,
            rtol=0,
            atol=1e-12 * scale,
            err_msg=message,
        )


@pytest.mark.float64
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [False, True])
def test_waves_detector_without_gpts_returns_scanned_waves(lazy, device, exit_waves):
    """No `gpts`, as `None` or an empty tuple, leaves ensemble waves as they are."""
    waves = exit_waves("scan", 0.1, device)
    for gpts in (None, ()):
        with abtem.config.set(
            {"precision": "float64", "fft": "numpy", "device": device}
        ):
            source = waves.copy().ensure_lazy() if lazy else waves
            result = abtem.WavesDetector(gpts=gpts).detect(source)
            result = result.compute() if lazy else result

        assert result.shape == waves.shape, gpts
        assert np.array_equal(result.sampling, waves.sampling), gpts
        assert np.array_equal(to_host_array(result), to_host_array(waves)), gpts


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_waves_detector_without_gpts_declares_the_dtype_of_the_waves(device):
    """Without resampling the detected waves are the waves themselves, so a
    lazy detection must declare their dtype, not the configured precision.
    `gpts=()` declared complex128 for complex64 waves under float64 precision
    (it tested `gpts is not None`) and then computed complex64."""
    rng = np.random.default_rng(0)
    array = (rng.standard_normal((2, 8, 8)) + 1j * rng.standard_normal((2, 8, 8)))
    waves = abtem.Waves(
        array.astype(np.complex64),
        energy=100e3,
        extent=(4.0, 4.0),
        ensemble_axes_metadata=[abtem.core.axes.OrdinalAxis(values=(0, 1))],
    )
    waves = waves.copy_to_device(device)

    for gpts in (None, ()):
        with abtem.config.set({"precision": "float64"}):
            result = abtem.WavesDetector(gpts=gpts).detect(waves.copy().ensure_lazy())
            declared = result.array.dtype
            computed = result.compute().array.dtype
        assert declared == computed == np.complex64, (gpts, declared, computed)
