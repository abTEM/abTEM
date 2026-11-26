import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import assert_array_matches_device, gpu

from abtem import GridScan, WavesDetector
from abtem.core.backend import cp

from hypothesis import HealthCheck, settings

settings.register_profile(
    "my_profile", suppress_health_check=[HealthCheck.filter_too_much]
)
settings.load_profile("my_profile")

@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_prism_matches_probe(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )
    probe_diffraction_patterns = (
        probe.build(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )

    s_matrix_diffraction_patterns.compute()
    probe_diffraction_patterns.compute()
    assert np.allclose(
        s_matrix_diffraction_patterns.compute().array,
        probe_diffraction_patterns.compute().array,
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe_with_interpolation(data, lazy, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device, max_interpolation=3))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )
    probe_diffraction_patterns = (
        probe.build(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
    )

    assert np.allclose(
        s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prism_matches_probe_with_multislice(data, lazy, device):
    potential = data.draw(abtem_st.potential(device=device))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    s_matrix_diffraction_patterns = (
        s_matrix.reduce(lazy=lazy).diffraction_patterns(None).to_cpu().compute()
    )
    probe_diffraction_patterns = (
        probe.multislice(potential=potential, lazy=lazy)
        .diffraction_patterns(None)
        .to_cpu()
        .compute()
    )

    if np.all(probe_diffraction_patterns.array < 1e-7):
        pass
    else:
        assert np.allclose(
            s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
        )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True])
@pytest.mark.parametrize("device", ["cpu", gpu])
# @pytest.mark.parametrize('interpolation', [True, False])
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
        abtem_st.annular_detector,
    ],
)
def test_s_matrix_matches_probe_no_interpolation(data, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))

    detector = data.draw(detector())
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )

    s_matrix_measurement = s_matrix.scan(
        scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()

    assert s_matrix_measurement == probe_measurement


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize(
    "downsample", [True, False], ids=["downsample", "no_downsample"]
)
@pytest.mark.parametrize("device", [gpu, "cpu"], ids=["gpu", "cpu"])
@pytest.mark.parametrize(
    "interpolation", [False, True], ids=["no_interpolation", "interpolation"]
)
@pytest.mark.parametrize(
    "frozen_phonons", [True, False], ids=["frozen_phonons", "no_frozen_phonons"]
)
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
        abtem_st.annular_detector,
    ],
)
# @settings(verbosity=Verbosity.verbose, phases=[Phase.generate])
def test_prism_scan(
    data, interpolation, detector, downsample, lazy, frozen_phonons, device
):
    potential = data.draw(
        abtem_st.potential(
            device=device, no_frozen_phonons=not frozen_phonons, ensemble_mean=False
        )
    )
    ctf = data.draw(abtem_st.ctf(partial_coherence=False))

    max_interpolation = 3 if interpolation else 1

    s_matrix = data.draw(
        abtem_st.s_matrix(
            potential=potential,
            max_interpolation=max_interpolation,
            device=device,
            downsample=downsample,
        )
    )

    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.build(lazy=True).dummy_probes()

    scan = GridScan()
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )
    scan.match_probe(probe)
    measurement_shape = detector._out_shape(probe)[0]
    # Sizing the detector for these probes leaves the angle on it. Make it explicit,
    # so the scan uses the angle the expected shape was computed with (a detector
    # without a given outer angle takes the one of the waves it is applied to).
    if hasattr(detector, "_outer_is_explicit"):
        detector.outer = detector.outer

    measurement = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=lazy)

    assert (
        measurement.shape
        == potential.ensemble_shape
        + ctf.ensemble_shape
        + scan.ensemble_shape
        + measurement_shape
    )
    assert measurement.dtype == detector._out_dtype(probe)[0]

    # measurement = measurement.compute()
    # assert (
    #     measurement.shape
    #     == potential.ensemble_shape + scan.ensemble_shape + measurement_shape
    # )
    # assert measurement.dtype == detector._out_dtype(probe)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.skipif(cp is None, reason="no gpu")
def test_s_matrix_store_on_host(data, lazy):
    potential = data.draw(abtem_st.potential(device="gpu", ensemble_mean=False))
    s_matrix = data.draw(abtem_st.s_matrix(potential=potential))
    s_matrix = s_matrix.build().compute()
    assert_array_matches_device(s_matrix.array, "cpu")


@pytest.mark.parametrize("store_on_host", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_s_matrix_build_dtype_follows_the_precision(precision, lazy, store_on_host):
    from ase.build import bulk

    import abtem
    from abtem.core.utils import get_dtype

    with abtem.config.set({"precision": precision, "device": "cpu"}):
        potential = abtem.Potential(
            bulk("Si", cubic=True), gpts=32, slice_thickness=2.7
        )
        s_matrix = abtem.SMatrix(
            energy=100e3,
            semiangle_cutoff=20,
            potential=potential,
            interpolation=1,
            store_on_host=store_on_host,
        )
        expected = get_dtype(complex=True)
        built = s_matrix.build(lazy=lazy)

        assert expected == (np.complex128 if precision == "float64" else np.complex64)
        assert built.array.dtype == expected
        assert built.compute().array.dtype == expected


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "detector",
    [
        abtem_st.segmented_detector,
        abtem_st.flexible_annular_detector,
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
        abtem_st.annular_detector,
    ],
)
def test_prism_scan_match_probe_scan(data, detector, lazy, device):
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )
    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    scan.match_probe(probe)

    prism_measurement = s_matrix.scan(
        scan=scan, detectors=detector, lazy=lazy
    ).compute()
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    ).compute()

    # assert prism_measurement.shape == probe_measurement.shape
    # assert prism_measurement.to_cpu() == probe_measurement.to_cpu()


# @given(atoms=abtem_st.atoms(min_side_length=5, max_side_length=10),
#        gpts=abtem_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(10, 15),
#        interpolation=st.integers(min_value=1, max_value=4),
#        energy=st.floats(100e3, 200e3),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu'])
# def test_prism_interpolation(data, atoms, gpts, planewave_cutoff, energy, lazy, device, interpolation):
#     detectors = data.draw(abtem_st.detectors(max_detectors=1))
#
#     potential = Potential(atoms, gpts=gpts, device=device).build(lazy=lazy)
#     # scan = GridScan(start=(0, 0), end=potential.extent)
#     #
#     # probe = Probe(semiangle_cutoff=planewave_cutoff, energy=energy, device=device)
#     # probe.grid.match(potential)
#
#     # tiled_potential = potential.tile((interpolation,) * 2)
#     s_matrix = SMatrix(potential=potential, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device, downsample=False)
#
#     # diffraction_pattern = probe.build(lazy=False).diffraction_patterns()
#     prism_diffraction_pattern = s_matrix.build(stop=0, lazy=False).reduce().diffraction_patterns()
#
#     # xp = get_array_module(device)
#     # assume(xp.abs(diffraction_pattern.array - prism_diffraction_pattern.array).max() < 1e-6)
#     # assume_valid_probe_and_detectors(probe, detectors)
#     #
#     # measurement = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
#     # prism_measurement = s_matrix.scan(scan=scan, detectors=detectors, lazy=lazy)
#     #
#     # assert np.allclose(measurement.array, prism_measurement.array, atol=1e-6)
#
# #
#

#
#
#
# @given(atoms=abtem_st.atoms(min_side_length=5, max_side_length=10),
#        gpts=abtem_st.gpts(min_value=32, max_value=64),
#        planewave_cutoff=st.floats(5, 15),
#        energy=st.floats(100e3, 200e3),
#        interpolation=st.integers(min_value=2, max_value=4),
#        distribute_scan=st.tuples(st.integers(min_value=1, max_value=3), st.integers(min_value=1, max_value=3)),
#        data=st.data())
# @pytest.mark.parametrize('lazy', [True])
# @pytest.mark.parametrize('device', ['cpu', gpu])
# @pytest.mark.parametrize('store_on_host', [False, True])
# def test_distribute_scan(data, atoms, gpts, planewave_cutoff, energy, lazy, distribute_scan, interpolation,
#                          device, store_on_host):
#     detectors = data.draw(abtem_st.detectors(allow_detect_every=lazy, max_detectors=1))
#
#     s_matrix = SMatrix(potential=atoms, gpts=gpts, interpolation=interpolation, planewave_cutoff=planewave_cutoff,
#                        energy=energy, device=device, store_on_host=True)
#
#     probe = s_matrix.build(stop=0, lazy=True).comparable_probe()
#
#     assert isinstance(s_matrix.build().compute().array, np.ndarray)
#
#     assume_valid_probe_and_detectors(probe, detectors)
#
#     scan = GridScan()
#     measurements = s_matrix.scan(scan=scan, detectors=detectors, distribute_scan=distribute_scan, lazy=lazy,
#                                  downsample=False)
#     measurements.compute()
#
#     assert_scanned_measurement_as_expected(measurements, atoms, probe, detectors, scan)


def test_prism_aberrated_ctf_matches_probe():
    # regression test for two bugs in the reduction coefficients: the
    # normalization used the complex square instead of the absolute square, and
    # the azimuthal angle convention must match polar_spatial_frequencies,
    # ie. arctan2(ky, kx). the real-space intensity of an aberrated probe is
    # sensitive to both; the vacuum diffraction intensity is sensitive to
    # neither.
    from abtem import CTF, Probe, SMatrix

    ctf = CTF(
        energy=100e3,
        semiangle_cutoff=20,
        defocus=50,
        astigmatism=40,
        astigmatism_angle=0.5236,
        coma=3e3,
        coma_angle=1.0,
    )

    probe = Probe._from_ctf(extent=20, gpts=128, ctf=ctf, energy=100e3)
    probe_intensity = probe.build(lazy=False).intensity().array

    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        downsample=False,
    )
    prism_intensity = s_matrix.reduce(ctf=ctf, lazy=False).intensity().array

    assert np.allclose(
        np.squeeze(prism_intensity),
        np.squeeze(probe_intensity),
        atol=1e-5 * probe_intensity.max(),
    )


class TestWrappedCrop2D:
    """``wrapped_slices`` assumed ``-n <= start < n`` and ``0 < stop <= 2n``.

    Outside that range it returned a wrongly shaped array instead of failing:
    for a window entirely in negative indices, ``slice(0, stop)`` is Python's
    negative indexing and selects ``n + stop`` rows rather than none. The crop
    then reached the caller intact and surfaced as a broadcast error frames
    away, in ``prism_transition_potential_scan``. Reachable from any model with
    atoms far enough outside the cell.
    """

    @staticmethod
    def _reference(array, corner, size):
        """The crop by modulo index -- the definition of a periodic crop."""
        i = (corner[0] + np.arange(size[0])) % array.shape[-2]
        j = (corner[1] + np.arange(size[1])) % array.shape[-1]
        return array[..., i[:, None], j[None, :]]

    @pytest.mark.parametrize("n, size", [(192, 64), (17, 5), (8, 8)])
    def test_every_corner_gives_the_right_shape_and_values(self, n, size):
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(0)
        array = rng.standard_normal((2, n, n))
        # Several periods either side: the old code was correct only in a
        # narrow band around zero, and wrong by n + size rows outside it.
        for corner in range(-3 * n, 3 * n):
            got = wrapped_crop_2d(array, (corner, 0), (size, size))
            assert got.shape == (2, size, size)
            assert np.array_equal(
                got, self._reference(array, (corner, 0), (size, size))
            )

    def test_both_axes_wrap_independently(self):
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(1)
        array = rng.standard_normal((3, 24, 20))
        for corner in [(-30, -25), (-5, 18), (23, -1), (47, 39), (-48, -40)]:
            got = wrapped_crop_2d(array, corner, (7, 9))
            assert got.shape == (3, 7, 9)
            assert np.array_equal(got, self._reference(array, corner, (7, 9)))

    @pytest.mark.parametrize("size", [16, 24, 32, 48])
    def test_windows_wider_than_one_period(self, size):
        """Two slices cannot express these; the general path must.

        The old code was correct here only for corners in a narrow band around
        zero -- e.g. n=16, corner=-23, size=24 came back as 8x8 -- so sweep the
        corner rather than picking one.
        """
        from abtem.prism.utils import wrapped_crop_2d

        n = 16
        rng = np.random.default_rng(2)
        array = rng.standard_normal((1, n, n))
        for corner in range(-2 * n, 2 * n, 3):
            got = wrapped_crop_2d(array, (corner, corner), (size, size))
            assert got.shape == (1, size, size)
            assert np.array_equal(
                got, self._reference(array, (corner, corner), (size, size))
            )

    def test_wrapped_slices_refuses_more_than_one_period(self):
        from abtem.prism.utils import wrapped_slices

        with pytest.raises(RuntimeError, match="exceeds the period"):
            wrapped_slices(0, 25, 24)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_matches_on_device(self, device):
        from abtem.core.backend import copy_to_device
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(3)
        array = rng.standard_normal((2, 32, 32))
        for corner in [(-40, 7), (-9, -9), (70, -70)]:
            got = wrapped_crop_2d(copy_to_device(array, device), corner, (11, 11))
            assert_array_matches_device(got, device)
            assert np.allclose(
                np.asarray(copy_to_device(got, "cpu")),
                self._reference(array, corner, (11, 11)),
            )


@pytest.mark.parametrize("ensemble_mean", [True, False])
def test_reduce_of_a_built_s_matrix_follows_the_ensemble_mean_of_the_frozen_phonons(
    ensemble_mean,
):
    # An eager SMatrixArray.reduce returned the frozen-phonon axis unreduced, while
    # a lazy one averaged it, as SMatrix.scan and multislice do, eager and lazy.
    import ase.build

    import abtem

    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=2, sigmas=0.1, seed=1, ensemble_mean=ensemble_mean
    )
    potential = abtem.Potential(frozen_phonons, sampling=0.15, slice_thickness=2)
    detector = abtem.AnnularDetector(30, 60)
    scan = GridScan((0, 0), (1, 1), gpts=(3, 5), fractional=True, potential=potential)
    s_matrix = abtem.SMatrix(potential=potential, energy=70e3, semiangle_cutoff=20)

    def axis_names(measurement):
        return [type(a).__name__ for a in measurement.axes_metadata]

    eager = s_matrix.build(lazy=False).reduce(scan, detectors=detector)
    lazy = s_matrix.build(lazy=True).reduce(scan, detectors=detector).compute()
    scanned = s_matrix.scan(scan, detector, lazy=False)

    expected = ["RealSpaceAxis", "RealSpaceAxis"]
    if not ensemble_mean:
        expected = ["FrozenPhononsAxis"] + expected
    for measurement in (eager, lazy, scanned):
        assert axis_names(measurement) == expected

    scale = np.abs(scanned.array).max()
    np.testing.assert_allclose(eager.array, scanned.array, rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(lazy.array, scanned.array, rtol=0, atol=1e-5 * scale)


def test_prism_reduce_averages_a_ctf_ensemble_eagerly_as_lazily():
    # With a frozen-phonon ensemble in the potential, a CTF ensemble flagged
    # ensemble_mean is averaged by the lazy branch of SMatrix.reduce, but the
    # eager branch returned its axis unreduced, so a 4-member defocus
    # distribution gave a (4, 3, 3) result instead of (3, 3).
    import ase

    import abtem
    from abtem import distributions
    from abtem.transfer import CTF

    atoms = ase.Atoms(
        "CO", positions=[(2.0, 2.0, 1.0), (5.5, 4.0, 2.5)], cell=(8.0, 8.0, 4.0)
    )
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=3, sigmas=0.05, seed=1, ensemble_mean=True
    )
    potential = abtem.Potential(
        frozen_phonons, gpts=48, slice_thickness=2.0, projection="finite"
    )
    s_matrix = abtem.SMatrix(potential=potential, energy=100e3, semiangle_cutoff=20)
    scan = abtem.GridScan((0, 0), (4, 4), gpts=(3, 3))
    detector = abtem.AnnularDetector(10, 60)
    defocus = distributions.from_values(
        np.array([-40.0, 0.0, 25.0, 60.0]), ensemble_mean=True
    )
    ctf = CTF(semiangle_cutoff=20, defocus=defocus)

    eager = s_matrix.reduce(scan, detectors=detector, ctf=ctf, lazy=False)
    lazy = s_matrix.reduce(scan, detectors=detector, ctf=ctf, lazy=True).compute()

    assert eager.shape == lazy.shape == (3, 3)
    scale = np.abs(lazy.array).max()
    np.testing.assert_allclose(eager.array, lazy.array, rtol=0, atol=1e-5 * scale)


def _cutoff_cells():
    # a square cell sampled differently along x and y, and a rectangular cell
    from ase.build import bulk

    return {
        "square-cell": (bulk("Si", cubic=True) * (1, 1, 2), (56, 64)),
        "rectangular-cell": (bulk("Si", cubic=True) * (1, 2, 1), (48, 100)),
    }


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("cell", ["square-cell", "rectangular-cell"])
@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_downsampled_prism_detects_to_the_antialias_cutoff_of_multislice(
    cell, lazy, device
):
    # downsample="cutoff" keeps everything the full grid's antialias aperture
    # passes, so at interpolation 1 the patterns, the radial bins and the
    # automatic outer angle are those of multislice on the full grid; 1.0.11 gave
    # (26, 22) patterns where multislice gives (38, 38) on the square cell
    import abtem

    atoms, gpts = _cutoff_cells()[cell]
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=2, device=device)
    scan = abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        gpts=(3, 5),
        fractional=True,
        endpoint=False,
        potential=potential,
    )
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        downsample="cutoff",
        device=device,
    )
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)

    def detectors():
        return [
            abtem.PixelatedDetector(max_angle="cutoff"),
            abtem.FlexibleAnnularDetector(),
            abtem.AnnularDetector(50),
        ]

    measured = s_matrix.scan(scan=scan, detectors=detectors(), lazy=lazy)
    if lazy:
        measured = measured.compute()
    expected = probe.scan(potential, scan=scan, detectors=detectors(), lazy=False)

    probe.grid.match(potential)
    assert s_matrix.dummy_probes().cutoff_angles == pytest.approx(probe.cutoff_angles)
    assert measured[0].max_angles == pytest.approx(probe.cutoff_angles)
    scale = max(float(np.abs(m.to_cpu().array).max()) for m in expected)
    for measured_i, expected_i in zip(measured, expected):
        # PRISM results carry the band limit of the built S-matrix in their
        # metadata, which multislice results have no reason to; the rest agrees
        assert measured_i.shape == expected_i.shape
        assert measured_i.axes_metadata == expected_i.axes_metadata
        np.testing.assert_allclose(
            measured_i.to_cpu().array,
            expected_i.to_cpu().array,
            rtol=1e-5,
            atol=1e-5 * scale,
        )


@pytest.mark.filterwarnings("ignore:The interpolation factor does not exactly divide")
@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_prism_cutoff_angles_are_equal_along_x_and_y_on_an_odd_grid(device):
    # a cubic cell on a (57, 63) grid, interpolated by 2: the extent of the
    # waves in units of the full sampling is 28.5 along x and 31.5 along y, which
    # floating point rounds to either neighbour, so the angles within the antialias
    # cutoff must not follow that rounding
    import ase.build

    import abtem

    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
    potential = abtem.Potential(atoms, gpts=(57, 63), slice_thickness=2, device=device)
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        device=device,
    )

    cutoff_angles = s_matrix.dummy_probes().cutoff_angles
    built_cutoff_angles = s_matrix.build(lazy=True).dummy_probes().cutoff_angles

    assert cutoff_angles[0] == pytest.approx(cutoff_angles[1])
    assert built_cutoff_angles == pytest.approx(cutoff_angles)


@pytest.mark.parametrize("device", [gpu, "cpu"])
def test_dummy_probes_keep_the_built_band_limit_with_supplied_metadata(device):
    import ase.build

    import abtem

    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
    potential = abtem.Potential(atoms, gpts=(56, 64), slice_thickness=2, device=device)
    s_matrix = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, device=device
    )
    assert s_matrix.downsampled_gpts != s_matrix.gpts

    probes = s_matrix.dummy_probes(metadata={"label": "supplied"})

    assert probes.metadata["label"] == "supplied"
    assert probes.metadata["antialias_sampling"] == s_matrix.sampling
    assert probes.cutoff_angles == pytest.approx(
        s_matrix.build(lazy=True).dummy_probes().cutoff_angles
    )


def _batch_s_matrix_array(gpts, interpolation, lazy=False):
    import ase.build

    import abtem

    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=2)
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=interpolation,
        downsample=False,
    )
    scan = abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        gpts=(12, 20),
        fractional=True,
        endpoint=False,
        potential=potential,
    )
    return s_matrix.build(lazy=lazy), scan


def test_auto_reduction_batch_follows_the_chunk_size_of_the_device(monkeypatch):
    """CPU only: the device label is set by hand, the batch size is only computed."""
    import abtem

    # the window is the 48 x 48 grid (interpolation 1), 18432 bytes per position
    s_matrix_array, scan = _batch_s_matrix_array(gpts=48, interpolation=1)

    with abtem.config.set({"dask.chunk-size": "1 MB", "dask.chunk-size-gpu": "180 kB"}):
        cpu = s_matrix_array._validate_max_batch_reduction(scan, "auto")
        monkeypatch.setattr(s_matrix_array, "_device", "gpu")
        gpu = s_matrix_array._validate_max_batch_reduction(scan, "auto")

    assert cpu == 1_000_000 // 18432
    assert gpu == 180_000 // 18432


@pytest.mark.parametrize("lazy", [False, True])
def test_auto_reduction_batch_holds_the_crop_box(monkeypatch, lazy):
    """CPU only: the automatic batch follows dask.chunk-size, not chunk-size-gpu."""
    import abtem
    from abtem.prism.s_matrix import SMatrixArray
    from abtem.prism.utils import minimum_crop

    # window 32 x 32 on the 64 x 64 grid; a batch of positions is contracted over
    # a box that spans the batch and one window
    s_matrix_array, scan = _batch_s_matrix_array(gpts=64, interpolation=2, lazy=lazy)
    detector = abtem.AnnularDetector(30, 90)
    expected = s_matrix_array.reduce(
        scan=scan, detectors=detector, max_batch_reduction=len(scan)
    ).compute()

    # the contraction forms an array of one box per position, see _reduce_to_waves
    contracted = []
    reduce_to_waves = SMatrixArray._reduce_to_waves

    def spy(self, array, positions, coefficients):
        pixel_positions = positions / np.array(self.waves.sampling) - np.asarray(
            self.window_offset
        )
        _, box, _ = minimum_crop(pixel_positions, self.window_gpts)
        contracted.append(positions.size // 2 * int(np.prod(box)) * 8)
        return reduce_to_waves(self, array, positions, coefficients)

    monkeypatch.setattr(SMatrixArray, "_reduce_to_waves", spy)
    with abtem.config.set({"dask.chunk-size": "400 kB"}):
        measured = s_matrix_array.reduce(scan=scan, detectors=detector).compute()

    assert max(contracted) <= 400_000
    assert len(contracted) > 1
    np.testing.assert_allclose(
        measured.array,
        expected.array,
        rtol=1e-5,
        atol=1e-5 * float(np.abs(expected.array).max()),
    )


def test_auto_reduction_batch_holds_the_crop_box_of_the_device(monkeypatch):
    """CPU only: the device label is set by hand, the batch size is only computed."""
    import abtem

    s_matrix_array, scan = _batch_s_matrix_array(gpts=64, interpolation=2)

    with abtem.config.set({"dask.chunk-size": "400 kB"}):
        cpu = s_matrix_array._validate_max_batch_reduction(scan, "auto")
    # the 32 x 32 windows alone would allow 400000 // 8192 positions
    assert cpu < 400_000 // 8192

    monkeypatch.setattr(s_matrix_array, "_device", "gpu")
    with abtem.config.set(
        {"dask.chunk-size": "100 MB", "dask.chunk-size-gpu": "400 kB"}
    ):
        gpu = s_matrix_array._validate_max_batch_reduction(scan, "auto")

    assert gpu == cpu


@pytest.fixture
def multislice_blocks(monkeypatch):
    """The (start, stop) plane-wave ranges of the multislice runs of an S-matrix
    build, appended in the order they run."""
    from abtem.prism.s_matrix import SMatrix

    blocks = []
    build_s_matrix = SMatrix._build_s_matrix

    def recording(s_matrix, wave_vector_range=slice(None), pbar=False):
        if isinstance(wave_vector_range, slice):
            blocks.append((wave_vector_range.start, wave_vector_range.stop))
        else:
            blocks.append(tuple(wave_vector_range.item()))
        return build_s_matrix(s_matrix, wave_vector_range, pbar)

    monkeypatch.setattr(SMatrix, "_build_s_matrix", staticmethod(recording))
    return blocks


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "lazy, disable_s_matrix_chunks",
    [(False, "auto"), (True, "auto"), (True, True)],
    ids=["eager", "lazy", "lazy, one task per S-matrix"],
)
def test_reduce_multislices_max_batch_multislice_plane_waves_at_a_time(
    device, lazy, disable_s_matrix_chunks, multislice_blocks
):
    # 9 plane waves, which a batch of 4 does not divide. The oracle is the
    # multislice of the same probe, to which PRISM at interpolation 1 is equal.
    import ase.build

    import abtem

    potential = abtem.Potential(
        ase.build.bulk("Si", cubic=True),
        gpts=(16, 20),
        slice_thickness=2,
        device=device,
    )
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=10,
        interpolation=1,
        device=device,
    )
    assert len(s_matrix) == 9
    scan = abtem.GridScan(gpts=(2, 3))
    detector = abtem.AnnularDetector(0, 20)

    def reduce(**kwargs):
        multislice_blocks.clear()
        measurement = s_matrix.reduce(
            scan=scan,
            detectors=detector,
            lazy=lazy,
            disable_s_matrix_chunks=disable_s_matrix_chunks,
            **kwargs,
        )
        if lazy:
            measurement = measurement.compute(progress_bar=False)
        return np.asarray(measurement.to_cpu().array), sorted(multislice_blocks)

    default, default_blocks = reduce()
    batched, batched_blocks = reduce(max_batch_multislice=4)

    assert default_blocks == [(0, 9)]
    assert batched_blocks == [(0, 4), (4, 8), (8, 9)]
    np.testing.assert_allclose(batched, default, rtol=1e-5)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=10, device=device)
    expected = probe.scan(potential, scan, detector, lazy=False)
    np.testing.assert_allclose(batched, np.asarray(expected.to_cpu().array), atol=1e-5)


@pytest.mark.parametrize(
    "lazy, disable_s_matrix_chunks",
    [(False, "auto"), (True, True)],
    ids=["eager", "lazy, one task per S-matrix"],
)
def test_the_plane_wave_batches_of_an_unseeded_crystal_see_one_potential(
    lazy, disable_s_matrix_chunks
):
    """Every batch of max_batch_multislice plane waves is propagated through the
    same frozen-phonon crystal, so the result is that of one batch."""
    import ase.build

    import abtem
    from abtem.potentials.iam import CrystalPotential

    atoms = ase.build.bulk("Si", cubic=True)
    unit = abtem.Potential(
        abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=None),
        gpts=(16, 20),
        slice_thickness=atoms.cell[2, 2] / 2,
    )
    crystal = CrystalPotential(unit, (1, 2, 2))
    s_matrix = abtem.SMatrix(
        potential=crystal, energy=100e3, semiangle_cutoff=10, interpolation=1
    )

    first_slices = []
    generate_slices = CrystalPotential.generate_slices

    def recording(self, *args, **kwargs):
        for i, slic in enumerate(generate_slices(self, *args, **kwargs)):
            if i == 0:
                first_slices.append(np.asarray(slic.array).copy())
            yield slic

    def reduce(**kwargs):
        first_slices.clear()
        measurement = s_matrix.reduce(
            scan=abtem.GridScan(gpts=(2, 3)),
            detectors=abtem.AnnularDetector(0, 20),
            lazy=lazy,
            disable_s_matrix_chunks=disable_s_matrix_chunks,
            **kwargs,
        )
        if lazy:
            measurement = measurement.compute(scheduler="synchronous")
        return np.asarray(measurement.array)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(CrystalPotential, "generate_slices", recording)
        default = reduce()
        batched = reduce(max_batch_multislice=4)

    assert len(first_slices) == -(-len(s_matrix) // 4) > 1
    for first_slice in first_slices[1:]:
        np.testing.assert_array_equal(first_slice, first_slices[0])
    np.testing.assert_allclose(
        batched, default, rtol=0, atol=1e-5 * np.abs(default).max()
    )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_a_small_max_batch_multislice_keeps_the_seeded_crystal(device):
    """An eager reduction in plane-wave batches of 4 of a seeded frozen-phonon
    crystal with 3 members gives the result of the default batch."""
    import ase.build

    import abtem
    from abtem.potentials.iam import CrystalPotential

    atoms = ase.build.bulk("Si", cubic=True)
    unit = abtem.Potential(
        abtem.FrozenPhonons(atoms, 2, sigmas=0.1, seed=1),
        gpts=(16, 20),
        slice_thickness=atoms.cell[2, 2] / 2,
        device=device,
    )
    crystal = CrystalPotential(unit, (1, 2, 2), seeds=(1, 2, 3))
    s_matrix = abtem.SMatrix(
        potential=crystal,
        energy=100e3,
        semiangle_cutoff=10,
        interpolation=1,
        device=device,
    )
    assert len(s_matrix) == 21

    def reduce(**kwargs):
        measurement = s_matrix.reduce(
            scan=abtem.GridScan(gpts=(2, 3)),
            detectors=abtem.AnnularDetector(0, 20),
            lazy=False,
            **kwargs,
        )
        return np.asarray(measurement.to_cpu().array)

    default = reduce()
    batched = reduce(max_batch_multislice=4)
    np.testing.assert_allclose(
        batched, default, rtol=0, atol=1e-5 * np.abs(default).max()
    )


@pytest.mark.parametrize("exit_planes", [2, 3], ids=["5 planes", "4 planes"])
@pytest.mark.parametrize(
    "lazy, disable_s_matrix_chunks",
    [(False, "auto"), (True, True), (True, False)],
    ids=["eager", "lazy, one task per S-matrix", "lazy"],
)
def test_prism_with_a_ctf_ensemble_and_exit_planes_and_frozen_phonons_equals_multislice(
    exit_planes, lazy, disable_s_matrix_chunks
):
    # the per-member eager reduction averaged the CTF ensemble before the caller
    # stored it in an allocation that keeps that axis, so the exit-plane axis of a
    # member landed on the CTF axis: a broadcast ValueError for 5 planes against 4
    # defocus values and wrong values for 4 against 4
    import abtem
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (1, 1, 2)
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=(1, 2))
    potential = abtem.Potential(
        frozen_phonons, sampling=0.1, slice_thickness=1.8, exit_planes=exit_planes
    )
    defocus = abtem.distributions.gaussian(
        standard_deviation=30, num_samples=4, sampling_limit=2
    )
    scan = abtem.GridScan((0, 0), (2, 2), gpts=(2, 2))
    detector = abtem.AnnularDetector(30, 120)

    reference = abtem.Probe(semiangle_cutoff=20, defocus=defocus, energy=100e3).scan(
        potential, scan=scan, detectors=detector, lazy=False
    )
    s_matrix = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1
    )
    kwargs = {"disable_s_matrix_chunks": disable_s_matrix_chunks} if lazy else {}
    measurement = s_matrix.scan(
        scan=scan,
        ctf=abtem.CTF(semiangle_cutoff=20, defocus=defocus),
        detectors=detector,
        lazy=lazy,
        **kwargs,
    ).compute()

    assert measurement.shape == reference.shape
    np.testing.assert_allclose(
        measurement.array, reference.array, rtol=0, atol=1e-5 * reference.array.max()
    )
