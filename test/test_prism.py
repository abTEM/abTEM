import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from utils import (assert_array_matches_device, assert_array_objects_equal,
                   devices, requires_gpu, to_host_array)

from abtem import GridScan, WavesDetector
from abtem.core.backend import cp


def _aberrated_ctf():
    """An aberrated CTF used to regression-test two bugs in the reduction
    coefficients: the normalization used the complex square instead of the
    absolute square, and the azimuthal angle convention must match
    polar_spatial_frequencies, ie. arctan2(ky, kx). The real-space intensity
    of an aberrated probe is sensitive to both; the vacuum diffraction
    intensity is sensitive to neither.
    """
    from abtem import CTF

    return CTF(
        energy=100e3,
        semiangle_cutoff=20,
        defocus=50,
        astigmatism=40,
        astigmatism_angle=0.5236,
        coma=3e3,
        coma_angle=1.0,
    )


@given(data=st.data())
@devices
def test_prism_matches_probe(data, device):
    s_matrix = data.draw(abtem_st.s_matrix(device=device, max_interpolation=3))

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    for lazy in (False, True):
        s_matrix_diffraction_patterns = (
            s_matrix.reduce(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
        )
        probe_diffraction_patterns = (
            probe.build(lazy=lazy).diffraction_patterns(max_angle=None).to_cpu()
        )

        assert np.allclose(
            s_matrix_diffraction_patterns.compute().array,
            probe_diffraction_patterns.compute().array,
        ), f"lazy={lazy}"


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@devices
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

    assert np.allclose(
        s_matrix_diffraction_patterns.array, probe_diffraction_patterns.array
    )


# PRISM and multislice are different code paths, so their results agree only
# to within float round-off. These are the tolerances of np.allclose, which is
# what `==` applied to the eager array before the comparison was made
# value-based for lazy results too.
PRISM_VS_PROBE_TOLERANCES = dict(rtol=1e-5, atol=1e-8)

PRISM_SCAN_DETECTORS = [
    abtem_st.segmented_detector,
    abtem_st.flexible_annular_detector,
    abtem_st.pixelated_detector,
    abtem_st.waves_detector,
    abtem_st.annular_detector,
]


@pytest.mark.slow
@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize(
    "downsample", [True, False], ids=["downsample", "no_downsample"]
)
@devices
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

    measurement = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=lazy)

    assert (
        measurement.shape
        == potential.ensemble_shape
        + ctf.ensemble_shape
        + scan.ensemble_shape
        + measurement_shape
    )
    assert measurement.dtype == detector._out_dtype(probe)[0]

    # A lazy result only promises its shape and dtype; computing checks them.
    if lazy:
        computed = measurement.compute()
        assert computed.shape == measurement.shape
        assert computed.dtype == measurement.dtype


@given(data=st.data())
@pytest.mark.parametrize("lazy", [False, True])
@requires_gpu
def test_s_matrix_store_on_host(data, lazy):
    potential = data.draw(abtem_st.potential(device="gpu", ensemble_mean=False))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, device="gpu", store_on_host=True)
    )
    s_matrix = s_matrix.build(lazy=lazy).compute()
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


def _single_precision_plane_waves(wave_vectors, extent, gpts, interpolation, xp):
    """The plane-wave expansion as it was built before it followed the precision:
    float32 wave vectors and coordinates throughout."""
    from abtem.core.complex import complex_exponential

    wave_vectors = xp.asarray(wave_vectors, dtype=xp.float32)
    x = xp.linspace(0, extent[0], gpts[0], endpoint=False, dtype=np.float32)
    y = xp.linspace(0, extent[1], gpts[1], endpoint=False, dtype=np.float32)
    array = complex_exponential(
        1.0 * 2 * np.pi * wave_vectors[:, 0, None, None] * x[:, None]
    ) * complex_exponential(
        1.0 * 2 * np.pi * wave_vectors[:, 1, None, None] * y[None, :]
    )
    array *= np.prod(interpolation) / np.prod(array.shape[-2:])
    return array


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("interpolation", [1, 2])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_plane_wave_expansion_follows_the_precision(
    precision, interpolation, lazy, device
):
    # Without a potential the S-matrix is its plane-wave expansion,
    # exp(2 pi i k.r) normalised by f^2 / N. In double precision it is the
    # analytic value to round-off; in single precision it is bitwise the float32
    # expansion the S-matrix has always had.
    import abtem
    from abtem.core.backend import get_array_module

    extent, gpts = (8.2, 9.4), (64, 72)
    with abtem.config.set({"precision": precision}):
        s_matrix = abtem.SMatrix(
            energy=100e3, semiangle_cutoff=20, extent=extent, gpts=gpts,
            interpolation=interpolation, downsample=False, device=device,
        )
        complex_dtype = np.complex128 if precision == "float64" else np.complex64
        built = s_matrix.build(lazy=lazy)
        assert built.array.dtype == complex_dtype
        array = to_host_array(built)
        assert array.dtype == complex_dtype

        wave_vectors = to_host_array(s_matrix.wave_vectors)
        if precision == "float32":
            xp = get_array_module(device)
            expected = _single_precision_plane_waves(
                wave_vectors, extent, gpts, (interpolation,) * 2, xp
            )
            np.testing.assert_array_equal(array, to_host_array(expected))
            return

    assert wave_vectors.dtype == np.float64
    x = np.linspace(0, extent[0], gpts[0], endpoint=False)
    y = np.linspace(0, extent[1], gpts[1], endpoint=False)
    phase = (
        wave_vectors[:, 0, None, None] * x[:, None]
        + wave_vectors[:, 1, None, None] * y[None, :]
    )
    expected = np.exp(2j * np.pi * phase) * interpolation**2 / np.prod(gpts)
    np.testing.assert_allclose(
        array, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_prism_scan_match_probe_scan(data, lazy, device):
    # The scan is either matched to the probe explicitly (GridScan.match_probe)
    # or left for s_matrix.scan()/probe.scan() to match on their own: both
    # workflows must agree between the two reduction paths. The detector is
    # drawn, so each is compared across the run.
    potential = data.draw(abtem_st.potential(device=device, ensemble_mean=False))
    s_matrix = data.draw(
        abtem_st.s_matrix(potential=potential, max_interpolation=1, device=device)
    )
    detector = data.draw(st.sampled_from(PRISM_SCAN_DETECTORS), label="detector")
    detector = data.draw(detector())

    s_matrix = s_matrix.round_gpts_to_interpolation()
    probe = s_matrix.dummy_probes()

    scan = GridScan()
    if data.draw(st.booleans(), label="match_probe"):
        scan.match_probe(probe)
    assume(
        (max(detector.angular_limits(probe)) < min(probe.cutoff_angles))
        or isinstance(detector, WavesDetector)
    )

    prism_measurement = s_matrix.scan(
        scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()
    probe_measurement = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=lazy
    ).to_cpu()

    assert prism_measurement.shape == probe_measurement.shape
    assert_array_objects_equal(
        prism_measurement, probe_measurement, **PRISM_VS_PROBE_TOLERANCES
    )


def test_prism_aberrated_ctf_matches_probe():
    # regression test for two bugs in the reduction coefficients: see
    # _aberrated_ctf's docstring.
    from abtem import Probe, SMatrix

    ctf = _aberrated_ctf()

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
            assert np.array_equal(got, self._reference(array, (corner, 0), (size, size)))

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

    @devices
    def test_matches_on_device(self, device):
        from abtem.core.backend import copy_to_device
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(3)
        array = rng.standard_normal((2, 32, 32))
        for corner in [(-40, 7), (-9, -9), (70, -70)]:
            got = wrapped_crop_2d(copy_to_device(array, device), corner, (11, 11))
            assert_array_matches_device(got, device)
            assert np.allclose(
                to_host_array(copy_to_device(got, "cpu")),
                self._reference(array, corner, (11, 11)),
            )


class TestWindowCrop:
    """``window_crop`` is ``minimum_crop``, except that a window as large as the
    grid is the whole grid for every position."""

    GPTS = (64, 48)

    @staticmethod
    def _positions(device):
        from abtem.core.backend import copy_to_device

        # Pixel positions inside the grid, and far outside it.
        positions = np.array([[3.2, 40.7], [31.5, 10.1], [-20.0, 90.0], [70.3, 5.5]])
        return copy_to_device(positions, device)

    @devices
    def test_a_window_the_size_of_the_grid_is_the_whole_grid(self, device):
        from abtem.prism.utils import window_crop

        positions = self._positions(device)
        crop_corner, size, corners = window_crop(positions, self.GPTS, self.GPTS)

        assert crop_corner == (0, 0)
        assert size == self.GPTS
        assert_array_matches_device(corners, device)
        assert corners.shape == positions.shape
        assert not np.any(to_host_array(corners))

    @devices
    def test_a_window_the_size_of_the_grid_in_one_direction_is_cropped(self, device):
        """A window equal to the grid along one axis only is centred on each
        position, as ``interpolation=(1, 2)`` gives."""
        from abtem.prism.utils import minimum_crop, window_crop

        positions = self._positions(device)
        window = (self.GPTS[0], self.GPTS[1] // 2)
        got = window_crop(positions, window, self.GPTS)
        expected = minimum_crop(positions, window)

        assert got[0] == expected[0]
        assert got[1] == expected[1]
        assert np.array_equal(to_host_array(got[2]), to_host_array(expected[2]))
        assert np.any(to_host_array(got[2]))

    @devices
    def test_a_smaller_window_is_cropped_around_the_positions(self, device):
        from abtem.prism.utils import minimum_crop, window_crop

        positions = self._positions(device)
        window = (16, 12)
        got = window_crop(positions, window, self.GPTS)
        expected = minimum_crop(positions, window)

        assert got[0] == expected[0]
        assert got[1] == expected[1]
        assert np.array_equal(to_host_array(got[2]), to_host_array(expected[2]))

    def test_a_single_position(self):
        from abtem.prism.utils import window_crop

        positions = np.array([[10.0, 20.0]])
        assert window_crop(positions, self.GPTS, self.GPTS)[2].shape == (1, 2)
        crop_corner, size, corners = window_crop(positions, (16, 12), self.GPTS)
        assert size == (16, 12)
        assert not np.any(corners)


class TestBatchCrop2D:
    @devices
    def test_a_window_the_size_of_the_array_is_not_copied(self, device):
        from abtem.core.backend import copy_to_device
        from abtem.prism.utils import batch_crop_2d

        array = copy_to_device(np.arange(3 * 4 * 6 * 5.0).reshape(3, 4, 6, 5), device)
        corners = copy_to_device(np.zeros((3, 2), dtype=int), device)
        assert batch_crop_2d(array, corners, (6, 5)) is array

    @devices
    def test_smaller_windows_are_cropped_at_their_corners(self, device):
        from abtem.core.backend import copy_to_device
        from abtem.prism.utils import batch_crop_2d

        array = np.arange(3 * 6 * 5.0).reshape(3, 6, 5)
        corners = np.array([[0, 0], [2, 1], [3, 3]])
        got = batch_crop_2d(
            copy_to_device(array, device), copy_to_device(corners, device), (3, 2)
        )
        for i, (cx, cy) in enumerate(corners):
            assert np.array_equal(
                to_host_array(got)[i], array[i, cx : cx + 3, cy : cy + 2]
            )
