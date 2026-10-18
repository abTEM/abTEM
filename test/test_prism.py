import itertools

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


@given(data=st.data())
@devices
@pytest.mark.parametrize("detector", PRISM_SCAN_DETECTORS)
def test_prism_scan(data, detector, device):
    # Laziness, downsampling, interpolation (1 to 3) and a frozen-phonon
    # ensemble are drawn rather than crossed: hypothesis explores their
    # combinations across runs. Two frozen-phonon configurations make the
    # ensemble axis one that a dropped or duplicated axis would change.
    lazy = data.draw(st.booleans(), label="lazy")
    frozen_phonons = data.draw(st.booleans(), label="frozen_phonons")
    potential = data.draw(
        abtem_st.potential(
            device=device,
            no_frozen_phonons=not frozen_phonons,
            min_frozen_phonons=2,
            max_frozen_phonons=2,
            ensemble_mean=False,
        )
    )
    ctf = data.draw(abtem_st.ctf(partial_coherence=False))

    s_matrix = data.draw(
        abtem_st.s_matrix(
            potential=potential,
            max_interpolation=3,
            device=device,
            downsample=data.draw(st.booleans(), label="downsample"),
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


def _hexagonal_carbon(reps, a=3.0, cz=2.0, angle=60.0):
    import ase

    cell = np.array(
        [
            [a, 0, 0],
            [a * np.cos(np.deg2rad(angle)), a * np.sin(np.deg2rad(angle)), 0],
            [0, 0, cz],
        ]
    )
    return ase.Atoms("C", cell=cell, pbc=True, positions=[(0, 0, 0)]) * reps


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_prism_non_orthogonal_matches_probe(lazy):
    """On a non-orthogonal (hexagonal) cell, PRISM with interpolation=1 reconstructs the
    probe exit wave (the S-matrix reduction at a position equals a probe multislice at
    that position). This exercises the reciprocal mapping of the plane-wave basis, the
    metric Fresnel propagation of every plane wave, and the reduction phases -- a wrong
    (orthogonal) choice in any of these breaks the equivalence."""
    import abtem

    # 3 x 3 x 3 cells on 96 x 96 (was 3 x 3 x 6 on 160 x 160): measured rel 7.4e-7
    # against the 1e-5 bound, as before (8.2e-7)
    pot = abtem.Potential(
        _hexagonal_carbon((3, 3, 3)), gpts=(96, 96), slice_thickness=1.0
    )
    assert not pot.grid.is_orthogonal

    px, py = 4.1, 3.3

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20.0)
    probe.grid.match(pot)
    ref = (
        probe.multislice(pot, scan=abtem.CustomScan([(px, py)]))
        .compute()
        .array.reshape((-1, 96, 96))[0]
    )

    s_matrix = abtem.SMatrix(
        potential=pot,
        energy=100e3,
        semiangle_cutoff=20.0,
        interpolation=1,
        downsample=False,
    )
    assert not s_matrix.grid.is_orthogonal

    w = (
        s_matrix.build(lazy=lazy)
        .reduce(scan=abtem.CustomScan([(px, py)]), detectors=WavesDetector())
        .compute()
        .array.reshape((-1, 96, 96))[0]
    )

    # remove the global phase, then compare the exit waves directly
    overlap = np.vdot(np.asarray(w).ravel(), np.asarray(ref).ravel())
    w = np.asarray(w) * np.conj(overlap / np.abs(overlap))
    rel = np.max(np.abs(w - np.asarray(ref))) / np.max(np.abs(np.asarray(ref)))
    assert rel < 1e-5


def test_prism_non_orthogonal_matches_multislice_haadf():
    """A skew PRISM HAADF scan matches the multislice HAADF scan (the ground truth)."""
    import abtem

    # 3 x 3 x 4 cells on 128 x 128 (was 4 x 4 x 8 on 208 x 208), still sampled
    # finely enough to resolve the 100 mrad outer angle: measured rel 1.7e-6 against
    # the 1e-4 bound
    pot = abtem.Potential(
        _hexagonal_carbon((3, 3, 4)), gpts=(128, 128), slice_thickness=1.0
    )
    assert not pot.grid.is_orthogonal

    scan = abtem.CustomScan([(2.0, 1.5), (4.1, 3.3), (6.0, 5.5), (3.0, 7.0)])
    detector = abtem.AnnularDetector(inner=30, outer=100)

    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20.0)
    probe.grid.match(pot)
    ground_truth = np.asarray(
        probe.scan(pot, scan=scan, detectors=detector).compute().array
    )

    prism = np.asarray(
        abtem.SMatrix(
            potential=pot, energy=100e3, semiangle_cutoff=20.0, interpolation=1
        )
        .scan(scan=scan, detectors=detector)
        .compute()
        .array
    )

    rel = np.max(np.abs(prism - ground_truth)) / np.max(np.abs(ground_truth))
    assert rel < 1e-4


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
    """``wrapped_crop_2d`` is the crop of a periodic array: for any corner, also
    several periods outside the array, and any size, also wider than one period.
    A wrongly shaped crop used to surface as a broadcast error frames away, in
    ``prism_transition_potential_scan``.
    """

    @staticmethod
    def _reference(array, corner, size):
        """The crop by explicit periodic tiling, independent of how the
        implementation splits the window."""
        reps = (
            -(-(abs(corner[0]) + size[0]) // array.shape[-2]) * 2 + 1,
            -(-(abs(corner[1]) + size[1]) // array.shape[-1]) * 2 + 1,
        )
        tiled = np.tile(array, (1,) * (array.ndim - 2) + reps)
        origin = (
            (reps[0] // 2) * array.shape[-2] + corner[0],
            (reps[1] // 2) * array.shape[-1] + corner[1],
        )
        return tiled[
            ..., origin[0] : origin[0] + size[0], origin[1] : origin[1] + size[1]
        ]

    @pytest.mark.parametrize(
        "corner, size",
        [((3, 5), (7, 9)), ((-50, 61), (7, 9)), ((-23, 30), (40, 33))],
        ids=["in-range", "far-outside", "wider-than-period"],
    )
    def test_matches_the_periodic_tiling(self, corner, size):
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(1)
        array = rng.standard_normal((3, 24, 20))
        got = wrapped_crop_2d(array, corner, size)
        assert got.shape == (3, *size)
        assert np.array_equal(got, self._reference(array, corner, size))

    @staticmethod
    def _by_modulo(array, corner, size):
        """The crop by modulo index: the definition of a periodic crop, and a
        second oracle independent of both the implementation and the tiling."""
        i = (corner[0] + np.arange(size[0])) % array.shape[-2]
        j = (corner[1] + np.arange(size[1])) % array.shape[-1]
        return array[..., i[:, None], j[None, :]]

    @staticmethod
    def _edge_values(n):
        """Corners (and sizes) at which an off-by-one in the split into runs
        shows: zero, the last row, the first row past the period, one period
        either side, and several periods out."""
        return sorted({0, n - 1, n, n + 1, -1, -n, 2 * n + 1, -3 * n + 2})

    @staticmethod
    def _sizes(n):
        return sorted({1, n - 1, n, n + 1, 2 * n + 2, 3 * n + 1})

    def test_sweep_of_corners_and_sizes(self):
        """Every combination of edge corners and sizes on both axes, so windows
        that wrap on one axis only, on both, and on neither (including exactly
        at the boundary) are all covered; then every corner from -3n to 3n on
        each axis alone, which finds an error confined to a band of corners that
        the edge corners step over. An odd and an even period, and leading axes.
        """
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(0)
        shape = (17, 12)
        array = rng.standard_normal((2, 3, *shape))

        def check(corner, size):
            got = wrapped_crop_2d(array, corner, size)
            assert got.shape == array.shape[:-2] + size
            assert np.array_equal(got, self._by_modulo(array, corner, size)), (
                corner,
                size,
            )

        rows = [
            (c, s) for c in self._edge_values(shape[0]) for s in self._sizes(shape[0])
        ]
        columns = [
            (c, s) for c in self._edge_values(shape[1]) for s in self._sizes(shape[1])
        ]
        for (corner_0, size_0), (corner_1, size_1) in itertools.product(rows, columns):
            check((corner_0, corner_1), (size_0, size_1))

        # The other axis: not wrapping, wrapping, and one pixel wide.
        for axis, n in enumerate(shape):
            other = shape[1 - axis]
            others = [(0, other), (-1, 4), (other, 1)]
            for size in self._sizes(n):
                for corner in range(-3 * n, 3 * n + 1):
                    for other_corner, other_size in others:
                        if axis == 0:
                            check((corner, other_corner), (size, other_size))
                        else:
                            check((other_corner, corner), (other_size, size))

    @pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
    def test_the_crop_keeps_the_dtype(self, dtype):
        """Both the view and the copy: the copy is allocated here, so a hardcoded
        dtype would show."""
        from abtem.prism.utils import wrapped_crop_2d

        array = np.arange(2 * 8 * 9).reshape(2, 8, 9).astype(dtype)
        for corner in [(1, 1), (-1, -1)]:  # a view, and a copy
            got = wrapped_crop_2d(array, corner, (5, 5))
            assert got.dtype == array.dtype
            assert np.array_equal(got, self._by_modulo(array, corner, (5, 5)))

    def test_a_window_that_does_not_wrap_is_a_view(self):
        """No copy: a window as large as the array (PRISM interpolation 1) must
        not double the memory of the S-matrix. A window that wraps is a copy."""
        from abtem.prism.utils import wrapped_crop_2d

        array = np.zeros((3, 24, 20))
        whole = wrapped_crop_2d(array, (0, 0), (24, 20))
        assert whole.shape == array.shape and np.shares_memory(whole, array)
        assert np.shares_memory(wrapped_crop_2d(array, (24, -20), (7, 9)), array)
        # Exactly at the boundary: ends on the last row, and starts on the next period.
        assert np.shares_memory(wrapped_crop_2d(array, (17, 11), (7, 9)), array)
        assert np.shares_memory(wrapped_crop_2d(array, (-24, 20), (24, 20)), array)
        # One row or column past the boundary wraps, and wraps on one axis only.
        assert not np.shares_memory(wrapped_crop_2d(array, (18, 0), (7, 9)), array)
        assert not np.shares_memory(wrapped_crop_2d(array, (0, 12), (7, 9)), array)
        assert not np.shares_memory(wrapped_crop_2d(array, (-1, -1), (7, 9)), array)
        assert not np.shares_memory(wrapped_crop_2d(array, (0, 0), (25, 20)), array)

    @staticmethod
    def _is_view(crop_of, array):
        """Whether ``crop_of(array)`` is a view of ``array``: whether writing
        into the array shows in the crop. Backend-neutral, as the torch backend
        has no ``shares_memory``. Writes into ``array``: pass a throwaway."""
        from abtem.core.backend import copy_to_device

        got = crop_of(array)
        before = to_host_array(copy_to_device(got, "cpu")).copy()
        array += 1
        after = to_host_array(copy_to_device(got, "cpu")).copy()
        return not np.array_equal(before, after)

    @devices
    def test_matches_on_device(self, device):
        from abtem.core.backend import copy_to_device
        from abtem.prism.utils import wrapped_crop_2d

        rng = np.random.default_rng(3)
        array = rng.standard_normal((2, 32, 32))
        on_device = copy_to_device(array, device)
        size = (11, 11)
        # Hand-listed corners (window 11 on a period of 32) and whether the window
        # wraps: it does when its far edge passes the end of a period.
        corners = [
            # wrapping on both axes, far outside the array
            ((-40, 7), True), ((-9, -9), True),
            # wrapping on the columns only, rows several periods out
            ((70, -70), True),
            # not wrapping: inside, ending on the last pixel, and whole periods out
            ((0, 0), False), ((21, 21), False), ((32, -32), False),
            ((-32, 64), False),
            # one period plus one pixel: starts on pixel 1 again, no wrap
            ((33, 1), False), ((1, 33), False),
            # wrapping on one axis only, or by one pixel
            ((22, 5), True), ((5, 22), True), ((-1, 0), True), ((0, -1), True),
            ((22, 22), True),
        ]  # fmt: skip
        for corner, wraps in corners:
            got = wrapped_crop_2d(on_device, corner, size)
            assert_array_matches_device(got, device)
            assert np.allclose(
                to_host_array(copy_to_device(got, "cpu")),
                self._reference(array, corner, size),
            )
            is_view = self._is_view(
                lambda a: wrapped_crop_2d(a, corner, size),
                copy_to_device(array, device),
            )
            assert is_view != wraps

        # a window wider than the period, wrapping on both axes
        got = wrapped_crop_2d(on_device, (-5, 40), (45, 33))
        assert_array_matches_device(got, device)
        assert np.allclose(
            to_host_array(copy_to_device(got, "cpu")),
            self._reference(array, (-5, 40), (45, 33)),
        )


def test_c_prism_on_a_skewed_cell_points_to_the_orthogonal_path():
    """A hexagonal cell is simulated on a skewed grid by default, where C-PRISM
    (upsample=True) is not supported: building it must say how to get dev's
    orthogonalised supercell back, and that path must still build."""
    import ase.build

    import abtem
    from abtem.prism.s_matrix import CompressedSMatrixArray

    atoms = ase.build.mx2("WSe2", vacuum=2)
    kwargs = dict(energy=60e3, semiangle_cutoff=20, interpolation=(2, 2), upsample=True)

    skewed = abtem.Potential(atoms, sampling=0.2, slice_thickness=2)
    assert skewed.non_orthogonal
    with pytest.raises(NotImplementedError, match="non_orthogonal=False"):
        abtem.SMatrix(potential=skewed, **kwargs).build(lazy=False)

    orthogonal = abtem.Potential(
        atoms, sampling=0.2, slice_thickness=2, non_orthogonal=False
    )
    built = abtem.SMatrix(potential=orthogonal, **kwargs).build(lazy=False)
    assert isinstance(built, CompressedSMatrixArray)


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
