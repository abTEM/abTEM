"""Tests of the upsampled (C-PRISM) scattering matrix, ``SMatrix(upsample=True)``."""

import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem import CompressedSMatrixArray, CustomScan, GridScan, Potential, Probe, SMatrix
from abtem.prism.s_matrix import SMatrixArray


def _small_potential(gpts=96, repetitions=(2, 2, 3)):
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(atoms, gpts=gpts, slice_thickness=2)


def _relative_error(measurement, reference):
    return (
        np.sqrt(((measurement.array - reference.array) ** 2).mean())
        / reference.array.mean()
    )


@pytest.mark.parametrize("interpolation", [(1, 1), (2, 2), (2, 4)])
def test_upsample_matches_probe(interpolation):
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=interpolation,
        upsample=True,
        tolerance=1e-6,
    )

    s_matrix_array = s_matrix.build(lazy=False)

    probe_diffraction_patterns = (
        s_matrix.dummy_probes().build(lazy=False).diffraction_patterns(max_angle=None)
    )
    diffraction_patterns = s_matrix_array.reduce().diffraction_patterns(max_angle=None)

    assert np.allclose(
        np.squeeze(diffraction_patterns.array),
        np.squeeze(probe_diffraction_patterns.array),
        atol=1e-5,
    )


def test_upsample_rank_one_in_vacuum():
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
        tolerance=1e-6,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    assert isinstance(s_matrix_array, CompressedSMatrixArray)
    assert s_matrix_array.rank == 1


def test_upsample_matches_multislice_no_interpolation():
    potential = _small_potential()

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    scan = CustomScan([(2.2, 3.3), (5.0, 5.0)])

    probe_diffraction_patterns = probe.multislice(
        potential=potential, scan=scan, lazy=False
    ).diffraction_patterns(max_angle=50)

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        upsample=True,
        tolerance=1e-6,
        max_rank=10_000,
    )

    diffraction_patterns = s_matrix.reduce(scan=scan, lazy=False).diffraction_patterns(
        max_angle=50
    )

    assert np.allclose(
        diffraction_patterns.array, probe_diffraction_patterns.array, atol=1e-5
    )


def test_upsample_beats_prism_at_same_interpolation():
    potential = _small_potential(repetitions=(2, 2, 4))

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(
        start=(0, 0), end=potential.extent, sampling=probe.aperture.nyquist_sampling
    )

    reference = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=False
    )

    prism_measurement = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    ).scan(scan=scan, detectors=detector, lazy=False)

    upsampled_measurement = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    ).scan(scan=scan, detectors=detector, lazy=False)

    prism_error = _relative_error(prism_measurement, reference)
    upsampled_error = _relative_error(upsampled_measurement, reference)

    assert upsampled_error < prism_error
    # the annular detector integrates any aliased ghost probes of the
    # interpolation as error; the band-limited interpolant must keep the
    # absolute error small, not just smaller than PRISM
    assert upsampled_error < 0.05


def test_upsample_off_grid_positions():
    potential = _small_potential()

    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)

    scan = CustomScan([(2.6180339, 4.1415926), (7.7182818, 3.3025851)])

    probe_diffraction_patterns = probe.multislice(
        potential=potential, scan=scan, lazy=False
    ).diffraction_patterns(max_angle=50)

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        upsample=True,
        tolerance=1e-6,
        max_rank=10_000,
    )

    diffraction_patterns = s_matrix.reduce(scan=scan, lazy=False).diffraction_patterns(
        max_angle=50
    )

    assert np.allclose(
        diffraction_patterns.array, probe_diffraction_patterns.array, atol=1e-5
    )


def test_upsample_windowed():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

    full = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    ).scan(scan=scan, detectors=detector, lazy=False)

    windowed = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        window_gpts=48,
    ).scan(scan=scan, detectors=detector, lazy=False)

    # the cropping window truncates the high-angle scattering tails, hence the
    # annular dark field signal is slightly reduced
    assert np.allclose(windowed.array, full.array, rtol=0.12)


def test_upsample_window_normalization():
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
        tolerance=1e-6,
        window_gpts=48,
    )

    diffraction_patterns = (
        s_matrix.build(lazy=False).reduce().diffraction_patterns(max_angle=None)
    )

    assert np.allclose(diffraction_patterns.array.sum(), 1.0, atol=1e-2)


@pytest.mark.parametrize("lazy", [False, True])
def test_upsample_frozen_phonons(lazy):
    atoms = bulk("Si", cubic=True) * (2, 2, 3)
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=13)
    potential = Potential(frozen_phonons, gpts=96, slice_thickness=2)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )

    measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)

    if lazy:
        measurement = measurement.compute()

    assert measurement.shape == (3, 3)
    assert np.all(measurement.array >= 0.0)


def test_upsample_lazy_matches_eager():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )

    eager = SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=False)
    lazy = SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=True).compute()

    assert np.allclose(eager.array, lazy.array, rtol=1e-4, atol=1e-8)


def test_upsample_detectors():
    potential = _small_potential()

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))

    detectors = [
        abtem.AnnularDetector(inner=40, outer=100),
        abtem.FlexibleAnnularDetector(),
        abtem.PixelatedDetector(max_angle=100),
        abtem.WavesDetector(),
    ]

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )

    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=False)

    assert len(measurements) == len(detectors)
    for measurement in measurements:
        assert measurement.shape[:2] == (2, 2)


def test_upsample_ctf_ensemble():
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))

    ctf = abtem.CTF(defocus=np.linspace(0, 50, 3))

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )

    measurement = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=False)

    assert measurement.shape == (3, 2, 2)


def test_upsample_downsampled_gpts_independent_of_interpolation():
    potential = _small_potential()

    downsampled_gpts = set()
    for interpolation in [(1, 1), (2, 2), (2, 4), (4, 4)]:
        s_matrix = SMatrix(
            potential=potential,
            energy=100e3,
            semiangle_cutoff=20,
            interpolation=interpolation,
            upsample=True,
        )
        downsampled_gpts.add(s_matrix.downsampled_gpts)

    assert len(downsampled_gpts) == 1


def test_upsample_aberrated_ctf_matches_probe():
    # the azimuthal angle convention of the reduction coefficients must match
    # polar_spatial_frequencies, ie. arctan2(ky, kx); the real-space intensity
    # of an aberrated probe is sensitive to the convention
    ctf = abtem.CTF(
        energy=100e3,
        semiangle_cutoff=20,
        defocus=50,
        astigmatism=40,
        astigmatism_angle=0.5236,
        coma=3e3,
        coma_angle=1.0,
    )

    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
        tolerance=1e-6,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    probe = Probe._from_ctf(
        extent=20, gpts=s_matrix_array.gpts, ctf=ctf, energy=100e3
    )
    probe_intensity = probe.build(lazy=False).intensity().array

    upsampled_intensity = s_matrix_array.reduce(ctf=ctf).intensity().array

    assert np.allclose(
        np.squeeze(upsampled_intensity),
        np.squeeze(probe_intensity),
        atol=1e-3 * probe_intensity.max(),
    )


def test_upsample_identical_to_prism_without_interpolation():
    # at an interpolation factor of (1, 1) the plane-wave expansion is complete
    # and no compression is performed, hence the upsampled scattering matrix is
    # identical to PRISM
    potential = _small_potential()

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    prism_measurement = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=1
    ).scan(scan=scan, detectors=detector, lazy=False)

    upsampled = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        upsample=True,
    )

    assert isinstance(upsampled.build(lazy=False), SMatrixArray)

    upsampled_measurement = upsampled.scan(scan=scan, detectors=detector, lazy=False)

    assert np.allclose(upsampled_measurement.array, prism_measurement.array)


def test_upsample_defocus_phase():
    # the propagation phase is factored out before the interpolation and added
    # back on the dense plane waves; it is a pure phase, unity for a
    # zero-thickness (vacuum) expansion and non-trivial for a real potential.
    # the exact round-trip is pinned by the vacuum and aberrated-ctf tests
    vacuum = SMatrix(
        extent=20,
        gpts=64,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )
    assert np.allclose(vacuum._defocus_phase(vacuum.wave_vectors), 1.0)

    thick = SMatrix(
        potential=_small_potential(),
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
    )
    phase = thick._defocus_phase(thick.wave_vectors)
    assert np.allclose(np.abs(phase), 1.0)
    assert not np.allclose(phase, 1.0)


def test_upsample_only_options_require_upsample():
    kwargs = dict(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, max_rank=64)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, position_quantization=16)


def test_upsample_kwargs_do_not_change_prism():
    # the PRISM scattering matrix and its copies are unchanged by the upsampling
    # options at their defaults
    potential = _small_potential()

    s_matrix = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20, interpolation=2
    )

    copied = SMatrix(
        potential=potential, **s_matrix._copy_kwargs(exclude=("potential",))
    )

    assert copied.window_gpts == s_matrix.window_gpts
    assert copied.downsampled_gpts == s_matrix.downsampled_gpts
    assert not copied.upsample


def test_upsample_streamed_expansion_matches_expanded():
    # the streamed full-window reduction never materializes the expanded
    # scattering matrix; it must agree with the expanded reduction to floating
    # point precision for on-grid scans, off-grid positions and aberrated CTFs
    potential = _small_potential()

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 3))
    detector = abtem.PixelatedDetector(max_angle=None)

    expanded = s_matrix_array.reduce(scan=scan, detectors=detector)

    for max_batch_expansion in (7, 10_000):
        streamed = s_matrix_array.reduce(
            scan=scan, detectors=detector, max_batch_expansion=max_batch_expansion
        )
        assert np.allclose(
            streamed.array, expanded.array, atol=1e-5 * expanded.array.max()
        )

    positions = CustomScan(np.array([[1.234, 2.345], [3.001, 0.777]]))
    expanded = s_matrix_array.reduce(scan=positions, detectors=detector)
    streamed = s_matrix_array.reduce(
        scan=positions, detectors=detector, max_batch_expansion=33
    )
    assert np.allclose(
        streamed.array, expanded.array, atol=1e-5 * expanded.array.max()
    )

    ctf = abtem.CTF(semiangle_cutoff=20, defocus=50, Cs=1e5)
    expanded = s_matrix_array.reduce(scan=scan, detectors=detector, ctf=ctf)
    streamed = s_matrix_array.reduce(
        scan=scan, detectors=detector, ctf=ctf, max_batch_expansion=50
    )
    assert np.allclose(
        streamed.array, expanded.array, atol=1e-5 * expanded.array.max()
    )


def test_upsample_streamed_expansion_through_s_matrix():
    # the constructor argument streams the one-shot scan without changing the
    # result, both eagerly and lazily
    potential = _small_potential()
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))
    detector = abtem.PixelatedDetector(max_angle=None)
    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
    )

    expanded = SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=False)
    streamed = SMatrix(**kwargs, max_batch_expansion=41).scan(
        scan=scan, detectors=detector, lazy=False
    )
    assert np.allclose(
        streamed.array, expanded.array, atol=1e-5 * expanded.array.max()
    )

    lazy_streamed = (
        SMatrix(**kwargs, max_batch_expansion=41)
        .scan(scan=scan, detectors=detector, lazy=True)
        .compute()
    )
    assert np.allclose(
        lazy_streamed.array, expanded.array, atol=1e-5 * expanded.array.max()
    )


def test_upsample_streamed_expansion_validation():
    kwargs = dict(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, max_batch_expansion=8)

    with pytest.raises(ValueError, match="window_gpts"):
        SMatrix(
            **kwargs,
            interpolation=2,
            upsample=True,
            window_gpts=32,
            max_batch_expansion=8,
        )

    # a window covering the full grid is no window, hence compatible with the
    # streamed expansion (copies pass the derived window back through here)
    s_matrix = SMatrix(
        **kwargs, interpolation=2, upsample=True, max_batch_expansion=8
    )
    copied = SMatrix(**s_matrix._copy_kwargs(exclude=("potential",)))
    assert copied.max_batch_expansion == 8
    assert copied.window_gpts == s_matrix.window_gpts

    s_matrix_array = SMatrix(
        **kwargs, interpolation=2, upsample=True, window_gpts=32
    ).build(lazy=False)
    with pytest.raises(ValueError, match="window_gpts"):
        s_matrix_array.reduce(max_batch_expansion=8)


def test_upsample_singular_values_spectrum():
    # the full spectrum is kept for choosing the tolerance; it extends the
    # retained singular values and is sorted in descending order
    potential = _small_potential()
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-2,
    ).build(lazy=False)

    singular_values = s_matrix_array.singular_values
    assert len(singular_values) >= s_matrix_array.rank
    assert np.allclose(
        singular_values[: s_matrix_array.rank], s_matrix_array.sigma
    )
    assert np.all(np.diff(singular_values) <= 0)
    assert singular_values[s_matrix_array.rank - 1] >= 1e-2 * singular_values[0]


def test_upsample_modes_reduction_matches_expand():
    # the full-window mode contraction (the GPU path) must match the expanded
    # reduction to floating point precision, for commensurate, incommensurate
    # and off-grid positions, aberrated CTFs, and the complex wave functions
    potential = _small_potential()
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
    ).build(lazy=False)
    detector = abtem.PixelatedDetector(max_angle=None)

    scans = [
        GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3)),
        GridScan(start=(0.3, 0.11), end=(9.1, 8.7), gpts=(2, 3)),
        CustomScan(np.array([[1.234, 2.345], [3.001, 0.777]])),
    ]
    for scan in scans:
        expanded = s_matrix_array.reduce(scan=scan, detectors=detector, method="expand")
        modes = s_matrix_array.reduce(scan=scan, detectors=detector, method="modes")
        assert np.allclose(
            modes.array, expanded.array, atol=1e-4 * expanded.array.max()
        )

    ctf = abtem.CTF(semiangle_cutoff=20, defocus=50, Cs=1e5)
    expanded = s_matrix_array.reduce(
        scan=scans[0], detectors=detector, ctf=ctf, method="expand"
    )
    modes = s_matrix_array.reduce(
        scan=scans[0], detectors=detector, ctf=ctf, method="modes"
    )
    assert np.allclose(modes.array, expanded.array, atol=1e-4 * expanded.array.max())

    # complex waves in the absolute frame (registration must match, not just
    # the intensities)
    expanded = s_matrix_array.reduce(
        scan=(5.15, 4.85), detectors=abtem.WavesDetector(), method="expand"
    )
    modes = s_matrix_array.reduce(
        scan=(5.15, 4.85), detectors=abtem.WavesDetector(), method="modes"
    )
    assert np.allclose(
        modes.array, expanded.array, atol=1e-4 * np.abs(expanded.array).max()
    )


def test_upsample_batched_windowed_reduction_matches_loop():
    # the vectorized (GPU) windowed contraction is exactly the per-position loop
    potential = _small_potential()
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=24,
    ).build(lazy=False)

    ctf = abtem.CTF(semiangle_cutoff=20, energy=100e3)
    values = s_matrix_array._coefficient_values(
        s_matrix_array._calculate_ctf_coefficients(ctf)
    )
    kernel = np.ascontiguousarray(
        s_matrix_array._window_kernel(values, np.zeros(2)).transpose(1, 2, 0)
    )
    u_windows = np.ascontiguousarray(np.asarray(s_matrix_array._u).transpose(1, 2, 0))
    snapped = np.array([[0, 0], [3, 45], [40, 2], [46, 47], [11, 30]])

    loop = s_matrix_array._reduce_to_waves(u_windows, snapped, kernel)
    batched = s_matrix_array._reduce_to_waves_batched(u_windows, snapped, kernel)
    assert np.allclose(loop, batched)


def test_upsample_reduction_method_validation():
    potential = _small_potential()
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, upsample=True, tolerance=1e-3)
    full = SMatrix(**kwargs).build(lazy=False)
    windowed = SMatrix(**kwargs, window_gpts=24).build(lazy=False)
    detector = abtem.PixelatedDetector(max_angle=None)

    with pytest.raises(ValueError, match="method"):
        full.reduce(detectors=detector, method="bogus")
    with pytest.raises(ValueError, match="window_gpts"):
        windowed.reduce(detectors=detector, method="expand")
    with pytest.raises(ValueError, match="modes"):
        full.reduce(detectors=detector, method="modes", max_batch_expansion=16)


def test_upsample_lattice_reduction_matches_general():
    # the lattice (commensurate-scan) reduction must reproduce the general
    # windowed reduction for tiling scans, partial regions and fractional
    # scan origins, and must decline scans it cannot represent
    from abtem.prism.s_matrix import CompressedSMatrixArray

    potential = _small_potential()
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=24,
    ).build(lazy=False)

    detector = abtem.PixelatedDetector(max_angle=None)
    extent = potential.extent
    sampling = s_matrix_array.sampling[0]

    scans = [
        GridScan(start=(0, 0), end=extent, gpts=(16, 16)),  # tiles the grid
        GridScan(  # partial region, still a whole-pixel step
            start=(extent[0] * 4 / 64, extent[1] * 8 / 64),
            end=(extent[0] * 24 / 64, extent[1] * 28 / 64),
            gpts=(5, 5),
        ),
        GridScan(  # fractional origin, applied as a sub-pixel kernel shift
            start=(sampling * 0.3, sampling * 0.3),
            end=(sampling * 32.3, sampling * 32.3),
            gpts=(8, 8),
        ),
    ]

    original = CompressedSMatrixArray._lattice_geometry
    try:
        for scan in scans:
            assert original(s_matrix_array, scan) is not None
            fast = s_matrix_array.reduce(scan=scan, detectors=detector)

            CompressedSMatrixArray._lattice_geometry = (
                lambda self, scan, warn=False: None
            )
            general = s_matrix_array.reduce(scan=scan, detectors=detector)
            CompressedSMatrixArray._lattice_geometry = original

            assert np.allclose(
                fast.array, general.array, atol=1e-5 * general.array.max()
            )
    finally:
        CompressedSMatrixArray._lattice_geometry = original

    # scans whose step is not a whole number of pixels fall back and warn
    incommensurate = GridScan(start=(0, 0), end=extent, gpts=(12, 12))
    assert s_matrix_array._lattice_geometry(incommensurate) is None
    with pytest.warns(UserWarning, match="whole number of pixels"):
        s_matrix_array._lattice_geometry(incommensurate, warn=True)


def test_upsample_blend_angle():
    # blending switches to the plane-wave (PRISM) reduction of the built beams
    # above the blend angle; 'auto' resolves it from the aliasing limit of the
    # interpolation, extent / (2 * interpolation * thickness)
    potential = _small_potential(repetitions=(2, 2, 6))
    thickness = potential.thickness

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        blend_angle="auto",
    )
    expected = 1e3 * min(
        extent / (2 * interpolation * thickness)
        for extent, interpolation in zip(s_matrix.extent, s_matrix.interpolation)
    )
    assert np.isclose(s_matrix._resolved_blend_angle(), expected)

    s_matrix_array = s_matrix.build(lazy=False)
    assert np.isclose(s_matrix_array.blend_angle, expected)

    detector = abtem.PixelatedDetector(max_angle=None)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

    blended = s_matrix_array.reduce(scan=scan, detectors=detector)
    unblended = s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=0)
    plain = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=2, upsample=True, tolerance=1e-4,
    ).build(lazy=False).reduce(scan=scan, detectors=detector)

    # disabling recovers the plain interpolated reduction exactly
    assert np.allclose(unblended.array, plain.array, atol=1e-6 * plain.array.max())
    # blending changes the high-angle content but not the total intensity much
    assert not np.allclose(blended.array, plain.array, atol=1e-6 * plain.array.max())
    assert np.isclose(blended.array.sum(), plain.array.sum(), rtol=0.05)

    # a blend angle beyond the maximum simulated angle is a no-op
    very_high = s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=1e4)
    assert np.allclose(very_high.array, plain.array, atol=1e-6 * plain.array.max())

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20,
                interpolation=2, blend_angle=10.0)
