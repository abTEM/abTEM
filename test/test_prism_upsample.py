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
    # the full spectrum is kept for choosing the tolerance and is sorted in
    # descending order. The retained modes are NOT the leading singular vectors
    # — the row space of the built beams is retained whole so that the
    # plane-wave branch is exact — so their amplitudes track the spectrum
    # without reproducing it, and carry no more energy than the optimal
    # subspace of the same size.
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
    sigma = s_matrix_array.sigma
    rank = s_matrix_array.rank

    assert len(singular_values) >= rank
    assert np.all(np.diff(singular_values) <= 0)
    assert np.all(np.diff(sigma) <= 0)
    assert np.isclose(sigma[0], singular_values[0], rtol=1e-3)

    optimal = float((singular_values[:rank] ** 2).sum())
    assert float((sigma**2).sum()) <= optimal * (1.0 + 1e-5)
    assert float((sigma**2).sum()) >= optimal * 0.99
    # every mode above the tolerance is retained; the rank exceeds that count
    # because the built beams' row space is kept whole
    assert rank >= int((singular_values >= 1e-2 * singular_values[0]).sum())


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

    blended = s_matrix_array.reduce(scan=scan, detectors=detector,
                                    blend_angle=expected)
    unblended = s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=0)
    plain = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=2, upsample=True, tolerance=1e-4, blend_angle=0,
    ).build(lazy=False).reduce(scan=scan, detectors=detector)

    # disabling recovers the plain interpolated reduction exactly
    assert np.allclose(unblended.array, plain.array, atol=1e-6 * plain.array.max())
    # the default blend acts through the detector routing; a full diffraction
    # pattern is not routable, hence the default reduction is the plain
    # interpolated one and Fourier blending must be requested explicitly
    default = s_matrix_array.reduce(scan=scan, detectors=detector)
    assert np.allclose(default.array, plain.array, atol=1e-6 * plain.array.max())
    # blending changes the high-angle content but not the total intensity much
    assert not np.allclose(blended.array, plain.array, atol=1e-6 * plain.array.max())
    assert np.isclose(blended.array.sum(), plain.array.sum(), rtol=0.05)

    # a blend angle beyond the maximum simulated angle is a no-op
    very_high = s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=1e4)
    assert np.allclose(very_high.array, plain.array, atol=1e-6 * plain.array.max())

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20,
                interpolation=2, blend_angle=10.0)


def test_upsample_blend_aperture_and_clamp():
    # 'aperture' weights the blend by the probe-forming aperture; 'auto' clamps
    # to the aperture edge (with a warning) when the aliasing angle falls inside
    # the bright-field disk, where blending would import the periodized ghosts
    potential = _small_potential(repetitions=(2, 2, 6))

    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, upsample=True, tolerance=1e-4)
    s_matrix_array = SMatrix(**kwargs, blend_angle="aperture").build(lazy=False)
    assert s_matrix_array.blend_angle == "aperture"

    detector = abtem.PixelatedDetector(max_angle=None)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    # a full diffraction pattern is not routable, hence the Fourier-weighted
    # blend must be requested explicitly in the reduction
    blended = s_matrix_array.reduce(scan=scan, detectors=detector,
                                    blend_angle="aperture")
    plain = s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=0)
    assert not np.allclose(blended.array, plain.array, atol=1e-6 * plain.array.max())
    assert np.isclose(blended.array.sum(), plain.array.sum(), rtol=0.05)

    # a thick specimen at a large factor pushes the aliasing angle inside the
    # disk: 'auto' must clamp to the aperture and warn
    thick = _small_potential(repetitions=(2, 2, 24))
    s_matrix = SMatrix(potential=thick, energy=100e3, semiangle_cutoff=20,
                       interpolation=4, upsample=True, blend_angle="auto")
    with pytest.warns(UserWarning, match="aliased"):
        resolved = s_matrix._resolved_blend_angle()
    assert resolved == "aperture"


def test_upsample_composite_blend():
    # the composite blend reduces the interpolated branch on the full window and
    # the plane-wave branch on one period of its periodized wave functions,
    # adding the detected intensities; it requires window-independent detectors
    potential = _small_potential(repetitions=(2, 2, 6))
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        blend_angle="auto",
    ).build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    detectors = [
        abtem.AnnularDetector(inner=0, outer=15),
        abtem.AnnularDetector(inner=30, outer=60),
    ]

    # explicit blend angle between the two detectors, so the first stays the
    # interpolated reduction and the second becomes the plane-wave branch
    composite = s_matrix_array.scan(
        scan=scan, detectors=detectors, blend_angle=25.0,
        blend_window_gpts="period",
    )
    plain = s_matrix_array.scan(scan=scan, detectors=detectors, blend_angle=0)

    # below the blend angle the composite is the interpolated reduction
    assert np.allclose(
        composite[0].array, plain[0].array, rtol=0.02
    )
    # the high-angle branch changes the dark-field values
    assert not np.allclose(composite[1].array, plain[1].array, rtol=0.02)
    assert np.all(composite[1].array >= 0)

    with pytest.raises(NotImplementedError, match="intensity"):
        s_matrix_array.scan(
            scan=scan,
            detectors=abtem.PixelatedDetector(max_angle=None),
            blend_angle=25.0,
            blend_window_gpts="period",
        )


def test_upsample_lattice_detection_chunking():
    # the lattice row block is sized for the matrix products, but the blend and
    # the detectors transform what it produces, so they walk the block in
    # smaller chunks; the chunking must not change any measured value
    potential = _small_potential(repetitions=(2, 2, 6))
    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=32,
    )
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(16, 16))
    detectors = [
        abtem.AnnularDetector(inner=0, outer=15),
        abtem.AnnularDetector(inner=30, outer=60),
    ]

    for blend in (None, "auto"):
        s_matrix_array = SMatrix(**kwargs, blend_angle=blend).build(lazy=False)
        assert s_matrix_array._lattice_geometry(scan) is not None

        whole = s_matrix_array.scan(scan=scan, detectors=detectors)

        # one scan row per detection chunk, the whole scan per row block
        s_matrix_array._REDUCE_BATCH_BYTES = (
            scan.gpts[1] * int(np.prod(s_matrix_array.window_gpts)) * 8
        )
        chunked = s_matrix_array.scan(scan=scan, detectors=detectors)

        for a, b in zip(whole, chunked):
            assert np.array_equal(a.array, b.array)


def test_upsample_blend_snaps_to_detector_boundary():
    # above the blend angle the composite reduction is the plane-wave branch
    # alone, which is the PRISM algorithm exactly. A band straddling the blend
    # angle would mix the branches, so the angle is snapped down to a boundary
    # and cut sharply there; every band then lies wholly on one side.
    from abtem.prism.s_matrix import CompressedSMatrixArray as C

    detectors = [
        abtem.AnnularDetector(inner=0, outer=15),
        abtem.AnnularDetector(inner=21, outer=50),
        abtem.AnnularDetector(inner=50, outer=90),
    ]
    assert C._snapped_blend_angle(37.5, detectors) == 21.0
    assert C._snapped_blend_angle(60.0, detectors) == 50.0
    assert C._snapped_blend_angle(10.0, detectors) == 10.0   # no boundary below
    assert C._snapped_blend_angle("aperture", detectors) == "aperture"
    assert C._snapped_blend_angle(None, detectors) is None
    assert C._snapped_blend_angle(37.5, detectors[1]) == 21.0  # not a list

    potential = _small_potential(repetitions=(2, 2, 6))
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2)
    prism = SMatrix(**kwargs).build(lazy=False)
    cp = SMatrix(**kwargs, upsample=True, tolerance=1e-3,
                 window_gpts=int(prism.window_gpts[0]) * 2).build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    composite = cp.scan(scan=scan, detectors=detectors, blend_angle=37.5,
                        blend_window_gpts="period")
    reference = prism.scan(scan=scan, detectors=detectors)

    # snapped to 21 mrad: both dark-field bands sit above it and must be PRISM
    for a, b in zip(composite[1:], reference[1:]):
        assert np.allclose(a.array, b.array, rtol=1e-4, atol=1e-9)
    # the bright field is below it and is the interpolated reduction
    assert not np.allclose(composite[0].array, reference[0].array, rtol=1e-4)


def test_upsample_plane_wave_branch_is_prism():
    # the plane-wave branch reduced on one interpolation period must BE the
    # PRISM reduction, at any tolerance: the compression retains the row space
    # of the built beams whole rather than by singular value
    potential = _small_potential(repetitions=(2, 2, 6))
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2)
    prism = SMatrix(**kwargs).build(lazy=False)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    expected = prism.reduce(scan=scan).array

    for tolerance in (1e-1, 1e-2, 1e-3):
        cp = SMatrix(**kwargs, upsample=True, tolerance=tolerance,
                     window_gpts=int(prism.window_gpts[0])).build(lazy=False)
        # a vanishing blend angle leaves the plane-wave branch alone
        branch = cp.reduce(scan=scan, blend_angle=1e-6).array
        error = np.abs(branch - expected).max() / np.abs(expected).max()
        assert error < 1e-4, f"tolerance {tolerance}: {error}"
