"""Tests of the upsampled (C-PRISM) scattering matrix, ``SMatrix(upsample=True)``."""

import numpy as np
import pytest
from ase.build import bulk
from utils import gpu

import abtem
from abtem import CompressedSMatrixArray, CustomScan, GridScan, Potential, Probe, SMatrix
from abtem.core.backend import asnumpy, get_array_module
from abtem.prism.s_matrix import SMatrixArray

# Every test that runs array code is parametrized over the device: the
# compression (``eigh`` on the Gram matrix), the mode contractions and the
# stitched pattern assembly all branch on the array module, and those branches
# are only exercised when the tests also run on the GPU. Tests of pure Python
# validation or grid arithmetic are left unparametrized and say so.
devices = pytest.mark.parametrize("device", [gpu, "cpu"])


def _small_potential(gpts=96, repetitions=(2, 2, 3), device="cpu"):
    atoms = bulk("Si", cubic=True) * repetitions
    return Potential(atoms, gpts=gpts, slice_thickness=2, device=device)


def _array(measurement):
    """The array of a measurement or wave function, on the host.

    Detectors return host arrays by default, but reductions to wave functions
    and their derived measurements stay on the device, so every assertion goes
    through this.
    """
    array = measurement.array if hasattr(measurement, "array") else measurement
    return asnumpy(array)


def _relative_error(measurement, reference):
    measurement, reference = _array(measurement), _array(reference)
    return np.sqrt(((measurement - reference) ** 2).mean()) / reference.mean()


def _assert_chunking_unchanged(whole, chunked, what):
    """The chunked reduction must reproduce the unchunked one.

    Not bit for bit: the chunk size sets the row count of the lattice matrix
    products, and BLAS picks its kernel (hence its summation order) from the
    operand shape, so the last bits legitimately move on some builds. A real
    chunking bug misplaces whole rows and lands far above this bound.
    """
    whole, chunked = _array(whole), _array(chunked)
    scale = max(float(np.abs(whole).max()), 1e-30)
    error = float(np.abs(whole - chunked).max()) / scale
    assert error < 1e-6, f"{what}: chunking moved values by {error:.3e} relative"


@devices
@pytest.mark.parametrize("interpolation", [(1, 1), (2, 2), (2, 4)])
def test_upsample_matches_probe(interpolation, device):
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=interpolation,
        upsample=True,
        tolerance=1e-6,
        device=device,
    )

    s_matrix_array = s_matrix.build(lazy=False)

    probe_diffraction_patterns = (
        s_matrix.dummy_probes().build(lazy=False).diffraction_patterns(max_angle=None)
    )
    diffraction_patterns = s_matrix_array.reduce().diffraction_patterns(max_angle=None)

    assert np.allclose(
        np.squeeze(_array(diffraction_patterns)),
        np.squeeze(_array(probe_diffraction_patterns)),
        atol=1e-5,
    )


@devices
def test_upsample_rank_one_in_vacuum(device):
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
        tolerance=1e-6,
        device=device,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    assert isinstance(s_matrix_array, CompressedSMatrixArray)
    assert s_matrix_array.rank == 1


@devices
@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([(2.2, 3.3), (5.0, 5.0)], id="on_grid"),
        pytest.param(
            [(2.6180339, 4.1415926), (7.7182818, 3.3025851)], id="off_grid"
        ),
    ],
)
def test_upsample_matches_multislice_no_interpolation(device, positions):
    potential = _small_potential(device=device)

    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    scan = CustomScan(positions)

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
        device=device,
    )

    diffraction_patterns = s_matrix.reduce(scan=scan, lazy=False).diffraction_patterns(
        max_angle=50
    )

    assert np.allclose(
        _array(diffraction_patterns), _array(probe_diffraction_patterns), atol=1e-5
    )


@devices
def test_upsample_beats_prism_at_same_interpolation(device):
    potential = _small_potential(repetitions=(2, 2, 4), device=device)

    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(
        start=(0, 0), end=potential.extent, sampling=probe.aperture.nyquist_sampling
    )

    reference = probe.scan(
        potential=potential, scan=scan, detectors=detector, lazy=False
    )

    prism_measurement = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        device=device,
    ).scan(scan=scan, detectors=detector, lazy=False)

    upsampled_measurement = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    ).scan(scan=scan, detectors=detector, lazy=False)

    prism_error = _relative_error(prism_measurement, reference)
    upsampled_error = _relative_error(upsampled_measurement, reference)

    assert upsampled_error < prism_error
    # the annular detector integrates any aliased ghost probes of the
    # interpolation as error; the band-limited interpolant must keep the
    # absolute error small, not just smaller than PRISM
    assert upsampled_error < 0.05


@devices
def test_upsample_windowed(device):
    potential = _small_potential(device=device)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

    full = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    ).scan(scan=scan, detectors=detector, lazy=False)

    windowed = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        window_gpts=48,
        device=device,
    ).scan(scan=scan, detectors=detector, lazy=False)

    # the cropping window truncates the high-angle scattering tails, hence the
    # annular dark field signal is slightly reduced
    assert np.allclose(_array(windowed), _array(full), rtol=0.12)


@devices
def test_upsample_window_normalization(device):
    s_matrix = SMatrix(
        extent=20,
        gpts=128,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=(2, 2),
        upsample=True,
        tolerance=1e-6,
        window_gpts=48,
        device=device,
    )

    diffraction_patterns = (
        s_matrix.build(lazy=False).reduce().diffraction_patterns(max_angle=None)
    )

    assert np.allclose(_array(diffraction_patterns).sum(), 1.0, atol=1e-2)


@devices
@pytest.mark.parametrize("lazy", [False, True])
def test_upsample_frozen_phonons(lazy, device):
    atoms = bulk("Si", cubic=True) * (2, 2, 3)
    frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=13)
    potential = Potential(
        frozen_phonons, gpts=96, slice_thickness=2, device=device
    )

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    )

    measurement = s_matrix.scan(scan=scan, detectors=detector, lazy=lazy)

    if lazy:
        measurement = measurement.compute()

    assert measurement.shape == (3, 3)
    assert np.all(_array(measurement) >= 0.0)


@devices
def test_upsample_lazy_matches_eager(device):
    potential = _small_potential(device=device)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    )

    eager = SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=False)
    lazy = SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=True).compute()

    assert np.allclose(_array(eager), _array(lazy), rtol=1e-4, atol=1e-8)


@devices
def test_upsample_detectors(device):
    potential = _small_potential(device=device)

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
        device=device,
    )

    measurements = s_matrix.scan(scan=scan, detectors=detectors, lazy=False)

    assert len(measurements) == len(detectors)
    for measurement in measurements:
        assert measurement.shape[:2] == (2, 2)


@devices
def test_upsample_ctf_ensemble(device):
    potential = _small_potential(device=device)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(2, 2))

    ctf = abtem.CTF(defocus=np.linspace(0, 50, 3))

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    )

    measurement = s_matrix.scan(scan=scan, detectors=detector, ctf=ctf, lazy=False)

    assert measurement.shape == (3, 2, 2)


def test_upsample_downsampled_gpts_independent_of_interpolation():
    # grid arithmetic only, no array code: device-independent
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


@devices
def test_upsample_aberrated_ctf_matches_probe(device):
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
        device=device,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    probe = Probe._from_ctf(
        extent=20, gpts=s_matrix_array.gpts, ctf=ctf, energy=100e3, device=device
    )
    probe_intensity = _array(probe.build(lazy=False).intensity())

    upsampled_intensity = _array(s_matrix_array.reduce(ctf=ctf).intensity())

    assert np.allclose(
        np.squeeze(upsampled_intensity),
        np.squeeze(probe_intensity),
        atol=1e-3 * probe_intensity.max(),
    )


@devices
def test_upsample_identical_to_prism_without_interpolation(device):
    # at an interpolation factor of (1, 1) the plane-wave expansion is complete
    # and no compression is performed, hence the upsampled scattering matrix is
    # identical to PRISM
    potential = _small_potential(device=device)

    detector = abtem.AnnularDetector(inner=40, outer=100)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))

    prism_measurement = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        device=device,
    ).scan(scan=scan, detectors=detector, lazy=False)

    upsampled = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=1,
        upsample=True,
        device=device,
    )

    assert isinstance(upsampled.build(lazy=False), SMatrixArray)

    upsampled_measurement = upsampled.scan(scan=scan, detectors=detector, lazy=False)

    assert np.allclose(_array(upsampled_measurement), _array(prism_measurement))


@devices
def test_upsample_defocus_phase(device):
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
        device=device,
    )
    assert np.allclose(_array(vacuum._defocus_phase(vacuum.wave_vectors)), 1.0)

    thick = SMatrix(
        potential=_small_potential(device=device),
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        device=device,
    )
    phase = _array(thick._defocus_phase(thick.wave_vectors))
    assert np.allclose(np.abs(phase), 1.0)
    assert not np.allclose(phase, 1.0)


def test_upsample_only_options_require_upsample():
    # constructor validation only: device-independent
    kwargs = dict(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, max_rank=64)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, position_quantization=16)

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(**kwargs, interpolation=2, window_gpts=32)


def test_upsample_kwargs_do_not_change_prism():
    # the PRISM scattering matrix and its copies are unchanged by the upsampling
    # options at their defaults. Copies pass the derived PRISM cropping window
    # back through ``window_gpts``, which must not trip its validation
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


@devices
def test_upsample_streamed_expansion_matches_expanded(device):
    # the streamed full-window reduction never materializes the expanded
    # scattering matrix; it must agree with the expanded reduction to floating
    # point precision for on-grid scans, off-grid positions and aberrated CTFs
    potential = _small_potential(device=device)

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        device=device,
    )
    s_matrix_array = s_matrix.build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 3))
    detector = abtem.PixelatedDetector(max_angle=None)

    expanded = _array(s_matrix_array.reduce(scan=scan, detectors=detector))

    for max_batch_expansion in (7, 10_000):
        streamed = _array(
            s_matrix_array.reduce(
                scan=scan, detectors=detector, max_batch_expansion=max_batch_expansion
            )
        )
        assert np.allclose(streamed, expanded, atol=1e-5 * expanded.max())

    positions = CustomScan(np.array([[1.234, 2.345], [3.001, 0.777]]))
    expanded = _array(s_matrix_array.reduce(scan=positions, detectors=detector))
    streamed = _array(
        s_matrix_array.reduce(
            scan=positions, detectors=detector, max_batch_expansion=33
        )
    )
    assert np.allclose(streamed, expanded, atol=1e-5 * expanded.max())

    ctf = abtem.CTF(semiangle_cutoff=20, defocus=50, Cs=1e5)
    expanded = _array(s_matrix_array.reduce(scan=scan, detectors=detector, ctf=ctf))
    streamed = _array(
        s_matrix_array.reduce(
            scan=scan, detectors=detector, ctf=ctf, max_batch_expansion=50
        )
    )
    assert np.allclose(streamed, expanded, atol=1e-5 * expanded.max())


@devices
def test_upsample_streamed_expansion_through_s_matrix(device):
    # the constructor argument streams the one-shot scan without changing the
    # result, both eagerly and lazily
    potential = _small_potential(device=device)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3))
    detector = abtem.PixelatedDetector(max_angle=None)
    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        device=device,
    )

    expanded = _array(SMatrix(**kwargs).scan(scan=scan, detectors=detector, lazy=False))
    streamed = _array(
        SMatrix(**kwargs, max_batch_expansion=41).scan(
            scan=scan, detectors=detector, lazy=False
        )
    )
    assert np.allclose(streamed, expanded, atol=1e-5 * expanded.max())

    lazy_streamed = _array(
        SMatrix(**kwargs, max_batch_expansion=41)
        .scan(scan=scan, detectors=detector, lazy=True)
        .compute()
    )
    assert np.allclose(lazy_streamed, expanded, atol=1e-5 * expanded.max())


def test_upsample_streamed_expansion_validation():
    # constructor and reduction validation only: device-independent
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


@devices
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_upsample_honours_the_precision_config(precision, device):
    # the compression and every array it derives follow config['precision'].
    # The expanded scattering matrix is the trap: its wave vectors multiply
    # into the plane-wave coefficients, so a wider dtype there silently
    # promotes the whole reduction and doubles its memory
    complex_dtype = np.dtype(f"complex{2 * int(precision[5:])}")
    real_dtype = np.dtype(precision)

    with abtem.config.set({"precision": precision}):
        s_matrix = SMatrix(
            potential=_small_potential(device=device),
            energy=100e3,
            semiangle_cutoff=20,
            interpolation=2,
            upsample=True,
            tolerance=1e-3,
            device=device,
        )
        assert s_matrix.wave_vectors.dtype == real_dtype

        built = s_matrix.build(lazy=False)
        assert built.u.dtype == complex_dtype
        assert built.vh_dense.dtype == complex_dtype
        assert built.wave_vectors.dtype == real_dtype

        expanded = built._expanded_s_matrix_array()
        assert expanded.array.dtype == complex_dtype
        scan = GridScan(start=(0, 0), end=s_matrix.extent, gpts=(3, 3))
        coefficients = expanded._calculate_positions_coefficients(scan)
        assert coefficients.dtype == complex_dtype

        waves = built.reduce(scan=scan, detectors=abtem.WavesDetector())
        assert waves.array.dtype == complex_dtype


@devices
def test_upsample_singular_values_spectrum(device):
    # the full spectrum is kept for choosing the tolerance and is sorted in
    # descending order. The retained modes are NOT the leading singular vectors
    # — the row space of the built beams is retained whole so that the
    # plane-wave branch is exact — so their amplitudes track the spectrum
    # without reproducing it, and carry no more energy than the optimal
    # subspace of the same size.
    potential = _small_potential(device=device)
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-2,
        device=device,
    ).build(lazy=False)

    singular_values = _array(s_matrix_array.singular_values)
    sigma = _array(s_matrix_array.sigma)
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


@devices
def test_upsample_modes_reduction_matches_expand(device):
    # the full-window mode contraction (the GPU path) must match the expanded
    # reduction to floating point precision, for commensurate, incommensurate
    # and off-grid positions, aberrated CTFs, and the complex wave functions
    potential = _small_potential(device=device)
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        device=device,
    ).build(lazy=False)
    detector = abtem.PixelatedDetector(max_angle=None)

    scans = [
        GridScan(start=(0, 0), end=potential.extent, gpts=(3, 3)),
        GridScan(start=(0.3, 0.11), end=(9.1, 8.7), gpts=(2, 3)),
        CustomScan(np.array([[1.234, 2.345], [3.001, 0.777]])),
    ]
    for scan in scans:
        expanded = _array(
            s_matrix_array.reduce(scan=scan, detectors=detector, method="expand")
        )
        modes = _array(
            s_matrix_array.reduce(scan=scan, detectors=detector, method="modes")
        )
        assert np.allclose(modes, expanded, atol=1e-4 * expanded.max())

    ctf = abtem.CTF(semiangle_cutoff=20, defocus=50, Cs=1e5)
    expanded = _array(
        s_matrix_array.reduce(
            scan=scans[0], detectors=detector, ctf=ctf, method="expand"
        )
    )
    modes = _array(
        s_matrix_array.reduce(
            scan=scans[0], detectors=detector, ctf=ctf, method="modes"
        )
    )
    assert np.allclose(modes, expanded, atol=1e-4 * expanded.max())

    # complex waves in the absolute frame (registration must match, not just
    # the intensities)
    expanded = _array(
        s_matrix_array.reduce(
            scan=(5.15, 4.85), detectors=abtem.WavesDetector(), method="expand"
        )
    )
    modes = _array(
        s_matrix_array.reduce(
            scan=(5.15, 4.85), detectors=abtem.WavesDetector(), method="modes"
        )
    )
    assert np.allclose(modes, expanded, atol=1e-4 * np.abs(expanded).max())


@devices
def test_upsample_batched_windowed_reduction_matches_loop(device):
    # the vectorized (GPU) windowed contraction is exactly the per-position
    # loop. On the GPU ``_reduce_to_waves`` dispatches to the batched
    # implementation, so this also pins that dispatch
    potential = _small_potential(device=device)
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=24,
        device=device,
    ).build(lazy=False)

    xp = get_array_module(s_matrix_array._u)

    ctf = abtem.CTF(semiangle_cutoff=20, energy=100e3)
    values = s_matrix_array._coefficient_values(
        s_matrix_array._calculate_ctf_coefficients(ctf)
    )
    # both operands carry the mode axis FIRST, the layout the modes are stored
    # in, so neither needs a transposed copy
    kernel = s_matrix_array._window_kernel(values, np.zeros(2))
    u_windows = xp.asarray(s_matrix_array._u)
    assert kernel.shape[0] == u_windows.shape[0] == s_matrix_array.rank
    snapped = xp.asarray([[0, 0], [3, 45], [40, 2], [46, 47], [11, 30]])

    loop = s_matrix_array._reduce_to_waves(u_windows, snapped, kernel)
    batched = s_matrix_array._reduce_to_waves_batched(u_windows, snapped, kernel)
    assert np.allclose(_array(loop), _array(batched))


def test_upsample_reduction_method_validation():
    # reduction argument validation only: device-independent
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


@devices
def test_upsample_lattice_reduction_matches_general(device):
    # the lattice (commensurate-scan) reduction must reproduce the general
    # windowed reduction for tiling scans, partial regions and fractional
    # scan origins, and must decline scans it cannot represent
    potential = _small_potential(device=device)
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=24,
        device=device,
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
            fast = _array(s_matrix_array.reduce(scan=scan, detectors=detector))

            CompressedSMatrixArray._lattice_geometry = (
                lambda self, scan, warn=False: None
            )
            general = _array(s_matrix_array.reduce(scan=scan, detectors=detector))
            CompressedSMatrixArray._lattice_geometry = original

            assert np.allclose(fast, general, atol=1e-5 * general.max())
    finally:
        CompressedSMatrixArray._lattice_geometry = original

    # scans whose step is not a whole number of pixels fall back and warn
    incommensurate = GridScan(start=(0, 0), end=extent, gpts=(12, 12))
    assert s_matrix_array._lattice_geometry(incommensurate) is None
    with pytest.warns(UserWarning, match="whole number of pixels"):
        s_matrix_array._lattice_geometry(incommensurate, warn=True)


@devices
def test_upsample_blend_angle(device):
    # blending switches to the plane-wave (PRISM) reduction of the built beams
    # above the blend angle; 'auto' resolves it from the aliasing limit of the
    # interpolation. The beams are referenced to the middle of the specimen,
    # which centres the tilt spectrum, hence the limit is
    # extent / (interpolation * thickness) — twice the uncentred one.
    # thick enough that the centred limit still falls inside the simulated
    # angular range, where blending is not a no-op
    potential = _small_potential(repetitions=(2, 2, 16), device=device)
    thickness = potential.thickness

    s_matrix = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        blend_angle="auto",
        device=device,
    )
    expected = 1e3 * min(
        extent / (interpolation * thickness)
        for extent, interpolation in zip(s_matrix.extent, s_matrix.interpolation)
    )
    assert np.isclose(s_matrix._resolved_blend_angle(), expected)
    # centring the tilt spectrum is what buys the factor of two
    uncentred = 1e3 * min(
        extent / (2 * interpolation * thickness)
        for extent, interpolation in zip(s_matrix.extent, s_matrix.interpolation)
    )
    assert np.isclose(expected, 2 * uncentred)

    s_matrix_array = s_matrix.build(lazy=False)
    assert np.isclose(s_matrix_array.blend_angle, expected)

    detector = abtem.PixelatedDetector(max_angle=None)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))

    blended = _array(
        s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=expected)
    )
    unblended = _array(
        s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=0)
    )
    plain = _array(
        SMatrix(
            potential=potential, energy=100e3, semiangle_cutoff=20,
            interpolation=2, upsample=True, tolerance=1e-4, blend_angle=0,
            device=device,
        ).build(lazy=False).reduce(scan=scan, detectors=detector)
    )
    assert expected < 1e3 * min(s_matrix_array.dummy_probes().cutoff_angles)

    # disabling recovers the plain interpolated reduction exactly
    assert np.allclose(unblended, plain, atol=1e-6 * plain.max())
    # the default blend acts through the detector routing; a full diffraction
    # pattern is not routable, hence the default reduction is the plain
    # interpolated one and Fourier blending must be requested explicitly
    default = _array(s_matrix_array.reduce(scan=scan, detectors=detector))
    assert np.allclose(default, plain, atol=1e-6 * plain.max())
    # blending changes the high-angle content but not the total intensity much
    assert not np.allclose(blended, plain, atol=1e-6 * plain.max())
    assert np.isclose(blended.sum(), plain.sum(), rtol=0.05)

    # a blend angle beyond the maximum simulated angle is a no-op
    very_high = _array(
        s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=1e4)
    )
    assert np.allclose(very_high, plain, atol=1e-6 * plain.max())

    with pytest.raises(ValueError, match="upsample=True"):
        SMatrix(extent=20, gpts=128, energy=100e3, semiangle_cutoff=20,
                interpolation=2, blend_angle=10.0)


@devices
def test_upsample_blend_aperture_and_clamp(device):
    # 'aperture' weights the blend by the probe-forming aperture; 'auto' clamps
    # to the aperture edge (with a warning) when the aliasing angle falls inside
    # the bright-field disk, where blending would import the periodized ghosts
    potential = _small_potential(repetitions=(2, 2, 6), device=device)

    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, upsample=True, tolerance=1e-4, device=device)
    s_matrix_array = SMatrix(**kwargs, blend_angle="aperture").build(lazy=False)
    assert s_matrix_array.blend_angle == "aperture"

    detector = abtem.PixelatedDetector(max_angle=None)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    # a full diffraction pattern is not routable, hence the Fourier-weighted
    # blend must be requested explicitly in the reduction
    blended = _array(
        s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle="aperture")
    )
    plain = _array(
        s_matrix_array.reduce(scan=scan, detectors=detector, blend_angle=0)
    )
    assert not np.allclose(blended, plain, atol=1e-6 * plain.max())
    assert np.isclose(blended.sum(), plain.sum(), rtol=0.05)

    # a thick specimen at a large factor pushes the aliasing angle inside the
    # disk: 'auto' must clamp to the aperture and warn
    thick = _small_potential(repetitions=(2, 2, 48), device=device)
    s_matrix = SMatrix(potential=thick, energy=100e3, semiangle_cutoff=20,
                       interpolation=4, upsample=True, blend_angle="auto",
                       device=device)
    with pytest.warns(UserWarning, match="aliased"):
        resolved = s_matrix._resolved_blend_angle()
    assert resolved == "aperture"


@devices
def test_upsample_composite_blend(device):
    # the composite blend reduces the interpolated branch on the full window and
    # the plane-wave branch on one period of its periodized wave functions,
    # adding the detected intensities; it requires window-independent detectors
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    s_matrix_array = SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        blend_angle="auto",
        device=device,
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
    assert np.allclose(_array(composite[0]), _array(plain[0]), rtol=0.02)
    # the high-angle branch changes the dark-field values
    assert not np.allclose(_array(composite[1]), _array(plain[1]), rtol=0.02)
    assert np.all(_array(composite[1]) >= 0)

    with pytest.raises(NotImplementedError, match="intensity"):
        s_matrix_array.scan(
            scan=scan,
            detectors=abtem.PixelatedDetector(max_angle=None),
            blend_angle=25.0,
            blend_window_gpts="period",
        )


@devices
def test_upsample_lattice_detection_chunking(device):
    # the lattice row block is sized for the matrix products, but the blend and
    # the detectors transform what it produces, so they walk the block in
    # smaller chunks; the chunking must not change any measured value
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    kwargs = dict(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20,
        interpolation=2,
        upsample=True,
        tolerance=1e-4,
        window_gpts=32,
        device=device,
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
            _assert_chunking_unchanged(a, b, f"blend={blend}")


def test_upsample_blend_snapping():
    # snapping the blend angle onto a detector boundary is pure angular
    # arithmetic: device-independent
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


@devices
def test_upsample_blend_snaps_to_detector_boundary(device):
    # above the blend angle the composite reduction is the plane-wave branch
    # alone, which is the PRISM algorithm exactly. A band straddling the blend
    # angle would mix the branches, so the angle is snapped down to a boundary
    # and cut sharply there; every band then lies wholly on one side.
    detectors = [
        abtem.AnnularDetector(inner=0, outer=15),
        abtem.AnnularDetector(inner=21, outer=50),
        abtem.AnnularDetector(inner=50, outer=90),
    ]

    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, device=device)
    prism = SMatrix(**kwargs).build(lazy=False)
    compressed = SMatrix(**kwargs, upsample=True, tolerance=1e-3,
                         window_gpts=int(prism.window_gpts[0]) * 2).build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    composite = compressed.scan(scan=scan, detectors=detectors, blend_angle=37.5,
                                blend_window_gpts="period")
    reference = prism.scan(scan=scan, detectors=detectors)

    # snapped to 21 mrad: both dark-field bands sit above it and must be PRISM
    for a, b in zip(composite[1:], reference[1:]):
        assert np.allclose(_array(a), _array(b), rtol=1e-4, atol=1e-9)
    # the bright field is below it and is the interpolated reduction
    assert not np.allclose(
        _array(composite[0]), _array(reference[0]), rtol=1e-4
    )


@devices
def test_upsample_plane_wave_branch_is_prism(device):
    # the plane-wave branch reduced on one interpolation period must BE the
    # PRISM reduction, at any tolerance: the compression retains the row space
    # of the built beams whole rather than by singular value
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, device=device)
    prism = SMatrix(**kwargs).build(lazy=False)
    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    expected = _array(prism.reduce(scan=scan))

    for tolerance in (1e-1, 1e-2, 1e-3):
        compressed = SMatrix(**kwargs, upsample=True, tolerance=tolerance,
                             window_gpts=int(prism.window_gpts[0])).build(lazy=False)
        # a vanishing blend angle leaves the plane-wave branch alone
        branch = _array(compressed.reduce(scan=scan, blend_angle=1e-6))
        error = np.abs(branch - expected).max() / np.abs(expected).max()
        assert error < 1e-4, f"tolerance {tolerance}: {error}"


@devices
def test_upsample_blend_taper_routing(device):
    # with a taper, a narrow band overlapping [cut - taper, cut] reads a convex
    # combination of the two branch intensities — the angular density is then
    # continuous across the cut — while bands clear of the zone stay pure
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    built = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=2, upsample=True, tolerance=1e-4, device=device,
    ).build(lazy=False)

    scan = GridScan(start=(0, 0), end=potential.extent, gpts=(4, 4))
    cut, taper = 40.0, 8.0
    rings = [
        abtem.AnnularDetector(inner=24, outer=32),   # below the zone -> low
        abtem.AnnularDetector(inner=34, outer=40),   # inside the zone -> taper
        abtem.AnnularDetector(inner=40, outer=48),   # above the cut -> high
    ]

    tapered = built.scan(scan=scan, detectors=rings, blend_angle=cut,
                         blend_taper=taper)
    sharp = built.scan(scan=scan, detectors=rings, blend_angle=cut)

    # outside the zone the taper changes nothing
    assert np.allclose(_array(tapered[0]), _array(sharp[0]), rtol=1e-5)
    assert np.allclose(_array(tapered[2]), _array(sharp[2]), rtol=1e-5)

    # inside the zone the value lies between the two pure branches
    low = _array(built.scan(scan=scan, detectors=rings[1], blend_angle=0))
    high = _array(
        built._with_window(
            tuple(-(-g // i) for g, i in zip(built.gpts, built._interpolation))
        ).reduce(scan=scan, detectors=rings[1], blend_angle=cut - taper,
                 _blend_component="high", _blend_taper=0.0)
    )
    lower = np.minimum(low, high) * (1 - 1e-4) - 1e-12
    upper = np.maximum(low, high) * (1 + 1e-4) + 1e-12
    assert np.all(_array(tapered[1]) >= lower)
    assert np.all(_array(tapered[1]) <= upper)
    assert not np.allclose(_array(tapered[1]), _array(sharp[1]), rtol=1e-5)


@devices
def test_upsample_lattice_product_chunking(device):
    # at small scan steps the lattice reduction's per-offset product tensor is
    # large; it is processed in row chunks within the batch budget, which must
    # not change any value — including at step 2, where the sub-step groups
    # hold half the window each
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    built = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=2, upsample=True, tolerance=1e-3, window_gpts=32,
        device=device,
    ).build(lazy=False)

    detector = abtem.PixelatedDetector(max_angle=None)
    for scan_gpts in (32, 16):  # steps 2 and 4 on the 64 grid
        scan = GridScan(start=(0, 0), end=potential.extent,
                        gpts=(scan_gpts, scan_gpts))
        assert built._lattice_geometry(scan) is not None

        whole = _array(built.reduce(scan=scan, detectors=detector))
        built._REDUCE_BATCH_BYTES = 65536  # a few product rows per chunk
        chunked = _array(built.reduce(scan=scan, detectors=detector))
        del built._REDUCE_BATCH_BYTES

        _assert_chunking_unchanged(whole, chunked, f"scan {scan_gpts}")


@devices
def test_commensurate_scan(device):
    # GridScan.commensurate steps a whole number of pixels of the grid the
    # scattering matrix is reduced on, snapping the request to a divisor of
    # the span, so the lattice reduction always engages
    potential = _small_potential(repetitions=(2, 2, 3), device=device)  # 96 -> 64
    s_matrix = SMatrix(potential=potential, energy=100e3, semiangle_cutoff=20,
                       interpolation=2, upsample=True, tolerance=1e-3,
                       window_gpts=32, device=device)
    built = s_matrix.build(lazy=False)
    assert built.gpts == (64, 64)

    for source in (potential, s_matrix, built):
        scan = GridScan.commensurate(source, gpts=16)
        assert scan.gpts == (16, 16)
        assert np.allclose(scan.end, potential.extent)
        assert built._lattice_geometry(scan) is not None

    # 64 / 13 is not whole: snapped to the nearest divisor step (4 -> 16 gpts)
    assert GridScan.commensurate(built, gpts=13).gpts == (16, 16)

    # a requested sampling snaps to the nearest whole-pixel step
    sampled = GridScan.commensurate(built, sampling=potential.extent[0] / 16.4)
    assert sampled.gpts == (16, 16)

    # a sub-region with a fractional span: the end moves onto the lattice
    sub = GridScan.commensurate(built, gpts=4, start=(1.234, 1.234),
                                end=(1.234 + 3.03, 1.234 + 3.03))
    pixels = np.asarray(sub.get_positions()) / (potential.extent[0] / 64)
    steps = np.diff(pixels[:, 0, 0])
    assert np.allclose(steps, np.round(steps), atol=1e-3)
    assert built._lattice_geometry(sub) is not None

    with pytest.raises(ValueError):
        GridScan.commensurate(built)
    with pytest.raises(ValueError):
        GridScan.commensurate(built, gpts=8, sampling=1.0)


@devices
def test_upsample_pixelated_detector_is_stitched(device):
    # a pixelated detector straddles the blend angle by construction, so the
    # pattern is stitched on the full angular grid: the interpolated reduction
    # below the cut, the plane-wave (PRISM) reduction scattered onto its
    # lattice pixels at and above it — exact, with zeros in between
    from abtem.prism.s_matrix import _FullGridPixelatedDetector

    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    kwargs = dict(potential=potential, energy=100e3, semiangle_cutoff=20,
                  interpolation=2, device=device)
    prism = SMatrix(**kwargs).build(lazy=False)
    built = SMatrix(**kwargs, upsample=True, tolerance=1e-4,
                    window_gpts=32).build(lazy=False)

    scan = GridScan.commensurate(built, gpts=4)
    detector = abtem.PixelatedDetector(max_angle=None)
    cut = 40.0
    assert built._routing_sides(cut, [detector]) == ["pattern"]

    stitched = built.reduce(scan=scan, detectors=detector, blend_angle=cut)
    stitched_array = _array(stitched)
    assert stitched_array.shape[-2:] == tuple(built.gpts)
    assert np.allclose(stitched_array.sum((-2, -1)), 1.0, atol=0.05)

    reference = prism.reduce(scan=scan, detectors=detector)
    reference_array = _array(reference)
    factor = int(round(reference.angular_sampling[0]
                       / stitched.angular_sampling[0]))
    size, period = built.gpts[0], reference_array.shape[-1]
    index = size // 2 + factor * (np.arange(period) - period // 2)
    coarse = (np.arange(period) - period // 2) * reference.angular_sampling[0]
    above = np.hypot(coarse[:, None], coarse[None, :]) >= cut

    lattice_pixels = stitched_array[..., index[:, None], index[None, :]]
    error = (np.abs(lattice_pixels - reference_array)[..., above].max()
             / reference_array[..., above].max())
    assert error < 1e-3

    fine = (np.arange(size) - size // 2) * stitched.angular_sampling[0]
    theta = np.hypot(fine[:, None], fine[None, :])
    off_lattice = np.ones((size, size), bool)
    off_lattice[np.ix_(index, index)] = False
    assert np.abs(stitched_array[..., off_lattice & (theta >= cut)]).max() == 0.0

    # below the cut the stitched pattern is the interpolated reduction,
    # detected on the padded grid
    padded = _FullGridPixelatedDetector(detector, tuple(built.gpts))
    plain = built.reduce(scan=scan, detectors=padded, blend_angle=0.0)
    assert plain.angular_sampling == stitched.angular_sampling
    below = theta < cut
    assert np.allclose(_array(plain)[..., below], stitched_array[..., below])


@devices
def test_upsample_pixelated_patterns_use_the_simulation_grid(device):
    # a windowed reduction detects diffraction patterns on the full
    # simulation grid whatever the blend angle covers: the window is an
    # internal accuracy device, and its coarser reciprocal sampling is not
    # one the user asked for. A detector collecting entirely below the cut
    # used to return window-grid patterns instead, which cannot be binned
    # onto a multislice comparison.
    potential = _small_potential(repetitions=(2, 2, 6), device=device)
    built = SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=2, upsample=True, tolerance=1e-4, window_gpts=32,
        device=device,
    ).build(lazy=False)
    assert tuple(built.window_gpts) != tuple(built.gpts)

    scan = GridScan.commensurate(built, gpts=4)
    reference = built._with_window(tuple(built.gpts)).reduce(
        scan=scan, detectors=abtem.PixelatedDetector(max_angle=None))

    cut = 60.0
    for max_angle in (None, 30.0):  # below the cut, and straddling it
        detector = abtem.PixelatedDetector(max_angle=max_angle)
        assert built._routing_sides(cut, [detector]) == ["pattern"]
        patterns = built.reduce(scan=scan, detectors=detector, blend_angle=cut)
        assert np.allclose(patterns.angular_sampling,
                           reference.angular_sampling)

    # nothing lies above a cut beyond the detected angles, so the pattern is
    # the interpolated branch alone — on the simulation grid all the same
    beyond = built.reduce(scan=scan, detectors=abtem.PixelatedDetector(),
                          blend_angle=1e4)
    assert np.allclose(beyond.angular_sampling, reference.angular_sampling)
