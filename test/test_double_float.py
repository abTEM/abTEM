"""Tests for emulated double precision on Apple Metal (``abtem.core._double_float``).

The kernels need a real Metal device, so these are skipped everywhere else,
including torch's CPU device (``ABTEM_TORCH__DEVICE=cpu``), which cannot compile
Metal shaders. Accuracy is asserted against NumPy's float64 at tolerances far
below float32's ~1e-7, which is the point of the module.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import requires_mps

import abtem
from abtem.core import backend, config

pytestmark = [
    *requires_mps,
    pytest.mark.skipif(
        str(config.get("torch.device")).lower() != "mps",
        reason="Metal shaders need the real MPS device",
    ),
]


@pytest.fixture
def double_float():
    from abtem.core import _double_float

    return _double_float


def _random_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _relative_error(x, reference):
    return np.abs(x - reference).max() / np.abs(reference).max()


def test_round_trip_keeps_float64_precision(double_float):
    rng = np.random.default_rng(0)
    x = _random_complex(rng, (3, 16, 16))

    y = double_float.to_numpy(double_float.from_numpy(x))

    # A float32 round trip is off by ~6e-8 relative, here ~1e-15.
    assert _relative_error(y, x) < 1e-14


@pytest.mark.parametrize(
    "shape",
    [(8, 8), (3, 8, 16), (2, 64, 128), (1, 256, 256), (4, 24, 24), (2, 45, 100)],
)
def test_fft2_matches_numpy_float64(double_float, shape):
    rng = np.random.default_rng(1)
    x = _random_complex(rng, shape)
    # The input the kernels actually see: x rounded to a pair of float32.
    seen = double_float.to_numpy(double_float.from_numpy(x))

    forward = double_float.to_numpy(double_float.fft2(double_float.from_numpy(x)))

    assert _relative_error(forward, np.fft.fft2(seen)) < 1e-13


@pytest.mark.parametrize(
    "length",
    [2, 3, 5, 6, 7, 9, 10, 12, 15, 21, 25, 35, 49, 60, 120, 245, 343, 360, 500, 1000]
    + [1024, 1536, 2048],
)
def test_fft2_is_accurate_for_every_supported_length(double_float, length):
    rng = np.random.default_rng(length)
    # Both axes get the length in turn, so the row and the column kernels run.
    for shape in [(2, 4, length), (2, length, 4)]:
        x = _random_complex(rng, shape)
        seen = double_float.to_numpy(double_float.from_numpy(x))

        out = double_float.to_numpy(double_float.fft2(double_float.from_numpy(x)))

        assert _relative_error(out, np.fft.fft2(seen)) < 1e-13


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 2560), (2, 2560, 4), (1, 8, 4096), (1, 4096, 8), (1, 2100, 64)],
)
@pytest.mark.filterwarnings("ignore::abtem.core.fft.SlowFFTSizeWarning")
def test_axes_beyond_one_threadgroup_use_a_four_step_transform(double_float, shape):
    rng = np.random.default_rng(sum(shape))
    x = _random_complex(rng, shape)
    seen = double_float.to_numpy(double_float.from_numpy(x))
    transformed = double_float.fft2(double_float.from_numpy(x))

    out = double_float.to_numpy(transformed)
    back = double_float.to_numpy(double_float.fft2(transformed, inverse=True))

    assert _relative_error(out, np.fft.fft2(seen)) < 1e-13
    assert _relative_error(back, seen) < 1e-13


@pytest.mark.parametrize("shape", [(4, 32, 32), (2, 128, 64), (2, 45, 100)])
def test_inverse_fft2_undoes_fft2(double_float, shape):
    rng = np.random.default_rng(2)
    x = double_float.from_numpy(_random_complex(rng, shape))

    back = double_float.fft2(double_float.fft2(x), inverse=True)

    assert (
        _relative_error(double_float.to_numpy(back), double_float.to_numpy(x)) < 1e-13
    )


@pytest.mark.parametrize("shape", [(3, 16, 24), (2, 45, 8), (2, 8, 2560), (1, 2560, 8)])
def test_fft2_applies_its_factors_like_separate_products(double_float, shape):
    rng = np.random.default_rng(sum(shape))
    x = double_float.from_numpy(_random_complex(rng, shape))
    pre = double_float.from_numpy(_random_complex(rng, shape[-2:]))
    post = double_float.from_numpy(_random_complex(rng, shape[-2:]))

    fused = double_float.fft2(x, pre=pre, post=post)
    separate = double_float.multiply(
        double_float.fft2(double_float.multiply(x, pre)), post
    )

    assert (
        _relative_error(double_float.to_numpy(fused), double_float.to_numpy(separate))
        < 1e-13
    )


def test_fft2_rejects_a_factor_of_the_wrong_shape(double_float):
    x = double_float.from_numpy(np.ones((2, 8, 8), dtype=np.complex128))
    factor = double_float.from_numpy(np.ones((8, 4), dtype=np.complex128))

    with pytest.raises(ValueError, match="does not fit"):
        double_float.fft2(x, pre=factor)


def test_fft2_leaves_its_input_untouched(double_float):
    rng = np.random.default_rng(3)
    x = double_float.from_numpy(_random_complex(rng, (2, 16, 16)))
    before = double_float.to_numpy(x)

    double_float.fft2(x)

    np.testing.assert_array_equal(double_float.to_numpy(x), before)


def test_multiply_broadcasts_a_kernel_over_the_batch(double_float):
    rng = np.random.default_rng(4)
    x = _random_complex(rng, (5, 32, 16))
    kernel = _random_complex(rng, (32, 16))
    xd, kd = double_float.from_numpy(x), double_float.from_numpy(kernel)

    product = double_float.to_numpy(double_float.multiply(xd, kd))

    seen_x, seen_k = double_float.to_numpy(xd), double_float.to_numpy(kd)
    assert _relative_error(product, seen_x * seen_k) < 1e-13


@pytest.mark.parametrize("shape", [(11, 8), (16, 13), (1, 8), (8, 22)])
def test_unsupported_sizes_raise(double_float, shape):
    x = double_float.from_numpy(np.ones(shape, dtype=np.complex128))

    with pytest.raises(ValueError, match="no prime factor above 7"):
        double_float.fft2(x)


def test_axes_too_long_to_split_in_two_raise(double_float):
    # 2**23 = 2048 * 4096: one of any two factors exceeds a threadgroup.
    with pytest.raises(ValueError, match="no prime factor above 7"):
        double_float._check_axis(2**23)
    double_float._check_axis(2**22)


def test_arrays_too_large_to_address_raise(double_float):
    class Huge:
        shape = (2**10, 2**10, 2**10)

    with pytest.raises(ValueError, match="smaller chunks"):
        double_float._check_size(Huge())


def _exit_wave(emulate, gpts):
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 8)
    with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
        potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=1.0)
        wave = abtem.PlaneWave(energy=200e3)
        wave.grid.match(potential)
        return np.asarray(wave.multislice(potential).compute().array)


@pytest.mark.parametrize("gpts", [64, 60, 72, 75])
def test_emulated_multislice_matches_float64_on_the_cpu(gpts):
    reference = _exit_wave(False, gpts)
    emulated = _exit_wave(True, gpts)

    assert emulated.dtype == np.complex128
    # The float32 multislice is off by ~1e-6 at this depth; emulated double
    # precision stays near float64 rounding.
    assert np.abs(emulated - reference).max() < 1e-10


def test_emulated_multislice_matches_float64_with_a_detector():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 8)

    def measure(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=64, slice_thickness=1.0)
            probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
            probe.grid.match(potential)
            scan = abtem.GridScan(
                start=(0, 0), end=(2, 2), gpts=(2, 2), potential=potential
            )
            detector = abtem.PixelatedDetector(max_angle=40)
            return np.asarray(probe.scan(potential, scan, detector).compute().array)

    reference, emulated = measure(False), measure(True)

    assert np.abs(emulated - reference).max() < 1e-10 * np.abs(reference).max()


def test_the_emulation_does_nothing_in_single_precision():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    with config.set({"precision": "float32", "torch.emulate-float64": True}):
        assert not backend.emulates_float64()
        potential = abtem.Potential(atoms, gpts=32, slice_thickness=1.0)
        wave = abtem.PlaneWave(energy=200e3)
        wave.grid.match(potential)

        result = wave.multislice(potential).compute()

    assert result.array.dtype == np.complex64


def _multislice(gpts, algorithm=None, device="cpu"):
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 2)
    potential = abtem.Potential(atoms, gpts=gpts, slice_thickness=1.0, device=device)
    wave = abtem.PlaneWave(energy=200e3, device=device)
    wave.grid.match(potential)
    kwargs = {} if algorithm is None else {"algorithm": algorithm}
    return np.asarray(wave.multislice(potential, **kwargs).compute().array)


@pytest.mark.parametrize(
    "gpts, algorithm, reason",
    [
        ((74, 74), None, "prime factor above 7"),  # 74 = 2 * 37
        (
            (64, 64),
            abtem.multislice.RealSpaceMultislice(),
            "only the Fourier multislice",
        ),
        (
            (64, 64),
            abtem.multislice.FourierMultislice(transpose=True),
            "transpose",
        ),
    ],
)
def test_double_precision_on_mps_runs_on_the_host_where_the_gpu_cannot(
    gpts, algorithm, reason
):
    from abtem.core.fft import SlowFFTSizeWarning, _warned_slow_fft_shapes

    with config.set({"precision": "float64", "device": "cpu"}):
        reference = _multislice(gpts, algorithm)

    _warned_slow_fft_shapes.clear()
    with config.set({"precision": "float64", "device": "mps"}):
        with pytest.warns(SlowFFTSizeWarning, match=reason):
            emulated = _multislice(gpts, algorithm, device="mps")

    # The host's float64, so the same result.
    np.testing.assert_allclose(emulated, reference, rtol=0, atol=1e-12)


def test_emulated_potential_matches_float64_on_the_host():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 4)

    def build(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=96, slice_thickness=1.0)
            return np.asarray(potential.build().compute().array)

    reference, emulated = build(False), build(True)

    assert emulated.dtype == np.float64
    assert np.abs(emulated - reference).max() < 1e-12 * np.abs(reference).max()


def test_emulated_potential_keeps_its_cache_out_of_the_pickle():
    import pickle

    from abtem.integrals import ScatteringFactorProjectionIntegrals

    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    integrator = ScatteringFactorProjectionIntegrals()
    with config.set({"precision": "float64", "torch.emulate-float64": True}):
        integrator.integrate_on_grid(atoms, 0.0, 1.0, (32, 32), (0.17, 0.17))
        assert len(integrator._emulated_kernels) == 1

        restored = pickle.loads(pickle.dumps(integrator))

    assert len(restored._emulated_kernels) == 0


def test_emulated_multislice_uses_a_transmission_function_as_given():
    # Not band-limited: the multislice must use it exactly as it is, as it does
    # on the host.
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 4)

    def exit_wave(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=64, slice_thickness=1.0)
            transmission = potential.build().compute().transmission_function(200e3)
            wave = abtem.PlaneWave(energy=200e3)
            wave.grid.match(potential)
            return np.asarray(wave.multislice(transmission).compute().array)

    reference, emulated = exit_wave(False), exit_wave(True)

    assert np.abs(emulated - reference).max() < 1e-10


def test_emulation_refuses_to_load_torch_after_pyfftw():
    # Loading PyTorch after pyfftw's OpenMP runtime makes torch's CPU work crash
    # the process, so the emulation must refuse like the 'mps' device does,
    # without importing torch.
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        import pyfftw
        import ase.build, abtem
        from abtem.core import config
        config.set({"precision": "float64", "torch.emulate-float64": True})
        potential = abtem.Potential(
            ase.build.bulk("Si", "diamond", a=5.431, cubic=True),
            gpts=32, slice_thickness=1.0,
        )
        wave = abtem.PlaneWave(energy=200e3)
        wave.grid.match(potential)
        try:
            wave.multislice(potential).compute()
        except RuntimeError as error:
            assert "pyfftw" in str(error), error
            assert "torch" not in sys.modules
            print("refused")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 0, result.stderr
    assert "refused" in result.stdout


def test_the_emulated_fft_warns_on_a_slow_grid(double_float):
    from abtem.core.fft import SlowFFTSizeWarning, _warned_slow_fft_shapes

    _warned_slow_fft_shapes.clear()
    x = double_float.from_numpy(np.ones((1, 448, 448), dtype=np.complex128))

    with pytest.warns(SlowFFTSizeWarning, match="448 x 448"):
        double_float.fft2(x)


def _count_fft2(monkeypatch, double_float):
    calls = []
    original = double_float.fft2

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(double_float, "fft2", counted)
    return calls


@pytest.mark.parametrize("double_channel", [False, True])
def test_emulated_core_loss_multislice_matches_float64_on_the_host(
    double_channel, monkeypatch
):
    import ase
    from utils import synthetic_transition_potential

    from abtem.multislice import _EmulatedDoubleMultislice

    atoms = ase.Atoms(
        "Si2", positions=[(2.0, 2.0, 1.0), (5.0, 5.0, 3.0)], cell=(8, 8, 6), pbc=True
    )
    gpts = (64, 64)
    forks = []
    original_fork = _EmulatedDoubleMultislice.fork

    def fork(self, waves):
        forks.append(1)
        return original_fork(self, waves)

    monkeypatch.setattr(_EmulatedDoubleMultislice, "fork", fork)

    def run(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(
                atoms, gpts=gpts, slice_thickness=1.0, exit_planes=2
            )
            transition = synthetic_transition_potential(
                Z=14, gpts=gpts, extent=(8.0, 8.0), energy=100e3
            )
            waves = abtem.PlaneWave(energy=100e3, gpts=gpts, extent=(8.0, 8.0)).build(
                lazy=False
            )
            result = waves.transition_potential_multislice(
                potential,
                transition,
                detectors=abtem.WavesDetector(),
                double_channel=double_channel,
            )
            return np.asarray(result.compute().array)

    reference, emulated = run(False), run(True)

    assert emulated.dtype == np.complex128
    assert np.abs(emulated - reference).max() < 1e-12 * np.abs(reference).max()
    # The scattered waves of the double channel go on through the slices on the
    # device, one fork per site batch.
    assert bool(forks) == double_channel


@pytest.mark.parametrize("shape", [(4, 128, 128), (1, 256, 256), (3, 160, 192)])
def test_host_ffts_run_on_the_gpu_when_large_enough(double_float, monkeypatch, shape):
    from abtem.core.fft import fft2, fft2_convolve, ifft2

    rng = np.random.default_rng(sum(shape))
    x = _random_complex(rng, shape)
    kernel = np.exp(1j * rng.standard_normal((1, *shape[-2:])))
    calls = _count_fft2(monkeypatch, double_float)

    with config.set({"precision": "float64", "torch.emulate-float64": True}):
        forward, inverse = fft2(x), ifft2(x)
        convolved = fft2_convolve(x, kernel)

    # fft2 and ifft2 are one transform each, the convolution is two.
    assert len(calls) == 1 + 1 + 2
    assert _relative_error(forward, np.fft.fft2(x)) < 1e-13
    assert _relative_error(inverse, np.fft.ifft2(x)) < 1e-13
    reference = np.fft.ifft2(np.fft.fft2(x) * kernel)
    assert _relative_error(convolved, reference) < 1e-13


@pytest.mark.parametrize(
    "shape, kwargs",
    [
        ((1, 64, 64), {}),  # too small for the round trip to pay
        ((16, 64, 64), {}),  # too little in all, though many images
        ((2, 192, 192), {"axes": (0, 1)}),  # asked for other axes
        ((2, 448, 448), {}),  # factors of 7, where the emulation is slow
        ((2, 11, 22000), {}),  # lengths it cannot do
    ],
)
def test_host_ffts_stay_on_the_host_where_the_gpu_does_not_pay(
    double_float, monkeypatch, shape, kwargs
):
    from abtem.core.fft import fft2

    x = _random_complex(np.random.default_rng(0), shape)
    calls = _count_fft2(monkeypatch, double_float)

    with config.set({"precision": "float64", "torch.emulate-float64": True}):
        fft2(x, **kwargs)

    assert not calls


def test_host_ffts_are_untouched_without_the_option(double_float, monkeypatch):
    from abtem.core.fft import fft2

    calls = _count_fft2(monkeypatch, double_float)
    x = _random_complex(np.random.default_rng(0), (4, 256, 256))

    with config.set({"precision": "float64", "torch.emulate-float64": False}):
        fft2(x)

    assert not calls


@pytest.mark.parametrize(
    "batch, factor_batch",
    [((2, 3), (3,)), ((2, 3), (2, 1)), ((2, 3), (2, 3)), ((4,), (1, 4)), ((3,), ())],
)
def test_fft2_applies_a_factor_per_image_of_the_batch(
    double_float, batch, factor_batch
):
    rng = np.random.default_rng(len(batch) + len(factor_batch))
    n0, n1 = 12, 20
    x = _random_complex(rng, (*batch, n0, n1))
    factor = _random_complex(rng, (*factor_batch, n0, n1))
    xd, fd = double_float.from_numpy(x), double_float.from_numpy(factor)

    out = double_float.to_numpy(double_float.fft2(xd, post=fd))

    seen_x, seen_f = double_float.to_numpy(xd), double_float.to_numpy(fd)
    assert _relative_error(out, np.fft.fft2(seen_x) * seen_f) < 1e-13


def test_fft2_rejects_a_factor_that_does_not_broadcast_over_the_batch(double_float):
    x = double_float.from_numpy(np.ones((2, 3, 8, 8), dtype=np.complex128))
    factor = double_float.from_numpy(np.ones((4, 8, 8), dtype=np.complex128))

    with pytest.raises(ValueError, match="broadcast"):
        double_float.fft2(x, post=factor)


@pytest.mark.parametrize(
    "tilt", [([0.0, 5.0, 0.0], [0.0, 0.0, 8.0]), ([0.0, 4.0], 0.0)]
)
def test_emulated_multislice_handles_a_tilt_axis(tilt):
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 4)

    def exit_wave(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            potential = abtem.Potential(atoms, gpts=64, slice_thickness=1.0)
            wave = abtem.PlaneWave(energy=200e3, tilt=tilt)
            wave.grid.match(potential)
            return np.asarray(wave.multislice(potential).compute().array)

    reference, emulated = exit_wave(False), exit_wave(True)

    assert emulated.ndim >= 3
    assert np.abs(emulated - reference).max() < 1e-10


def test_emulated_multislice_runs_on_the_processes_of_a_distributed_cluster():
    # The option has to reach the worker processes, and each of them drives its
    # own Metal context from two threads.
    distributed = pytest.importorskip("distributed")
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (1, 1, 2)

    def scan(emulate):
        with config.set(
            {
                "precision": "float64",
                "torch.emulate-float64": emulate,
                "dask.chunk-size": "128 kB",
            }
        ):
            potential = abtem.Potential(
                abtem.FrozenPhonons(atoms, num_configs=3, sigmas=0.05, seed=1),
                gpts=64,
                slice_thickness=1.0,
            )
            probe = abtem.Probe(energy=200e3, semiangle_cutoff=20)
            probe.grid.match(potential)
            grid = abtem.GridScan(
                start=(0, 0), end=(4, 4), gpts=(4, 4), potential=potential
            )
            measurement = probe.scan(potential, grid, abtem.PixelatedDetector(40))
            return np.asarray(measurement.compute().array)

    reference = scan(False)
    with (
        distributed.LocalCluster(
            n_workers=2, threads_per_worker=2, processes=True, dashboard_address=":0"
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        emulated = scan(True)
        loaded = list(client.run(lambda: "torch" in __import__("sys").modules).values())

    assert any(loaded), "no worker ran the Metal kernels"
    assert np.abs(emulated - reference).max() < 1e-12 * np.abs(reference).max()


def test_bloch_waves_run_on_the_host_with_the_option_on():
    # Bloch waves have no multislice and their linear algebra is faster on the
    # host's BLAS and LAPACK than it could be in double-float on the GPU, so the
    # option leaves them alone: the same float64 result, to the last bit.
    atoms = bulk("Si", cubic=True)

    def exit_waves(emulate):
        with config.set({"precision": "float64", "torch.emulate-float64": emulate}):
            bloch = abtem.BlochWaves(atoms, energy=200e3, sg_max=0.1, g_max=3.0)
            waves = bloch.calculate_exit_waves(
                thicknesses=[50.0, 100.0], gpts=(48, 48), lazy=False
            )
            patterns = bloch.calculate_diffraction_patterns(
                thicknesses=[50.0, 100.0], lazy=False
            )
            return np.asarray(waves.array), np.asarray(patterns.array)

    waves_host, patterns_host = exit_waves(False)
    waves_gpu, patterns_gpu = exit_waves(True)

    assert waves_gpu.dtype == np.complex128
    np.testing.assert_allclose(waves_gpu, waves_host, rtol=0, atol=1e-12)
    np.testing.assert_allclose(patterns_gpu, patterns_host, rtol=0, atol=1e-12)
