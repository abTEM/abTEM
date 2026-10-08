"""Tests for the experimental Metal (MPS) backend on Apple silicon.

Skipped unless PyTorch is installed on Apple silicon, or the backend is pointed
at torch's CPU device. Nothing needs enabling on a Mac: PyTorch is imported on
the first use of the 'mps' device.

On any other machine with PyTorch installed, ``ABTEM_TORCH__DEVICE=cpu`` runs the
backend's layer (array wrapper, dispatch registries, dtype narrowing, the lock)
on torch's CPU device. Select every test that exercises the backend, here and in
the device-parametrized tests of the other files (their ``[torch]`` cases), with
``-m torch``::

    ABTEM_TORCH__DEVICE=cpu pytest test -m torch

CuPy takes precedence where both are present, so hide the GPU
(``CUDA_VISIBLE_DEVICES=``) on a machine that has one. Metal's rules still apply
on the CPU device (single precision only), so most failures there also fail on a
Mac; Metal kernel numerics and its thread safety are not exercised.

Metal is single precision, so every comparison against the CPU reference is made
at float32 tolerances rather than exactly.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from ase.build import bulk
from utils import requires_mps

import abtem
from abtem.core.backend import (
    asnumpy,
    copy_to_device,
    device_name_from_array_module,
    get_array_module,
)

pytestmark = requires_mps

# Loading torch's OpenMP runtime ahead of pyfftw's is only done on macOS on Apple
# silicon (`_preload_torch_openmp`), so nothing can be asserted of it elsewhere.
macos_only = pytest.mark.skipif(
    sys.platform != "darwin", reason="concerns macOS's libomp.dylib handling"
)


@pytest.fixture
def atoms():
    return bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 3)


@pytest.fixture(autouse=True)
def eager():
    with abtem.config.set({"dask.lazy": False, "precision": "float32"}):
        yield


def test_array_module_round_trip():
    xp = get_array_module("mps")

    assert device_name_from_array_module(xp) == "mps"
    assert get_array_module("metal") is xp
    assert get_array_module("torch") is xp

    array = np.random.RandomState(0).randn(4, 5).astype(np.float32)
    on_device = copy_to_device(array, "mps")

    assert get_array_module(on_device) is xp
    assert on_device.dtype == np.float32
    assert on_device.shape == array.shape
    assert np.array_equal(asnumpy(on_device), array)
    assert np.array_equal(copy_to_device(on_device, "cpu"), array)


def test_scalar_keeps_zero_dimensions():
    # np.ascontiguousarray promotes a 0-d scalar to shape (1,); a spurious
    # dimension there propagates into every broadcast against it.
    xp = get_array_module("mps")

    assert xp.asarray(20.0).shape == ()
    assert xp.asarray(np.float32(20.0)).shape == ()
    assert xp.asarray([20.0]).shape == (1,)


def test_inferred_double_precision_is_narrowed():
    # A Python float is float64 to NumPy, but Metal has no float64; an inferred
    # dtype narrows rather than failing.
    xp = get_array_module("mps")

    assert xp.asarray((0.1, 0.2)).dtype == np.float32

    with pytest.raises(RuntimeError, match="single-precision"):
        xp.asarray(np.zeros(4), dtype=np.float64)


def test_double_precision_configuration_is_rejected():
    with abtem.config.set({"precision": "float64"}):
        with pytest.raises(RuntimeError, match="single-precision"):
            get_array_module("mps")


def test_metal_device_rejects_double_precision_when_configured():
    # The combination is refused where it is set, not later inside a
    # computation, and a refused set leaves the configuration untouched.
    before = (abtem.config.get("device"), abtem.config.get("precision"))

    with pytest.raises(ValueError, match="single-precision"):
        abtem.config.set({"device": "mps", "precision": "float64"})

    assert (abtem.config.get("device"), abtem.config.get("precision")) == before

    with abtem.config.set({"device": "mps"}):
        with pytest.raises(ValueError, match="single-precision"):
            abtem.config.set({"precision": "float64"})

        assert abtem.config.get("precision") == "float32"

    assert (abtem.config.get("device"), abtem.config.get("precision")) == before


def test_unsupported_operation_names_itself():
    xp = get_array_module("mps")

    with pytest.raises(AttributeError, match="not_a_real_ufunc"):
        xp.not_a_real_ufunc


def test_fft_round_trip():
    array = (np.random.RandomState(1).randn(2, 32, 32) * (1 + 1j)).astype(np.complex64)

    on_device = copy_to_device(array, "mps")
    restored = asnumpy(abtem.core.fft.ifft2(abtem.core.fft.fft2(on_device)))

    assert np.allclose(restored, array, atol=1e-4)
    assert np.allclose(
        asnumpy(abtem.core.fft.fft2(on_device)), np.fft.fft2(array), atol=1e-3
    )


def test_scatter_add_matches_numpy():
    xp = get_array_module("mps")
    rng = np.random.RandomState(2)

    rows = rng.randint(0, 16, 32)
    cols = rng.randint(0, 16, 32)
    values = rng.randn(32).astype(np.float32)

    expected = np.zeros((16, 16), dtype=np.float32)
    np.add.at(expected, (rows, cols), values)

    result = xp.zeros((16, 16), dtype=np.float32)
    xp.add.at(result, (xp.asarray(rows), xp.asarray(cols)), xp.asarray(values))

    assert np.allclose(asnumpy(result), expected, atol=1e-5)


def test_potential_matches_cpu(atoms):
    arrays = [
        asnumpy(abtem.Potential(atoms, gpts=128, device=device).build().array)
        for device in ("cpu", "mps")
    ]

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_plane_wave_multislice_matches_cpu(atoms):
    arrays = []
    for device in ("cpu", "mps"):
        potential = abtem.Potential(atoms, gpts=128, device=device)
        waves = abtem.PlaneWave(energy=100e3, device=device).multislice(potential)
        arrays.append(asnumpy(waves.array))

    assert arrays[1].dtype == np.complex64
    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_stem_scan_matches_cpu(atoms):
    arrays = []
    for device in ("cpu", "mps"):
        potential = abtem.Potential(atoms, gpts=128, device=device)
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device=device)
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(3, 3))
        detector = abtem.AnnularDetector(inner=50, outer=150)
        arrays.append(
            asnumpy(probe.scan(potential, scan=scan, detectors=detector).array)
        )

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_angle_and_round_take_their_second_argument_positionally():
    # numpy.angle(z, deg) and numpy.round(a, decimals) both accept a second
    # positional argument that torch either spells as a keyword or does not
    # take at all; dask's own angle() passes deg positionally.
    xp = get_array_module("mps")
    rng = np.random.RandomState(3)

    z = (rng.randn(4, 5) + 1j * rng.randn(4, 5)).astype(np.complex64)
    on_device = copy_to_device(z, "mps")

    assert np.allclose(asnumpy(xp.angle(on_device)), np.angle(z), atol=1e-5)
    assert np.allclose(asnumpy(xp.angle(on_device, True)), np.angle(z, True), atol=1e-3)

    x = rng.randn(6).astype(np.float32)
    assert np.allclose(asnumpy(xp.round(copy_to_device(x, "mps"), 2)), np.round(x, 2))


def test_lazy_phase_matches_cpu():
    arrays = []
    for device in ("cpu", "mps"):
        with abtem.config.set({"dask.lazy": True}):
            probe = abtem.Probe(
                energy=100e3, semiangle_cutoff=20, gpts=64, extent=10, device=device
            )
            arrays.append(asnumpy(probe.build().phase().compute().array))

    # Compared as a wrapped difference: a probe's phase sits on the branch cut
    # over much of the plane, where a float32 rounding either way flips the
    # value between +pi and -pi.
    difference = np.angle(np.exp(1j * (arrays[0] - arrays[1])))

    assert np.abs(difference).max() < 1e-3


@pytest.mark.parametrize("lazy_unit", [False, True])
def test_crystal_potential_from_built_unit_matches_cpu(atoms, lazy_unit):
    # A unit potential the caller built themselves is used as-is; built
    # lazily, its array is a dask array rather than one of the device's own.
    arrays = []
    for device in ("cpu", "mps"):
        with abtem.config.set({"dask.lazy": True}):
            unit = abtem.Potential(atoms, gpts=64, device=device).build(lazy=lazy_unit)
            crystal = abtem.CrystalPotential(unit, repetitions=(2, 2, 2))
            waves = abtem.PlaneWave(energy=100e3, device=device).multislice(crystal)
            arrays.append(asnumpy(waves.compute().array))

    assert np.allclose(*arrays, atol=1e-3 * np.abs(arrays[0]).max())


def test_where_without_x_and_y():
    # numpy.where(condition) is numpy.nonzero(condition); torch.where's
    # one-argument form is spelled the same way but the wrapper required all
    # three. Reached from e.g. LineProfiles.width, via a sign-change search.
    xp = get_array_module("mps")

    array = copy_to_device(np.array([0.0, 1.0, 0.0, 2.0, 3.0], np.float32), "mps")
    (indices,) = xp.where(array > 0.5)

    assert np.array_equal(asnumpy(indices), np.array([1, 3, 4]))

    with pytest.raises(ValueError, match="both or neither"):
        xp.where(array > 0.5, array)


def test_line_profile_width_matches_cpu():
    widths = []
    for device in ("cpu", "mps"):
        probe = abtem.Probe(
            energy=100e3, semiangle_cutoff=20, gpts=128, extent=20, device=device
        )
        profile = (
            probe.build()
            .intensity()
            .interpolate_line_at_position(center=(10, 10), angle=0, extent=10)
        )
        widths.append(float(asnumpy(profile.width(height=0.5))))

    assert widths[1] == pytest.approx(widths[0], rel=1e-4)


@pytest.mark.parametrize(
    "dtypes",
    [(np.complex64, np.float32), (np.float32, np.bool_), (np.float32, np.int64)],
    ids=["complex-float", "float-bool", "float-int"],
)
def test_contractions_promote_mixed_dtypes(dtypes):
    # torch's elementwise operators promote the way NumPy's do, but its
    # contractions refuse mismatched operands outright.
    xp = get_array_module("mps")
    rng = np.random.RandomState(4)
    a = rng.randn(3, 4) + (1j * rng.randn(3, 4) if dtypes[0] is np.complex64 else 0)
    a = a.astype(dtypes[0])
    b = (rng.randn(4, 5) > 0).astype(dtypes[1])
    a_dev, b_dev = copy_to_device(a, "mps"), copy_to_device(b, "mps")

    expected = a @ b
    results = {
        "@": a_dev @ b_dev,
        "host @ device": a @ b_dev,
        "np.matmul": np.matmul(a_dev, b_dev),
        "xp.matmul": xp.matmul(a_dev, b_dev),
        "xp.dot": xp.dot(a_dev, b_dev),
        "xp.tensordot": xp.tensordot(a_dev, b_dev, axes=([1], [0])),
        "xp.einsum": xp.einsum("ij,jk->ik", a_dev, b_dev),
    }
    for name, result in results.items():
        assert np.allclose(asnumpy(result), expected, atol=1e-5), name


def test_diag_and_fill_diagonal_match_numpy():
    xp = get_array_module("mps")
    matrix = np.arange(16, dtype=np.float32).reshape(4, 4)
    values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    on_device = copy_to_device(matrix, "mps")
    assert np.array_equal(asnumpy(xp.diag(on_device)), np.diag(matrix))
    assert np.array_equal(asnumpy(np.diag(on_device)), np.diag(matrix))
    assert np.array_equal(asnumpy(xp.diag(xp.asarray(values))), np.diag(values))

    expected = matrix.copy()
    np.fill_diagonal(expected, values)
    xp.fill_diagonal(on_device, xp.asarray(values))
    assert np.array_equal(asnumpy(on_device), expected)

    np.fill_diagonal(expected, 0.0)
    np.fill_diagonal(on_device, 0.0)
    assert np.array_equal(asnumpy(on_device), expected)


def test_flip_and_broadcasting_match_numpy():
    xp = get_array_module("mps")
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    on_device = copy_to_device(array, "mps")

    for axis in (None, 0, -1, (0, 2)):
        assert np.array_equal(
            asnumpy(xp.flip(on_device, axis=axis)), np.flip(array, axis)
        )
        assert np.array_equal(asnumpy(np.flip(on_device, axis)), np.flip(array, axis))

    column = np.arange(3, dtype=np.float32)[:, None]
    row = np.arange(4, dtype=np.int64)[None, :]
    expected = np.broadcast_arrays(column, row)
    for result in (
        xp.broadcast_arrays(xp.asarray(column), xp.asarray(row)),
        np.broadcast_arrays(xp.asarray(column), xp.asarray(row)),
    ):
        assert len(result) == len(expected)
        for got, want in zip(result, expected):
            assert get_array_module(got) is xp
            assert np.array_equal(asnumpy(got), want)

    assert np.array_equal(
        asnumpy(np.broadcast_to(xp.asarray(column), (3, 5))),
        np.broadcast_to(column, (3, 5)),
    )


@pytest.mark.parametrize("mode", ["constant", "wrap", "reflect", "symmetric"])
def test_pad_matches_numpy(mode):
    xp = get_array_module("mps")
    array = np.arange(15, dtype=np.float32).reshape(3, 5)
    # widths beyond the axis length exercise the folding of each mode
    pad_width = ((4, 1), (0, 7))

    result = xp.pad(xp.asarray(array), pad_width, mode=mode)

    assert np.array_equal(asnumpy(result), np.pad(array, pad_width, mode=mode))


def test_minimum_and_maximum_take_a_scalar():
    # NumPy accepts a scalar on either side; torch's binary functions do not.
    xp = get_array_module("mps")
    indices = np.array([0, 3, 7, 9])
    on_device = xp.asarray(indices)

    assert np.array_equal(
        asnumpy(xp.minimum(on_device + 1, 8)), np.minimum(indices + 1, 8)
    )
    assert np.array_equal(asnumpy(xp.maximum(2, on_device)), np.maximum(2, indices))


def test_dask_constant_boundary_overlap_stays_on_device():
    # A constant boundary pads with chunks dask builds from the array's meta
    # through np.full_like. Unimplemented, that raised TypeError, which dask's
    # curried creation wrapper took for missing arguments and returned a
    # partial function as the chunk.
    import dask.array as da

    xp = get_array_module("mps")
    array = np.random.RandomState(5).rand(2, 8, 8).astype(np.float32)
    lazy = da.from_array(xp.asarray(array), chunks=(1, 4, 8))

    result = da.overlap.overlap(
        lazy, depth={0: 0, 1: 2, 2: 2}, boundary={0: 1.5, 1: 1.5, 2: 1.5}
    ).compute(scheduler="synchronous")
    expected = da.overlap.overlap(
        da.from_array(array, chunks=(1, 4, 8)),
        depth={0: 0, 1: 2, 2: 2},
        boundary={0: 1.5, 1: 1.5, 2: 1.5},
    ).compute()

    assert get_array_module(result) is xp
    assert np.array_equal(asnumpy(result), expected)

    filled = np.full_like(xp.asarray(array), 2.0, shape=(3, 1), order="C")
    assert get_array_module(filled) is xp
    assert np.array_equal(asnumpy(filled), np.full((3, 1), 2.0, np.float32))


def test_diffraction_pattern_bilinear_resampling_matches_cpu():
    # The CPU routine writes into host buffers through `out=`; Metal has its
    # own gather-based path.
    patterns = []
    for device in ("cpu", "mps"):
        # a rectangular cell, so the two axes resample by different factors
        probe = abtem.Probe(
            energy=100e3,
            semiangle_cutoff=20,
            gpts=(64, 80),
            extent=(10, 13),
            device=device,
        )
        diffraction = probe.build().diffraction_patterns(max_angle=None)
        resampled = diffraction.interpolate(sampling=0.137)
        patterns.append(asnumpy(resampled.array))

    assert patterns[1].shape == patterns[0].shape
    np.testing.assert_allclose(
        patterns[1], patterns[0], rtol=0, atol=1e-5 * np.abs(patterns[0]).max()
    )


def _run_isolated(script, hang="the script hung"):
    """Run ``script`` in a fresh interpreter, for what only a new process shows.

    Library load order is fixed once per process, and a crash or deadlock here
    would take the test session down with it.
    """
    try:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(hang)


def test_importing_abtem_does_not_import_torch():
    # PyTorch costs about two seconds and 180 MB to import; abTEM defers it to
    # the first use of the 'mps' device.
    completed = _run_isolated(
        """
        import sys
        import abtem
        assert "torch" not in sys.modules
        """
    )

    assert completed.returncode == 0, completed.stderr


@macos_only
def test_torch_openmp_runtime_is_loaded_ahead_of_pyfftws():
    # torch and pyfftw each bundle libomp.dylib, and torch's has to initialize
    # first; importing abTEM loads it, without torch, before pyfftw.
    completed = _run_isolated(
        """
        import ctypes
        import abtem

        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        images = [
            dyld._dyld_get_image_name(i).decode()
            for i in range(dyld._dyld_image_count())
        ]
        runtimes = [image for image in images if image.endswith("/libomp.dylib")]
        print(runtimes)
        assert runtimes and "/torch/lib/" in runtimes[0], runtimes
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_metal_after_threaded_cpu_ffts():
    # pyfftw's threaded FFTs run first and torch only arrives afterwards -- the
    # order that loading torch lazily creates. What then crashes, without
    # torch's OpenMP runtime loaded first, is torch work on the CPU: the dtype
    # cast asarray makes before uploading a double-precision host array, and
    # the CPU fallback torch takes for a large eigendecomposition. Metal-only
    # operations do not touch OpenMP and would pass either way.
    completed = _run_isolated(
        """
        import numpy as np
        import abtem

        abtem.config.set({"fftw.threads": 4})
        waves = abtem.PlaneWave(energy=100e3, gpts=128, extent=10, device="cpu")
        abtem.core.fft.fft2(waves.build(lazy=False).array)

        xp = abtem.core.backend.get_array_module("mps")
        assert xp.asarray(np.random.rand(2048, 2048)).shape == (2048, 2048)

        hermitian = np.random.rand(600, 600).astype(np.complex64)
        hermitian = hermitian + hermitian.conj().T
        xp.linalg.eigh(xp.asarray(hermitian))
        """
    )

    # a segfault shows up as a negative return code, with nothing on stderr
    assert completed.returncode == 0, (completed.returncode, completed.stderr)


@macos_only
def test_pyfftw_imported_before_abtem_is_refused_rather_than_crashing():
    # Too late to load torch's runtime first: refuse with a reason instead of
    # importing torch into a process where its operations would segfault.
    completed = _run_isolated(
        """
        import sys
        import pyfftw
        import abtem

        try:
            abtem.core.backend.get_array_module("mps")
        except RuntimeError as error:
            assert "pyfftw was imported before abTEM" in str(error)
        else:
            raise SystemExit("the Metal backend loaded after pyfftw")
        assert "torch" not in sys.modules
        """
    )

    assert completed.returncode == 0, (completed.returncode, completed.stderr)


def test_materializing_a_lazy_array_does_not_deadlock():
    # Every Metal operation holds _TORCH_LOCK. Materializing a lazy array whose
    # graph has Metal work of its own used to hand that graph to dask's
    # threaded scheduler from inside the lock, where the workers waited for
    # the lock forever. Run in a subprocess: a deadlock in this process would
    # leave the lock held, and hang every Metal test after this one too.
    script = textwrap.dedent(
        """
        import dask.array as da
        import numpy as np
        import abtem
        from abtem.core.backend import copy_to_device, get_array_module

        xp = get_array_module("mps")
        on_device = copy_to_device(np.ones((4, 8, 8), np.float32), "mps")
        lazy = da.from_array(on_device, chunks=(1, 8, 8)).map_blocks(
            xp.exp, meta=np.array((), np.float32)
        )
        assert xp.asarray(lazy).shape == (4, 8, 8)
        assert np.allclose(xp.asnumpy(lazy), np.e)
        """
    )
    completed = _run_isolated(
        script, hang="materializing a lazy Metal array deadlocked"
    )

    assert completed.returncode == 0, completed.stderr


def test_per_object_device_keeps_the_metal_scheduler(atoms, monkeypatch):
    # With the device given per object and the global one left at 'cpu', a
    # detector's host-resident result was labelled 'cpu', so its computation --
    # still full of Metal work -- went to dask's threaded scheduler.
    import abtem.array

    chosen = []
    resolve = abtem.array._resolve_mps_scheduler

    def spy(kwargs):
        kwargs = resolve(kwargs)
        chosen.append(kwargs.get("scheduler"))
        return kwargs

    monkeypatch.setattr(abtem.array, "_resolve_mps_scheduler", spy)

    with abtem.config.set({"device": "cpu", "dask.lazy": True}):
        potential = abtem.Potential(atoms, gpts=128, device="mps")
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20, device="mps")
        scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), gpts=(2, 2))
        result = probe.scan(
            potential, scan=scan, detectors=abtem.AnnularDetector(50, 150)
        )

        assert result.device == "mps"
        result.compute()

    assert chosen == ["synchronous"]


@pytest.mark.parametrize("threads_per_worker", [1, 2])
def test_metal_runs_on_a_suitable_distributed_client(threads_per_worker):
    # As for CUDA: a running client whose workers are each single-threaded is
    # left in charge of a Metal computation; anything else would drive the
    # device from several threads at once, so it gets the synchronous scheduler.
    distributed = pytest.importorskip("distributed")
    from abtem.array import _resolve_mps_scheduler

    with (
        distributed.LocalCluster(
            n_workers=1,
            threads_per_worker=threads_per_worker,
            processes=False,
            dashboard_address=":0",
        ) as cluster,
        distributed.Client(cluster),
    ):
        kwargs = _resolve_mps_scheduler({})

    if threads_per_worker == 1:
        assert "scheduler" not in kwargs
    else:
        assert kwargs["scheduler"] == "synchronous"

    assert _resolve_mps_scheduler({})["scheduler"] == "synchronous"
    assert _resolve_mps_scheduler({"scheduler": "threads"}) == {"scheduler": "threads"}
