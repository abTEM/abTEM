import os
from numbers import Number

import hypothesis.extra.numpy as numpy_st
import hypothesis.strategies as st
import pytest
import strategies as abtem_st
from hypothesis import assume, given, settings
# from abtem.core.test.strategies import random_chunks, random_array_object
from utils import (assert_array_matches_device, assert_array_matches_laziness,
                   devices, gpu, lazy_params, remove_dummy_dimensions,
                   requires_gpu, si_cubic_atoms)

from abtem.array import concatenate  # , concat_array_object_ensemble_blocks
from abtem.array import stack
from abtem.core.axes import OrdinalAxis

# The full set of `has_array` strategies exercised by most array-object tests.
ALL_HAS_ARRAY = [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array,
]

# The subset used by tests that don't support potential arrays or S-matrix
# arrays (e.g. from_array_and_metadata, concatenation).
HAS_ARRAY_NO_POTENTIAL = [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
]


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_indexing(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))

    indices = data.draw(
        numpy_st.basic_indices(
            has_array.ensemble_shape, allow_newaxis=False, allow_ellipsis=False
        )
    )

    if isinstance(indices, Number):
        num_lost_axes = 1
    elif isinstance(indices, slice):
        num_lost_axes = 0
    else:
        num_lost_axes = sum(1 for i in indices if isinstance(i, Number))

    assert len(has_array[indices].shape) == len(has_array.shape) - num_lost_axes


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", [abtem_st.potential_array])
def test_indexing_potential(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    indices = data.draw(
        numpy_st.basic_indices(
            has_array.shape[:-2], allow_newaxis=False, allow_ellipsis=False
        )
    )

    if isinstance(indices, Number):
        num_lost_axes = 1
    elif isinstance(indices, slice):
        num_lost_axes = 0
    else:
        ensemble_indices = indices[: len(has_array.shape) - 3]
        num_lost_axes = sum(1 for i in ensemble_indices if isinstance(i, Number))

    assert len(has_array[indices].shape) == max(len(has_array.shape) - num_lost_axes, 3)


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.s_matrix_array,
    ],
)
def test_indexing_raises(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    indices = data.draw(
        numpy_st.basic_indices(
            has_array.shape, allow_newaxis=False, allow_ellipsis=False
        )
    )

    if (
        isinstance(indices, tuple) and len(indices) > len(has_array.ensemble_shape)
    ) or (isinstance(indices, int) and len(has_array.ensemble_shape) == 0):
        with pytest.raises(RuntimeError):
            has_array[indices]


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_shape(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    assert len(has_array.base_shape) == has_array._base_dims
    assert has_array.shape == has_array.ensemble_shape + has_array.base_shape
    assert len(has_array.base_axes_metadata) == len(has_array.base_shape)
    assert len(has_array.ensemble_axes_metadata) == len(has_array.ensemble_shape)


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_ensure_lazy(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.ensure_lazy()
    assert has_array.is_lazy
    assert_array_matches_laziness(has_array.array, True)


@settings(max_examples=5)
@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_to_zarr(data, has_array, url, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    waves.to_zarr(url)


@settings(max_examples=5)
@given(data=st.data(), url=abtem_st.temporary_path_zip(allow_none=False))
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_to_zarr_zip(data, has_array, url, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    waves.to_zarr(url)


@settings(max_examples=5)
@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_to_zarr_from_zarr(data, has_array, url, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array.to_zarr(url)
    has_array_from_zarr = (
        has_array.from_zarr(url).copy_to_device(has_array.device).compute()
    )
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()
    has_array_from_zarr.compute()
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()


@settings(max_examples=5)
@given(data=st.data(), url=abtem_st.temporary_path_zip(allow_none=False))
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_to_zarr_from_zarr_zip(data, has_array, url, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array.to_zarr(url)
    has_array_from_zarr = (
        has_array.from_zarr(url).copy_to_device(has_array.device).compute()
    )
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()
    has_array_from_zarr.compute()
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()


@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.waves,
        abtem_st.potential_array,
    ],
)
def test_from_zarr_legacy_format(data, has_array, url):
    # Files written by abTEM <= 1.0.9 store per-object "kwargs{i}"/"type{i}"
    # attributes instead of the canonical "metadata{i}"; they may hold any
    # ArrayObject subclass (e.g. Waves, PotentialArray), not just measurements.
    import zarr

    from abtem.array import from_zarr

    has_array = data.draw(has_array(lazy=False, device="cpu"))

    root = zarr.open(url, mode="w")
    root.create_array(name="array0", data=has_array.array, chunks=has_array.shape)
    root.attrs["kwargs0"] = has_array._pack_kwargs(
        has_array._copy_kwargs(exclude=("array",))
    )
    root.attrs["type0"] = has_array.__class__.__name__

    has_array_from_zarr = from_zarr(url).compute()
    assert has_array_from_zarr == has_array


# ---- large-array zarr chunking (regression: whole-array single chunk hit a
# codec's 2**31-1 byte buffer limit, and a write that failed partway left a
# store with valid-looking metadata but silently-all-zero data) -------------


def test_safe_zarr_chunks_stays_under_budget():
    from abtem.array import _safe_zarr_chunks

    shape = (1025, 512, 512)
    chunks = _safe_zarr_chunks(shape, itemsize=8, max_bytes=10_000_000)

    nbytes = 8
    for c in chunks:
        nbytes *= c
    assert nbytes <= 10_000_000
    assert all(0 < c <= s for c, s in zip(chunks, shape))


def test_safe_zarr_chunks_no_op_when_already_small():
    from abtem.array import _safe_zarr_chunks

    shape = (4, 8, 8)
    assert _safe_zarr_chunks(shape, itemsize=8) == shape


def test_safe_zarr_chunks_never_splits_trailing_axes_when_avoidable():
    """Splitting an ArrayObject's base (measurement) axes -- e.g. a
    DiffractionPatterns' 2D image plane -- doesn't just change chunk
    granularity: several of abTEM's own lazy dask operations assume those
    axes are never chunked and silently compute wrong results if they are
    (see test_measure.py's interpolate_line regression). n_fixed_trailing_axes
    must be respected whenever shrinking the leading (ensemble) axis alone is
    enough to fit the budget."""
    from abtem.array import _safe_zarr_chunks

    shape = (1024, 256, 256)
    chunks = _safe_zarr_chunks(
        shape, itemsize=8, max_bytes=1_000_000, n_fixed_trailing_axes=2
    )
    assert chunks[-2:] == shape[-2:]
    assert chunks[0] < shape[0]


def test_safe_zarr_chunks_falls_back_to_trailing_axes_if_unavoidable():
    """If even a single element along every leading axis still exceeds the
    budget, there is no choice but to also split the trailing axes -- this
    must not raise or loop forever."""
    from abtem.array import _safe_zarr_chunks

    shape = (1, 2048, 2048)
    chunks = _safe_zarr_chunks(
        shape, itemsize=8, max_bytes=1_000_000, n_fixed_trailing_axes=2
    )
    nbytes = 8
    for c in chunks:
        nbytes *= c
    assert nbytes <= 1_000_000
    assert chunks[-2:] != shape[-2:]


def _make_dp(n_energy, gpts, seed=0):
    import dask.array as da
    import numpy as np

    import abtem
    from abtem.measurements import DiffractionPatterns

    rng = np.random.default_rng(seed)
    array = rng.random((n_energy, gpts, gpts))
    lazy_array = da.from_array(array, chunks=(1, gpts, gpts))
    dp = DiffractionPatterns.from_array_and_metadata(
        lazy_array,
        axes_metadata=[
            OrdinalAxis(label="energy", values=tuple(range(n_energy))),
            abtem.core.axes.ReciprocalSpaceAxis(sampling=0.1, label="x", units="1/A"),
            abtem.core.axes.ReciprocalSpaceAxis(sampling=0.1, label="y", units="1/A"),
        ],
    )
    return dp, array


def test_from_zarr_auto_chunks_never_splits_base_axes(tmp_path):
    """from_zarr(url, chunks="auto") must not let dask's own auto-chunking
    heuristic split an ArrayObject's base (measurement) axes -- several of
    abTEM's own lazy operations (e.g. interpolate_line) assume those are
    never chunked. chunks=None (the default) already avoids this by
    mirroring whatever to_zarr actually wrote (which itself never splits
    base axes); explicitly requesting "auto" used to bypass that protection
    since dask's own heuristic doesn't know which axes are which."""
    import dask

    import abtem.array as abtem_array_module

    dp, _ = _make_dp(n_energy=2, gpts=64)
    url = str(tmp_path / "dp_auto.zarr")
    dp.to_zarr(url)

    with dask.config.set({"array.chunk-size": "1KiB"}):
        loaded = abtem_array_module.from_zarr(url, chunks="auto")

    assert loaded.array.chunks[-2:] == ((64,), (64,))


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_to_zarr_never_chunks_base_axes(tmp_path, monkeypatch, suffix):
    """Regression: the spatial (base) axes of a DiffractionPatterns are much
    larger than its ensemble (energy) axis here, so a naive "always shrink
    the largest axis" policy would chunk the spatial plane first -- which
    several lazy dask operations (e.g. interpolate_line) silently compute
    wrong results against. to_zarr must chunk only the ensemble axis."""
    import zarr

    import abtem.array as abtem_array_module

    monkeypatch.setattr(abtem_array_module, "_MAX_ZARR_CHUNK_BYTES", 1_000_000)

    dp, _ = _make_dp(n_energy=8, gpts=256)
    url = str(tmp_path / f"dp_base{suffix}")
    dp.to_zarr(url)

    if suffix == ".zip":
        store = zarr.storage.ZipStore(url, mode="r")
        root = zarr.open(store=store, mode="r")
    else:
        root = zarr.open(url, mode="r")

    zarr_array = root["array0"]
    assert zarr_array.chunks[-2:] == zarr_array.shape[-2:]
    assert zarr_array.chunks[0] < zarr_array.shape[0]

    if suffix == ".zip":
        store.close()


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_to_zarr_writes_multiple_chunks_not_one_giant_chunk(tmp_path, monkeypatch, suffix):
    """A single whole-array zarr chunk hits a codec's 2**31-1 byte buffer
    limit for any reasonably large array (regression: this used to always
    happen via chunks=computed_array.shape). Force a tiny budget so a small
    test array reproduces the same "must be split" condition, and check the
    written array is actually split -- and still round-trips correctly."""
    import numpy as np
    import zarr

    import abtem.array as abtem_array_module

    monkeypatch.setattr(abtem_array_module, "_MAX_ZARR_CHUNK_BYTES", 10_000)

    dp, array = _make_dp(n_energy=8, gpts=16)
    url = str(tmp_path / f"dp{suffix}")
    dp.to_zarr(url)

    if suffix == ".zip":
        store = zarr.storage.ZipStore(url, mode="r")
        root = zarr.open(store=store, mode="r")
    else:
        root = zarr.open(url, mode="r")

    zarr_array = root["array0"]
    assert zarr_array.chunks != zarr_array.shape

    if suffix == ".zip":
        store.close()

    loaded = abtem_array_module.from_zarr(url).compute()
    np.testing.assert_allclose(loaded.array, array)


@pytest.mark.parametrize("suffix", ["", ".zip"])
def test_to_zarr_cleans_up_on_failed_write(tmp_path, monkeypatch, suffix):
    """A write that fails partway (e.g. a chunk still over a codec's buffer
    limit) must not leave behind a store with valid-looking metadata but
    missing chunk data -- previously silently readable back as all zeros
    (zarr's fill_value for a declared-but-never-written chunk)."""
    import zarr

    def _raising_create_array(self, *args, **kwargs):
        raise ValueError("Codec does not support buffers of > 2147483647 bytes")

    monkeypatch.setattr(zarr.Group, "create_array", _raising_create_array)

    dp, _ = _make_dp(n_energy=4, gpts=8)
    url = str(tmp_path / f"dp{suffix}")
    with pytest.raises(ValueError, match="Codec does not support buffers"):
        dp.to_zarr(url)

    assert not os.path.exists(url)


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_expand_dims(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    expanded = waves.expand_dims((0,))
    assert expanded.shape[0] == 1
    expanded = expanded.expand_dims((1,))
    assert expanded.shape[1] == 1


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_squeeze(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    squeezed = waves.squeeze()
    assert (
        remove_dummy_dimensions(waves.ensemble_shape) + waves.base_shape
        == squeezed.shape
    )


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("destination", ["cpu", gpu])
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_to_cpu(data, has_array, lazy, device, destination):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.copy_to_device(device=destination)
    assert_array_matches_device(has_array.array, destination)
    has_array.compute()
    assert_array_matches_device(has_array.array, destination)


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", ALL_HAS_ARRAY)
def test_stacks_with_self(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    stacked = stack(
        (has_array, has_array), axis_metadata=OrdinalAxis(values=(1, 1)), axis=0
    )
    stacked.compute()
    has_array._metadata = stacked[1].metadata
    assert stacked[0].to_cpu() == stacked[1].to_cpu() == has_array.to_cpu()


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize("has_array", HAS_ARRAY_NO_POTENTIAL)
def test_from_array_and_metadata(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    new = has_array.__class__.from_array_and_metadata(
        has_array.array, has_array.axes_metadata, has_array.metadata
    )
    assert new.to_cpu() == has_array.to_cpu()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True])
@devices
@pytest.mark.parametrize("has_array", HAS_ARRAY_NO_POTENTIAL)
def test_concatenates_with_self(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))

    axis = data.draw(st.integers(min_value=0, max_value=len(has_array.ensemble_shape)))
    assume(has_array.axes_metadata[axis]._concatenate)

    concatenated = concatenate((has_array, has_array), axis=axis)
    concatenated.compute()

    assume(axis < len(has_array.ensemble_shape))
    indices = (slice(None),) * axis + (slice(0, has_array.shape[axis]),)
    assert concatenated[indices].to_cpu() == has_array.to_cpu()


# @given(data=st.data())
# def test_ensemble_blocks(data):
#     array_object = data.draw(random_array_object(lazy=True))
#
#     blocks = array_object.ensemble_blocks().compute()
#
#     concat_array_object = concat_array_object_ensemble_blocks(blocks)
#
#     assert array_object.compute() == concat_array_object

# array_object = data.draw(random_array_object())
#
# array_object = array_object.ensure_lazy()
#
# chunks = data.draw(random_chunks(array_object.ensemble_shape)).example()
#
# array_object = array_object.rechunk(chunks=chunks)
#
# blocks = array_object.ensemble_blocks().compute()
#
# concat_array_object = concat_array_object_ensemble_blocks(blocks)
#
# assert array_object.compute() == concat_array_object


class TestStackAndHyperspyTrustTheRealArrayType:
    """A detector's `to_cpu=True` (the default) moves a measurement's array
    to `numpy` without updating the object's own `.device` label, so
    `.device` can say `"gpu"` while `.array` is already a plain `ndarray`.
    `ArrayObject._stack` and `.to_hyperspy` used to pick their array module
    from that possibly-stale `.device` label instead of the real `.array`
    type, and crashed handing a `numpy.ndarray` to `cupy.stack`/
    `cupy.moveaxis`. Fixing the underlying label inconsistency itself is out
    of scope here (see the issue file) -- these tests only pin down that the
    two consumers no longer trust it.
    """

    @staticmethod
    def _stale_label_measurement():
        import ase

        import abtem

        atoms = ase.Atoms(
            "BN", positions=[(2.0, 2.0, 1.0), (4.0, 4.0, 1.0)], cell=(8, 8, 4),
            pbc=True,
        )
        with abtem.config.set({"device": "gpu"}):
            pot = abtem.Potential(
                atoms, gpts=(32, 32), slice_thickness=2.0, device="gpu"
            )
            probe = abtem.Probe(
                semiangle_cutoff=20, energy=60e3, extent=(8.0, 8.0), gpts=(32, 32)
            )
            scan = abtem.GridScan(
                start=(0, 0), end=(1, 1), gpts=(2, 2), fractional=True,
                potential=pot,
            )
            # to_cpu=True is the AnnularDetector default; spelled out here
            # since it's the whole reason .array and .device disagree.
            return probe.scan(
                potential=pot, scan=scan,
                detectors=abtem.AnnularDetector(inner=0, outer=30, to_cpu=True),
                lazy=False,
            )

    @requires_gpu
    def test_precondition_device_label_disagrees_with_array_type(self):
        """Pins down the setup every test below depends on, so a future fix
        to the underlying label inconsistency (out of scope here) doesn't
        silently turn these into tests of nothing."""
        import numpy as np

        m = self._stale_label_measurement()
        assert isinstance(m.array, np.ndarray)
        assert m.device == "gpu"

    @requires_gpu
    def test_stack_does_not_crash_on_a_stale_device_label(self):
        import numpy as np

        m = self._stale_label_measurement()
        stacked = stack(
            (m, m), axis_metadata=OrdinalAxis(values=(0, 1)), axis=0
        )
        assert np.array_equal(
            np.asarray(stacked.array), np.stack([np.asarray(m.array)] * 2, axis=0)
        )

    @requires_gpu
    def test_to_hyperspy_does_not_crash_on_a_stale_device_label(self, monkeypatch):
        """hyperspy isn't installed in every environment this suite runs
        in; stubbing its two signal classes lets this test exercise the
        real to_hyperspy code path -- including the line that crashed --
        everywhere, rather than only wherever hyperspy happens to be
        installed. A version of this test gated behind hyperspy's presence
        (test_hyperspy.py's own skipif) would silently skip in exactly the
        environments where this regression would go unnoticed."""
        import types

        import numpy as np

        import abtem.array as abtem_array_module

        class _FakeSignal:
            def __init__(self, data, axes=None):
                self.data = data

            def as_lazy(self):
                return self

        monkeypatch.setattr(
            abtem_array_module,
            "hs",
            types.SimpleNamespace(
                signals=types.SimpleNamespace(
                    Signal1D=_FakeSignal, Signal2D=_FakeSignal
                )
            ),
        )
        m = self._stale_label_measurement()
        sig = m.to_hyperspy()
        # transpose=True (the default) is what exercises the crashing line
        # (xp.moveaxis); for this measurement -- base_dims=2, no ensemble
        # axes -- that reverses the two base axes, i.e. a plain transpose.
        assert np.array_equal(np.asarray(sig.data), np.asarray(m.array).T)

    def test_get_array_module_receives_the_array_not_the_device_label(
        self, monkeypatch
    ):
        """CPU-runnable complement to the two GPU-only tests above. Those
        need get_array_module("gpu") to actually resolve to cupy to
        reproduce the crash, so (like every @requires_gpu test) they never
        run in CI -- no GPU runner is configured -- and only ever execute
        on a workstation with cupy. This doesn't reproduce the crash, but
        it runs everywhere and directly asserts the fix's actual invariant
        -- _stack and to_hyperspy call get_array_module with the real
        array, never with .device -- independent of cupy or a GPU being
        present at all.

        Deliberately does not use _stale_label_measurement: that needs a
        real GPU to produce a genuine numpy/cupy mismatch, but the
        invariant under test here (which argument gets passed) doesn't
        care what .device or .array actually contain, only that they
        disagree -- so an arbitrary marker string standing in for .device
        is enough, and keeps this test runnable without a GPU.
        """
        import types

        import numpy as np

        import abtem.array as abtem_array_module
        from abtem.measurements import Images

        array = np.random.default_rng(0).random((4, 4)).astype(np.float32)
        m = Images(array=array, sampling=(0.1, 0.1))
        m._device = "not-a-real-device"  # disagrees with .array on purpose

        real_get_array_module = abtem_array_module.get_array_module
        calls = []

        def recording_get_array_module(x):
            calls.append(x)
            return real_get_array_module("cpu" if isinstance(x, str) else x)

        monkeypatch.setattr(
            abtem_array_module, "get_array_module", recording_get_array_module
        )

        stack((m, m), axis_metadata=OrdinalAxis(values=(0, 1)), axis=0)
        assert any(c is m.array for c in calls)
        assert not any(isinstance(c, str) for c in calls)

        calls.clear()

        class _FakeSignal:
            def __init__(self, data, axes=None):
                self.data = data

            def as_lazy(self):
                return self

        monkeypatch.setattr(
            abtem_array_module,
            "hs",
            types.SimpleNamespace(
                signals=types.SimpleNamespace(
                    Signal1D=_FakeSignal, Signal2D=_FakeSignal
                )
            ),
        )
        m.to_hyperspy()
        assert any(c is m.array for c in calls)
        assert not any(isinstance(c, str) for c in calls)


class TestBaseLessArrayObject:
    """`-len(self.base_shape)` is `-0` for a base-less object (`base_shape == ()`),
    and Python has no negative zero: `[: -0]` is `[:0]`, always empty, and
    `[-0 :]` is `[0:]`, always everything. Five sites in `abtem/array.py` used
    that form; `MeasurementsEnsemble` (`abtem/measurements.py`, `_base_dims = 0`)
    is the base-less class, reachable from public API via
    `Images.to_measurement_ensemble()`.
    """

    @staticmethod
    def _ensemble(chunks=None):
        import abtem

        atoms = si_cubic_atoms()
        potential = abtem.Potential(atoms, gpts=(64, 64), slice_thickness=2.0)
        probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
        scan = abtem.GridScan(start=(0, 0), end=(2, 2), sampling=1.0)
        with abtem.config.set({"fft": "numpy"}):
            images = probe.scan(
                potential,
                scan=scan,
                detectors=abtem.AnnularDetector(inner=50, outer=150),
                lazy=True,
            )
            m = images.to_measurement_ensemble()
        assert m.base_shape == ()
        if chunks is not None:
            m = m.rechunk(chunks)
        return m

    def test_squeeze_removes_a_length_one_ensemble_axis(self):
        m = self._ensemble()
        sliced = m[0:1]
        assert sliced.shape == (1, 2)
        assert sliced.squeeze().shape == (2,)

    def test_has_base_chunks_is_false_with_no_base_dims(self):
        m = self._ensemble(chunks=(1, 1))
        assert m.array.chunks == ((1, 1), (1, 1))
        assert m._has_base_chunks is False

    def test_no_base_chunks_is_a_no_op(self):
        m = self._ensemble(chunks=(1, 1))
        before = m.array.chunks
        after = m.no_base_chunks().array.chunks
        assert after == before

    def test_no_base_chunks_own_arithmetic_is_correct_even_if_reached(
        self, monkeypatch
    ):
        """`no_base_chunks()`'s early return on a correct `_has_base_chunks`
        already keeps the buggy line from firing for a base-less object --
        the test above pins that. This pins the line itself: forcing the
        guard open (as a stale or differently-computed `_has_base_chunks`
        might) must not resurrect the -0 collapse into one block."""
        m = self._ensemble(chunks=(1, 1))
        monkeypatch.setattr(
            type(m), "_has_base_chunks", property(lambda self: True)
        )
        after = m.no_base_chunks().array.chunks
        assert after == ((1, 1), (1, 1))

    def test_partition_args_does_not_raise(self):
        m = self._ensemble()
        m._partition_args()  # used to raise ValueError

    def test_apply_transform_reaches_every_ensemble_axis(self):
        """`ArrayObject.apply_transform`'s blockwise callback sliced ensemble
        axes off with the same -0 bug (`_apply_transform`'s `base_ndims`
        argument), so applying any `ArrayObjectTransform` to a base-less
        object either crashed (axes-metadata/array-ndim mismatch) or silently
        dropped every ensemble axis."""
        from abtem.transform import TransformFromFunc

        m = self._ensemble()

        def double(array_object, **kwargs):
            return array_object.array * 2

        transformed = TransformFromFunc(func=double, func_kwargs={}).apply(m)
        assert transformed.shape == m.shape

        import numpy as np

        assert np.allclose(
            np.asarray(transformed.compute().array),
            np.asarray(m.compute().array) * 2,
        )
