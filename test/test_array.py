import os
from numbers import Number

import hypothesis.extra.numpy as numpy_st
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
# from abtem.core.test.strategies import random_chunks, random_array_object
from utils import (assert_array_matches_device, assert_array_matches_laziness,
                   gpu, remove_dummy_dimensions)

from abtem.array import concatenate  # , concat_array_object_ensemble_blocks
from abtem.array import stack
from abtem.core.axes import OrdinalAxis
from abtem.measurements import Images


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_shape(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    assert len(has_array.base_shape) == has_array._base_dims
    assert has_array.shape == has_array.ensemble_shape + has_array.base_shape
    assert len(has_array.base_axes_metadata) == len(has_array.base_shape)
    assert len(has_array.ensemble_axes_metadata) == len(has_array.ensemble_shape)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_ensure_lazy(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.ensure_lazy()
    assert has_array.is_lazy
    assert_array_matches_laziness(has_array.array, True)


@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_to_zarr(data, has_array, url, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    waves.to_zarr(url)


@given(data=st.data(), url=abtem_st.temporary_path_zip(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", [gpu, "cpu"])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_to_zarr_zip(data, has_array, url, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    waves.to_zarr(url)


@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_to_zarr_from_zarr(data, has_array, url, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array.to_zarr(url)
    has_array_from_zarr = (
        has_array.from_zarr(url).copy_to_device(has_array.device).compute()
    )
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()
    has_array_from_zarr.compute()
    assert has_array_from_zarr.to_cpu() == has_array.to_cpu()


@given(data=st.data(), url=abtem_st.temporary_path_zip(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
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

    has_array_from_zarr = from_zarr(url, chunks="auto").compute()
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
def test_to_zarr_writes_multiple_chunks_not_one_giant_chunk(
    tmp_path, monkeypatch, suffix
):
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
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_expand_dims(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    expanded = waves.expand_dims((0,))
    assert expanded.shape[0] == 1
    expanded = expanded.expand_dims((1,))
    assert expanded.shape[1] == 1


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_squeeze(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    squeezed = waves.squeeze()
    assert (
        remove_dummy_dimensions(waves.ensemble_shape) + waves.base_shape
        == squeezed.shape
    )


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("destination", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_to_cpu(data, has_array, lazy, device, destination):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.copy_to_device(device=destination)
    assert_array_matches_device(has_array.array, destination)
    has_array.compute()
    assert_array_matches_device(has_array.array, destination)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        abtem_st.potential_array,
        abtem_st.s_matrix_array,
    ],
)
def test_stacks_with_self(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    stacked = stack(
        (has_array, has_array), axis_metadata=OrdinalAxis(values=(1, 1)), axis=0
    )
    stacked.compute()
    has_array._metadata = stacked[1].metadata
    assert stacked[0].to_cpu() == stacked[1].to_cpu() == has_array.to_cpu()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        # abtem_st.potential_array
    ],
)
def test_from_array_and_metadata(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    new = has_array.__class__.from_array_and_metadata(
        has_array.array, has_array.axes_metadata, has_array.metadata
    )
    assert new.to_cpu() == has_array.to_cpu()


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "has_array",
    [
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
        abtem_st.waves,
        # abtem_st.potential_array
    ],
)
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
        import ase.build

        import abtem

        atoms = ase.build.bulk("Si", cubic=True)
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

    def test_partition_args_does_not_raise(self):
        m = self._ensemble()
        m._partition_args()  # used to raise ValueError

    def test_apply_transform_reaches_every_ensemble_axis(self):
        """`ArrayObject.apply_transform`'s blockwise callback sliced ensemble
        axes off with the same -0 bug (`_apply_transform`'s `base_ndims`
        argument), so applying any `ArrayObjectTransform` to a base-less
        object either crashed (axes-metadata/array-ndim mismatch) or silently
        dropped every ensemble axis."""
        import numpy as np

        from abtem.transform import TransformFromFunc

        m = self._ensemble()

        def double(array_object, **kwargs):
            return array_object.array * 2

        transformed = TransformFromFunc(func=double, func_kwargs={}).apply(m)
        assert transformed.shape == m.shape

        assert np.allclose(
            np.asarray(transformed.compute().array),
            np.asarray(m.compute().array) * 2,
        )


class _FakeDataArray:
    """Stands in for xarray.DataArray, which is an optional dependency."""

    def __init__(self, array, dims=None, coords=None, attrs=None, **kwargs):
        self.array, self.dims, self.coords, self.attrs = array, dims, coords, attrs


@pytest.fixture
def fake_xarray(monkeypatch):
    import types

    import abtem.array

    monkeypatch.setattr(
        abtem.array, "xr", types.SimpleNamespace(DataArray=_FakeDataArray)
    )


def test_to_data_array_without_a_label_leaves_the_metadata_alone(fake_xarray):
    import abtem

    waves = abtem.Waves(
        np.ones((16, 20), dtype=np.complex64), energy=100e3, sampling=0.1
    )
    assert "label" not in waves.metadata
    metadata_before = dict(waves.metadata)

    data_array = waves.to_data_array()

    assert isinstance(data_array, _FakeDataArray)
    assert "long_name" not in data_array.attrs
    assert waves.metadata == metadata_before


def test_to_data_array_with_a_label_sets_long_name_on_the_copy_only(fake_xarray):
    import abtem

    waves = abtem.Waves(
        np.ones((16, 20), dtype=np.complex64),
        energy=100e3,
        sampling=0.1,
        metadata={"label": "psi"},
    )
    metadata_before = dict(waves.metadata)

    data_array = waves.to_data_array()

    assert data_array.attrs["long_name"] == "psi"
    assert waves.metadata == metadata_before


def test_to_data_array_does_not_share_nested_metadata_values(fake_xarray):
    # elastic_diffuse_diffraction_patterns records its component names as a list
    # in the metadata.
    import abtem

    waves = abtem.Waves(
        np.ones((16, 20), dtype=np.complex64),
        energy=100e3,
        sampling=0.1,
        metadata={"components": ["total", "elastic"]},
    )

    waves.to_data_array().attrs["components"].append("diffuse")

    assert waves.metadata["components"] == ["total", "elastic"]


def test_set_ensemble_axes_metadata_refuses_an_axis_that_does_not_fit():
    """The axis is checked against the array: a mismatch raises and leaves the
    old axis in place. The test sets metadata only, so it pins no device."""
    old_axis = OrdinalAxis(label="a", values=(1, 2, 3))
    images = Images(
        np.zeros((3, 4, 4)), sampling=0.1, ensemble_axes_metadata=[old_axis]
    )

    with pytest.raises(RuntimeError, match="ordinal axis"):
        images.set_ensemble_axes_metadata(OrdinalAxis(label="b", values=(10, 20)), 0)
    assert images.ensemble_axes_metadata[0] == old_axis

    new_axis = OrdinalAxis(label="b", values=(10, 20, 30))
    assert images.set_ensemble_axes_metadata(new_axis, 0) is images
    assert images.ensemble_axes_metadata[0] == new_axis


def test_set_ensemble_axes_metadata_leaves_a_list_shared_with_another_object_alone():
    """A refused axis is never written into the list the caller passed, and an
    accepted one is installed in a list of the object's own, so another object
    built from the same list keeps its axes. The test sets metadata only, so it
    pins no device."""
    axes = [OrdinalAxis(label="a", values=(1, 2, 3))]
    first = Images(np.zeros((3, 4, 4)), sampling=0.1, ensemble_axes_metadata=axes)
    second = Images(np.zeros((3, 4, 4)), sampling=0.1, ensemble_axes_metadata=axes)

    with pytest.raises(RuntimeError, match="ordinal axis"):
        first.set_ensemble_axes_metadata(OrdinalAxis(label="b", values=(10, 20)), 0)
    assert second.ensemble_axes_metadata[0].label == "a"
    assert axes[0].label == "a"

    first.set_ensemble_axes_metadata(OrdinalAxis(label="b", values=(10, 20, 30)), 0)
    assert first.ensemble_axes_metadata[0].label == "b"
    assert second.ensemble_axes_metadata[0].label == "a"
    assert axes[0].label == "a"


def _images_file(tmp_path):
    import abtem

    images = abtem.Images(
        np.zeros((2, 4, 3)),
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(label="a", values=(1, 2))],
    )
    url = str(tmp_path / "images.zarr")
    images.to_zarr(url)
    return url


def test_a_file_whose_stored_axes_do_not_fit_its_array_is_refused(tmp_path):
    """An object is rebuilt from the axes the file stores, so a file whose axes do
    not fit its array raises instead of loading the axes of its constructor
    kwargs. CPU only: zarr is written and read on the host."""
    import zarr

    import abtem

    url = _images_file(tmp_path)
    root = zarr.open(url, mode="r+")
    metadata = root.attrs["metadata0"]
    metadata["axes"]["axis_0"]["values"] = {"_type": "tuple", "_value": [7, 8, 9]}
    root.attrs["metadata0"] = metadata

    with pytest.raises(RuntimeError, match="ordinal axis"):
        abtem.from_zarr(url)


def test_an_error_in_from_array_and_metadata_escapes_from_zarr(tmp_path, monkeypatch):
    """Only the `NotImplementedError` of a class that cannot rebuild from axes
    and metadata sends `from_zarr` to its constructor kwargs; any other error
    escapes. CPU only: zarr is written and read on the host."""
    import abtem

    url = _images_file(tmp_path)

    def broken(cls, array, axes_metadata, metadata):
        raise TypeError("broken from_array_and_metadata")

    monkeypatch.setattr(abtem.Images, "from_array_and_metadata", classmethod(broken))

    with pytest.raises(TypeError, match="broken from_array_and_metadata"):
        abtem.from_zarr(url)


@pytest.mark.parametrize(
    "cls",
    [
        "abtem.measurements.IndexedDiffractionPatterns",
        "abtem.prism.s_matrix.SMatrixArray",
        "abtem.inelastic.core_loss.TransitionPotentialArray",
    ],
)
def test_classes_that_cannot_rebuild_from_axes_say_so(cls):
    """The stubs that send `from_zarr` to the stored constructor kwargs raise
    `NotImplementedError` when called on the class; as instance methods or plain
    functions they raised a `TypeError` first."""
    import importlib

    module, name = cls.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)

    with pytest.raises(NotImplementedError):
        cls.from_array_and_metadata(np.zeros((2, 2)), axes_metadata=[], metadata={})


def test_s_matrix_and_indexed_patterns_still_read_from_their_kwargs(tmp_path):
    """The classes without `from_array_and_metadata` load from their stored
    constructor kwargs. CPU only: zarr is written and read on the host."""
    from ase.build import bulk

    import abtem

    potential = abtem.Potential(bulk("Si", cubic=True), sampling=0.2)
    s_matrix = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=10
    ).build(lazy=False)
    s_matrix.to_zarr(str(tmp_path / "s_matrix.zarr"))
    read = abtem.from_zarr(str(tmp_path / "s_matrix.zarr")).compute()
    assert type(read) is type(s_matrix)
    np.testing.assert_allclose(read.array, s_matrix.array)

    patterns = abtem.DiffractionPatterns(
        np.random.default_rng(0).random((48, 40)),
        sampling=(0.2, 0.25),
        metadata={"energy": 100e3},
        fftshift=True,
    )
    indexed = patterns.index_diffraction_spots(cell=bulk("Si", cubic=True).cell)
    indexed.to_zarr(str(tmp_path / "indexed.zarr"))
    read = abtem.from_zarr(str(tmp_path / "indexed.zarr")).compute()
    assert type(read) is type(indexed)
    np.testing.assert_allclose(np.asarray(read.array), np.asarray(indexed.array))


def _linear_axes():
    from abtem.core.axes import LinearAxis, RealSpaceAxis, ReciprocalSpaceAxis, ScanAxis

    return [
        LinearAxis(label="z", sampling=0.7, offset=-1.3),
        RealSpaceAxis(label="z", sampling=0.7, offset=-1.3),
        ScanAxis(label="z", sampling=0.7, offset=-1.3),
        ReciprocalSpaceAxis(label="z", sampling=0.7, offset=-(9 // 2) * 0.7),
    ]


# (item, step of the members it selects), over 9 members: slices with a positive
# step, negative bounds included, and index arrays of evenly increasing members
_LINEAR_AXIS_ITEMS = [
    (slice(1, 8, 2), 2),
    (slice(None, None, 3), 3),
    (slice(-4, None), 1),
    (slice(1, -2), 1),
    (slice(-7, -1, 2), 2),
    ([0, 3, 6], 3),
    ([2, 3, 4], 1),
    ([5], 1),
    (np.array([False, True, False, True, False, True, False, False, False]), 2),
]


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("axis", _linear_axes(), ids=lambda axis: type(axis).__name__)
@pytest.mark.parametrize("item, step", _LINEAR_AXIS_ITEMS)
def test_indexing_a_linear_ensemble_axis_keeps_the_coordinates_of_its_members(
    axis, item, step, lazy, device
):
    from abtem.core.axes import ReciprocalSpaceAxis
    from abtem.core.backend import asnumpy, copy_to_device
    from abtem.measurements import Images

    if isinstance(axis, ReciprocalSpaceAxis) and step != 1:
        pytest.skip("a ReciprocalSpaceAxis keeps its class only for a step of one")

    host = np.arange(9)[:, None, None] * 6.0 * np.ones((9, 3, 2))
    images = Images(
        copy_to_device(host, device), sampling=0.1, ensemble_axes_metadata=[axis]
    )
    if lazy:
        images = images.ensure_lazy(chunks=(4, -1, -1))

    indexed = images[item]
    new_axis = indexed.ensemble_axes_metadata[0]
    expected = np.asarray(axis.coordinates(9))[item]

    assert type(new_axis) is type(axis)
    assert new_axis.sampling == pytest.approx(0.7 * step)
    np.testing.assert_allclose(
        new_axis.coordinates(len(expected)), expected, rtol=0, atol=1e-12 * 9 * 0.7
    )
    np.testing.assert_array_equal(asnumpy(indexed.compute().array), host[item])


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_a_new_axis_before_a_stepped_linear_ensemble_axis(device):
    from abtem.core.axes import LinearAxis
    from abtem.core.backend import copy_to_device
    from abtem.measurements import Images

    axis = LinearAxis(label="z", sampling=0.7, offset=-1.3)
    images = Images(
        copy_to_device(np.zeros((9, 3, 2)), device),
        sampling=0.1,
        ensemble_axes_metadata=[axis],
    )

    new_axis = images[None, 1::2].ensemble_axes_metadata[1]

    np.testing.assert_allclose(
        new_axis.coordinates(4), np.asarray(axis.coordinates(9))[1::2], atol=1e-12
    )


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_lazy_chunks_of_a_stepped_linear_ensemble_axis_carry_their_own_coordinates(
    lazy, device
):
    """The partition path indexes the already indexed axis with contiguous
    forward slices; each block's coordinates are those of its members."""
    from abtem.core.axes import ScanAxis
    from abtem.core.backend import copy_to_device
    from abtem.measurements import Images

    images = Images(
        copy_to_device(np.zeros((9, 3, 2)), device),
        sampling=0.1,
        ensemble_axes_metadata=[ScanAxis(sampling=0.5, offset=1.0)],
    )
    if lazy:
        images = images.ensure_lazy(chunks=(4, -1, -1))
    indexed = images[1::2]
    expected = indexed.ensemble_axes_metadata[0].coordinates(4)

    blocks = indexed._partition_ensemble_axes_metadata(chunks=((2, 2),), lazy=False)

    for block, start in zip(blocks, (0, 2)):
        np.testing.assert_allclose(
            block[0].coordinates(2), expected[start : start + 2], atol=1e-12
        )
