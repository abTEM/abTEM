import os
from numbers import Number

import hypothesis.extra.numpy as numpy_st
import hypothesis.strategies as st
import pytest
import strategies as abtem_st
from hypothesis import assume, given, settings
# from abtem.core.test.strategies import random_chunks, random_array_object
from utils import (assert_array_matches_device, assert_array_matches_laziness,
                   gpu, remove_dummy_dimensions)

from abtem.array import concatenate  # , concat_array_object_ensemble_blocks
from abtem.array import stack
from abtem.core.axes import OrdinalAxis


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


@settings(max_examples=5)
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


@settings(max_examples=5)
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


@settings(max_examples=5)
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


@settings(max_examples=5)
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
