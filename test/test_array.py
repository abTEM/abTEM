from numbers import Number

import hypothesis.extra.numpy as numpy_st
import hypothesis.strategies as st
import pytest
from hypothesis import given, assume, reproduce_failure

import strategies as abtem_st
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import cp
from abtem.core.array import stack, concatenate
from utils import assert_array_matches_device, gpu, remove_dummy_dimensions, assert_array_matches_laziness


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_indexing(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))

    indices = data.draw(numpy_st.basic_indices(has_array.ensemble_shape, allow_newaxis=False, allow_ellipsis=False))

    if isinstance(indices, Number):
        num_lost_axes = 1
    elif isinstance(indices, slice):
        num_lost_axes = 0
    else:
        num_lost_axes = sum(1 for i in indices if isinstance(i, Number))

    assert len(has_array[indices].shape) == len(has_array.shape) - num_lost_axes


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [abtem_st.potential_array])
def test_indexing_potential(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    indices = data.draw(numpy_st.basic_indices(has_array.shape[:-2], allow_newaxis=False, allow_ellipsis=False))

    if isinstance(indices, Number):
        num_lost_axes = 1
    elif isinstance(indices, slice):
        num_lost_axes = 0
    else:
        ensemble_indices = indices[:len(has_array.shape) - 3]
        num_lost_axes = sum(1 for i in ensemble_indices if isinstance(i, Number))

    assert len(has_array[indices].shape) == max(len(has_array.shape) - num_lost_axes, 3)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.s_matrix_array
])
def test_indexing_raises(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    indices = data.draw(numpy_st.basic_indices(has_array.shape, allow_newaxis=False, allow_ellipsis=False))

    if ((isinstance(indices, tuple) and len(indices) > len(has_array.ensemble_shape)) or
            (isinstance(indices, int) and len(has_array.ensemble_shape) == 0)):
        with pytest.raises(RuntimeError):
            indexed = has_array[indices]


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_shape(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    assert len(has_array.base_shape) == has_array._base_dims
    assert has_array.shape == has_array.ensemble_shape + has_array.base_shape
    assert len(has_array.base_axes_metadata) == len(has_array.base_shape)
    assert len(has_array.ensemble_axes_metadata) == len(has_array.ensemble_shape)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_ensure_lazy(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.ensure_lazy()
    assert has_array.is_lazy
    assert_array_matches_laziness(has_array.array, True)


@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_to_zarr(data, has_array, url, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    waves.to_zarr(url)


@given(data=st.data(), url=abtem_st.temporary_path(allow_none=False))
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_to_zarr_from_zarr(data, has_array, url, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array.to_zarr(url)
    has_array_from_zarr = has_array.from_zarr(url).copy_to_device(has_array.device).compute()
    assert has_array_from_zarr == has_array
    has_array_from_zarr.compute()
    assert has_array_from_zarr == has_array


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_expand_dims(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    expanded = waves.expand_dims((0,))
    assert expanded.shape[0] == 1
    expanded = expanded.expand_dims((1,))
    assert expanded.shape[1] == 1


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_squeeze(data, has_array, lazy, device):
    waves = data.draw(has_array(lazy=lazy, device=device))
    squeezed = waves.squeeze()
    assert remove_dummy_dimensions(waves.ensemble_shape) + waves.base_shape == squeezed.shape


@given(data=st.data(), destination=st.sampled_from(['cpu', 'gpu']))
@pytest.mark.parametrize('lazy', [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
@pytest.mark.skipif(cp is None, reason="no gpu")
def test_to_cpu(data, has_array, destination, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    has_array = has_array.copy_to_device(device=destination)
    assert_array_matches_device(has_array.array, destination)
    has_array.compute()
    assert_array_matches_device(has_array.array, destination)


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    abtem_st.potential_array,
    abtem_st.s_matrix_array
])
def test_stacks_with_self(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    stacked = stack((has_array, has_array), axes_metadata=OrdinalAxis(values=(1, 1)), axis=0)
    stacked.compute()
    assert has_array == stacked[0] == stacked[1]


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    # abtem_st.potential_array
])
def test_from_array_and_metadata(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))
    new = has_array.__class__.from_array_and_metadata(has_array.array, has_array.axes_metadata, has_array.metadata)
    assert new == has_array


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ['cpu', gpu])
@pytest.mark.parametrize("has_array", [
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
    abtem_st.waves,
    # abtem_st.potential_array
])
def test_concatenates_with_self(data, has_array, lazy, device):
    has_array = data.draw(has_array(lazy=lazy, device=device))

    axis = data.draw(st.integers(min_value=0, max_value=len(has_array.shape) - 1))
    assume(has_array.axes_metadata[axis]._concatenate)

    concatenated = concatenate((has_array, has_array), axis=axis)
    concatenated.compute()

    assume(axis < len(has_array.ensemble_shape))
    indices = (slice(None),) * axis + (slice(0, has_array.shape[axis]),)
    assert concatenated[indices] == has_array
