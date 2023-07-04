from functools import reduce
from operator import mul

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, reproduce_failure

import strategies as abtem_st
from abtem.core.ensemble import concatenate_array_blocks


@given(data=st.data())
@pytest.mark.parametrize(
    "ensemble",
    [
        abtem_st.frozen_phonons,
        abtem_st.dummy_frozen_phonons,
        abtem_st.md_frozen_phonons,
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.potential,
        abtem_st.potential_array,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        abtem_st.composite_wave_transform,
        abtem_st.ctf,
    ],
)
def test_ensemble_shape(data, ensemble):
    ensemble = data.draw(ensemble())
    #print(ensemble.ensemble_shape, ensemble.ensemble_axes_metadata)
    assert len(ensemble.ensemble_shape) == len(ensemble.ensemble_axes_metadata)
    assert len(ensemble.ensemble_shape) == len(ensemble._default_ensemble_chunks)


@given(data=st.data())
@pytest.mark.parametrize(
    "ensemble",
    [
        abtem_st.frozen_phonons,
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.waves,
        abtem_st.ctf,
        abtem_st.dummy_frozen_phonons,
        abtem_st.md_frozen_phonons,
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.potential,
        abtem_st.potential_array,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        abtem_st.composite_wave_transform,
    ],
)
def test_ensembles(data, ensemble):
    ensemble = data.draw(ensemble())
    if len(ensemble.ensemble_shape) > 0:
        chunks = data.draw(
            st.integers(min_value=1, max_value=reduce(mul, ensemble.ensemble_shape))
        )
    else:
        chunks = ()

    blocks = ensemble.ensemble_blocks(chunks).compute()

    if len(ensemble.ensemble_shape) > 0:
        assert len(blocks.shape) == len(ensemble.ensemble_shape)

    try:
        for i, _, fp in ensemble.generate_blocks(chunks):
            assert blocks[i] == fp.item()
    except NotImplementedError:
        pass


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
@pytest.mark.parametrize(
    "ensemble",
    [
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        abtem_st.ctf,
    ],
)
def test_array_waves_transform(data, ensemble, chunks):
    ensemble = data.draw(ensemble())

    blocks = ensemble.ensemble_blocks(1).compute()

    waves = data.draw(abtem_st.probe()).build()
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i]._evaluate_kernel(waves)
    blocks = concatenate_array_blocks(blocks)
    array = ensemble._evaluate_kernel(waves)

    assert array.shape[:-2] == ensemble.ensemble_shape
    assert blocks.shape[:-2] == ensemble.ensemble_shape

    assert np.allclose(blocks, array, atol=1e-5)


@given(data=st.data())
@pytest.mark.parametrize(
    "ensemble",
    [
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        abtem_st.composite_wave_transform,
        abtem_st.ctf,
    ],
)
@pytest.mark.parametrize("lazy", [True, False])
def test_apply_waves_transform(data, ensemble, lazy):
    ensemble = data.draw(ensemble())
    waves = data.draw(abtem_st.probe(allow_distribution=False)).build(lazy=lazy)

    ensemble_shape = ensemble._out_ensemble_shape(waves)
    waves = ensemble.apply(waves)

    assert waves.shape[:-2] == ensemble_shape
