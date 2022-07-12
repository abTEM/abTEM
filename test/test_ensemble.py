from functools import reduce
from operator import mul

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import strategies as abtem_st
from abtem.core.ensemble import concatenate_array_blocks


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [
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

])
def test_ensemble_shape(data, ensemble):
    ensemble = data.draw(ensemble())
    assert len(ensemble.ensemble_shape) == len(ensemble.ensemble_axes_metadata)
    assert len(ensemble.ensemble_shape) == len(ensemble.default_ensemble_chunks)


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [
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
])
def test_ensembles(data, ensemble):
    ensemble = data.draw(ensemble())
    if len(ensemble.ensemble_shape) > 0:
        chunks = data.draw(st.integers(min_value=1, max_value=reduce(mul, ensemble.ensemble_shape)))
    else:
        chunks = ()

    blocks = ensemble.ensemble_blocks(chunks).compute()

    if len(ensemble.ensemble_shape) > 0:
        assert len(blocks.shape) == len(ensemble.ensemble_shape)

    for i, _, fp in ensemble.generate_blocks(chunks):
        assert blocks[i] == fp


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
@pytest.mark.parametrize('ensemble', [
    abtem_st.grid_scan,
    abtem_st.line_scan,
    abtem_st.custom_scan,
    abtem_st.aberrations,
    abtem_st.aperture,
    abtem_st.temporal_envelope,
    abtem_st.spatial_envelope,
    abtem_st.ctf
])
def test_array_waves_transform(data, ensemble, chunks):
    ensemble = data.draw(ensemble())
    blocks = ensemble.ensemble_blocks(chunks).compute()

    waves = data.draw(abtem_st.probe())
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].evaluate(waves)

    array = ensemble.evaluate(waves)
    blocks = concatenate_array_blocks(blocks)

    assert array.shape[:-2] == ensemble.ensemble_shape
    assert blocks.shape[:-2] == ensemble.ensemble_shape
    assert np.allclose(blocks, array)


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [abtem_st.grid_scan,
                                      abtem_st.line_scan,
                                      abtem_st.custom_scan,
                                      abtem_st.aberrations,
                                      abtem_st.aperture,
                                      abtem_st.temporal_envelope,
                                      abtem_st.spatial_envelope,
                                      abtem_st.composite_wave_transform,
                                      abtem_st.ctf
                                      ])
@pytest.mark.parametrize('lazy', [False, True])
def test_apply_waves_transform(data, ensemble, lazy):
    ensemble = data.draw(ensemble())
    waves = data.draw(abtem_st.probe(allow_distribution=False)).build(lazy=lazy)
    waves = ensemble.apply(waves)
    assert waves.shape[:-2] == ensemble.ensemble_shape
