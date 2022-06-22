from functools import reduce
from operator import mul

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume, reproduce_failure

from abtem.core.ensemble import concatenate_array_blocks
import strategies.atoms as atoms_st
import strategies.potentials as potentials_st
import strategies.scan as scan_st
import strategies.transfer as transfer_st
import strategies.waves as waves_st


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [atoms_st.random_frozen_phonons,
                                      scan_st.grid_scan,
                                      scan_st.line_scan,
                                      scan_st.custom_scan,
                                      potentials_st.random_potential,
                                      potentials_st.random_potential_array,
                                      transfer_st.random_aberrations,
                                      transfer_st.random_aperture,
                                      transfer_st.random_temporal_envelope,
                                      transfer_st.random_spatial_envelope,
                                      transfer_st.random_composite_wave_transform,
                                      transfer_st.random_ctf,
                                      ])
def test_ensemble_shape(data, ensemble):
    ensemble = data.draw(ensemble())
    assert len(ensemble.ensemble_shape) == len(ensemble.ensemble_axes_metadata)
    assert len(ensemble.ensemble_shape) == len(ensemble.default_ensemble_chunks)


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [atoms_st.random_frozen_phonons,
                                      scan_st.grid_scan,
                                      scan_st.line_scan,
                                      scan_st.custom_scan,
                                      potentials_st.random_potential,
                                      potentials_st.random_potential_array,
                                      transfer_st.random_aberrations,
                                      transfer_st.random_aperture,
                                      transfer_st.random_temporal_envelope,
                                      transfer_st.random_spatial_envelope,
                                      transfer_st.random_composite_wave_transform,
                                      transfer_st.random_ctf, ])
def test_ensembles(data, ensemble):
    ensemble = data.draw(ensemble())
    assume(len(ensemble.ensemble_shape) > 0)
    chunks = data.draw(st.integers(min_value=1, max_value=reduce(mul, ensemble.ensemble_shape)))

    blocks = ensemble.ensemble_blocks(chunks).compute()

    assert len(blocks.shape) == len(ensemble.ensemble_shape)

    for i, _, fp in ensemble.generate_blocks(chunks):
        assert blocks[i] == fp


@given(data=st.data(), chunks=st.integers(min_value=1, max_value=10))
@pytest.mark.parametrize('ensemble', [scan_st.grid_scan,
                                      scan_st.line_scan,
                                      scan_st.custom_scan,
                                      transfer_st.random_aberrations,
                                      transfer_st.random_aperture,
                                      transfer_st.random_temporal_envelope,
                                      transfer_st.random_spatial_envelope,
                                      transfer_st.random_ctf
                                      ])
def test_array_waves_transform(data, ensemble, chunks):
    ensemble = data.draw(ensemble())
    blocks = ensemble.ensemble_blocks(chunks).compute()

    waves = data.draw(waves_st.random_probe())
    for i in np.ndindex(blocks.shape):
        blocks[i] = blocks[i].evaluate(waves)

    array = ensemble.evaluate(waves)
    blocks = concatenate_array_blocks(blocks)

    assert array.shape[:-2] == ensemble.ensemble_shape
    assert blocks.shape[:-2] == ensemble.ensemble_shape
    assert np.allclose(blocks, array)


@given(data=st.data())
@pytest.mark.parametrize('ensemble', [scan_st.grid_scan,
                                      scan_st.line_scan,
                                      scan_st.custom_scan,
                                      transfer_st.random_aberrations,
                                      transfer_st.random_aperture,
                                      transfer_st.random_temporal_envelope,
                                      transfer_st.random_spatial_envelope,
                                      transfer_st.random_composite_wave_transform,
                                      transfer_st.random_ctf
                                      ])
@pytest.mark.parametrize('lazy', [False, True])
def test_apply_waves_transform(data, ensemble, lazy):
    ensemble = data.draw(ensemble())
    waves = data.draw(waves_st.random_probe(allow_distribution=False)).build(lazy=lazy)
    waves = ensemble.apply(waves)
    assert waves.shape[:-2] == ensemble.ensemble_shape
