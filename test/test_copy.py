import hypothesis.strategies as st
import pytest
from hypothesis import given

from strategies.potentials import random_potential
import strategies.atoms as atoms_st
import strategies.potentials as potentials_st
import strategies.scan as scan_st
import strategies.transfer as transfer_st
import strategies.waves as waves_st
import strategies.prism as prism_st


@given(data=st.data())
@pytest.mark.parametrize('copyable', [
    atoms_st.random_frozen_phonons,
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
    waves_st.random_probe,
    waves_st.random_planewave,
    prism_st.random_s_matrix,
])
def test_copy_equals(data, copyable):
    original = data.draw(copyable())
    assert original.copy() == original
