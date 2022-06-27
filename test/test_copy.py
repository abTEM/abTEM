import hypothesis.strategies as st
import pytest
from hypothesis import given

import strategies as abtem_st


@given(data=st.data())
@pytest.mark.parametrize('copyable', [
    abtem_st.atoms,
    abtem_st.frozen_phonons,
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
    abtem_st.probe,
    abtem_st.plane_wave,
    abtem_st.waves,
    abtem_st.s_matrix,
    # # # prism_st.random_s_matrix,
    abtem_st.images,
    abtem_st.diffraction_patterns,
    abtem_st.line_profiles,
    abtem_st.polar_measurements,
])
def test_copy_equals(data, copyable):
    original = data.draw(copyable())
    assert original.copy() == original
