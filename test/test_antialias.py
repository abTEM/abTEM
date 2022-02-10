import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume, settings

import strats as abst
from abtem import Probe, SMatrix, GridScan, Potential
from abtem.core.antialias import AntialiasAperture
from abtem.potentials.temperature import AbstractFrozenPhonons
from utils import gpu, ensure_is_tuple


@given(gpts=abst.gpts(min_value=64, max_value=128),
       sampling=abst.sampling(min_value=.01, max_value=.2),
       downsample = st.just('valid') | st.just('cutoff') | st.floats(min_value=10, max_value=100),
       energy=st.floats(100e3, 200e3, width=32))
def test_downsample(gpts, sampling, energy, downsample):
    wave = Probe(energy=energy, gpts=gpts, sampling=sampling).build()
    old_valid_gpts = wave.antialias_valid_gpts
    old_cutoff_gpts = wave.antialias_cutoff_gpts

    if isinstance(downsample, float):
        assume(downsample < max(wave.cutoff_angles))

    wave = wave.downsample(downsample)

    assert wave.antialias_valid_gpts == old_valid_gpts
    assert wave.antialias_cutoff_gpts == old_cutoff_gpts
