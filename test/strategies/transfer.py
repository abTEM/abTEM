import hypothesis.strategies as st
import numpy as np

from abtem.core import distributions
from abtem.waves.transfer import polar_symbols, Aberrations, Aperture, TemporalEnvelope, CompositeWaveTransform, \
    SpatialEnvelope, CTF
from . import core as core_st
from . import scan as scan_st


@st.composite
def uniform_distribution(draw, min_value=-100., max_value=100.):
    num = draw(st.integers(min_value=1, max_value=10))
    low = draw(st.floats(min_value=min_value, max_value=max_value))
    extent = draw(st.floats(min_value=0., max_value=max_value - low))
    return distributions.uniform(low=low, high=low + extent, num_samples=num)


@st.composite
def parameter(draw, min_value=-100, max_value=100, allow_distribution=True):
    scalar_value = st.floats(min_value=min_value, max_value=max_value)
    if not allow_distribution:
        return draw(scalar_value)
    return draw(st.one_of(uniform_distribution(min_value=min_value, max_value=max_value), scalar_value))


@st.composite
def aberrations(draw, allow_distribution=True):
    n = draw(st.integers(min_value=0, max_value=2))
    symbols = draw(st.permutations(polar_symbols).map(lambda x: x[:n]))

    parameters = {}
    for symbol in symbols:
        parameters[symbol] = draw(parameter(allow_distribution=allow_distribution))

    return Aberrations(**parameters)


@st.composite
def aperture(draw, allow_distribution=True):
    semiangle_cutoff = draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution))
    normalize = draw(st.booleans())
    energy = draw(core_st.energy())
    taper = draw(st.floats(min_value=0., max_value=1.))
    return Aperture(semiangle_cutoff=semiangle_cutoff, energy=energy, taper=taper)


@st.composite
def temporal_envelope(draw, allow_distribution=True):
    focal_spread = draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution))
    normalize = draw(st.booleans())
    energy = draw(core_st.energy())
    return TemporalEnvelope(focal_spread=focal_spread, energy=energy, normalize=normalize)


@st.composite
def spatial_envelope(draw, allow_distribution=True):
    return SpatialEnvelope(
        angular_spread=draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution)),
        energy=draw(core_st.energy()),
        normalize=draw(st.booleans()),
        aberrations=draw(aberrations()))


@st.composite
def composite_wave_transform(draw, allow_distribution=True):
    n = draw(st.integers(min_value=1, max_value=2))
    wave_transforms = [
        aberrations(allow_distribution=allow_distribution),
        aperture(allow_distribution=allow_distribution),
        temporal_envelope(allow_distribution=allow_distribution),
        spatial_envelope(allow_distribution=allow_distribution),
        scan_st.grid_scan(),
        scan_st.line_scan(),
        scan_st.custom_scan(),
    ]
    sampled_wave_transforms = draw(st.lists(st.one_of(wave_transforms), min_size=1, max_size=n))
    return CompositeWaveTransform(sampled_wave_transforms)


@st.composite
def ctf(draw, allow_distribution=True):
    return CTF(
        aberrations=draw(aberrations(allow_distribution=allow_distribution)),
        semiangle_cutoff=draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution)),
        angular_spread=draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution)),
        focal_spread=draw(parameter(min_value=5, max_value=20, allow_distribution=allow_distribution)),
        energy=draw(core_st.energy()))
