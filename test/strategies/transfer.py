import hypothesis.strategies as st
import numpy as np

from abtem import ParameterSeries
from abtem.waves.transfer import polar_symbols, Aberrations, Aperture, TemporalEnvelope, CompositeWaveTransform, \
    SpatialEnvelope, CTF
from . import core as core_st
from . import scan as scan_st


@st.composite
def random_linspace(draw, min_value=-100., max_value=100.):
    num = draw(st.integers(min_value=1, max_value=10))
    start = draw(st.floats(min_value=min_value, max_value=max_value))
    extent = draw(st.floats(min_value=0., max_value=max_value - start))
    return np.linspace(start=start, stop=start + extent, num=num)


@st.composite
def random_parameter_series(draw, min_value=-100, max_value=100):
    return ParameterSeries(draw(random_linspace(min_value=min_value, max_value=max_value)))


@st.composite
def random_parameter_value(draw, min_value=-100, max_value=100, allow_distribution=True):
    scalar_value = st.floats(min_value=min_value, max_value=max_value)
    if not allow_distribution:
        return draw(scalar_value)
    return draw(st.one_of(random_parameter_series(min_value=min_value, max_value=max_value), scalar_value))


@st.composite
def random_aberrations(draw, allow_distribution=True):
    n = draw(st.integers(min_value=1, max_value=2))
    symbols = draw(st.permutations(polar_symbols).map(lambda x: x[:n]))

    parameters = {}
    for symbol in symbols:
        parameters[symbol] = draw(random_parameter_value(allow_distribution=allow_distribution))

    return Aberrations(**parameters)


@st.composite
def random_aperture(draw, allow_distribution=True):
    semiangle_cutoff = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    normalize = draw(st.booleans())
    energy = draw(core_st.energy())
    taper = draw(st.floats(min_value=0., max_value=1.))
    return Aperture(semiangle_cutoff=semiangle_cutoff, energy=energy, taper=taper, normalize=normalize)


@st.composite
def random_temporal_envelope(draw, allow_distribution=True):
    focal_spread = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    normalize = draw(st.booleans())
    energy = draw(core_st.energy())
    return TemporalEnvelope(focal_spread=focal_spread, energy=energy, normalize=normalize)


@st.composite
def random_spatial_envelope(draw, allow_distribution=True, aberrations=None):
    angular_spread = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    aberrations = draw(random_aberrations())
    normalize = draw(st.booleans())
    energy = draw(core_st.energy())
    return SpatialEnvelope(angular_spread=angular_spread, energy=energy, normalize=normalize, aberrations=aberrations)


@st.composite
def random_composite_wave_transform(draw, allow_distribution=True):
    n = draw(st.integers(min_value=1, max_value=2))
    wave_transforms = [random_aberrations(allow_distribution=allow_distribution),
                       random_aperture(allow_distribution=allow_distribution),
                       random_temporal_envelope(allow_distribution=allow_distribution),
                       random_spatial_envelope(allow_distribution=allow_distribution),
                       scan_st.grid_scan(),
                       scan_st.line_scan(),
                       scan_st.custom_scan(),
                       ]
    sampled_wave_transforms = draw(st.lists(st.one_of(wave_transforms), min_size=1, max_size=n))
    return CompositeWaveTransform(sampled_wave_transforms)


@st.composite
def random_ctf(draw, allow_distribution=True):
    aberrations = draw(random_aberrations(allow_distribution=allow_distribution))
    semiangle_cutoff = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    angular_spread = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    focal_spread = draw(random_parameter_value(min_value=5, max_value=20, allow_distribution=allow_distribution))
    energy = draw(core_st.energy())
    return CTF(aberrations=aberrations, semiangle_cutoff=semiangle_cutoff, angular_spread=angular_spread,
               focal_spread=focal_spread, energy=energy)
