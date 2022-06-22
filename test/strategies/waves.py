from hypothesis import given

from abtem.waves.waves import Probe, PlaneWave
from abtem.core.energy import energy2wavelength
from . import core as core_st
from . import transfer as transfer_st
import hypothesis.strategies as st


def antialias_cutoff_angle(sampling, energy):
    return energy2wavelength(energy) / sampling * 1e3 / 2 * 2 / 3


@st.composite
def random_probe(draw,
                 # min_gpts=32,
                 # max_gpts=64,
                 # min_extent=5,
                 # max_extent=10,
                 device='cpu',
                 allow_distribution=True):
    gpts = draw(core_st.gpts())
    extent = draw(core_st.extent())
    energy = draw(core_st.energy())
    aberrations = draw(transfer_st.random_aberrations(allow_distribution=allow_distribution))
    aperture = draw(transfer_st.random_aperture(allow_distribution=allow_distribution))
    return Probe(gpts=gpts, extent=extent, energy=energy, aberrations=aberrations, aperture=aperture, device=device)


@st.composite
def random_planewave(draw, device='cpu', allow_distribution=True):
    gpts = draw(core_st.gpts())
    extent = draw(core_st.extent())
    energy = draw(core_st.energy())
    # aberrations = draw(transfer_st.random_aberrations(allow_distribution=allow_distribution))
    # aperture = draw(transfer_st.random_aperture(allow_distribution=allow_distribution))
    return PlaneWave(gpts=gpts, extent=extent, energy=energy, device=device)


@st.composite
def random_waves(draw, device='cpu', allow_distribution=True, lazy=True):
    # aberrations = draw(transfer_st.random_aberrations(allow_distribution=allow_distribution))
    # aperture = draw(transfer_st.random_aperture(allow_distribution=allow_distribution))
    return draw(random_probe(device=device, allow_distribution=allow_distribution)).build(lazy=lazy)
