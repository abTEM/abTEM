from abtem.waves.prism import SMatrix
from abtem.core.energy import energy2wavelength
from . import core as core_st
from . import transfer as transfer_st
import hypothesis.strategies as st


def antialias_cutoff_angle(sampling, energy):
    return energy2wavelength(energy) / sampling * 1e3 / 2 * 2 / 3


@st.composite
def random_s_matrix(draw, device='cpu', interpolation=False, downsample=False, potential=None):
    gpts = draw(core_st.gpts())
    extent = draw(core_st.extent())
    energy = draw(core_st.energy())
    planewave_cutoff = draw(st.floats(5, 15))

    s_matrix = SMatrix(potential=potential,
                       gpts=gpts,
                       extent=extent,
                       energy=energy,
                       planewave_cutoff=planewave_cutoff,
                       downsample=downsample,
                       device=device)
    return s_matrix

    # aberrations = draw(transfer_st.random_aberrations(allow_distribution=allow_distribution))
    # aperture = draw(transfer_st.random_aperture(allow_distribution=allow_distribution))
    # return Probe(gpts=gpts, extent=extent, energy=energy, aberrations=aberrations, aperture=aperture, device=device)
