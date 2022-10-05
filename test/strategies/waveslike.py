import hypothesis.strategies as st
import numpy as np

from abtem.core.energy import energy2wavelength
from abtem.prism.s_matrix import SMatrix
from abtem.waves import Probe, PlaneWave, Waves
from . import core as core_st
from . import transfer as transfer_st


def antialias_cutoff_angle(sampling, energy):
    return energy2wavelength(energy) / sampling * 1e3 / 2 * 2 / 3


@st.composite
def probe(
    draw,
    # min_gpts=32,
    # max_gpts=64,
    # min_extent=5,
    # max_extent=10,
    device="cpu",
    allow_distribution=True,
):
    gpts = draw(core_st.gpts())
    extent = draw(core_st.extent())
    energy = draw(core_st.energy())
    aberrations = draw(transfer_st.aberrations(allow_distribution=allow_distribution))
    aperture = draw(transfer_st.aperture(allow_distribution=allow_distribution))
    return Probe(
        semiangle_cutoff=30,
        gpts=gpts,
        extent=extent,
        energy=energy,
        aberrations=aberrations,
        aperture=aperture,
        device=device,
    )


@st.composite
def plane_wave(draw, device="cpu", normalize=False, allow_distribution=True):
    gpts = draw(core_st.gpts())
    extent = draw(core_st.extent())
    energy = draw(core_st.energy())
    # aberrations = draw(transfer_st.random_aberrations(allow_distribution=allow_distribution))
    # aperture = draw(transfer_st.random_aperture(allow_distribution=allow_distribution))
    return PlaneWave(gpts=gpts, extent=extent, energy=energy, device=device, normalize=normalize)


@st.composite
def waves(draw, device="cpu", lazy=True, min_scan_dims=0, min_base_side=8):
    shape = draw(
        core_st.shape(
            min_base_side=min_base_side,
            min_ensemble_dims=min_scan_dims,
            max_ensemble_dims=3,
        )
    )

    axes = []
    if min_scan_dims:
        axes = draw(
            st.lists(core_st.scan_axis_metadata(), min_size=1, max_size=len(shape) - 2)
        )

    axes = draw(core_st.axes_metadata(shape[: -len(axes) - 2])) + axes

    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 2) + (-1,) * 2))
    else:
        chunks = None

    array = core_st.random_array(
        shape, chunks=chunks, device=device, min_value=0.0, dtype=np.complex64
    )
    sampling = draw(core_st.sampling())
    energy = draw(core_st.energy())
    return Waves(
        array=array, sampling=sampling, ensemble_axes_metadata=axes, energy=energy
    )


@st.composite
def s_matrix(
    draw,
    device="cpu",
    min_interpolation=1,
    max_interpolation=1,
    min_planewave_cutoff=5.0,
    max_planewave_cutoff=20.0,
    downsample=False,
    potential=None,
    allow_distribution=False,
    store_on_host=False,
):

    if potential is None:
        gpts = draw(core_st.gpts())
        extent = draw(core_st.extent())
    else:
        gpts = None
        extent = None

    energy = draw(core_st.energy())
    interpolation = draw(
        st.integers(min_value=min_interpolation, max_value=max_interpolation)
    )
    planewave_cutoff = draw(
        st.floats(min_value=min_planewave_cutoff, max_value=max_planewave_cutoff)
    )

    s_matrix = SMatrix(
        potential=potential,
        gpts=gpts,
        extent=extent,
        energy=energy,
        planewave_cutoff=planewave_cutoff,
        interpolation=interpolation,
        downsample=downsample,
        device=device,
        store_on_host=store_on_host,
    )
    return s_matrix


@st.composite
def s_matrix_array(draw, lazy, **kwargs):
    return draw(s_matrix(**kwargs)).build(lazy=lazy)
