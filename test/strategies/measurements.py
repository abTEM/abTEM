import dask
import dask.array as da
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as numpy_strats

from abtem.core.axes import ScanAxis
from abtem.core.backend import get_array_module
from abtem.measurements.core import Images, DiffractionPatterns, RealSpaceLineProfiles, PolarMeasurements
from . import core as core_st


@st.composite
def images(draw, lazy=False, device='cpu', min_value=0., min_base_side=1):
    sampling = draw(core_st.sampling(allow_none=False))
    energy = draw(core_st.energy())
    metadata = {'energy': energy}

    shape = draw(core_st.shape(min_base_side=min_base_side))
    axes = draw(core_st.axes_metadata(shape[:-2]))

    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 2) + (8,) * 2))
    else:
        chunks = None

    array = core_st.random_array(shape, chunks=chunks, device=device, min_value=min_value)

    return Images(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def diffraction_patterns(draw, lazy=False, device='cpu', min_scan_dims=0, min_value=0., min_base_side=1):
    sampling = draw(core_st.sampling(allow_none=False))
    metadata = {'energy': draw(core_st.energy())}
    fftshift = draw(st.booleans())

    shape = draw(core_st.shape(min_base_side=min_base_side, min_ensemble_dims=min_scan_dims, max_ensemble_dims=3))

    axes = []
    if min_scan_dims:
        axes = draw(st.lists(core_st.scan_axis_metadata(), min_size=1, max_size=len(shape) - 2))

    axes = draw(core_st.axes_metadata(shape[:-len(axes) - 2])) + axes

    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 2) + (-1,) * 2))
    else:
        chunks = None

    array = core_st.random_array(shape, chunks=chunks, device=device, min_value=min_value)

    return DiffractionPatterns(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata,
                               fftshift=fftshift)


@st.composite
def line_profiles(draw, lazy=False, device='cpu', min_value=0., min_base_side=1):
    sampling = draw(core_st.sensible_floats(min_value=0.01, max_value=0.1))
    metadata = {'energy': draw(core_st.energy())}

    shape = draw(core_st.shape(base_dims=1, min_base_side=min_base_side))

    axes = draw(core_st.axes_metadata(shape[:-1]))

    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 1) + (8,)))
    else:
        chunks = None

    array = core_st.random_array(shape, chunks=chunks, device=device, min_value=min_value)

    return RealSpaceLineProfiles(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def polar_measurements(draw, lazy=False, device='cpu', min_scan_dims=0, min_value=0., min_base_side=1):
    radial_sampling = draw(core_st.sensible_floats(min_value=1, max_value=10.))
    radial_offset = draw(core_st.sensible_floats(min_value=0., max_value=10.))

    azimuthal_sampling = draw(core_st.sensible_floats(min_value=0.01, max_value=np.pi))
    azimuthal_offset = draw(core_st.sensible_floats(min_value=0., max_value=np.pi))

    metadata = {'energy': draw(core_st.energy())}

    shape = draw(core_st.shape(min_base_side=min_base_side, min_ensemble_dims=min_scan_dims, max_ensemble_dims=3))

    axes = []
    if min_scan_dims:
        axes = draw(st.lists(core_st.scan_axis_metadata(), min_size=1, max_size=len(shape) - 2))

    axes = draw(core_st.axes_metadata(shape[:-len(axes) - 2])) + axes
    if lazy:
        chunks = draw(core_st.chunks(shape, (1,) * (len(shape) - 2) + (-1,) * 2))
    else:
        chunks = None

    array = core_st.random_array(shape, chunks=chunks, device=device, min_value=min_value)

    return PolarMeasurements(array=array, radial_sampling=radial_sampling, radial_offset=radial_offset,
                             azimuthal_sampling=azimuthal_sampling, azimuthal_offset=azimuthal_offset,
                             ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def all_measurements(draw, lazy, device):
    return draw(st.one_of(diffraction_patterns(lazy=lazy, device=device), images(lazy=lazy, device=device)))
