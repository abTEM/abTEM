import dask
import dask.array as da
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as numpy_strats

from abtem.core.axes import ScanAxis
from abtem.core.backend import get_array_module
from abtem.measure.measure import Images, DiffractionPatterns, LineProfiles, PolarMeasurements
from . import core as core_st


@st.composite
def scan_axis_metadata(draw):
    sampling = st.floats(min_value=0.01, max_value=.1)
    offset = st.floats(min_value=0., max_value=10.)
    endpoint = draw(st.booleans())
    return ScanAxis(offset=draw(offset), sampling=draw(sampling), endpoint=endpoint)


@st.composite
def axes_metadata(draw):
    axes = draw(st.lists(elements=scan_axis_metadata(), min_size=0, max_size=2))
    return axes


@st.composite
def measurement_array(draw,
                      num_dims,
                      lazy=False,
                      device='cpu',
                      min_value=0.,
                      max_value=1.,
                      min_side=1,
                      max_side=24,
                      base_axes_chunks=True):
    shape = draw(numpy_strats.array_shapes(min_dims=num_dims, max_dims=num_dims, min_side=min_side, max_side=max_side))

    xp = get_array_module(device)

    def make_array(random_state, shape):
        return (random_state.rand(*shape) * (max_value - min_value) + min_value).astype(np.float32)

    if lazy:
        random_state = dask.delayed(xp.random.RandomState)(seed=13)
        array = dask.delayed(make_array)(random_state, shape)
        array = da.from_delayed(array, shape=shape, meta=xp.array((), dtype=np.float32))

        if base_axes_chunks:
            chunks = draw(st.tuples(*tuple(st.integers(min_value=8, max_value=max(n, 8)) for n in shape)))
        else:
            chunks = draw(st.tuples(*tuple(st.integers(min_value=8, max_value=max(n, 8)) for n in shape[:-2])))
            chunks += ((array.shape[-2]), (array.shape[-1]))

        array = array.rechunk(chunks)
    else:
        random_state = xp.random.RandomState(seed=13)
        array = make_array(random_state, shape)

    return array


@st.composite
def images(draw, lazy=False, device='cpu', **kwargs):
    sampling = draw(core_st.sampling(allow_none=False))
    energy = draw(core_st.energy())
    metadata = {'energy': energy}
    axes = draw(axes_metadata())
    array = draw(measurement_array(len(axes) + 2, lazy, device, **kwargs))
    return Images(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def diffraction_patterns(draw, lazy=False, device='cpu', **kwargs):
    sampling = draw(core_st.sampling(allow_none=False))
    metadata = {'energy': draw(core_st.energy())}
    axes = draw(axes_metadata())
    fftshift = draw(st.booleans())
    array = draw(measurement_array(len(axes) + 2, lazy, device, base_axes_chunks=False, **kwargs))
    return DiffractionPatterns(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata,
                               fftshift=fftshift)


@st.composite
def line_profiles(draw, lazy=False, device='cpu'):
    sampling = draw(core_st.sensible_floats(min_value=0.01, max_value=0.1))
    metadata = {'energy': draw(core_st.energy())}
    axes = draw(axes_metadata())
    #
    # start_floats = core_st.sensible_floats(min_value=-10., max_value=10.)
    # start = draw(st.tuples(start_floats, start_floats))
    #
    # end = draw(st.tuples(core_st.sensible_floats(min_value=start[0] + 1., max_value=start[0] + 10.),
    #                      core_st.sensible_floats(min_value=start[1] + 1., max_value=start[1] + 10.)))

    array = draw(measurement_array(len(axes) + 1, lazy, device))
    return LineProfiles(array=array, sampling=sampling, ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def polar_measurements(draw, lazy=False, device='cpu', **kwargs):
    radial_sampling = draw(core_st.sensible_floats(min_value=1, max_value=10.))
    radial_offset = draw(core_st.sensible_floats(min_value=0., max_value=10.))

    azimuthal_sampling = draw(core_st.sensible_floats(min_value=0.01, max_value=np.pi))
    azimuthal_offset = draw(core_st.sensible_floats(min_value=0., max_value=np.pi))

    metadata = {'energy': draw(core_st.energy())}
    axes = draw(axes_metadata())

    array = draw(measurement_array(len(axes) + 2, lazy, device, base_axes_chunks=False, **kwargs))
    return PolarMeasurements(array=array, radial_sampling=radial_sampling, radial_offset=radial_offset,
                             azimuthal_sampling=azimuthal_sampling, azimuthal_offset=azimuthal_offset,
                             ensemble_axes_metadata=axes, metadata=metadata)


@st.composite
def all_measurements(draw, lazy, device):
    return draw(st.one_of(diffraction_patterns(lazy=lazy, device=device), images(lazy=lazy, device=device)))
