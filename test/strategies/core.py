import dask
import dask.array as da
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as numpy_strats
import os
import tempfile
import uuid
from abtem.core.axes import ScanAxis, OrdinalAxis
from abtem.core.backend import get_array_module


def round_to_multiple(x, base=5):
    return base * round(x / base)


def sensible_floats(allow_nan=False, allow_infinity=False, **kwargs):
    return st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity, **kwargs)


@st.composite
def gpts(draw,
         min_value=32,
         max_value=64,
         allow_none=False,
         base=None):
    gpts = st.integers(min_value=min_value, max_value=max_value)
    gpts = gpts | st.tuples(gpts, gpts)
    if allow_none:
        gpts = gpts | st.none()
    gpts = st.one_of(gpts)
    gpts = draw(gpts)

    if base is not None:
        if isinstance(gpts, int):
            return round_to_multiple(round_to_multiple(gpts, base))
        else:
            return tuple(round_to_multiple(n, base) for n in gpts)

    return gpts


@st.composite
def sampling(draw, min_value=0.01, max_value=0.1, allow_none=False):
    sampling = st.floats(min_value=min_value, max_value=max_value)
    sampling = sampling | st.tuples(sampling, sampling)
    if allow_none:
        sampling = sampling | st.none()
    sampling = st.one_of(sampling)
    sampling = draw(sampling)
    return sampling


@st.composite
def extent(draw, min_value=1., max_value=10., allow_none=False):
    extent = st.floats(min_value=min_value, max_value=max_value)
    extent = extent | st.tuples(extent, extent)
    if allow_none:
        extent = extent | st.none()
    extent = st.one_of(extent)
    extent = draw(extent)
    return extent


@st.composite
def energy(draw, min_value=80e3, max_value=300e3, allow_none=False):
    energy = st.floats(min_value=min_value, max_value=max_value)
    if allow_none:
        energy = energy | st.none()
    energy = draw(energy)
    return energy


@st.composite
def temporary_path(draw, allow_none=True):
    path = st.just(os.path.join(tempfile.gettempdir(), f'abtem-test-{str(uuid.uuid4())}.zarr'))
    if allow_none:
        path = st.one_of(st.just(path), st.none())
    return draw(path)


@st.composite
def scan_axis_metadata(draw):
    sampling = st.floats(min_value=0.01, max_value=.1)
    offset = st.floats(min_value=0., max_value=10.)
    endpoint = draw(st.booleans())
    return ScanAxis(offset=draw(offset), sampling=draw(sampling), endpoint=endpoint)


@st.composite
def ordinal_axis_metadata(draw, n):
    values = draw(st.lists(st.integers(), min_size=n, max_size=n))
    return OrdinalAxis(values=values)


@st.composite
def axes_metadata(draw, shape):
    axes = []
    for n in shape:
        axes += [draw(st.one_of(scan_axis_metadata(), ordinal_axis_metadata(n)))]
    return axes


@st.composite
def shape(draw,
          base_dims=2,
          min_base_side=1,
          max_base_side=32,
          min_ensemble_dims=0,
          max_ensemble_dims=2,
          min_ensemble_side=2,
          max_ensemble_side=4):
    base_shape = draw(numpy_strats.array_shapes(min_dims=base_dims, max_dims=base_dims,
                                                min_side=min_base_side, max_side=max_base_side))
    ensemble_shape = draw(numpy_strats.array_shapes(min_dims=min_ensemble_dims, max_dims=max_ensemble_dims,
                                                    min_side=min_ensemble_side, max_side=max_ensemble_side))
    return ensemble_shape + base_shape


@st.composite
def chunks(draw, shape, min_chunk_size):
    validated_chunks = ()
    for n, c in zip(shape, min_chunk_size):
        if c == -1:
            validated_chunks += (-1,)
        else:
            validated_chunks += (draw(st.integers(min_value=min(c, n), max_value=n)),)

    return validated_chunks


def random_array(shape,
                 min_value=0.,
                 max_value=1.,
                 chunks=None,
                 device='cpu',
                 dtype=np.float32,
                 random_state=None):
    xp = get_array_module(device)

    if chunks is not None:
        random_state = dask.delayed(xp.random.RandomState)(seed=13)

        array = dask.delayed(random_array)(
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            chunks=None,
            device=device,
            dtype=dtype,
            random_state=random_state)

        array = da.from_delayed(array, shape=shape, meta=xp.array((), dtype=dtype))

        return array.rechunk(chunks)

    if random_state is None:
        random_state = xp.random.RandomState(seed=13)

    array = (random_state.rand(*shape) * (max_value - min_value) + min_value).astype(dtype)

    if np.iscomplexobj(array):
        array.imag = (random_state.rand(*shape) * (max_value - min_value) + min_value).astype(np.float32)

    return array
