import hypothesis.strategies as st


def round_to_multiple(x, base=5):
    return base * round(x / base)


def sensible_floats(allow_nan=False, allow_infinity=False, **kwargs):
    return st.floats(allow_nan=allow_nan, allow_infinity=allow_infinity, **kwargs)


@st.composite
def gpts(draw, min_value=32, max_value=64, allow_none=False, base=None):
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
