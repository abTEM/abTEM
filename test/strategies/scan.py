import dask
import dask.array as da
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra import numpy as numpy_strats

from abtem.core.axes import ScanAxis
from abtem.core.backend import get_array_module
from abtem.measure.measure import Images
from . import core as core_strats
from abtem.waves.scan import CustomScan, LineScan, GridScan

@st.composite
def linescan(draw):
    sampling = draw(core_strats.sampling(allow_none=False))

    

