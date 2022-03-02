import os
import tempfile
import uuid

import hypothesis.strategies as st
import numpy as np

from abtem.measure.detect import AnnularDetector, FlexibleAnnularDetector, SegmentedDetector, PixelatedDetector, \
    WavesDetector


@st.composite
def temporary_path(draw, allow_none=True):
    path = st.just(os.path.join(tempfile.gettempdir(), f'abtem-test-{str(uuid.uuid4())}.zarr'))
    if allow_none:
        path = st.one_of(st.just(path), st.none())
    return draw(path)


@st.composite
def detect_every(draw, allow=True):
    if allow:
        return draw(st.integers(min_value=1, max_value=5) | st.just(False))
    else:
        return False


@st.composite
def annular_detector(draw, max_angle=50., allow_detect_every=True):
    inner = draw(st.floats(min_value=0, max_value=max_angle - 1))
    outer = draw(st.floats(min_value=inner + 1, max_value=max_angle))
    offset = st.tuples(st.floats(min_value=-10, max_value=10), st.floats(min_value=-10, max_value=10))
    offset = draw(offset | st.none())
    return AnnularDetector(inner=inner,
                           outer=outer,
                           offset=offset,
                           ensemble_mean=draw(st.booleans()),
                           to_cpu=draw(st.booleans()),
                           url=draw(temporary_path()),
                           detect_every=draw(detect_every(allow_detect_every)))


@st.composite
def flexible_annular_detector(draw, allow_detect_every=True):
    return FlexibleAnnularDetector(ensemble_mean=draw(st.booleans()),
                                   to_cpu=draw(st.booleans()),
                                   url=draw(temporary_path()),
                                   detect_every=draw(detect_every(allow_detect_every)))


@st.composite
def segmented_detector(draw, max_angle=20, allow_detect_every=True):
    inner = draw(st.floats(min_value=0, max_value=max_angle - 1))
    outer = draw(st.floats(min_value=inner + 1, max_value=max_angle))
    nbins_radial = draw(st.integers(min_value=1, max_value=4))
    nbins_azimuthal = draw(st.integers(min_value=1, max_value=4))
    rotation = draw(st.floats(min_value=0., max_value=2 * np.pi))
    return SegmentedDetector(inner=inner,
                             outer=outer,
                             nbins_radial=nbins_radial,
                             nbins_azimuthal=nbins_azimuthal,
                             rotation=rotation,
                             ensemble_mean=draw(st.booleans()),
                             to_cpu=draw(st.booleans()),
                             url=draw(temporary_path()),
                             detect_every=draw(detect_every(allow_detect_every)))


@st.composite
def pixelated_detector(draw, allow_detect_every=True):
    max_angle = draw(st.one_of(st.just('valid'), st.just('cutoff'), st.none(), st.floats(min_value=5, max_value=20)))
    return PixelatedDetector(max_angle=max_angle,
                             ensemble_mean=draw(st.booleans()),
                             to_cpu=draw(st.booleans()),
                             url=draw(temporary_path()),
                             detect_every=draw(detect_every(allow_detect_every)))


@st.composite
def waves_detector(draw, allow_detect_every=True):
    return WavesDetector(to_cpu=draw(st.booleans()), url=draw(temporary_path()),
                         detect_every=draw(detect_every(allow_detect_every)))


@st.composite
def detectors(draw, max_detectors=2, allow_detect_every=True):
    possible_detectors = st.one_of([annular_detector(allow_detect_every=allow_detect_every),
                                    flexible_annular_detector(allow_detect_every=allow_detect_every),
                                    segmented_detector(allow_detect_every=allow_detect_every),
                                    pixelated_detector(allow_detect_every=allow_detect_every),
                                    waves_detector(allow_detect_every=allow_detect_every)
                                    ])

    detectors = st.lists(possible_detectors, min_size=1, max_size=max_detectors)
    return draw(detectors)
