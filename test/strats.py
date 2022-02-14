import os
import tempfile
import uuid

import hypothesis.strategies as st
import numpy as np
from ase import Atoms

from abtem import FrozenPhonons, Probe
from abtem.measure.detect import AnnularDetector, FlexibleAnnularDetector, SegmentedDetector, PixelatedDetector, \
    WavesDetector
from hypothesis.extra.numpy import arrays, array_shapes


def round_to_multiple(x, base=5):
    return base * round(x / base)


@st.composite
def gpts(draw, min_value=32, max_value=128, allow_none=False, base=None):
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


# @st.composite
# def tilt(draw, min_value=0., max_value=10.):
#     energy = st.floats(min_value=min_value, max_value=max_value)
#     energy = energy | st.none()
#     energy = draw(energy)
#     return energy


@st.composite
def empty_atoms_data(draw,
                     min_side_length=1.,
                     max_side_length=5.,
                     min_thickness=.5,
                     max_thickness=5.):
    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))
    return {
        'numbers': [],
        'positions': [],
        'cell': cell
    }


@st.composite
def random_atoms_data(draw,
                      min_side_length=1.,
                      max_side_length=5.,
                      min_thickness=.5,
                      max_thickness=5.,
                      max_atoms=10):
    n = draw(st.integers(1, max_atoms))

    numbers = draw(st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n))

    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))

    position = st.tuples(st.floats(min_value=0, max_value=cell[0]),
                         st.floats(min_value=0, max_value=cell[1]),
                         st.floats(min_value=0, max_value=cell[2]))

    positions = draw(st.lists(elements=position, min_size=n, max_size=n))

    return {
        'numbers': numbers,
        'positions': positions,
        'cell': cell
    }


@st.composite
def random_atoms(draw,
                 min_side_length=1.,
                 max_side_length=5.,
                 min_thickness=.5,
                 max_thickness=5.,
                 max_atoms=10):
    n = draw(st.integers(1, max_atoms))
    numbers = draw(st.lists(elements=st.integers(min_value=1, max_value=80), min_size=n, max_size=n))
    cell = draw(st.tuples(st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_side_length, max_value=max_side_length),
                          st.floats(min_value=min_thickness, max_value=max_thickness)))
    position = st.tuples(st.floats(min_value=0, max_value=cell[0]),
                         st.floats(min_value=0, max_value=cell[1]),
                         st.floats(min_value=0, max_value=cell[2]))
    positions = draw(st.lists(elements=position, min_size=n, max_size=n))
    return Atoms(numbers=numbers, positions=positions, cell=cell)


@st.composite
def random_frozen_phonons(draw,
                          min_side_length=1.,
                          max_side_length=5.,
                          min_thickness=.5,
                          max_thickness=5.,
                          max_atoms=10,
                          max_configs=2):
    atoms = draw(random_atoms(min_side_length=min_side_length,
                              max_side_length=max_side_length,
                              min_thickness=min_thickness,
                              max_thickness=max_thickness,
                              max_atoms=max_atoms))
    num_configs = draw(st.integers(min_value=1, max_value=max_configs))
    sigmas = draw(st.floats(min_value=0., max_value=.2))
    return FrozenPhonons(atoms, num_configs=num_configs, sigmas=sigmas, seed=13)


@st.composite
def temporary_path(draw):
    path = os.path.join(tempfile.gettempdir(), f'abtem-test-{str(uuid.uuid4())}.zarr')
    path = draw(st.one_of(st.just(path), st.none()))
    return path


@st.composite
def annular_detector(draw, max_angle=50.):
    inner = draw(st.floats(min_value=0, max_value=max_angle - 1))
    outer = draw(st.floats(min_value=inner + 1, max_value=max_angle))
    ensemble_mean = draw(st.booleans())
    offset = st.tuples(st.floats(min_value=-10, max_value=10), st.floats(min_value=-10, max_value=10))
    offset = draw(offset | st.none())
    to_cpu = draw(st.booleans())
    url = draw(temporary_path())
    return AnnularDetector(inner=inner, outer=outer, offset=offset, ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)


@st.composite
def flexible_annular_detector(draw):
    ensemble_mean = draw(st.booleans())
    to_cpu = draw(st.booleans())
    url = draw(temporary_path())
    return FlexibleAnnularDetector(ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)


@st.composite
def segmented_detector(draw, max_angle=20):
    inner = draw(st.floats(min_value=0, max_value=max_angle - 1))
    outer = draw(st.floats(min_value=inner + 1, max_value=max_angle))
    nbins_radial = draw(st.integers(min_value=1, max_value=4))
    nbins_azimuthal = draw(st.integers(min_value=1, max_value=4))
    rotation = draw(st.floats(min_value=0., max_value=2 * np.pi))
    ensemble_mean = draw(st.booleans())
    to_cpu = draw(st.booleans())
    url = draw(temporary_path())
    return SegmentedDetector(inner=inner,
                             outer=outer,
                             nbins_radial=nbins_radial,
                             nbins_azimuthal=nbins_azimuthal,
                             rotation=rotation,
                             ensemble_mean=ensemble_mean,
                             to_cpu=to_cpu,
                             url=url)


@st.composite
def pixelated_detector(draw):
    max_angle = draw(st.one_of(st.just('valid'), st.just('cutoff'), st.none(), st.floats(min_value=5, max_value=20)))
    ensemble_mean = draw(st.booleans())
    to_cpu = draw(st.booleans())
    url = draw(temporary_path())
    return PixelatedDetector(max_angle=max_angle, ensemble_mean=ensemble_mean, to_cpu=to_cpu, url=url)


@st.composite
def waves_detector(draw):
    to_cpu = draw(st.booleans())
    url = draw(temporary_path())
    return WavesDetector(to_cpu=to_cpu, url=url)


@st.composite
def detectors(draw, max_detectors=2):
    possible_detectors = st.one_of([annular_detector(),
                                    flexible_annular_detector(),
                                    segmented_detector(),
                                    pixelated_detector()])

    detectors = st.lists(possible_detectors, min_size=1, max_size=max_detectors)
    return draw(detectors)


@st.composite
def probe(draw,
          min_gpts=64,
          max_gpts=128,
          max_semiangle_cutoff=30,
          ):
    gpts = draw(gpts(min_value=64, max_value=128))
    semiangle_cutoff = draw(st.floats(min_value=5, max_value=max_semiangle_cutoff))
    return Probe(gpts=gpts, semiangle_cutoff=semiangle_cutoff)


@st.composite
def images(draw):
    sampling = draw(sampling)
    return sampling