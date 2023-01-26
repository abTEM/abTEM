import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, assume, reproduce_failure
from matplotlib import pyplot as plt

from test_grid import check_grid_consistent
from utils import (
    assert_array_matches_device,
    gpu,
    remove_dummy_dimensions,
    assert_array_matches_laziness,
)

import strategies as abtem_st


# @pytest.mark.parametrize("builder", [Probe, plane_wave, SMatrix])
# @given(grid_data=grid_data())
# def test_grid_raises(grid_data, builder):
#     probe = builder(**grid_data)
#     try:
#         probe.grid.check_is_defined()
#     except GridUndefinedError:
#         with pytest.raises(GridUndefinedError):
#             probe.build()
#
#
# @given(grid_data=grid_data(), energy=core_st.energy(allow_none=True))
# @pytest.mark.parametrize("builder", [Probe, plane_wave, SMatrix])
# def test_energy_raises(grid_data, energy, builder):
#     probe = builder(energy=energy, **grid_data)
#     assume(energy is None)
#     with pytest.raises(EnergyUndefinedError):
#         probe.build()



@pytest.mark.parametrize(
    "waves_builder", [abtem_st.probe]
)
@pytest.mark.parametrize("device", [gpu])
@pytest.mark.parametrize("lazy", [False])
@given(data=st.data())
def test_can_build(data, waves_builder, device, lazy):
    waves_builder = data.draw(waves_builder(device=device))

    waves = waves_builder.build(lazy=lazy)#.compute()

    assert_array_matches_device(waves.array, device)

    assert waves.gpts == waves_builder.gpts
    assert waves_builder.ensemble_shape == waves.ensemble_shape
    assert waves.array.shape[-2:] == waves.gpts
    assert waves.array.shape[: -len(waves.base_shape)] == waves.ensemble_shape
    assert waves.array.dtype == np.complex64

    assert np.all(np.isclose(waves_builder.extent, waves.extent))
    check_grid_consistent(waves.extent, waves.gpts, waves.sampling)

    assert np.isclose(waves.energy, waves_builder.energy)


@given(data=st.data())
@pytest.mark.parametrize(
    "waves_builder", [abtem_st.probe, abtem_st.plane_wave, abtem_st.s_matrix]
)
@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("lazy", [True, False])
def test_can_compute(data, waves_builder, device, lazy):
    waves_builder = data.draw(waves_builder(device=device))
    waves = waves_builder.build(lazy=lazy)

    assert_array_matches_laziness(waves.array, lazy=lazy)
    waves.compute()

    assert_array_matches_laziness(waves.array, lazy=False)
    assert waves.array.shape[-2:] == waves_builder.gpts
    assert waves_builder.shape == waves.shape
    assert waves.array.shape[: -len(waves.base_shape)] == waves.ensemble_shape
    assert waves.array.dtype == np.complex64


@given(data=st.data(), potential=abtem_st.potential())
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.s_matrix,
    ],
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_can_multislice(data, potential, waves_builder, lazy, device):
    waves_builder = data.draw(waves_builder(device=device))
    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy)

    assert potential.ensemble_shape + waves_builder.shape == waves.shape
    assert_array_matches_laziness(waves.array, lazy=lazy)
    waves.compute()
    assert potential.ensemble_shape + waves_builder.shape == waves.shape
    assert_array_matches_laziness(waves.array, lazy=False)
    assert waves.array.dtype == np.complex64


def assert_is_normalized(waves):
    assert np.allclose(
        waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)), 1.0
    )


@given(data=st.data())
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.s_matrix,
    ],
)
@pytest.mark.parametrize("lazy", [False])
def test_normalized(data, waves_builder, lazy):
    try:
        waves_builder = data.draw(waves_builder(normalize=True))
    except TypeError:
        waves_builder = data.draw(waves_builder())

    waves = waves_builder.build(lazy=lazy)
    try:
        waves = waves.reduce()
    except AttributeError:
        pass
    waves.compute()
    assert_is_normalized(waves)


@given(data=st.data(), atoms=abtem_st.atoms())
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.s_matrix,
    ],
)
@pytest.mark.parametrize("lazy", [True, False])
def test_empty_multislice_normalized(data, atoms, waves_builder, lazy):
    try:
        waves_builder = data.draw(waves_builder(normalize=True))
    except TypeError:
        waves_builder = data.draw(waves_builder())

    atoms = atoms[:0]

    waves = waves_builder.multislice(atoms, lazy=lazy)
    try:
        waves = waves.reduce()
    except AttributeError:
        pass
    waves.compute()
    assert_is_normalized(waves)


@given(data=st.data(), potential=abtem_st.potential())
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
        #abtem_st.s_matrix,
    ],
)
def test_multislice_scatter(data, potential, waves_builder, lazy):
    try:
        waves_builder = data.draw(waves_builder(normalize=True))
    except TypeError:
        waves_builder = data.draw(waves_builder())

    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy).compute()

    try:
        waves = waves.reduce()
    except AttributeError:
        pass

    assert np.all(
        waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)) < 1.0001
    ) or np.allclose(
        waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)), 1.00002
    )


@given(data=st.data(), potential=abtem_st.potential())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.s_matrix,
    ],
)
def test_build_then_multislice(data, waves_builder, potential, lazy):
    waves_builder = data.draw(waves_builder())
    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy).compute()

    build_waves = waves_builder.build(lazy=lazy)
    build_waves = build_waves.multislice(potential).compute()

    try:
        build_waves = build_waves.reduce()
        waves = waves.reduce()
    except AttributeError:
        pass
    assert np.allclose(build_waves.array, waves.array)


@given(data=st.data())
@pytest.mark.parametrize(
    "transform",
    [
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        abtem_st.composite_wave_transform,
        abtem_st.ctf,
    ],
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_apply_transform(data, transform, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    transform = data.draw(transform())
    assume(len(transform.ensemble_shape + waves.shape) < 6)
    transformed_waves = waves.apply_transform(transform)
    assert transformed_waves.shape == transform.ensemble_shape + waves.shape


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_intensity(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    images = waves.intensity()
    assert images.shape == waves.shape
    assert images.array.dtype == np.float32


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_images(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    images = waves.images()
    assert images.shape == waves.shape
    assert images.array.dtype == np.complex64


@given(
    data=st.data(),
    max_angle=st.just("valid")
    | st.just("cutoff")
    | st.floats(min_value=10, max_value=100),
    normalization=st.sampled_from(["intensity", "values"]),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_downsample(data, max_angle, normalization, lazy, device):
    probe = data.draw(abtem_st.probe(device=device, allow_distribution=False))
    waves = probe.build(lazy=lazy)
    old_gpts = waves.gpts
    valid_gpts = waves.antialias_valid_gpts
    cutoff_gpts = waves.antialias_cutoff_gpts
    old_max = waves.intensity().array.max(axis=(-2, -1))

    waves = waves.downsample(max_angle=max_angle, normalization=normalization)

    if isinstance(max_angle, float):
        assume(max_angle < 0.9 * max(waves.cutoff_angles))
        assume(max_angle > 1.1 * probe.aperture.semiangle_cutoff)
    elif max_angle == "valid":
        assume(
            min(probe.rectangle_cutoff_angles) > 1.1 * probe.aperture.semiangle_cutoff
        )
    elif max_angle == "cutoff":
        assume(min(probe.cutoff_angles) > 1.1 * probe.aperture.semiangle_cutoff)

    assert waves.gpts != old_gpts
    assert waves.array.dtype == np.complex64

    if max_angle == "valid":
        assert waves.gpts == valid_gpts
    elif max_angle == "cutoff":
        assert waves.gpts == cutoff_gpts

    if normalization == "intensity":
        assert_is_normalized(waves)
    elif normalization == "values":
        np.allclose(old_max, waves.intensity().array.max(axis=(-2, -1)))


@given(
    data=st.data(),
    max_angle=st.sampled_from(["cutoff", "valid"]),
    fftshift=st.booleans(),
    block_direct=st.one_of(
        (st.floats(min_value=0.0, max_value=5.0), st.just(False), st.just(None))
    ),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_diffraction_patterns(data, max_angle, fftshift, block_direct, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))

    assume(min(waves._gpts_within_angle(max_angle)) > 0)

    diffraction_patterns = waves.diffraction_patterns(
        max_angle=max_angle, fftshift=fftshift, block_direct=block_direct
    )
    assert diffraction_patterns.array.dtype == np.float32


@given(
    data=st.data(),
    repetitions=st.tuples(
        st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
    ),
    renormalize=st.booleans(),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_tile(data, repetitions, renormalize, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    old_extent = waves.extent
    old_sum = waves.diffraction_patterns(max_angle=None).to_cpu().compute().array.sum((-2,-1))
    tiled = waves.tile(repetitions, renormalize=renormalize)


    assert np.allclose(
        (old_extent[0] * repetitions[0], old_extent[1] * repetitions[1]), tiled.extent
    )
    if renormalize:
        new_sum = tiled.diffraction_patterns(max_angle=None).to_cpu().compute().array.sum((-2, -1))
        assert np.allclose(old_sum, new_sum)
