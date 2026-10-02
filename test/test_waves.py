import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given, settings
from test_grid import check_grid_consistent
from utils import (
    assert_array_matches_device,
    assert_array_matches_laziness,
    devices,
    gpu,
)

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
    "waves_builder", [abtem_st.probe, abtem_st.plane_wave, abtem_st.s_matrix]
)
@devices
@pytest.mark.parametrize("lazy", [False, True])
@given(data=st.data())
def test_can_build(data, waves_builder, device, lazy):
    waves_builder = data.draw(waves_builder(device=device))

    waves = waves_builder.build(lazy=lazy)

    assert_array_matches_device(waves.array, device)

    assert waves.gpts == waves_builder.gpts
    assert waves_builder.ensemble_shape == waves.ensemble_shape
    assert waves.array.shape[-2:] == waves.gpts
    assert waves.array.shape[: -len(waves.base_shape)] == waves.ensemble_shape
    assert waves.array.dtype == np.complex64

    assert np.all(np.isclose(waves_builder.extent, waves.extent))
    check_grid_consistent(waves.extent, waves.gpts, waves.sampling)

    assert np.isclose(waves.energy, waves_builder.accelerator.energy)


@given(data=st.data())
@pytest.mark.parametrize(
    "waves_builder", [abtem_st.probe, abtem_st.plane_wave, abtem_st.s_matrix]
)
@devices
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
@devices
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
        abtem_st.s_matrix,
    ],
)
def test_multislice_scatter(data, potential, waves_builder, lazy):
    try:
        waves_builder = data.draw(waves_builder(normalize=True))
    except TypeError:
        waves_builder = data.draw(waves_builder())

    waves_builder.grid.match(potential)

    waves = waves_builder.build().compute()

    # old_sum = waves.diffraction_patterns(max_angle="full").array.sum()

    waves = waves.multislice(potential).compute()

    try:
        waves = waves.reduce()
    except AttributeError:
        pass

    # new_sum = waves.diffraction_patterns(max_angle="full").array.sum()

    # print(old_sum, new_sum, old_sum > new_sum, potential.array)
    # print(waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)))

    assert np.all(
        waves.diffraction_patterns(max_angle=None).array.sum(axis=(-2, -1)) < 1.0005
    )


@settings(max_examples=5)
@given(data=st.data(), potential=abtem_st.potential())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.probe,
        abtem_st.plane_wave,
    ],
)
@pytest.mark.parametrize(
    "detectors",
    [
        abtem_st.pixelated_detector,
        abtem_st.waves_detector,
        abtem_st.segmented_detector,
        None,
    ],
)
def test_build_then_multislice(data, waves_builder, detectors, potential, lazy):
    waves_builder = data.draw(waves_builder())
    waves_builder.grid.match(potential)

    if detectors is not None:
        detectors = data.draw(detectors())

    waves = waves_builder.multislice(
        potential, detectors=detectors, lazy=lazy
    ).compute()

    build_waves = waves_builder.build(lazy=lazy)
    build_waves = build_waves.multislice(potential, detectors=detectors).compute()

    assert np.allclose(build_waves.array, waves.array)


@given(data=st.data(), potential=abtem_st.potential())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize(
    "waves_builder",
    [
        abtem_st.s_matrix,
    ],
)
def test_build_then_multislice_s_matrix(data, waves_builder, potential, lazy):
    waves_builder = data.draw(waves_builder())
    waves_builder.grid.match(potential)

    waves = waves_builder.multislice(potential, lazy=lazy).compute()

    build_waves = waves_builder.build(lazy=lazy)
    build_waves = build_waves.multislice(potential).compute()

    build_waves = build_waves.reduce()
    waves = waves.reduce()

    assert np.allclose(build_waves.array, waves.array)


@settings(max_examples=5)
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
        # abtem_st.composite_wave_transform,
        abtem_st.ctf,
    ],
)
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_apply_transform(data, transform, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    transform = data.draw(transform())
    assume(len(transform.ensemble_shape + waves.shape) < 6)
    transformed_waves = waves.apply_transform(transform)
    assert transformed_waves.shape == transform.ensemble_shape + waves.shape


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_intensity(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    images = waves.intensity()
    assert images.shape == waves.shape
    assert images.array.dtype == np.float32


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_images(data, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    images = waves.to_images()
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
@devices
def test_downsample(data, max_angle, normalization, lazy, device):
    probe = data.draw(abtem_st.probe(device=device, allow_distribution=False))
    waves = probe.build(lazy=lazy)
    old_gpts = waves.gpts
    valid_gpts = waves.antialias_valid_gpts
    cutoff_gpts = waves.antialias_cutoff_gpts
    old_max = waves.intensity().array.max(axis=(-2, -1))

    downsampled_waves = waves.downsample(
        max_angle=max_angle, normalization=normalization
    )

    if isinstance(max_angle, float):
        assume(max_angle < 0.8 * max(downsampled_waves.cutoff_angles))
        assume(max_angle > 1.2 * probe.aperture.semiangle_cutoff)
    elif max_angle == "valid":
        assume(
            min(probe.rectangle_cutoff_angles) > 1.1 * probe.aperture.semiangle_cutoff
        )
    elif max_angle == "cutoff":
        assume(min(probe.cutoff_angles) > 1.1 * probe.aperture.semiangle_cutoff)

    assert downsampled_waves.gpts != old_gpts
    assert downsampled_waves.array.dtype == np.complex64

    assume(downsampled_waves.gpts[0] > 4)
    assume(downsampled_waves.gpts[1] > 4)

    if max_angle == "valid":
        assert downsampled_waves.gpts == valid_gpts
    elif max_angle == "cutoff":
        assert downsampled_waves.gpts == cutoff_gpts

    if normalization == "intensity":
        assert_is_normalized(downsampled_waves)
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
@devices
def test_diffraction_patterns(data, max_angle, fftshift, block_direct, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))

    assume(min(waves._gpts_within_angle(max_angle)) > 0)

    diffraction_patterns = waves.diffraction_patterns(
        max_angle=max_angle, fftshift=fftshift, block_direct=block_direct
    )
    assert diffraction_patterns.array.dtype == np.float32


@pytest.mark.parametrize("block_direct", [True, np.True_], ids=["bool", "numpy_bool"])
def test_diffraction_patterns_block_direct_true_blocks_up_to_the_semiangle_cutoff(
    block_direct,
):
    """block_direct=True blocks what DiffractionPatterns.block_direct() blocks: the
    bright-field disk of a 20 mrad probe and a margin, not a radius of 1 mrad."""
    import abtem

    probe = abtem.Probe(
        energy=100e3, semiangle_cutoff=20, extent=(4.8, 6.0), gpts=(48, 60)
    )
    waves = probe.build(lazy=False)
    patterns = waves.diffraction_patterns()
    expected = patterns.block_direct()
    one_mrad = patterns.block_direct(radius=1.0)
    assert np.abs(expected.array).max() < 1e-6 * np.abs(one_mrad.array).max()

    blocked = waves.diffraction_patterns(block_direct=block_direct)

    np.testing.assert_array_equal(blocked.array, expected.array)


@pytest.mark.parametrize(
    "semiangle_cutoff",
    [None, 0.0, 1e-6, 1e-3, 1.0, np.inf],
    ids=["plane_wave", "zero", "1e-6", "1e-3", "1.0", "infinite"],
)
def test_diffraction_patterns_block_direct_true_of_a_plane_wave_blocks_one_pixel(
    semiangle_cutoff,
):
    """Without a semiangle cutoff (a plane wave), or with one smaller than the
    angular sampling (6.4 mrad here) or an infinite one, block_direct=True blocks
    the zero-angle pixel alone. In a one-unit-cell SrTiO3 pattern the pixels next to
    it are the (100) and (010) reflections, which are kept."""
    from ase import Atoms

    import abtem

    a = 3.905
    atoms = Atoms(
        "SrTiO3",
        scaled_positions=[
            (0, 0, 0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    ) * (1, 1, 4)
    with abtem.config.set({"device": "cpu"}):
        potential = abtem.Potential(atoms, gpts=(48, 48), slice_thickness=a / 2)
        if semiangle_cutoff in (None, np.inf):
            beam = abtem.PlaneWave(energy=200e3)
        else:
            beam = abtem.Probe(energy=200e3, semiangle_cutoff=semiangle_cutoff)
        waves = beam.multislice(potential, lazy=False)
    if semiangle_cutoff == np.inf:
        # A CTF without an aperture records an infinite semiangle cutoff.
        waves = waves.apply_ctf(abtem.CTF())
        assert waves.metadata["semiangle_cutoff"] == np.inf
    patterns = waves.diffraction_patterns()
    center = tuple(n // 2 for n in patterns.shape[-2:])
    unblocked = np.asarray(patterns.array)
    assert unblocked[center[0] + 1, center[1]] > 0
    assert unblocked[center[0], center[1] + 1] > 0

    blocked = np.asarray(waves.diffraction_patterns(block_direct=True).array)

    assert blocked[center] == 0
    keep = np.ones(unblocked.shape, dtype=bool)
    keep[center] = False
    np.testing.assert_array_equal(blocked[keep], unblocked[keep])


@given(
    data=st.data(),
    repetitions=st.tuples(
        st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
    ),
    renormalize=st.booleans(),
)
@pytest.mark.parametrize("lazy", [True, False])
@devices
def test_tile(data, repetitions, renormalize, lazy, device):
    waves = data.draw(abtem_st.waves(lazy=lazy, device=device))
    old_extent = waves.extent
    old_sum = (
        waves.diffraction_patterns(max_angle=None)
        .to_cpu()
        .compute()
        .array.sum((-2, -1))
    )
    tiled = waves.tile(repetitions, renormalize=renormalize)

    assert np.allclose(
        (old_extent[0] * repetitions[0], old_extent[1] * repetitions[1]), tiled.extent
    )
    if renormalize:
        new_sum = (
            tiled.diffraction_patterns(max_angle=None)
            .to_cpu()
            .compute()
            .array.sum((-2, -1))
        )
        assert np.allclose(old_sum, new_sum)


def _build_exit_plane_waves(device="cpu", exit_planes=1):
    import abtem
    import ase

    silicon = ase.build.bulk("Si", cubic=True)
    atoms = silicon * (2, 2, 5)
    atoms.center(axis=2)

    potential = abtem.Potential(
        atoms,
        slice_thickness=2.0,
        gpts=(32, 32),
        exit_planes=exit_planes,
        device=device,
    )
    probe = abtem.Probe(energy=200e3, semiangle_cutoff=10, device=device)
    probe.match_grid(potential)

    pos = atoms.positions[0][:2]
    scan = abtem.CustomScan([pos])
    return probe.multislice(potential, scan).compute()


@pytest.fixture
def exit_plane_waves(request):
    """Create Waves with a ThicknessAxis for depth_profile tests.

    Indirectly parametrized over ``device`` (see the consuming tests below)
    so that a "gpu" run actually builds on the GPU instead of silently
    reusing the CPU build -- following the pattern ``test_system`` uses in
    test_realspace_multislice.py.
    """
    device = getattr(request, "param", "cpu")
    return _build_exit_plane_waves(device)


def _thickness_values(waves):
    from abtem.core.axes import ThicknessAxis

    (thickness_axis,) = [
        ax for ax in waves.ensemble_axes_metadata if isinstance(ax, ThicknessAxis)
    ]
    return np.array(thickness_axis.values)


@pytest.mark.parametrize("exit_plane_waves", [gpu, "cpu"], indirect=True)
def test_depth_profile_shape(exit_plane_waves):
    profile = exit_plane_waves.depth_profile()
    n_z = exit_plane_waves.shape[0]
    n_x = exit_plane_waves.shape[-1]
    assert profile.shape == (1, n_x, n_z)


@pytest.mark.parametrize("exit_plane_waves", [gpu, "cpu"], indirect=True)
def test_depth_profile_projection_axis_x(exit_plane_waves):
    profile = exit_plane_waves.depth_profile(projection_axis="x")
    n_z = exit_plane_waves.shape[0]
    n_y = exit_plane_waves.shape[-2]
    assert profile.shape == (1, n_y, n_z)


# The Si cell is 5 * 5.43 = 27.15 Å thick, cut into 14 slices of 1.939 Å.
# exit_planes=1 and 2 give evenly spaced planes from the entrance surface
# (15 and 8 planes). (3, 5, 7) gives evenly spaced planes that start below
# the entrance surface, at 4 slice thicknesses.
@pytest.mark.parametrize("exit_planes", [1, 2, (3, 5, 7)])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_depth_profile_sampling(device, exit_planes):
    waves = _build_exit_plane_waves(device, exit_planes)
    thickness = _thickness_values(waves)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = waves.depth_profile()

    assert np.isclose(profile.sampling[0], waves.sampling[0])

    # Oracle: row i of the depth profile is the wave at exit plane i, so the
    # z coordinates must be the exit-plane thicknesses. Images carry no
    # offset, so z is measured from the first exit plane.
    z = np.array(profile.base_axes_metadata[1].coordinates(profile.base_shape[1]))
    assert len(z) == len(thickness)
    assert np.allclose(z + thickness[0], thickness)


@pytest.mark.parametrize("exit_planes", [1, (3, 5, 7)])
def test_show_depth_profile_rows_at_exit_plane_thicknesses(exit_planes):
    import matplotlib.pyplot as plt

    waves = _build_exit_plane_waves("cpu", exit_planes)
    thickness = _thickness_values(waves)

    visualization = waves.show_depth_profile()
    (image,) = visualization.axes[0, 0].get_images()
    _, _, z_min, z_max = image.get_extent()
    plt.close("all")

    # imshow spreads n rows evenly over the extent, so the centre of row i is
    # at z_min + (i + 1/2) * (z_max - z_min) / n. Each row must be centred on
    # the thickness of its exit plane (absolute, including the offset).
    n = len(thickness)
    centres = z_min + (np.arange(n) + 0.5) * (z_max - z_min) / n
    assert np.allclose(centres, thickness)


def test_depth_profile_nonuniform_exit_planes_warns():
    # 14 slices with exit_planes=3: planes at 0, 3, 6, 9, 12 and 14 slices,
    # so the last spacing (2 slices) differs from the others (3 slices).
    waves = _build_exit_plane_waves("cpu", 3)
    thickness = _thickness_values(waves)

    with pytest.warns(UserWarning, match="not uniformly spaced"):
        profile = waves.depth_profile()

    # A uniform axis cannot hit every plane; the first and last rows must
    # still be at the entrance and exit surfaces.
    z = np.array(profile.base_axes_metadata[1].coordinates(profile.base_shape[1]))
    assert len(z) == len(thickness)
    assert np.isclose(z[0], thickness[0])
    assert np.isclose(z[-1], thickness[-1])


def test_depth_profile_no_thickness_axis_raises():
    import abtem

    probe = abtem.Probe(energy=200e3, semiangle_cutoff=10, extent=10, gpts=32)
    waves = probe.build([5.0, 5.0])
    with pytest.raises(ValueError, match="ThicknessAxis"):
        waves.depth_profile()


def test_depth_profile_invalid_axis_raises(exit_plane_waves):
    with pytest.raises(ValueError, match="projection_axis"):
        exit_plane_waves.depth_profile(projection_axis="z")


def test_depth_profile_finite_depth(exit_plane_waves):
    full = exit_plane_waves.depth_profile()
    partial = exit_plane_waves.depth_profile(depth=3.0)
    assert full.shape == partial.shape
    assert partial.array.sum() < full.array.sum()


@pytest.mark.parametrize("convert_complex", ["intensity", "phase", "real", "imag"])
def test_depth_profile_convert_complex(exit_plane_waves, convert_complex):
    profile = exit_plane_waves.depth_profile(convert_complex=convert_complex)
    assert profile.array.shape[-2:] == (exit_plane_waves.shape[-1], exit_plane_waves.shape[0])
