import warnings

import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import assume, given
from test_grid import check_grid_consistent
from utils import (
    assert_array_matches_device,
    assert_array_matches_laziness,
    devices,
    gpu,
)

import abtem
from abtem.core.backend import asnumpy
from abtem.waves import Waves

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
@pytest.mark.parametrize("device", [gpu, "cpu"])
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


@pytest.mark.parametrize(
    "chunks", [(1, 16, 20), (2, 16, 20), (3, 8, 10), (2, (10, 6), 20)]
)
@pytest.mark.parametrize("reciprocal_space", [False, True])
def test_lazy_normalize_matches_eager(chunks, reciprocal_space):
    from abtem.core.axes import OrdinalAxis
    from abtem.waves import Waves

    rng = np.random.default_rng(0)
    array = rng.normal(size=(3, 16, 20)) + 1j * rng.normal(size=(3, 16, 20))
    kwargs = dict(
        energy=100e3,
        sampling=0.1,
        reciprocal_space=reciprocal_space,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )
    eager = Waves(array, **kwargs).normalize()
    lazy = Waves(da.from_array(array, chunks=chunks), **kwargs).normalize()

    assert lazy.is_lazy
    lazy_result = lazy.copy().compute()
    assert lazy_result.reciprocal_space == reciprocal_space
    assert np.allclose(lazy_result.array, eager.array)


_GRID_SPLIT_OPERATIONS = {
    "apply_ctf": lambda waves, kernel, potential: waves.apply_ctf(
        defocus=50, semiangle_cutoff=30
    ),
    "normalize": lambda waves, kernel, potential: waves.normalize(),
    "ensure_reciprocal_space": lambda waves, kernel, potential: (
        waves.ensure_reciprocal_space()
    ),
    "ensure_real_space": lambda waves, kernel, potential: waves.ensure_real_space(),
    "convolve": lambda waves, kernel, potential: waves.convolve(kernel),
    "downsample": lambda waves, kernel, potential: waves.downsample(max_angle=20),
    "diffraction_patterns": lambda waves, kernel, potential: (
        waves.diffraction_patterns(max_angle=None)
    ),
    "diffraction_patterns_block_direct": lambda waves, kernel, potential: (
        waves.diffraction_patterns(block_direct=True)
    ),
    "multislice": lambda waves, kernel, potential: waves.multislice(potential),
}


@pytest.mark.parametrize("operation", list(_GRID_SPLIT_OPERATIONS))
@pytest.mark.parametrize("reciprocal_space", [False, True])
@pytest.mark.parametrize(
    "chunks",
    [
        (3, 16, 20),
        (3, 8, 20),
        (3, 16, 7),
        (1, (10, 6), (5, 5, 10)),
    ],
    ids=["unsplit", "split_first_axis", "split_second_axis", "split_both"],
)
@devices
def test_lazy_waves_split_along_the_grid_match_eager(
    device, chunks, reciprocal_space, operation
):
    """A lazy operation on waves whose grid is in several dask chunks gives the
    eager result: the grid is gathered into one chunk before the operation."""
    from ase import Atoms

    from abtem.core.axes import OrdinalAxis
    from abtem.core.backend import copy_to_device
    from abtem.potentials.iam import Potential

    rng = np.random.default_rng(0)
    shape = (3, 16, 20)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    kernel = (rng.normal(size=shape[1:]) + 1j * rng.normal(size=shape[1:])).astype(
        np.complex64
    )
    potential = Potential(
        Atoms("C", positions=[(0.8, 1.0, 1.0)], cell=(1.6, 2.0, 2.0)),
        gpts=(16, 20),
        slice_thickness=1,
        device=device,
    )
    kwargs = dict(
        energy=100e3,
        sampling=0.1,
        reciprocal_space=reciprocal_space,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )
    eager = Waves(copy_to_device(array, device), **kwargs)
    lazy = Waves(copy_to_device(da.from_array(array, chunks=chunks), device), **kwargs)
    apply = _GRID_SPLIT_OPERATIONS[operation]

    with abtem.config.set({"fft": "numpy"}):
        expected = apply(eager, kernel, potential)
        result = apply(lazy, kernel, potential)
        assert result.is_lazy
        computed = result.copy().compute(progress_bar=False)

    assert computed.array.shape == expected.array.shape
    expected_array = asnumpy(expected.array)
    np.testing.assert_allclose(
        asnumpy(computed.array),
        expected_array,
        rtol=0,
        atol=1e-5 * np.abs(expected_array).max(),
    )


@pytest.mark.parametrize(
    "operation", ["apply_ctf", "diffraction_patterns"], ids=lambda name: name
)
@devices
def test_lazy_waves_split_along_the_grid_by_auto_chunks_keep_several_blocks(
    device, operation
):
    """Gathering the grid of a stack that dask chunked automatically leaves the
    ensemble axes in several blocks, so the result is not one block that holds
    the whole stack."""
    import dask

    from abtem.core.axes import OrdinalAxis
    from abtem.core.backend import copy_to_device

    rng = np.random.default_rng(0)
    shape = (4, 3, 256, 256)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    kwargs = dict(
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[
            OrdinalAxis(values=(0, 1, 2, 3)),
            OrdinalAxis(values=(0, 1, 2)),
        ],
    )
    apply = {
        "apply_ctf": lambda waves: waves.apply_ctf(defocus=50, semiangle_cutoff=30),
        "diffraction_patterns": lambda waves: waves.diffraction_patterns(
            max_angle=None
        ),
    }[operation]
    eager = Waves(copy_to_device(array, device), **kwargs)

    with dask.config.set({"array.chunk-size": "1MiB"}):
        lazy = Waves(
            copy_to_device(da.from_array(array, chunks="auto"), device), **kwargs
        )
        assert lazy.array.numblocks[:2] == (1, 1)
        assert lazy.array.numblocks[2:] != (1, 1)
        result = apply(lazy)

    assert result.array.numblocks[2:] == (1, 1)
    assert all(n > 1 for n in result.array.numblocks[:2])
    with abtem.config.set({"fft": "numpy"}):
        expected = asnumpy(apply(eager).array)
        computed = asnumpy(result.copy().compute(progress_bar=False).array)
    np.testing.assert_allclose(
        computed, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )




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
    images = waves.to_images()
    assert images.shape == waves.shape
    assert images.array.dtype == np.complex64


@given(
    data=st.data(),
    max_angle=st.just("valid")
    | st.just("cutoff")
    | st.floats(min_value=10, max_value=100),
)
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_downsample(data, max_angle, lazy, device):
    probe = data.draw(abtem_st.probe(device=device, allow_distribution=False))
    waves = probe.build(lazy=lazy)
    old_gpts = waves.gpts
    valid_gpts = waves.antialias_valid_gpts
    cutoff_gpts = waves.antialias_cutoff_gpts

    downsampled_waves = waves.downsample(max_angle=max_angle)

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

    # a probe is normalized, and downsampling keeps it so
    assert_is_normalized(downsampled_waves)


def _probe_waves(device):
    # a grid sampled differently along x and y
    return abtem.Probe(
        energy=100e3, semiangle_cutoff=20, gpts=(56, 64), extent=(10, 12), device=device
    ).build(lazy=False)


@pytest.mark.parametrize("gpts", [None, (45, 45), (39, 41)])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_downsample_keeps_the_antialias_cutoff(gpts, device):
    # the pixel count within the antialias cutoff of the resampled waves still
    # means the same angles when the new grid holds them, and is capped at its own
    # grid otherwise
    waves = _probe_waves(device)

    downsampled = waves.downsample(gpts=gpts) if gpts else waves.downsample("cutoff")

    assert downsampled.gpts != waves.gpts
    expected = [
        min(angle, nyquist)
        for angle, nyquist in zip(
            waves.cutoff_angles,
            np.array(downsampled.gpts) // 2 * downsampled.angular_sampling,
        )
    ]
    assert downsampled.cutoff_angles == pytest.approx(expected)


@pytest.mark.parametrize("gpts", [(45, 45), (45, 50)])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_downsample_to_gpts_of_the_other_parity_keeps_the_cutoff_count(gpts, device):
    # the pixel count within the antialias cutoff takes the parity of the grid
    # the waves come from; (45, 45) and (45, 50) are not the parity of the (56, 64)
    # grid, and the angles, the "cutoff" pattern and the bins of a
    # FlexibleAnnularDetector keep the values that these calls have always given
    downsampled = _probe_waves(device).downsample(gpts=gpts)

    assert downsampled.cutoff_angles == pytest.approx([66.62585871, 64.77514041])
    assert downsampled.diffraction_patterns("cutoff").shape[-2:] == (37, 43)
    assert abtem.FlexibleAnnularDetector().detect(downsampled).shape == (64, 1)


@pytest.mark.parametrize("route", ["downsample", "detector"])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_resampling_onto_a_finer_grid_keeps_the_cap_of_the_coarser_grid(route, device):
    # the content of waves resampled to (30, 40) ends at the edge of that grid, and
    # resampling them onto the finer (45, 50) grid does not extend it
    waves = _probe_waves(device)

    if route == "downsample":
        coarse = waves.downsample(gpts=(30, 40))
    else:
        coarse = abtem.WavesDetector(gpts=(30, 40)).detect(waves)
    fine = coarse.downsample(gpts=(45, 50))

    assert fine.antialias_cutoff_gpts == (30, 40)
    assert fine.cutoff_angles == pytest.approx([55.52154893, 61.69060992])
    assert fine.diffraction_patterns("cutoff").shape[-2:] == (31, 41)
    assert fine.diffraction_patterns("valid").shape[-2:] == (21, 29)
    assert abtem.FlexibleAnnularDetector().detect(fine).shape == (55, 1)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_downsample_twice_keeps_the_antialias_cutoff(device):
    waves = _probe_waves(device)

    downsampled = waves.downsample("cutoff").downsample("cutoff")

    assert downsampled.cutoff_angles == pytest.approx(waves.cutoff_angles)


@pytest.mark.parametrize("repetitions", [(2, 2), (3, 1)])
@pytest.mark.parametrize("gpts", [None, (45, 45), (39, 41)])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_tiling_downsampled_waves_keeps_the_antialias_cutoff(gpts, repetitions, device):
    # tiling changes the extent at fixed sampling, so the angles within the
    # antialias cutoff stay those of the waves the resampled waves come from, up to
    # the Nyquist angle of their own grid and the parity of the pixel count
    waves = _probe_waves(device)
    downsampled = waves.downsample(gpts=gpts) if gpts else waves.downsample("cutoff")

    tiled = downsampled.tile(repetitions)

    assert "adjusted_antialias_cutoff_gpts" not in tiled.metadata
    expected = np.minimum(
        waves.cutoff_angles, np.array(tiled.gpts) // 2 * tiled.angular_sampling
    )
    assert tiled.cutoff_angles == pytest.approx(
        expected, abs=max(tiled.angular_sampling)
    )


def test_waves_read_a_stored_adjusted_antialias_cutoff(tmp_path):
    # waves written by released versions to zarr store the pixel count itself
    waves = abtem.Waves(
        np.zeros((38, 38), np.complex64),
        energy=100e3,
        extent=(5.43, 5.43),
        metadata={"adjusted_antialias_cutoff_gpts": (26, 22)},
    )
    waves.to_zarr(str(tmp_path / "waves.zarr"))

    loaded = abtem.from_zarr(str(tmp_path / "waves.zarr")).compute()

    assert waves.antialias_cutoff_gpts == (26, 22)
    assert tuple(loaded.antialias_cutoff_gpts) == (26, 22)


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
    "semiangle_cutoff", [None, np.inf], ids=["plane_wave", "infinite"]
)
def test_diffraction_patterns_block_direct_true_of_a_plane_wave_blocks_one_pixel(
    semiangle_cutoff,
):
    """Without a semiangle cutoff (a plane wave), or with an infinite one (a CTF
    without an aperture), block_direct=True blocks the zero-angle pixel alone. In
    a one-unit-cell SrTiO3 pattern the pixels next to it are the (100) and (010)
    reflections, which are kept. Cutoffs below half the angular sampling take the
    same branch; the synthetic zero-frequency-pixel test below covers them."""
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
        waves = abtem.PlaneWave(energy=200e3).multislice(potential, lazy=False)
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


def _zero_frequency_pixel(shape, fftshift):
    """The index of the zero frequency, from NumPy's FFT conventions alone."""
    index = []
    for n in shape:
        frequencies = np.fft.fftfreq(n)
        if fftshift:
            frequencies = np.fft.fftshift(frequencies)
        index.append(int(np.flatnonzero(frequencies == 0)[0]))
    return tuple(index)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("fftshift", [True, False], ids=["shifted", "unshifted"])
@pytest.mark.parametrize(
    "gpts, extent",
    [
        ((45, 64), (20.0, 26.0)),
        ((64, 45), (26.0, 20.0)),
        ((45, 51), (20.0, 23.0)),
        ((64, 80), (26.0, 32.0)),
    ],
    ids=["odd_even", "even_odd", "odd_odd", "even_even"],
)
def test_diffraction_patterns_block_direct_true_blocks_the_zero_frequency_pixel(
    gpts, extent, fftshift, device
):
    """Without a scalar semiangle cutoff larger than half the angular sampling,
    block_direct=True zeroes the zero-frequency pixel and nothing else, for every
    pattern shape, with and without fftshift.

    Only the shape parity and fftshift choose the pixel; every cutoff below
    takes the same branch of ``_apply_block_direct``, and laziness only wraps
    it, so they are looped over rather than parametrized."""
    from abtem.waves import Waves

    rng = np.random.default_rng(0)
    array = (5 + rng.normal(size=gpts) + 1j * rng.normal(size=gpts)).astype(
        np.complex64
    )
    pixel = _zero_frequency_pixel(gpts, fftshift)
    keep = np.ones(gpts, dtype=bool)
    keep[pixel] = False
    kwargs = dict(max_angle="full", parity="same", fftshift=fftshift)
    for semiangle_cutoff in [None, 0.0, 1e-3, np.inf, np.array([10.0, 20.0])]:
        metadata = (
            {} if semiangle_cutoff is None else {"semiangle_cutoff": semiangle_cutoff}
        )
        for lazy in (False, True):
            case = f"semiangle_cutoff={semiangle_cutoff}, lazy={lazy}"
            waves = Waves(array, energy=200e3, extent=extent, metadata=metadata)
            waves = waves.copy_to_device(device)
            if lazy:
                waves = waves.ensure_lazy()
            unblocked = waves.diffraction_patterns(**kwargs).compute().to_cpu().array
            assert unblocked.shape == gpts and np.all(unblocked > 0), case

            blocked = waves.diffraction_patterns(block_direct=True, **kwargs).compute()

            blocked = blocked.to_cpu().array
            assert blocked[pixel] == 0, case
            np.testing.assert_array_equal(blocked[keep], unblocked[keep], case)


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


@pytest.mark.parametrize("lazy", [True, False])
def test_tile_renormalize_keeps_the_configured_precision(lazy):
    # Dividing by a 0-d array or NumPy scalar (xp.asarray(np.prod(repetitions)))
    # promotes complex64 to complex128 under NEP 50, doubling the precision and
    # memory of a lazy wave; dividing by a Python int does not.
    import abtem

    waves = abtem.Probe(energy=100e3, semiangle_cutoff=20, extent=5, gpts=32).build(
        (0, 0), lazy=lazy
    )
    assert waves.array.dtype == np.complex64

    tiled = waves.tile((2, 2), renormalize=True)

    assert tiled.array.dtype == np.complex64
    assert tiled.compute().array.dtype == np.complex64


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
def exit_plane_waves():
    """Create Waves with a ThicknessAxis for depth_profile tests."""
    return _build_exit_plane_waves()


def _thickness_values(waves):
    from abtem.core.axes import ThicknessAxis

    (thickness_axis,) = [
        ax for ax in waves.ensemble_axes_metadata if isinstance(ax, ThicknessAxis)
    ]
    return np.array(thickness_axis.values)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_depth_profile_shape(exit_plane_waves, device):
    profile = exit_plane_waves.depth_profile()
    n_z = exit_plane_waves.shape[0]
    n_x = exit_plane_waves.shape[-1]
    assert profile.shape == (1, n_x, n_z)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_depth_profile_projection_axis_x(exit_plane_waves, device):
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


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("reciprocal_space", [False, True])
def test_phase_shift_keeps_the_space_of_the_waves(
    reciprocal_space, lazy, precision, device
):
    import dask.array as da

    import abtem
    from abtem.core.axes import OrdinalAxis
    from abtem.core.backend import asnumpy
    from abtem.core.utils import get_dtype
    from abtem.waves import Waves

    amount = 0.3
    with abtem.config.set({"precision": precision}):
        dtype = get_dtype(complex=True)
        rng = np.random.default_rng(0)
        psi = rng.normal(size=(2, 16, 20)) + 1j * rng.normal(size=(2, 16, 20))
        psi = psi.astype(dtype)
        array = np.fft.fft2(psi).astype(dtype) if reciprocal_space else psi
        if lazy:
            array = da.from_array(array, chunks=(1, -1, -1))
        waves = Waves(
            array,
            energy=100e3,
            sampling=0.1,
            reciprocal_space=reciprocal_space,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        ).copy_to_device(device)

        shifted = waves.phase_shift(amount)

        assert shifted.reciprocal_space == reciprocal_space
        assert shifted.is_lazy == lazy
        # The phase factor must not widen the waves: a NumPy complex128 scalar
        # promotes complex64 waves to complex128 under NEP 50.
        assert shifted.array.dtype == dtype
        result = shifted.ensure_real_space().compute()
        assert result.array.dtype == dtype
        result = asnumpy(result.array)

    expected = np.exp(1j * amount) * psi
    atol = 100 * np.finfo(dtype).eps * np.abs(psi).max()
    assert np.allclose(result, expected, rtol=0, atol=atol)


# automatic batches are sized by the chunk size; with 48 x 48 probes (18432 bytes
# each), 6 slices and exit_planes=2 (the entrance and 3 more planes), a 12 x 20
# scan and 4 planes: every axis of the output has a different length
_BATCH_CHUNK_SIZE = 400_000
_BATCH_SCAN = dict(gpts=(12, 20), start=(0, 0), end=(1, 1), fractional=True)


def _batch_atoms():
    import ase.build

    return ase.build.bulk("Si", cubic=True) * (1, 1, 2)


def _batch_probe_and_potential(exit_planes):
    potential = abtem.Potential(
        _batch_atoms(), gpts=48, slice_thickness=2, exit_planes=exit_planes
    )
    probe = abtem.Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    return probe, potential


def _largest_chunk_bytes(measurement):
    chunk = tuple(max(c) for c in measurement.array.chunks)
    return int(np.prod(chunk)) * measurement.array.dtype.itemsize


@pytest.mark.parametrize(
    "detector",
    [abtem.WavesDetector(), abtem.PixelatedDetector(max_angle="full")],
    ids=["waves", "pixelated"],
)
def test_auto_multislice_batch_holds_every_exit_plane(detector):
    """CPU only: the automatic batch follows dask.chunk-size."""
    probe, potential = _batch_probe_and_potential(exit_planes=2)
    assert potential.num_exit_planes == 4

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)
        measurement = probe.scan(potential, scan=scan, detectors=detector, lazy=True)

    assert measurement.array.shape[0] == 4
    assert _largest_chunk_bytes(measurement) <= _BATCH_CHUNK_SIZE


def test_auto_multislice_batch_holds_the_summed_output_of_several_detectors():
    """CPU only: the automatic batch follows dask.chunk-size.

    With a single exit plane, detectors whose outputs together exceed one wave
    (here 1.5 waves per position) also shrink the automatic batch.
    """
    probe, potential = _batch_probe_and_potential(exit_planes=None)
    assert potential.num_exit_planes == 1
    detectors = [abtem.WavesDetector(), abtem.PixelatedDetector(max_angle="full")]

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)
        measurements = probe.scan(potential, scan=scan, detectors=detectors, lazy=True)

    assert sum(_largest_chunk_bytes(m) for m in measurements) <= _BATCH_CHUNK_SIZE


@pytest.mark.parametrize(
    "detector",
    [abtem.WavesDetector(), abtem.AnnularDetector(30, 90)],
    ids=["waves", "annular"],
)
def test_auto_multislice_batch_does_not_change_the_values(detector):
    from abtem.core.backend import asnumpy

    probe, potential = _batch_probe_and_potential(exit_planes=2)
    scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)

    expected = probe.scan(
        potential, scan=scan, detectors=detector, max_batch=len(scan), lazy=True
    )
    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        measured = probe.scan(potential, scan=scan, detectors=detector, lazy=True)

    assert measured.array.numblocks != expected.array.numblocks
    expected = asnumpy(expected.compute().array)
    np.testing.assert_allclose(
        asnumpy(measured.compute().array),
        expected,
        rtol=1e-5,
        atol=1e-5 * float(np.abs(expected).max()),
    )


@pytest.mark.parametrize("exit_planes, max_batch", [(2, 5), (None, 5), (None, "auto")])
def test_multislice_batch_without_the_auto_exit_plane_rule_is_unchanged(
    exit_planes, max_batch
):
    """The output blocks follow the probe blocks of the same ``max_batch`` when it
    is explicit or when there is a single exit plane.

    CPU only: the automatic batch follows dask.chunk-size.
    """
    probe, potential = _batch_probe_and_potential(exit_planes=exit_planes)

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)
        probes = probe.build(scan=scan, lazy=True, max_batch=max_batch)
        measurement = probe.scan(
            potential,
            scan=scan,
            detectors=abtem.WavesDetector(),
            max_batch=max_batch,
            lazy=True,
        )

    assert measurement.array.chunks[-4:-2] == probes.array.chunks[:2]


def _synthetic_transition_potential(potential, num_transitions=1):
    # a synthetic transition potential stands in for GPAW
    from abtem.core.axes import OrdinalAxis
    from abtem.inelastic.core_loss import TransitionPotentialArray

    rng = np.random.default_rng(0)
    shape = (num_transitions,) + potential.gpts
    array = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return TransitionPotentialArray(
        Z=14,
        array=array.astype(np.complex64),
        energy=100e3,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(num_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def test_auto_transition_potential_scan_batch_holds_every_exit_plane():
    """CPU only: the automatic batch follows dask.chunk-size."""
    probe, potential = _batch_probe_and_potential(exit_planes=2)
    assert potential.num_exit_planes == 4

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)
        measurement = probe.transition_potential_scan(
            potential,
            _synthetic_transition_potential(potential),
            scan=scan,
            detectors=abtem.WavesDetector(),
            sites=_batch_atoms()[:2],
            lazy=True,
        )

    assert 4 in measurement.array.shape
    assert _largest_chunk_bytes(measurement) <= _BATCH_CHUNK_SIZE


def test_auto_plane_wave_ensemble_batch_holds_every_exit_plane():
    """CPU only: the automatic batch follows dask.chunk-size."""
    from abtem.distributions import uniform

    _, potential = _batch_probe_and_potential(exit_planes=2)
    # 7 tilts, the exit planes and the 48 x 48 grid differ in length
    plane_wave = abtem.PlaneWave(energy=100e3, tilt=(uniform(0, 5, 7), 0))

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        measurement = plane_wave.multislice(potential, lazy=True)

    assert measurement.array.shape[:2] == (4, 7)
    assert _largest_chunk_bytes(measurement) <= _BATCH_CHUNK_SIZE


@pytest.mark.parametrize(
    "detector",
    [abtem.AnnularDetector(30, 90), abtem.FlexibleAnnularDetector()],
    ids=["annular", "flexible"],
)
def test_auto_multislice_batch_of_an_integrating_detector_ignores_the_exit_planes(
    detector,
):
    """The output of an integrating detector is a few numbers per plane, so the
    probe blocks are those of a potential without exit planes.

    CPU only: the automatic batch follows dask.chunk-size.
    """
    probe, potential = _batch_probe_and_potential(exit_planes=2)
    assert potential.num_exit_planes == 4

    with abtem.config.set({"dask.chunk-size": f"{_BATCH_CHUNK_SIZE} B"}):
        scan = abtem.GridScan(**_BATCH_SCAN, potential=potential)
        probes = probe.build(scan=scan, lazy=True)
        measurement = probe.scan(potential, scan=scan, detectors=detector, lazy=True)

    assert measurement.array.shape[0] == 4
    assert measurement.array.chunks[1:3] == probes.array.chunks[:2]


def test_normalize_labels_the_waves_as_a_normalized_builder_does():
    # normalize() left the label 'values', so diffraction_patterns() divided by the
    # number of pixels once more and its intensity summed to 1 / N^2
    import abtem

    waves = abtem.PlaneWave(energy=100e3, gpts=32, extent=10.0).build(lazy=False)
    assert waves.metadata["normalization"] == "values"

    normalized = waves.normalize()

    assert normalized.metadata["normalization"] == "reciprocal_space"
    assert_is_normalized(normalized)
    np.testing.assert_allclose(
        normalized.diffraction_patterns(max_angle=None).array,
        waves.diffraction_patterns(max_angle=None).array,
        atol=1e-6,
    )


@pytest.mark.parametrize("normalize", [False, True], ids=["values", "normalized"])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_downsample_keeps_the_normalization_of_the_waves(normalize, lazy):
    # Downsampling used to preserve the pixel values of every wave, so the diffraction
    # patterns of a downsampled probe or normalized plane wave summed to
    # (N_new / N_old)^2 instead of one.
    import abtem

    old_gpts, new_gpts = (64, 64), (32, 32)
    waves = abtem.PlaneWave(
        energy=100e3, gpts=old_gpts, extent=10.0, normalize=normalize
    ).build(lazy=lazy)

    downsampled = waves.downsample(gpts=new_gpts)
    detected = abtem.WavesDetector(gpts=new_gpts).detect(waves)

    for result in (downsampled, detected):
        result = result.compute() if lazy else result
        assert result.metadata["normalization"] == waves.metadata["normalization"]
        assert_is_normalized(result)
        # the wave still has the normalization of its builder, on the new grid
        expected = 1 / np.prod(new_gpts) if normalize else 1.0
        np.testing.assert_allclose(np.abs(result.array), expected, rtol=1e-4)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_downsample_normalization_is_deprecated_but_honoured(lazy):
    # downsample(normalization=...) is in v1.0.11: it warns and still decides what
    # is preserved, so 'values' keeps the pixel values of a normalized probe-like wave
    # and 'amplitude' its total amplitude
    import abtem

    waves = abtem.PlaneWave(
        energy=100e3, gpts=(64, 64), extent=10.0, normalize=True
    ).build(lazy=lazy)

    def downsampled(**kwargs):
        result = waves.downsample(gpts=(32, 32), **kwargs)
        return np.abs(np.asarray(result.compute().array if lazy else result.array))

    default = downsampled()
    with pytest.warns(FutureWarning, match="normalization.*deprecated") as record:
        values = downsampled(normalization="values")
    # the warning points at the caller of downsample
    assert [w.filename for w in record] == [__file__]
    with pytest.warns(FutureWarning, match="normalization.*deprecated"):
        amplitude = downsampled(normalization="amplitude")

    np.testing.assert_allclose(amplitude, default, rtol=1e-5)
    # 'values' keeps the pixel values of the 64 x 64 wave, 'amplitude' scales them
    np.testing.assert_allclose(values, 1 / 64**2, rtol=1e-4)
    np.testing.assert_allclose(amplitude, 1 / 32**2, rtol=1e-4)

    # v1.0.11 also took 'intensity', which fft_interpolate treats as 'amplitude'
    with pytest.warns(FutureWarning, match="normalization.*deprecated"):
        intensity = downsampled(normalization="intensity")
    np.testing.assert_allclose(intensity, amplitude, rtol=1e-12)

    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="not recognized"):
        waves.downsample(gpts=(32, 32), normalization="nonsense")


def test_downsample_without_the_normalization_argument_does_not_warn():
    import abtem

    waves = abtem.PlaneWave(energy=100e3, gpts=(64, 64), extent=10.0).build(lazy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        waves.downsample(gpts=(32, 32))
