"""PRISM returns what multislice of the same probes returns.

At interpolation 1 the reduced S-matrix is the multislice exit wave of
``s_matrix.dummy_probes()`` to float round-off (multislice is linear in the
incident wave), so shape, axes metadata and values must agree with
``Probe.scan``. The sizes differ along every axis: 2 frozen-phonon
configurations, a 3 x 5 scan, a 64 x 64 grid and a 32 x 32 diffraction
pattern.
"""

import numpy as np
import pytest
from ase.build import bulk
from utils import assert_array_objects_equal, devices

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.potentials.iam import PotentialArray
from abtem.prism.s_matrix import SMatrix, SMatrixArray
from abtem.scan import validate_scan

ATOMS = bulk("Si", cubic=True) * (1, 1, 2)
POSITION = (2.0, 1.5)

# float32 round-off of two reduction orders; the largest difference over
# these cases is about 1e-6 of the maximum
TOLERANCE = 1e-5


def _potential(
    frozen_phonons=None, device="cpu", num_configs=2, exit_planes=None, gpts=64
):
    atoms = ATOMS
    if frozen_phonons is not None:
        atoms = abtem.FrozenPhonons(
            ATOMS,
            num_configs=num_configs,
            sigmas=0.05,
            seed=1,
            ensemble_mean=frozen_phonons,
        )
    return abtem.Potential(
        atoms, gpts=gpts, slice_thickness=2, exit_planes=exit_planes, device=device
    )


def _s_matrix(potential, **kwargs):
    kwargs = {"energy": 100e3, "interpolation": 1, "downsample": False, **kwargs}
    return abtem.SMatrix(potential=potential, semiangle_cutoff=20, **kwargs)


def _grid_scan(potential):
    return abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        gpts=(3, 5),
        fractional=True,
        endpoint=False,
        potential=potential,
    )


def _max_abs(measurements):
    measurements = measurements if isinstance(measurements, list) else [measurements]
    return max(float(np.abs(m.to_cpu().array).max()) for m in measurements)


def _assert_matches(measured, expected):
    assert_array_objects_equal(
        measured,
        expected,
        rtol=TOLERANCE,
        atol=TOLERANCE * _max_abs(expected),
    )


_PATHS = {
    "eager": dict(lazy=False),
    "lazy": dict(lazy=True, disable_s_matrix_chunks=False),
    "lazy-whole-s-matrix": dict(lazy=True, disable_s_matrix_chunks=True),
}


@pytest.mark.parametrize("path", list(_PATHS))
@devices
def test_prism_keeps_waves_of_every_frozen_phonon_configuration(path, device):
    # ensemble_mean averages the annular intensities over the configurations;
    # wave functions are never averaged, so they keep the configuration axis
    potential = _potential(frozen_phonons=True, device=device)
    s_matrix = _s_matrix(potential, device=device)
    scan = _grid_scan(potential)

    def detectors():
        return [abtem.WavesDetector(), abtem.AnnularDetector(30, 90)]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=scan, detectors=detectors(), lazy=False
    )
    measured = s_matrix.scan(scan=scan, detectors=detectors(), **_PATHS[path])
    measured = measured.compute()

    assert measured[0].shape == (2, 3, 5, 64, 64)
    assert measured[1].shape == (3, 5)
    _assert_matches(list(measured), list(expected))


@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("frozen_phonons", [None, False])
@pytest.mark.parametrize("detector", ["annular", "pixelated", "waves"])
@devices
def test_prism_single_position_matches_multislice(
    path, frozen_phonons, detector, device
):
    # a bare (x, y) loses its position axis, as Probe.scan drops it
    potential = _potential(frozen_phonons, device=device)
    s_matrix = _s_matrix(potential, device=device)
    make_detector = {
        "annular": lambda: abtem.AnnularDetector(30, 90),
        "pixelated": lambda: abtem.PixelatedDetector(max_angle=100),
        "waves": lambda: abtem.WavesDetector(),
    }[detector]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=POSITION, detectors=make_detector(), lazy=False
    )
    measured = s_matrix.scan(
        scan=POSITION, detectors=make_detector(), **_PATHS[path]
    ).compute()

    _assert_matches(measured, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_s_matrix_array_reduce_single_position_matches_multislice(lazy):
    potential = _potential(exit_planes=2)
    s_matrix = _s_matrix(potential)
    detector = abtem.AnnularDetector(30, 90)

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=POSITION, detectors=detector, lazy=False
    )
    measured = s_matrix.build(lazy=lazy).reduce(scan=POSITION, detectors=detector)
    measured = measured.compute()

    assert measured.shape == (4,)
    _assert_matches(measured, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_multi_energy_prism_single_position_drops_the_position_axis(lazy):
    potential = _potential()
    s_matrix = _s_matrix(potential, energy=[80e3, 100e3])
    detector = abtem.AnnularDetector(30, 90)

    measured = s_matrix.scan(scan=POSITION, detectors=detector, lazy=lazy).compute()

    assert measured.shape == (2,)
    for i, energy in enumerate((80e3, 100e3)):
        expected = (
            _s_matrix(potential, energy=energy)
            .dummy_probes()
            .scan(potential=potential, scan=POSITION, detectors=detector, lazy=False)
        )
        np.testing.assert_allclose(
            measured.array[i], expected.array, rtol=TOLERANCE, atol=0
        )


@pytest.mark.parametrize(
    "entry", ["scan-eager", "scan-lazy", "build-reduce", "build-reduce-modes"]
)
@pytest.mark.parametrize("pixelated", [False, True])
def test_upsampled_prism_single_position_drops_the_position_axis(entry, pixelated):
    # oracle: the same reduction at a one-position CustomScan, which keeps
    # its position axis
    potential = _potential()
    s_matrix = _s_matrix(potential, interpolation=2, upsample=True)
    # the default method on CPU expands to an SMatrixArray and reduces that
    method = "modes" if entry.endswith("modes") else "auto"
    s_matrix_array = s_matrix.build(lazy=False)

    def reduce(scan):
        if pixelated:
            detector = abtem.PixelatedDetector(max_angle=60)
        else:
            detector = abtem.AnnularDetector(30, 90)
        if entry.startswith("build-reduce"):
            return s_matrix_array.reduce(scan=scan, detectors=detector, method=method)
        return s_matrix.scan(
            scan=scan, detectors=detector, lazy=entry == "scan-lazy"
        ).compute()

    expected = reduce(abtem.CustomScan([POSITION]))
    measured = reduce(POSITION)

    assert measured.shape == expected.shape[1:]
    assert measured.axes_metadata == expected.axes_metadata[1:]
    np.testing.assert_array_equal(measured.array, expected.array[0])


def test_composite_blend_single_position_drops_the_position_axis():
    # the blend adds the intensities of two reductions at the one position
    potential = _potential()
    s_matrix_array = _s_matrix(potential, interpolation=2, upsample=True).build(
        lazy=False
    )

    def reduce(scan):
        return s_matrix_array.reduce(
            scan=scan,
            detectors=abtem.AnnularDetector(30, 90),
            blend_angle=40.0,
            blend_window_gpts="period",
        )

    expected = reduce(abtem.CustomScan([POSITION]))
    measured = reduce(POSITION)

    assert measured.shape == ()
    np.testing.assert_array_equal(measured.array, expected.array[0])


def _transition_potential(potential):
    # a synthetic transition potential stands in for GPAW
    rng = np.random.default_rng(0)
    array = rng.standard_normal((2, 64, 64)) + 1j * rng.standard_normal((2, 64, 64))
    return TransitionPotentialArray(
        Z=14,
        array=array.astype(np.complex64),
        energy=100e3,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def _core_loss_scan(s_matrix, potential, lazy):
    return s_matrix.transition_potential_scan(
        _transition_potential(potential),
        scan=_grid_scan(potential),
        detectors=abtem.AnnularDetector(0, 40),
        sites=ATOMS[:2],
        lazy=lazy,
    )


@pytest.mark.parametrize("exit_planes", [None, 2])
def test_lazy_upsampled_core_loss_does_not_compress(monkeypatch, exit_planes):
    # the core-loss reduction never reads the compressed basis; the eager
    # static scan never builds the S-matrix, so it is the oracle
    calls = []
    compress = SMatrix._compress

    def counting_compress(self, array):
        calls.append(1)
        return compress(self, array)

    monkeypatch.setattr(SMatrix, "_compress", counting_compress)

    potential = _potential(exit_planes=exit_planes)
    s_matrix = _s_matrix(potential, interpolation=2, upsample=True)

    measured = _core_loss_scan(s_matrix, potential, lazy=True)
    assert calls == []
    measured = measured.compute()
    expected = _core_loss_scan(s_matrix, potential, lazy=False)

    assert calls == []
    assert measured.shape == expected.shape
    assert measured.axes_metadata == expected.axes_metadata
    np.testing.assert_array_equal(measured.array, expected.array)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("ensemble_mean", [False, True])
def test_upsampled_core_loss_frozen_phonons_match_each_configuration(
    lazy, ensemble_mean
):
    # 4 configurations, a 3 x 5 scan: each configuration's result is the
    # static scan of that configuration's potential, which never builds
    potential = _potential(ensemble_mean, num_configs=4)
    s_matrix = _s_matrix(potential, interpolation=2, upsample=True)
    configurations = [block.item() for _, _, block in potential.generate_blocks(1)]

    expected = np.stack(
        [
            _core_loss_scan(
                _s_matrix(configuration, interpolation=2, upsample=True),
                configuration,
                lazy=False,
            ).array.reshape((3, 5))
            for configuration in configurations
        ]
    )
    if ensemble_mean:
        expected = expected.mean(0)

    measured = _core_loss_scan(s_matrix, potential, lazy=lazy).compute()

    assert measured.shape == expected.shape
    np.testing.assert_allclose(
        measured.array, expected, rtol=TOLERANCE, atol=TOLERANCE * expected.max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("frozen_phonons", [None, False, True])
@pytest.mark.parametrize("detector", ["pixelated", "flexible"])
def test_upsampled_prism_sizes_detectors_on_a_non_square_grid(
    lazy, frozen_phonons, detector
):
    # the size of a pixelated pattern or of the flexible annular bins follows
    # the antialias cutoff of the downsampled S-matrix, which differs from that
    # of the full grid on a non-square grid. The oracle is the eager scan of
    # each configuration as a static potential, which has no ensemble axis and
    # takes the size from the reduction.
    def make_detector():
        if detector == "pixelated":
            return abtem.PixelatedDetector()
        return abtem.FlexibleAnnularDetector()

    potential = _potential(frozen_phonons, gpts=(56, 64))
    s_matrix = _s_matrix(potential, interpolation=2, upsample=True, downsample="cutoff")
    scan = _grid_scan(potential)

    built = potential.build(lazy=False)
    slices = np.asarray(built.array)
    slices = slices[None] if frozen_phonons is None else slices
    configurations = [
        PotentialArray(
            configuration,
            slice_thickness=built.slice_thickness,
            extent=potential.extent,
        )
        for configuration in slices
    ]

    arrays = [
        _s_matrix(configuration, interpolation=2, upsample=True, downsample="cutoff")
        .scan(scan=scan, detectors=make_detector(), lazy=False)
        .array
        for configuration in configurations
    ]
    expected = np.stack(arrays)
    if frozen_phonons is None:
        expected = expected[0]
    elif frozen_phonons:
        expected = expected.mean(0)

    measured = s_matrix.scan(scan=scan, detectors=make_detector(), lazy=lazy)
    declared_shape = measured.shape
    measured = measured.compute()

    assert declared_shape == expected.shape
    assert measured.shape == expected.shape
    np.testing.assert_allclose(
        measured.array, expected, rtol=TOLERANCE, atol=TOLERANCE * expected.max()
    )


# automatic batches are sized by the chunk size; with a 64 x 64 grid, 37 beams and
# 4 exit planes (6 slices, exit_planes=2) one complex64 wave of one exit plane is
# 32768 bytes
CHUNK_SIZE = 1_000_000
WAVE_BYTES = 64 * 64 * 8


def test_auto_build_batch_holds_every_exit_plane():
    """CPU only: the automatic batch follows dask.chunk-size, not chunk-size-gpu."""
    potential = _potential(exit_planes=2)
    with abtem.config.set({"dask.chunk-size": "1 MB"}):
        built = _s_matrix(potential).build(lazy=True)

    n_planes = built.array.shape[0]
    assert n_planes == 4
    assert max(built.array.chunks[1]) * n_planes * WAVE_BYTES <= CHUNK_SIZE


def test_explicit_build_batch_counts_plane_waves():
    """CPU only: a GPU build caps the batch by chunk-size-gpu, not dask.chunk-size."""
    built = _s_matrix(_potential(exit_planes=2)).build(lazy=True, max_batch=5)
    assert built.array.chunks[1] == (5,) * 7 + (2,)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "frozen_phonons, exit_planes", [(None, 2), (False, None), (False, 2)]
)
def test_auto_reduction_batch_holds_every_exit_plane(
    monkeypatch, lazy, frozen_phonons, exit_planes
):
    """CPU only: the automatic batch follows dask.chunk-size, not chunk-size-gpu."""
    potential = _potential(frozen_phonons, exit_planes=exit_planes)
    s_matrix_array = _s_matrix(potential).build(lazy=lazy)
    scan = abtem.GridScan(
        start=(0, 0),
        end=(1, 1),
        gpts=(12, 20),
        fractional=True,
        endpoint=False,
        potential=potential,
    )
    detector = abtem.AnnularDetector(30, 90)
    expected = s_matrix_array.reduce(
        scan=scan, detectors=detector, max_batch_reduction=len(scan)
    ).compute()

    sizes = []
    reduce_to_waves = SMatrixArray._reduce_to_waves

    def spy(self, *args):
        waves = reduce_to_waves(self, *args)
        sizes.append(waves.nbytes)
        return waves

    monkeypatch.setattr(SMatrixArray, "_reduce_to_waves", spy)
    with abtem.config.set({"dask.chunk-size": "1 MB"}):
        measured = s_matrix_array.reduce(scan=scan, detectors=detector)
    measured = measured.compute()

    assert max(sizes) <= CHUNK_SIZE
    assert len(sizes) > 1
    # the batch size moves the values by float32 round-off of the reduction
    _assert_matches(measured, expected)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "make_detector",
    [
        lambda: abtem.AnnularDetector(30, 90),
        lambda: abtem.PixelatedDetector(max_angle=100),
    ],
    ids=["annular", "pixelated"],
)
def test_auto_reduction_batch_of_a_single_position_may_exceed_the_chunk_size(
    lazy, make_detector
):
    """CPU only: the automatic batch follows dask.chunk-size, not chunk-size-gpu."""
    # the 4 exit planes of one position (131 kB) exceed the chunk size
    s_matrix = _s_matrix(_potential(exit_planes=2))
    expected = s_matrix.reduce(detectors=make_detector(), lazy=lazy).compute()

    with abtem.config.set({"dask.chunk-size": "100 kB"}):
        measured = s_matrix.reduce(detectors=make_detector(), lazy=lazy).compute()

    assert measured.shape[0] == 4
    _assert_matches(measured, expected)


@pytest.mark.parametrize("path", list(_PATHS))
@pytest.mark.parametrize("exit_planes", [None, 2])
@devices
def test_reduce_without_a_scan_has_no_position_axis(path, exit_planes, device):
    potential = _potential(exit_planes=exit_planes, device=device)
    s_matrix = _s_matrix(potential, device=device)
    centre = (potential.extent[0] / 2, potential.extent[1] / 2)

    def detectors():
        return [abtem.AnnularDetector(30, 90), abtem.PixelatedDetector(max_angle=100)]

    expected = s_matrix.dummy_probes().scan(
        potential=potential, scan=centre, detectors=detectors(), lazy=False
    )
    measured = s_matrix.reduce(detectors=detectors(), **_PATHS[path]).compute()

    planes = () if exit_planes is None else (4,)
    assert measured[0].shape == planes
    assert measured[1].shape == planes + expected[1].base_shape
    _assert_matches(list(measured), list(expected))


# an upsampled S-matrix is always built eagerly
@pytest.mark.parametrize(
    "lazy, upsample", [(False, False), (True, False), (False, True)]
)
@devices
def test_array_reduce_squeezes_a_validated_bare_position(lazy, upsample, device):
    s_matrix = _s_matrix(
        _potential(device=device), interpolation=2, upsample=upsample, device=device
    )
    built = s_matrix.build(lazy=lazy)
    scan = validate_scan(POSITION, s_matrix)

    measured = built.reduce(scan=scan, detectors=abtem.AnnularDetector(30, 90))
    expected = built.reduce(scan=POSITION, detectors=abtem.AnnularDetector(30, 90))

    assert measured.shape == expected.shape == ()
