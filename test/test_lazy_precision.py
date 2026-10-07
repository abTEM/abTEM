"""A lazy result declares the dtype its blocks return, and the eager result has.

A dask reduction accumulates in the declared dtype, so a declaration below the
blocks loses precision, and one above them makes ``.dtype`` misreport the data.
Every case compares the declared dtype, the dtype of the computed blocks and the
dtype of the eager result of the same call.
"""

import dask.array as da
import numpy as np
import pytest

import abtem
from abtem import (
    AnnularDetector,
    Aperture,
    FlexibleAnnularDetector,
    PixelatedDetector,
    Probe,
    SegmentedDetector,
    SpectralSlitDetector,
    WavesDetector,
)
from abtem.core.axes import OrdinalAxis
from abtem.core.fft import fft2, fft2_convolve
from abtem.measurements import DiffractionPatterns, Images
from abtem.multislice import MultisliceTransform
from abtem.potentials.iam import PotentialArray
from abtem.tilt import BeamTilt
from abtem.waves import Waves
from utils import synthetic_transition_potential

CONFIG_AND_INPUT = [
    ("float32", np.complex128),
    ("float64", np.complex64),
]

GRID = (16, 20)
MEMBERS = 3


def _complex_data(dtype):
    rng = np.random.default_rng(7)
    shape = (MEMBERS,) + GRID
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


def _real_data(dtype):
    rng = np.random.default_rng(11)
    return (10 * rng.standard_normal((MEMBERS,) + GRID)).astype(dtype)


def _members():
    return [OrdinalAxis(values=tuple(range(MEMBERS)))]


def _waves(dtype, lazy):
    array = _complex_data(dtype)
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return Waves(array, energy=100e3, sampling=0.1, ensemble_axes_metadata=_members())


def _images(dtype, lazy):
    array = _real_data(dtype) if not np.iscomplexobj(dtype(0)) else _complex_data(dtype)
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return Images(array, sampling=0.1, ensemble_axes_metadata=_members())


def _patterns(dtype, lazy):
    array = np.abs(_real_data(dtype)) + 1
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return DiffractionPatterns(array, sampling=0.1, ensemble_axes_metadata=_members())


def _check(lazy, eager, member_offset_free=False):
    """The lazy result declares the dtype of its computed blocks and of the eager
    result. Values are compared up to each member's additive constant when
    `member_offset_free` is set."""
    declared = lazy.array.dtype
    computed = lazy.copy().compute().array
    assert declared == computed.dtype == eager.array.dtype
    expected = eager.array
    if member_offset_free:
        computed = computed - computed.min(axis=(-2, -1), keepdims=True)
        expected = expected - expected.min(axis=(-2, -1), keepdims=True)
    scale = np.abs(expected).max()
    np.testing.assert_allclose(
        computed,
        expected,
        rtol=0,
        atol=64 * np.finfo(expected.dtype).eps * scale,
    )


WAVES_ENTRY_POINTS = {
    "ensure_reciprocal_space": lambda w: w.ensure_reciprocal_space(),
    "apply_ctf": lambda w: w.apply_ctf(defocus=20),
    "aperture": lambda w: w.apply_transform(Aperture(20)),
    "beam_tilt": lambda w: w.apply_transform(BeamTilt((1.0, 2.0))),
    "annular": lambda w: AnnularDetector(0, 30).detect(w),
    "flexible_annular": lambda w: FlexibleAnnularDetector().detect(w),
    "segmented": lambda w: SegmentedDetector(2, 4, 0, 30).detect(w),
    "pixelated": lambda w: PixelatedDetector().detect(w),
    "pixelated_max_angle": lambda w: PixelatedDetector(max_angle=30).detect(w),
    "pixelated_resample": lambda w: PixelatedDetector(resample=0.5).detect(w),
    "spectral_slit": lambda w: SpectralSlitDetector(width=10.0, q_max=40.0).detect(w),
    "waves": lambda w: WavesDetector().detect(w),
    # consistent on the base: fft_interpolate works in the configured precision
    "waves_gpts": lambda w: WavesDetector(gpts=GRID).detect(w),
    "pixelated_real_space_resample": lambda w: PixelatedDetector(
        reciprocal_space=False, resample=0.05
    ).detect(w),
}


@pytest.mark.parametrize("config, input_dtype", CONFIG_AND_INPUT)
@pytest.mark.parametrize("name", WAVES_ENTRY_POINTS)
def test_lazy_waves_results_declare_their_block_precision(name, config, input_dtype):
    entry_point = WAVES_ENTRY_POINTS[name]
    with abtem.config.set({"precision": config, "fft": "numpy"}):
        lazy = entry_point(_waves(input_dtype, lazy=True))
        eager = entry_point(_waves(input_dtype, lazy=False))
        assert lazy.is_lazy and not eager.is_lazy
        _check(lazy, eager)


def test_a_lazy_sum_of_detected_float64_intensities_accumulates_in_float64():
    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        detector = AnnularDetector(0, 30)
        lazy = detector.detect(_waves(np.complex128, lazy=True))
        eager = detector.detect(_waves(np.complex128, lazy=False))
        np.testing.assert_allclose(
            lazy.array.sum(axis=0).compute(), eager.array.sum(axis=0), rtol=1e-12
        )


def _potential_array(dtype, lazy):
    array = np.abs(_real_data(dtype))[:, :, :]
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return PotentialArray(array, slice_thickness=1.0, sampling=0.1)


MEASUREMENTS = {
    "diffractograms_float64": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.diffractograms(),
    ),
    "diffractograms_int64": (
        lambda: _int_images(np.int64, True),
        lambda: _int_images(np.int64, False),
        lambda m: m.diffractograms(),
    ),
    # 16-bit detector images: the FFT of any integer is complex128
    "diffractograms_uint16": (
        lambda: _int_images(np.uint16, True),
        lambda: _int_images(np.uint16, False),
        lambda m: m.diffractograms(),
    ),
    "integrate_gradient_complex64": (
        lambda: _images(np.complex64, True),
        lambda: _images(np.complex64, False),
        lambda m: m.integrate_gradient(),
    ),
    "integrate_gradient_complex128": (
        lambda: _images(np.complex128, True),
        lambda: _images(np.complex128, False),
        lambda m: m.integrate_gradient(),
    ),
    "diffraction_patterns_interpolate": (
        lambda: _patterns(np.float64, True),
        lambda: _patterns(np.float64, False),
        lambda m: m.interpolate(sampling=0.25),
    ),
    # counts are interpolated in the configured precision
    "diffraction_patterns_interpolate_uint16": (
        lambda: _int_patterns(np.uint16, True),
        lambda: _int_patterns(np.uint16, False),
        lambda m: m.interpolate(sampling=0.25),
    ),
    "center_of_mass_float32": (
        lambda: _patterns(np.float32, True),
        lambda: _patterns(np.float32, False),
        lambda m: m.center_of_mass(),
    ),
    "center_of_mass_float64": (
        lambda: _patterns(np.float64, True),
        lambda: _patterns(np.float64, False),
        lambda m: m.center_of_mass(),
    ),
    "images_interpolate_fft_float64": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.interpolate(0.05),
    ),
    "images_interpolate_fft_float32": (
        lambda: _images(np.float32, True),
        lambda: _images(np.float32, False),
        lambda m: m.interpolate(0.05),
    ),
    # consistent on the base
    "images_interpolate_spline": (
        lambda: _images(np.float64, True),
        lambda: _images(np.float64, False),
        lambda m: m.interpolate(0.05, method="spline"),
    ),
}


def _int_images(dtype, lazy):
    array = np.abs(10 * _real_data(np.float64)).astype(dtype)
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return Images(array, sampling=0.1, ensemble_axes_metadata=_members())


def _int_patterns(dtype, lazy):
    array = np.abs(10 * _real_data(np.float64)).astype(dtype) + 1
    if lazy:
        array = da.from_array(array, chunks=(1,) + GRID)
    return DiffractionPatterns(array, sampling=0.1, ensemble_axes_metadata=_members())


MEASUREMENT_PRECISION = {
    "images_interpolate_fft_float64": "float32",
    "images_interpolate_fft_float32": "float64",
}


@pytest.mark.parametrize("name", MEASUREMENTS)
def test_lazy_measurements_declare_their_block_precision(name):
    make_lazy, make_eager, method = MEASUREMENTS[name]
    config = MEASUREMENT_PRECISION.get(name, "float32")
    with abtem.config.set({"precision": config, "fft": "numpy"}):
        lazy = method(make_lazy())
        eager = method(make_eager())
        assert lazy.is_lazy and not eager.is_lazy
        _check(lazy, eager, member_offset_free=name.startswith("integrate_gradient"))


def test_lazy_potential_array_transmission_function_declares_its_block_precision():
    with abtem.config.set({"precision": "float32"}):
        lazy = _potential_array(np.float64, True).transmission_function(100e3)
        eager = _potential_array(np.float64, False).transmission_function(100e3)
        assert lazy.is_lazy and not eager.is_lazy
        _check(lazy, eager)


@pytest.mark.parametrize("dtype", [np.complex128, np.uint16])
def test_lazy_fft2_declares_the_dtype_of_its_blocks(dtype):
    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        if dtype == np.complex128:
            data = _complex_data(dtype)
        else:
            data = _int_images(dtype, False).array
        result = fft2(da.from_array(data, chunks=(1,) + GRID))
        assert result.dtype == result.compute().dtype == np.complex128


@pytest.mark.parametrize("kernel_dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("x_dtype", [np.complex64, np.complex128, np.float32])
def test_lazy_fft2_convolve_declares_the_dtype_of_its_blocks(x_dtype, kernel_dtype):
    # The product is taken in place, so the kernel's dtype does not matter.
    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        if np.issubdtype(x_dtype, np.complexfloating):
            data = _complex_data(x_dtype)
        else:
            data = _real_data(x_dtype)
        kernel = np.ones(GRID, dtype=kernel_dtype)
        lazy = fft2_convolve(da.from_array(data, chunks=(1,) + GRID), kernel)
        eager = fft2_convolve(data, kernel)
        assert lazy.dtype == lazy.compute().dtype == eager.dtype


MULTISLICE_CASES = ["fp", "fp_member", "single", "single_annular"]


MULTISLICE_ROUTES = {
    "multislice": lambda waves, potential, detectors: waves.multislice(
        potential, detectors=detectors
    ),
    "transform_apply": lambda waves, potential, detectors: MultisliceTransform(
        potential, detectors
    ).apply(waves),
    "apply_transform": lambda waves, potential, detectors: waves.apply_transform(
        MultisliceTransform(potential, detectors)
    ),
}


def _multislice(case, dtype, lazy, route="multislice"):
    from ase.build import bulk

    atoms = bulk("Si", cubic=True) * (1, 1, 2)
    atoms.rattle(0.01, seed=3)
    extent = atoms.cell[0, 0]
    if case.startswith("fp"):
        atoms = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.05, seed=5)
    potential = abtem.Potential(atoms, gpts=(32, 40), slice_thickness=2)
    rng = np.random.default_rng(3)
    shape = (2, 32, 40)
    array = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if lazy:
        array = da.from_array(array, chunks=(1, 32, 40))
    waves = Waves(
        array,
        energy=100e3,
        extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
    )
    if case == "fp_member":
        waves = waves[0]
    detectors = AnnularDetector(0, 40) if case == "single_annular" else None
    return MULTISLICE_ROUTES[route](waves, potential, detectors)


@pytest.mark.parametrize("route", MULTISLICE_ROUTES)
@pytest.mark.parametrize("config, input_dtype", CONFIG_AND_INPUT)
@pytest.mark.parametrize("case", MULTISLICE_CASES)
def test_multislice_of_waves_runs_in_the_configured_precision(
    case, config, input_dtype, route
):
    with abtem.config.set({"precision": config, "fft": "numpy"}):
        lazy = _multislice(case, input_dtype, lazy=True, route=route)
        eager = _multislice(case, input_dtype, lazy=False, route=route)
        assert lazy.is_lazy and not eager.is_lazy
        expected = np.dtype(config)
        if np.iscomplexobj(eager.array):
            expected = np.result_type(expected, np.complex64)
        assert eager.array.dtype == expected
        _check(lazy, eager)


def test_waves_dtype_is_the_array_dtype():
    with abtem.config.set({"precision": "float32"}):
        assert _waves(np.complex128, lazy=False).dtype == np.complex128
        assert _waves(np.complex128, lazy=True).dtype == np.complex128
        assert Probe(energy=100e3, semiangle_cutoff=20).dtype == np.complex64


@pytest.mark.parametrize("lazy", [False, True])
def test_transition_potential_multislice_of_waves_runs_in_the_configured_precision(
    lazy,
):
    from ase.build import bulk

    atoms = bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, gpts=(32, 32), slice_thickness=1.4)
    transition_potential = synthetic_transition_potential(
        gpts=potential.gpts, extent=potential.extent, n_transitions=3
    )
    probe = Probe(energy=100e3, semiangle_cutoff=20)
    probe.grid.match(potential)
    built = probe.build(lazy=False)

    def run(dtype):
        array = built.array.astype(dtype)
        if lazy:
            array = da.from_array(array, chunks=array.shape)
        waves = Waves(array, **built._copy_kwargs(exclude=("array",)))
        return waves.transition_potential_multislice(
            potential,
            transition_potential,
            detectors=FlexibleAnnularDetector(),
            double_channel=False,
            sites=atoms,
        ).compute()

    with abtem.config.set({"precision": "float32", "fft": "numpy"}):
        in_configured_precision = run(np.complex64)
        in_double_precision = run(np.complex128)
    np.testing.assert_array_equal(
        in_double_precision.array, in_configured_precision.array
    )
