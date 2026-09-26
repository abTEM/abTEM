"""Lazy arrays must reach the array module's functions block by block.

``get_array_module`` on a dask array returns the module of its chunks. NumPy's
functions accept a dask array (they dispatch to dask), so code that hands the whole
dask array to ``xp.<function>`` works on CPU; CuPy's functions reject it, so the same
code fails for lazy GPU data. The CPU tests below replace the array module with
``_RejectsDask`` -- NumPy's functions, raising on a dask argument as CuPy's do -- in
the module under test, so they catch this without a GPU. The GPU tests run the same
paths on real CuPy chunks.
"""

import types

import ase.build
import dask.array as da
import numpy as np
import pytest
from utils import requires_gpu

import abtem
import abtem.array
import abtem.measurements
import abtem.potentials.iam
import abtem.waves
from abtem.core import backend
from abtem.core.axes import OrdinalAxis
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.measurements import DiffractionPatterns, Images, periodic_crop

ELEMENT_WISE = ("abs", "real", "imag", "phase", "intensity")
LABELS = {
    "abs": "amplitude",
    "real": "real",
    "imag": "imaginary",
    "phase": "phase",
    "intensity": "intensity",
}


def _contains_dask(values) -> bool:
    for value in values:
        if isinstance(value, da.Array):
            return True
        if isinstance(value, (list, tuple)) and _contains_dask(value):
            return True
    return False


class _RejectsDask(types.ModuleType):
    """NumPy's functions, raising on a dask argument like CuPy's.

    ``squeeze`` is let through: CuPy's is ``a.squeeze(axis)``, which works on a dask
    array.
    """

    _duck_typed = ("squeeze",)

    def __getattr__(self, name):
        attr = getattr(np, name)
        if not callable(attr) or isinstance(attr, type) or name in self._duck_typed:
            return attr

        def function(*args, **kwargs):
            if _contains_dask(args) or _contains_dask(kwargs.values()):
                raise TypeError(f"Unsupported type {da.Array}")
            return attr(*args, **kwargs)

        return function


@pytest.fixture
def rejects_dask(monkeypatch):
    """Return a function that makes `module`'s array module reject dask arrays."""
    stand_in = _RejectsDask("rejects_dask")

    def patch(module, device_strings=False):
        def get_array_module(x=None):
            if isinstance(x, da.Array) or (device_strings and isinstance(x, str)):
                return stand_in
            return backend.get_array_module(x)

        monkeypatch.setattr(module, "get_array_module", get_array_module)

    return patch


def _complex_diffraction_patterns(xp=np, lazy=True):
    rng = np.random.default_rng(0)
    array = (rng.random((3, 4, 8, 8)) + 1j * rng.random((3, 4, 8, 8))).astype(
        np.complex64
    )
    array = xp.asarray(array)
    if lazy:
        array = da.from_array(array, chunks=(1, 2, 8, 8))
    return DiffractionPatterns(
        array,
        sampling=(0.1, 0.1),
        fftshift=True,
        ensemble_axes_metadata=[
            OrdinalAxis(values=tuple(range(3))),
            OrdinalAxis(values=tuple(range(4))),
        ],
        metadata={"energy": 60e3, "label": "source label", "units": "source units"},
    )


def _potential():
    atoms = ase.build.mx2("WSe2", vacuum=2)
    return abtem.Potential(atoms, sampling=0.2, slice_thickness=2)


def _scan(detector, device="cpu"):
    probe = abtem.Probe(energy=60e3, semiangle_cutoff=20, device=device)
    return probe.scan(_potential(), scan=abtem.GridScan(sampling=1.0), detectors=detector)


def _crystal_potential(device="cpu", lazy_unit=True):
    atoms = ase.build.bulk("Si", cubic=True)
    unit = abtem.Potential(atoms, sampling=0.2, slice_thickness=1, device=device)
    unit = unit.build(lazy=lazy_unit)
    return abtem.CrystalPotential(unit, repetitions=(2, 2, 2))


def _to_numpy(array):
    return backend.asnumpy(array.compute() if isinstance(array, da.Array) else array)


@pytest.mark.parametrize("method", ELEMENT_WISE)
def test_element_wise_funcs_apply_per_block(rejects_dask, method):
    dp = _complex_diffraction_patterns()
    expected = getattr(_complex_diffraction_patterns(lazy=False), method)().array
    rejects_dask(abtem.measurements)

    result = getattr(dp, method)()

    assert result.is_lazy
    assert result.array.dtype == expected.dtype
    np.testing.assert_array_equal(result.compute().array, expected)


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("method", ELEMENT_WISE)
def test_element_wise_funcs_leave_the_source_metadata_alone(method, lazy):
    dp = _complex_diffraction_patterns(lazy=lazy)

    result = getattr(dp, method)()

    assert result.metadata["label"] == LABELS[method]
    assert dp.metadata["label"] == "source label"
    assert dp.metadata["units"] == "source units"


def test_tile_scan_applies_to_lazy_arrays(rejects_dask):
    patterns = _scan(abtem.PixelatedDetector(max_angle=30))
    expected = patterns.copy().compute().tile_scan((2, 3)).array
    rejects_dask(abtem.measurements)

    tiled = patterns.tile_scan((2, 3))

    assert tiled.is_lazy
    np.testing.assert_array_equal(tiled.compute().array, expected)


def test_to_image_ensemble_applies_to_lazy_arrays(rejects_dask):
    polar = _scan(abtem.SegmentedDetector(10, 50, 2, 4))
    expected = polar.copy().compute().to_image_ensemble().array
    rejects_dask(abtem.measurements)

    images = polar.to_image_ensemble()

    assert images.is_lazy
    np.testing.assert_array_equal(images.compute().array, expected)


def test_complex_differentials_stay_lazy(rejects_dask):
    polar = _scan(abtem.SegmentedDetector(10, 50, 2, 4))
    directions = (((0, 2), (1, 3)), ((4, 6), (5, 7)))
    expected = polar.copy().compute().differentials(*directions, return_complex=True).array
    rejects_dask(abtem.measurements)

    differentials = polar.differentials(*directions, return_complex=True)

    assert differentials.is_lazy
    assert differentials.array.dtype == expected.dtype
    np.testing.assert_array_equal(differentials.compute().array, expected)


def test_crystal_potential_tiles_a_lazily_built_unit(rejects_dask):
    waves = abtem.PlaneWave(energy=60e3)
    expected = waves.multislice(_crystal_potential(lazy_unit=False)).compute().array
    rejects_dask(abtem.potentials.iam, device_strings=True)

    exit_waves = waves.multislice(_crystal_potential(lazy_unit=True)).compute()

    np.testing.assert_array_equal(exit_waves.array, expected)



def test_crystal_potential_generates_eager_slices_from_a_lazily_built_unit():
    """Tiling a lazy unit yields dask-backed slices, which an eager consumer such as
    build(lazy=False) cannot place into a CuPy array."""
    lazy = _crystal_potential(lazy_unit=True)
    eager = _crystal_potential(lazy_unit=False)

    slices = list(lazy.generate_slices())

    assert slices and not any(isinstance(s.array, da.Array) for s in slices)
    # the unit is materialised internally, not computed in place
    assert lazy.potential_unit.is_lazy
    np.testing.assert_array_equal(
        lazy.build(lazy=False).array, eager.build(lazy=False).array
    )


def _relative_difference_inputs():
    rng = np.random.default_rng(0)
    a = rng.random((8, 8)).astype(np.float32) + 0.5
    b = rng.random((8, 8)).astype(np.float32) + 0.5
    a[0, 0] = 0.01  # below the threshold below: NaN there
    valid = np.abs(a) >= 0.1 * a.max()
    expected = np.where(valid, (a - b) / np.where(valid, a, 1), np.nan) * 100
    return a, b, expected


def test_relative_difference_of_eager_measurements_is_unchanged():
    a, b, expected = _relative_difference_inputs()

    result = Images(a, sampling=0.1).relative_difference(
        Images(b, sampling=0.1), min_relative_tol=0.1
    )

    np.testing.assert_array_equal(result.array, expected)


def test_relative_difference_of_lazy_measurements(rejects_dask):
    a, b, expected = _relative_difference_inputs()
    lazy_a = Images(da.from_array(a, chunks=4), sampling=0.1)
    lazy_b = Images(da.from_array(b, chunks=4), sampling=0.1)
    rejects_dask(abtem.measurements)

    result = lazy_a.relative_difference(lazy_b, min_relative_tol=0.1)

    assert result.is_lazy
    np.testing.assert_array_equal(result.compute().array, expected)


def _transmit(lazy_potential, lazy_waves, conjugate):
    atoms = ase.build.mx2("MoS2", vacuum=2)
    potential = abtem.Potential(atoms, gpts=64, slice_thickness=10)
    potential = potential.build(lazy=lazy_potential)
    waves = abtem.PlaneWave(energy=100e3).match_grid(potential).build(lazy=lazy_waves)
    return potential.transmit(waves, conjugate=conjugate)


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("lazy_waves", [False, True], ids=["eager_waves", "lazy_waves"])
def test_transmit_through_a_lazy_potential(lazy_waves, conjugate):
    expected = _transmit(False, False, conjugate).array

    transmitted = _transmit(True, lazy_waves, conjugate)

    assert transmitted.is_lazy == lazy_waves
    np.testing.assert_array_equal(_to_numpy(transmitted.array), expected)


def _synthetic_transition_potential(xp=np):
    rng = np.random.default_rng(0)
    array = rng.standard_normal((3, 32, 32)) + 1j * rng.standard_normal((3, 32, 32))
    return TransitionPotentialArray(
        Z=5,
        energy=100e3,
        extent=(8.0, 8.0),
        array=xp.asarray(array.astype(np.complex64)),
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )


def _core_loss_probe_waves(lazy, device="cpu"):
    probe = abtem.Probe(
        energy=100e3, semiangle_cutoff=20, extent=8.0, gpts=32, device=device
    )
    return probe.build(lazy=lazy)


def test_transition_potential_methods_accept_lazy_waves(rejects_dask):
    potential = _synthetic_transition_potential()
    sites = np.array([[2.0, 2.0], [6.0, 6.0]])
    eager = _core_loss_probe_waves(lazy=False)
    expected_threshold = potential.absolute_threshold(eager, 0.9)
    expected_sites = potential.filter_sites(eager, sites, threshold=1e-6)
    rejects_dask(abtem.inelastic.core_loss)

    waves = _core_loss_probe_waves(lazy=True)
    threshold = potential.absolute_threshold(waves, 0.9)
    filtered = potential.filter_sites(waves, sites, threshold=1e-6)

    assert threshold == expected_threshold
    np.testing.assert_array_equal(filtered, expected_sites)
    # the caller's waves are not computed in place
    assert waves.is_lazy


def _image_ensemble(xp=np, lazy=False):
    rng = np.random.default_rng(1)
    array = xp.asarray(rng.random((3, 40, 30)).astype(np.float32))
    if lazy:
        array = da.from_array(array, chunks=(1, 40, 30))
    return Images(
        array,
        sampling=(0.1, 0.12),
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )


@pytest.mark.parametrize("corner", [(-3, 25), (35, -4), (38, 27)])
def test_periodic_crop_across_the_border(corner):
    array = _image_ensemble().array
    shape = (10, 9)
    x = np.arange(corner[0], corner[0] + shape[0]) % array.shape[-2]
    y = np.arange(corner[1], corner[1] + shape[1]) % array.shape[-1]
    x, y = np.meshgrid(x, y, indexing="ij")
    expected = array[..., x.ravel(), y.ravel()].reshape(array.shape[:-2] + shape)

    eager = periodic_crop(array, corner, shape)
    lazy = periodic_crop(da.from_array(array, chunks=(1, 40, 30)), corner, shape)

    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(lazy.compute(), expected)


@pytest.mark.parametrize("position", [(2.0, 1.8), (0.1, 3.5), (3.95, 0.05)])
def test_integrate_disc_of_lazy_images(position):
    expected = np.asarray(_image_ensemble().integrate_disc(position, 0.5))

    result = _image_ensemble(lazy=True).integrate_disc(position, 0.5)

    assert isinstance(result, da.Array)
    # a lazy reduction sums in a different order
    np.testing.assert_allclose(result.compute(), expected, rtol=1e-12, atol=0)


def test_integrate_disc_raise_checks_the_image_axes_of_an_ensemble():
    images = _image_ensemble()
    inside = abtem.measurements.integrate_disc(images, (2.0, 1.8), 0.5, border="raise")

    np.testing.assert_array_equal(inside, images.integrate_disc((2.0, 1.8), 0.5))
    with pytest.raises(RuntimeError, match="outside the image"):
        abtem.measurements.integrate_disc(images, (0.1, 3.5), 0.5, border="raise")


def _polar_measurement():
    polar = _scan(abtem.SegmentedDetector(10, 50, 2, 4))
    return polar.compute()


def test_polar_to_diffraction_patterns_of_eager_measurements_is_unchanged():
    polar = _polar_measurement()
    regions = abtem.detectors._polar_detector_bins(
        gpts=(48, 48),
        sampling=tuple(
            (1 + 0.1) * polar.outer_angle / 48 * 2 for _ in range(2)
        ),
        inner=polar.radial_offset,
        outer=polar.outer_angle,
        nbins_radial=polar.base_shape[0],
        nbins_azimuthal=polar.base_shape[1],
        fftshift=True,
        rotation=polar.azimuthal_offset,
        offset=(0.0, 0.0),
        return_indices=False,
    )
    expected = np.zeros(polar.ensemble_shape + regions.shape, dtype=np.float32)
    for label in range(regions.max() + 1):
        radial, azimuthal = np.unravel_index(label, polar.base_shape)
        expected[..., regions == label] = polar.array[..., radial, azimuthal][..., None]
    expected[..., regions < 0] = np.nan

    patterns = polar.to_diffraction_patterns(48)

    assert patterns.array.dtype == np.float32
    np.testing.assert_array_equal(patterns.array, expected)


def test_polar_to_diffraction_patterns_stays_lazy_and_runs_the_graph_once():
    polar = _polar_measurement()
    expected = polar.to_diffraction_patterns(48).array
    evaluations = []

    def count(block):
        evaluations.append(1)
        return block

    lazy = polar.copy()
    chunks = (5,) + (-1,) * (polar.array.ndim - 1)
    lazy._array = da.from_array(polar.array, chunks=chunks).map_blocks(count)
    evaluations.clear()  # map_blocks itself calls count once, on the meta

    patterns = lazy.to_diffraction_patterns(48)

    assert patterns.is_lazy
    np.testing.assert_array_equal(patterns.compute().array, expected)
    assert len(evaluations) == lazy.array.numblocks[0]


def test_multi_energy_s_matrix_build_honours_lazy_false():
    atoms = ase.build.mx2("WSe2", vacuum=2)
    potential = abtem.Potential(atoms, sampling=0.2, slice_thickness=2)

    def build(lazy):
        s_matrix = abtem.SMatrix(
            potential=potential, energy=[60e3, 80e3], semiangle_cutoff=10
        )
        return s_matrix.build(lazy=lazy)

    eager = build(lazy=False)
    lazy = build(lazy=True).compute()

    assert not eager.is_lazy
    # the build is not bit-reproducible from run to run
    scale = np.abs(lazy.array).max()
    np.testing.assert_allclose(eager.array, lazy.array, rtol=0, atol=1e-5 * scale)


def test_eager_multislice_leaves_a_lazy_potential_lazy():
    atoms = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(atoms, sampling=0.2, slice_thickness=1).build(lazy=True)
    eager_potential = potential.copy().compute()
    waves = abtem.PlaneWave(energy=60e3)
    waves.grid.match(potential)
    expected = waves.build(lazy=False).multislice(eager_potential).array

    exit_waves = waves.build(lazy=False).multislice(potential)

    assert potential.is_lazy
    np.testing.assert_array_equal(exit_waves.array, expected)

def test_concatenate_eager_then_lazy():
    images = Images(
        np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8),
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
    )

    concatenated = abtem.array.concatenate([images, images.copy().ensure_lazy()])

    assert concatenated.is_lazy
    np.testing.assert_array_equal(
        concatenated.compute().array, np.concatenate([images.array] * 2)
    )


@requires_gpu
class TestLazyCuPy:
    """The paths above on real CuPy chunks."""

    @pytest.fixture(autouse=True)
    def _gpu(self):
        with abtem.config.set({"device": "gpu"}):
            yield

    @pytest.mark.parametrize("method", ELEMENT_WISE)
    def test_element_wise_funcs(self, method):
        import cupy as cp

        dp = _complex_diffraction_patterns(xp=cp)
        expected = getattr(_complex_diffraction_patterns(lazy=False), method)().array

        result = getattr(dp, method)()

        assert result.is_lazy
        assert isinstance(result.array._meta, cp.ndarray)
        np.testing.assert_allclose(
            _to_numpy(result.array), expected, rtol=1e-6, atol=1e-6
        )

    def test_tile_scan(self):
        patterns = _scan(abtem.PixelatedDetector(max_angle=30, to_cpu=False), "gpu")
        expected = _to_numpy(patterns.copy().compute().tile_scan((2, 3)).array)

        tiled = patterns.tile_scan((2, 3))

        assert tiled.is_lazy
        np.testing.assert_array_equal(_to_numpy(tiled.array), expected)

    def test_to_image_ensemble(self):
        polar = _scan(abtem.SegmentedDetector(10, 50, 2, 4, to_cpu=False), "gpu")
        expected = _to_numpy(polar.copy().compute().to_image_ensemble().array)

        images = polar.to_image_ensemble()

        assert images.is_lazy
        np.testing.assert_array_equal(_to_numpy(images.array), expected)

    def test_complex_differentials(self):
        polar = _scan(abtem.SegmentedDetector(10, 50, 2, 4, to_cpu=False), "gpu")
        directions = (((0, 2), (1, 3)), ((4, 6), (5, 7)))
        expected = polar.copy().compute().differentials(*directions, return_complex=True)

        differentials = polar.differentials(*directions, return_complex=True)

        assert differentials.is_lazy
        np.testing.assert_array_equal(
            _to_numpy(differentials.array), _to_numpy(expected.array)
        )

    def test_crystal_potential_with_a_lazily_built_unit(self):
        waves = abtem.PlaneWave(energy=60e3, device="gpu")
        eager_unit = _crystal_potential(device="gpu", lazy_unit=False)
        expected = _to_numpy(waves.multislice(eager_unit).array)

        lazy_unit = _crystal_potential(device="gpu", lazy_unit=True)
        exit_waves = waves.multislice(lazy_unit)

        np.testing.assert_array_equal(_to_numpy(exit_waves.array), expected)

    def test_crystal_potential_build_with_a_lazily_built_unit(self):
        expected = _to_numpy(
            _crystal_potential(device="gpu", lazy_unit=False).build(lazy=False).array
        )

        built = _crystal_potential(device="gpu", lazy_unit=True).build(lazy=False)

        np.testing.assert_array_equal(_to_numpy(built.array), expected)

    def test_blocked_scan_of_a_crystal_potential_with_a_lazily_built_unit(self):
        """A scan split into several blocks builds the potential through
        generate_slices rather than generate_chunked_slices."""

        def haadf(lazy_unit):
            probe = abtem.Probe(energy=60e3, semiangle_cutoff=20, device="gpu")
            scan = abtem.GridScan(start=(0, 0), end=(2.7, 2.7), sampling=0.3)
            detector = abtem.AnnularDetector(inner=40, outer=100)
            potential = _crystal_potential(device="gpu", lazy_unit=lazy_unit)
            measurement = probe.scan(
                potential, scan=scan, detectors=detector, max_batch=10
            )
            return _to_numpy(measurement.compute().array)

        np.testing.assert_allclose(
            haadf(lazy_unit=True), haadf(lazy_unit=False), rtol=1e-6, atol=0
        )

    def test_relative_difference(self):
        import cupy as cp

        a, b, expected = _relative_difference_inputs()
        lazy_a = Images(da.from_array(cp.asarray(a), chunks=4), sampling=0.1)
        lazy_b = Images(da.from_array(cp.asarray(b), chunks=4), sampling=0.1)

        result = lazy_a.relative_difference(lazy_b, min_relative_tol=0.1)

        assert result.is_lazy
        np.testing.assert_allclose(
            _to_numpy(result.array), expected, rtol=1e-6, equal_nan=True
        )

    @pytest.mark.parametrize("conjugate", [False, True])
    def test_transmit_through_a_lazy_potential_with_eager_waves(self, conjugate):
        expected = _to_numpy(_transmit(False, False, conjugate).array)

        transmitted = _transmit(True, False, conjugate)

        assert not transmitted.is_lazy
        np.testing.assert_allclose(
            _to_numpy(transmitted.array), expected, rtol=1e-6, atol=1e-6
        )

    def test_transition_potential_methods_with_lazy_waves(self):
        import cupy as cp

        potential = _synthetic_transition_potential(xp=cp)
        sites = np.array([[2.0, 2.0], [6.0, 6.0]])
        eager = _core_loss_probe_waves(lazy=False, device="gpu")
        expected_threshold = potential.absolute_threshold(eager, 0.9)
        expected_sites = potential.filter_sites(eager, sites, threshold=1e-6)
        expected_scattered = _to_numpy(potential.scatter(eager, sites).array)

        waves = _core_loss_probe_waves(lazy=True, device="gpu")

        assert potential.absolute_threshold(waves, 0.9) == pytest.approx(
            expected_threshold, rel=1e-5
        )
        np.testing.assert_array_equal(
            potential.filter_sites(waves, sites, threshold=1e-6), expected_sites
        )
        np.testing.assert_allclose(
            _to_numpy(potential.scatter(waves, sites).array),
            expected_scattered,
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_integrate_disc(self, lazy):
        import cupy as cp

        expected = np.asarray(_image_ensemble().integrate_disc((0.1, 3.5), 0.5))

        result = _image_ensemble(xp=cp, lazy=lazy).integrate_disc((0.1, 3.5), 0.5)

        np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-6)

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_polar_to_diffraction_patterns(self, lazy):
        detector = abtem.SegmentedDetector(10, 50, 2, 4, to_cpu=False)
        polar = _scan(detector, "gpu")
        if not lazy:
            polar = polar.compute()
        expected = _polar_measurement().to_diffraction_patterns(48).array

        patterns = polar.to_diffraction_patterns(48)

        assert patterns.is_lazy == lazy
        np.testing.assert_allclose(
            _to_numpy(patterns.array), expected, rtol=1e-5, atol=1e-7, equal_nan=True
        )

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_multi_energy_s_matrix_build(self, lazy):
        atoms = ase.build.mx2("WSe2", vacuum=2)

        def build(device, lazy):
            potential = abtem.Potential(
                atoms, sampling=0.2, slice_thickness=2, device=device
            )
            s_matrix = abtem.SMatrix(
                potential=potential,
                energy=[60e3, 80e3],
                semiangle_cutoff=10,
                device=device,
            )
            return s_matrix.build(lazy=lazy)

        expected = build("cpu", lazy=False).array

        built = build("gpu", lazy=lazy)

        assert built.is_lazy == lazy
        scale = np.abs(expected).max()
        np.testing.assert_allclose(
            _to_numpy(built.array), expected, rtol=0, atol=1e-5 * scale
        )

    def test_lazy_interpolate_keeps_the_device(self):
        import cupy as cp

        patterns = _complex_diffraction_patterns(xp=cp).intensity()
        expected = _complex_diffraction_patterns(lazy=False).intensity()
        expected = expected.interpolate(0.05).array

        interpolated = patterns.interpolate(0.05)

        assert interpolated.is_lazy
        assert interpolated.device == "gpu"
        on_cpu = interpolated.to_cpu().compute()
        assert isinstance(on_cpu.array, np.ndarray)
        np.testing.assert_allclose(on_cpu.array, expected, rtol=1e-5, atol=1e-6)

    def test_concatenate_eager_then_lazy(self):
        import cupy as cp

        images = Images(
            cp.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8),
            sampling=0.1,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )

        concatenated = abtem.array.concatenate([images, images.copy().ensure_lazy()])

        assert concatenated.is_lazy
        np.testing.assert_array_equal(
            _to_numpy(concatenated.array),
            np.concatenate([_to_numpy(images.array)] * 2),
        )
