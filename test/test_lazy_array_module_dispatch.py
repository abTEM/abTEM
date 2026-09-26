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
from abtem.measurements import DiffractionPatterns, Images

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
