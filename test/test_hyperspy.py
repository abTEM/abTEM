import sys
import warnings
from types import SimpleNamespace

import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import devices, lazy_params

import abtem.array
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import asnumpy
from abtem.waves import Waves

try:
    import hyperspy
except ImportError:
    hyperspy = None


@given(data=st.data())
@lazy_params
@devices
@pytest.mark.parametrize(
    "measurement",
    [
        abtem_st.images,
        abtem_st.line_profiles,
        abtem_st.diffraction_patterns,
        abtem_st.polar_measurements,
        abtem_st.potential_array,
        abtem_st.waves,
    ],
)
@pytest.mark.skipif("hyperspy" not in sys.modules, reason="requires hyperspy")
def test_hyperspy(data, measurement, lazy, device):
    measurement = data.draw(measurement(lazy=lazy, device=device))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        hyperspy_signal = measurement.to_hyperspy()
        expected = measurement.to_cpu().to_hyperspy().data

    signal_data = hyperspy_signal.data
    if lazy:
        signal_data, expected = signal_data.compute(), expected.compute()
    if device == "mps":
        # HyperSpy cannot hold torch backend arrays.
        assert isinstance(signal_data, np.ndarray)
    np.testing.assert_array_equal(asnumpy(signal_data), expected)


# Exports of reciprocal-space Waves and of unshifted DiffractionPatterns carry the
# zero frequency at the centre, on spatial-frequency axes. x and y differ in size
# (even and odd) and in sampling, so a swapped axis or an ifftshift used in place
# of fftshift fails.
GPTS = (16, 21)
SAMPLING = (0.1, 0.23)


def _waves(lazy: bool, device: str, reciprocal_space: bool) -> Waves:
    rng = np.random.default_rng(0)
    array = rng.standard_normal((3, *GPTS)) + 1j * rng.standard_normal((3, *GPTS))
    array = array.astype(np.complex64)
    if lazy:
        array = da.from_array(array, chunks=(1, -1, -1))
    waves = Waves(
        array,
        energy=100e3,
        sampling=SAMPLING,
        ensemble_axes_metadata=[OrdinalAxis(label="c", values=(1, 2, 3))],
        metadata={"label": "psi", "units": "arb. unit"},
    ).copy_to_device(device)
    return waves.ensure_reciprocal_space() if reciprocal_space else waves


def _host(array) -> np.ndarray:
    if isinstance(array, da.Array):
        array = array.compute()
    return asnumpy(array)


def _stored(waves) -> np.ndarray:
    return _host(waves.copy().compute().array if waves.is_lazy else waves.array)


def _assert_frequencies(coordinates, i: int):
    dk = 1 / (GPTS[i] * SAMPLING[i])
    expected = np.fft.fftshift(np.fft.fftfreq(GPTS[i], SAMPLING[i]))
    np.testing.assert_allclose(coordinates, expected, rtol=0, atol=1e-12 * dk)


@pytest.fixture
def quantem_records(monkeypatch):
    """Replace quantem by a stand-in whose Dataset.from_array records its kwargs."""
    captured = {}

    def from_array(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    fake = SimpleNamespace(
        core=SimpleNamespace(
            datastructures=SimpleNamespace(
                Dataset=SimpleNamespace(from_array=from_array)
            )
        )
    )
    monkeypatch.setattr(abtem.array, "em", fake)
    return captured


@lazy_params
@devices
def test_to_data_array_of_reciprocal_space_waves(lazy, device):
    xr = pytest.importorskip("xarray")
    waves = _waves(lazy, device, reciprocal_space=True)

    data_array = waves.to_data_array()

    assert isinstance(data_array, xr.DataArray)
    assert data_array.dims == ("c", "kx", "ky")
    for i, dim in enumerate(("kx", "ky")):
        _assert_frequencies(data_array.coords[dim].values, i)
        assert data_array.coords[dim].attrs["units"] == "1/Å"
    assert isinstance(data_array.data, da.Array) == lazy
    np.testing.assert_array_equal(
        _host(data_array.data), np.fft.fftshift(_stored(waves), axes=(-2, -1))
    )


@lazy_params
@devices
def test_to_hyperspy_of_reciprocal_space_waves(lazy, device):
    pytest.importorskip("hyperspy")
    waves = _waves(lazy, device, reciprocal_space=True)

    signal = waves.to_hyperspy()

    axes = signal.axes_manager.signal_axes
    assert [axis.name for axis in axes] == ["kx", "ky"]
    for i, axis in enumerate(axes):
        assert axis.units == "1/Å"
        _assert_frequencies(axis.axis, i)
    # to_hyperspy transposes the base axes by default
    expected = np.swapaxes(np.fft.fftshift(_stored(waves), axes=(-2, -1)), -1, -2)
    np.testing.assert_array_equal(_host(signal.data), expected)


@lazy_params
@devices
def test_to_quantem_of_reciprocal_space_waves(lazy, device, quantem_records):
    waves = _waves(lazy, device, reciprocal_space=True)
    expected = np.fft.fftshift(_stored(waves), axes=(-2, -1))

    waves.to_quantem()

    dk = tuple(1 / (n * d) for n, d in zip(GPTS, SAMPLING))
    np.testing.assert_allclose(quantem_records["sampling"][-2:], dk, rtol=1e-12)
    assert quantem_records["units"][-2:] == ("A^-1", "A^-1")
    assert quantem_records["name"] == "DiffractionPatterns"
    np.testing.assert_array_equal(_host(quantem_records["array"]), expected)


@lazy_params
@devices
def test_to_quantem_of_real_space_waves_is_unchanged(lazy, device, quantem_records):
    waves = _waves(lazy, device, reciprocal_space=False)
    expected = _stored(waves)

    waves.to_quantem()

    assert quantem_records["name"] == "Waves"
    np.testing.assert_allclose(quantem_records["sampling"][-2:], SAMPLING)
    assert quantem_records["units"][-2:] == ("A", "A")
    np.testing.assert_array_equal(_host(quantem_records["array"]), expected)


@devices
def test_to_quantem_leaves_a_lazy_object_lazy(device, quantem_records):
    waves = _waves(True, device, reciprocal_space=True)
    unshifted = _waves(True, device, reciprocal_space=False).diffraction_patterns(
        max_angle=None, fftshift=False
    )
    assert not unshifted.fftshift

    for lazy_object in (waves, unshifted):
        lazy_object.to_quantem()

        assert lazy_object.is_lazy
        assert isinstance(lazy_object.array, da.Array)


@devices
def test_exports_invert_to_the_stored_coefficients(device):
    pytest.importorskip("xarray")
    waves = _waves(False, device, reciprocal_space=True)

    data_array = waves.to_data_array()

    np.testing.assert_array_equal(
        np.fft.ifftshift(_host(data_array.data), axes=(-2, -1)), _stored(waves)
    )


@devices
def test_lazy_exports_match_eager_exports(device):
    pytest.importorskip("xarray")
    eager = _waves(False, device, reciprocal_space=True).to_data_array()
    lazy_export = _waves(True, device, reciprocal_space=True).to_data_array()

    np.testing.assert_array_equal(_host(lazy_export.data), _host(eager.data))
    for dim in eager.dims:
        np.testing.assert_array_equal(lazy_export.coords[dim], eager.coords[dim])


def test_zarr_round_trip_of_reciprocal_space_waves_is_unchanged(tmp_path):
    """The stored array, sampling and axes are those of the in-memory waves."""
    pytest.importorskip("xarray")
    waves = _waves(False, "cpu", reciprocal_space=True)
    url = str(tmp_path / "waves.zarr")

    waves.to_zarr(url)
    loaded = abtem.array.from_zarr(url).compute()

    assert loaded.reciprocal_space
    assert loaded.sampling == waves.sampling
    assert loaded.axes_metadata == waves.axes_metadata
    np.testing.assert_array_equal(loaded.array, waves.array)
    exported, expected = loaded.to_data_array(), waves.to_data_array()
    np.testing.assert_array_equal(exported.values, expected.values)
    for dim in expected.dims:
        np.testing.assert_array_equal(exported.coords[dim], expected.coords[dim])


@lazy_params
@devices
def test_real_space_waves_export_real_space_axes(lazy, device):
    pytest.importorskip("xarray")
    waves = _waves(lazy, device, reciprocal_space=False)

    data_array = waves.to_data_array()

    assert data_array.dims == ("c", "x", "y")
    for i, dim in enumerate(("x", "y")):
        np.testing.assert_allclose(
            data_array.coords[dim].values, np.arange(GPTS[i]) * SAMPLING[i]
        )
        assert data_array.coords[dim].attrs["units"] == "Å"
    np.testing.assert_array_equal(_host(data_array.data), _stored(waves))


@lazy_params
@devices
def test_unshifted_diffraction_patterns_export_fftshifted(lazy, device):
    pytest.importorskip("hyperspy")
    pytest.importorskip("xarray")
    waves = _waves(lazy, device, reciprocal_space=False)
    unshifted = waves.diffraction_patterns(max_angle=None, fftshift=False)
    shifted = waves.diffraction_patterns(max_angle=None, fftshift=True)
    assert not unshifted.fftshift

    for export in (unshifted.to_data_array(), shifted.to_data_array()):
        assert export.dims == ("c", "kx", "ky")
        for i, dim in enumerate(("kx", "ky")):
            _assert_frequencies(export.coords[dim].values, i)
    np.testing.assert_array_equal(
        _host(unshifted.to_data_array().data), _host(shifted.to_data_array().data)
    )
    for signal in (unshifted.to_hyperspy(), shifted.to_hyperspy()):
        for i, axis in enumerate(signal.axes_manager.signal_axes):
            _assert_frequencies(axis.axis, i)
    np.testing.assert_array_equal(
        _host(unshifted.to_hyperspy().data), _host(shifted.to_hyperspy().data)
    )
