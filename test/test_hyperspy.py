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
from abtem.core.axes import OrdinalAxis, ScanAxis
from abtem.core.backend import asnumpy
from abtem.measurements import DiffractionPatterns, Images, PolarMeasurements
from abtem.potentials.iam import PotentialArray
from abtem.waves import Waves

try:
    import hyperspy
except ImportError:
    hyperspy = None


@given(data=st.data())
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
@pytest.mark.skipif(hyperspy is None, reason="requires hyperspy")
def test_hyperspy(data, measurement, device):
    for lazy in (False, True):
        drawn = data.draw(measurement(lazy=lazy, device=device))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            hyperspy_signal = drawn.to_hyperspy()

        # HyperSpy orders the signal axes the other way round: the stored array
        # with its base axes reversed. A potential is exported as a stack of
        # images, one per slice, and unshifted diffraction patterns with the zero
        # frequency at the centre.
        stored = _host(drawn.array)
        base_dims = drawn._base_dims
        if isinstance(drawn, PotentialArray):
            base_dims = 2
        if isinstance(drawn, DiffractionPatterns) and not drawn.fftshift:
            stored = np.fft.fftshift(stored, axes=(-2, -1))
        assert len(hyperspy_signal.axes_manager.signal_axes) == base_dims, lazy
        base = tuple(range(stored.ndim - base_dims, stored.ndim))
        expected = np.moveaxis(stored, base, base[::-1])

        signal_data = hyperspy_signal.data
        if lazy:
            signal_data = signal_data.compute()
        if device == "mps":
            # HyperSpy cannot hold torch backend arrays.
            assert isinstance(signal_data, np.ndarray)
        np.testing.assert_array_equal(
            asnumpy(signal_data), expected, err_msg=f"lazy={lazy}"
        )


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


def _assert_quantem_calibration(records, expected):
    """origin + index * sampling of the trailing axes equal the expected coordinates."""
    origin, sampling = records["origin"], records["sampling"]
    for o, d, coordinates in zip(
        origin[-len(expected) :], sampling[-len(expected) :], expected
    ):
        calibrated = o + np.arange(len(coordinates)) * d
        np.testing.assert_allclose(calibrated, coordinates, rtol=0, atol=1e-12 * d)


@pytest.fixture
def quantem_records(monkeypatch):
    """Replace quantem by a stand-in whose Dataset.from_array records its kwargs,
    and the dataset's metadata under "metadata". The tests against the real
    package are marked `quantem`."""
    captured = {}

    def from_array(**kwargs):
        captured.update(kwargs)
        # Like quantem's Dataset, an empty metadata dict exposed as `metadata`.
        captured["metadata"] = {}
        return SimpleNamespace(metadata=captured["metadata"], **kwargs)

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
    assert quantem_records["origin"] == (0.0, 0.0, 0.0)
    _assert_quantem_calibration(
        quantem_records, [np.arange(n) * d for n, d in zip(GPTS, SAMPLING)]
    )
    np.testing.assert_array_equal(_host(quantem_records["array"]), expected)


def _frequency_exports(kind: str, lazy: bool, device: str):
    if kind == "reciprocal_space_waves":
        return _waves(lazy, device, reciprocal_space=True)
    waves = _waves(lazy, device, reciprocal_space=False)
    return waves.diffraction_patterns(
        max_angle=None, fftshift=kind == "diffraction_patterns_centred"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "reciprocal_space_waves",
        "diffraction_patterns_centred",
        "diffraction_patterns_unshifted",
    ],
)
@lazy_params
@devices
def test_to_quantem_origin_puts_zero_frequency_at_k_zero(
    kind, lazy, device, quantem_records
):
    exported = _frequency_exports(kind, lazy, device)

    exported.to_quantem()

    frequencies = [
        np.fft.fftshift(np.fft.fftfreq(n, d)) for n, d in zip(GPTS, SAMPLING)
    ]
    _assert_quantem_calibration(quantem_records, frequencies)
    assert quantem_records["origin"][0] == 0.0  # the ensemble axis
    for o, d, n in zip(
        quantem_records["origin"][-2:], quantem_records["sampling"][-2:], GPTS
    ):
        assert abs(o + (n // 2) * d) <= 1e-12 * d


@devices
def test_to_quantem_origin_is_the_offset_of_scan_axes(device, quantem_records):
    scan_axes = [
        ScanAxis(label="x", sampling=0.5, offset=1.0, units="Å"),
        ScanAxis(label="y", sampling=0.4, offset=-2.0, units="Å"),
    ]
    images = Images(
        np.zeros((4, 5, *GPTS), dtype=np.float32),
        sampling=SAMPLING,
        ensemble_axes_metadata=scan_axes,
    ).copy_to_device(device)

    images.to_quantem()

    assert quantem_records["origin"][:2] == (1.0, -2.0)
    assert quantem_records["origin"][2:] == (0.0, 0.0)
    _assert_quantem_calibration(
        quantem_records,
        [axis.coordinates(n) for axis, n in zip(images.axes_metadata, images.shape)],
    )


def _lazy_exports(device: str) -> dict:
    """A lazy object of each class (and export path) that to_quantem handles."""
    waves = _waves(True, device, reciprocal_space=False)
    centred = waves.diffraction_patterns(max_angle=None, fftshift=True)
    return {
        "real_space_waves": waves,
        "reciprocal_space_waves": _waves(True, device, reciprocal_space=True),
        "images": waves.intensity(),
        "line_profiles": waves.intensity().interpolate_line(
            start=(0, 0), end=(1, 1), gpts=8
        ),
        "diffraction_patterns_centred": centred,
        "diffraction_patterns_unshifted": waves.diffraction_patterns(
            max_angle=None, fftshift=False
        ),
        "polar_measurements": centred.polar_binning(
            nbins_radial=3, nbins_azimuthal=4, inner=1.0, outer=10.0
        ),
        "potential_array": PotentialArray(
            da.zeros((2, *GPTS), chunks=(1, -1, -1), dtype=np.float32),
            sampling=SAMPLING,
            slice_thickness=1.0,
        ).copy_to_device(device),
    }


@devices
@pytest.mark.parametrize("kind", list(_lazy_exports("cpu")))
def test_to_quantem_leaves_a_lazy_object_lazy(kind, device, quantem_records):
    lazy_object = _lazy_exports(device)[kind]
    assert lazy_object.is_lazy

    lazy_object.to_quantem()

    assert lazy_object.is_lazy
    assert isinstance(lazy_object.array, da.Array)
    assert type(quantem_records["array"]) is np.ndarray


@pytest.mark.parametrize(
    "metadata, signal_units",
    [({"units": "eV / e"}, "eV / e"), ({}, "arb. units")],
)
def test_to_quantem_signal_units_are_the_metadata_units(
    metadata, signal_units, quantem_records
):
    images = Images(
        np.zeros(GPTS, dtype=np.float32), sampling=SAMPLING, metadata=metadata
    )

    images.to_quantem()

    assert quantem_records["signal_units"] == signal_units


def test_to_quantem_copies_the_metadata(quantem_records):
    metadata = {"label": "intensity", "units": "arb. unit", "energy": 100e3}
    images = Images(
        np.zeros(GPTS, dtype=np.float32), sampling=SAMPLING, metadata=metadata
    )

    dataset = images.to_quantem()

    assert dataset.metadata == metadata
    dataset.metadata["energy"] = 0.0
    assert images.metadata == metadata


@devices
def test_to_quantem_exports_a_host_array(device, quantem_records):
    # quantem's Dataset holds NumPy arrays only and rejects CuPy arrays.
    images = Images(
        np.arange(np.prod(GPTS), dtype=np.float32).reshape(GPTS), sampling=SAMPLING
    ).copy_to_device(device)

    images.to_quantem()

    assert type(quantem_records["array"]) is np.ndarray
    np.testing.assert_array_equal(quantem_records["array"], _host(images.array))


# The tests below run against the real quantem package rather than the
# quantem_records stand-in: the calibration is read back from the Dataset as
# quantem stores it, and its own slicing is used to check that `origin` is the
# coordinate of index 0.


def _quantem_dataset_class():
    return pytest.importorskip("quantem.core.datastructures").Dataset


def _quantem_coordinates(dataset, axis: int) -> np.ndarray:
    return (
        dataset.origin[axis] + np.arange(dataset.shape[axis]) * (dataset.sampling[axis])
    )


def _abtem_coordinates(array_object) -> list:
    return [
        np.asarray(axis.coordinates(n))
        for axis, n in zip(array_object.axes_metadata, array_object.shape)
    ]


@pytest.mark.quantem
@devices
@pytest.mark.parametrize("kind", list(_lazy_exports("cpu")))
def test_quantem_accepts_every_export(kind, device):
    Dataset = _quantem_dataset_class()
    exported = _lazy_exports(device)[kind]
    eager = exported.copy().compute()

    dataset = exported.to_quantem()

    assert type(dataset) is Dataset
    # Reciprocal-space waves are exported as diffraction patterns.
    if kind == "reciprocal_space_waves":
        assert dataset.name == "DiffractionPatterns"
    else:
        assert dataset.name == type(exported).__name__
    assert dataset.shape == exported.shape
    assert len(dataset.origin) == len(dataset.sampling) == len(dataset.units)
    assert dataset.signal_units == exported.metadata["units"]
    assert dataset.metadata == exported.metadata
    assert dataset.metadata is not exported.metadata
    assert exported.is_lazy
    assert type(dataset.array) is np.ndarray
    if kind in ("reciprocal_space_waves", "diffraction_patterns_unshifted"):
        expected = np.fft.fftshift(_host(eager.array), axes=(-2, -1))
    else:
        expected = _host(eager.array)
    np.testing.assert_array_equal(dataset.array, expected)


@pytest.mark.quantem
@pytest.mark.parametrize(
    "kind",
    [
        "reciprocal_space_waves",
        "diffraction_patterns_centred",
        "diffraction_patterns_unshifted",
    ],
)
@lazy_params
@devices
def test_quantem_calibration_of_frequency_axes(kind, lazy, device):
    _quantem_dataset_class()
    exported = _frequency_exports(kind, lazy, device)

    dataset = exported.to_quantem()

    assert dataset.units == ["pixels", "A^-1", "A^-1"]
    np.testing.assert_array_equal(dataset.origin[0], 0.0)
    np.testing.assert_array_equal(dataset.sampling[0], 1.0)
    for i, axis in enumerate((-2, -1)):
        _assert_frequencies(_quantem_coordinates(dataset, axis), i)
        # quantem's own slicing moves the origin to the coordinate of the first
        # kept index: starting at the centre pixel puts k = 0 at the origin.
        index = [slice(None)] * dataset.ndim
        index[axis] = slice(GPTS[i] // 2, None)
        centre_on = dataset[tuple(index)]
        dk = 1 / (GPTS[i] * SAMPLING[i])
        assert abs(centre_on.origin[axis]) <= 1e-12 * dk


@pytest.mark.quantem
@devices
def test_quantem_calibration_of_real_space_and_scan_axes(device):
    _quantem_dataset_class()
    scan_axes = [
        ScanAxis(label="x", sampling=0.5, offset=1.0, units="Å"),
        ScanAxis(label="y", sampling=0.4, offset=-2.0, units="Å"),
    ]
    images = Images(
        np.zeros((4, 5, *GPTS), dtype=np.float32),
        sampling=SAMPLING,
        ensemble_axes_metadata=scan_axes,
    ).copy_to_device(device)

    dataset = images.to_quantem()

    assert dataset.units == ["A"] * 4
    np.testing.assert_allclose(dataset.origin, (1.0, -2.0, 0.0, 0.0), rtol=0)
    np.testing.assert_allclose(dataset.sampling, (0.5, 0.4, *SAMPLING), rtol=0)
    for axis, coordinates in enumerate(_abtem_coordinates(images)):
        np.testing.assert_allclose(
            _quantem_coordinates(dataset, axis), coordinates, rtol=0, atol=1e-12
        )
    # quantem's crop shifts the origin by the cropped samples, as abTEM's
    # coordinates of the kept scan positions do.
    cropped = dataset.crop(((1, 0), (2, 0)), axes=(0, 1))
    np.testing.assert_allclose(cropped.origin[:2], (1.5, -1.2), rtol=0, atol=1e-12)


@pytest.mark.quantem
@devices
def test_quantem_calibration_of_polar_measurements(device):
    _quantem_dataset_class()
    inner, outer, rotation = 5.0, 25.0, 0.3
    nbins_radial, nbins_azimuthal = 4, 6
    waves = _waves(False, device, reciprocal_space=False)
    diffraction_patterns = waves.diffraction_patterns(max_angle=None, fftshift=True)
    assert min(diffraction_patterns.max_angles) > outer
    polar = diffraction_patterns.polar_binning(
        nbins_radial=nbins_radial,
        nbins_azimuthal=nbins_azimuthal,
        inner=inner,
        outer=outer,
        rotation=rotation,
    )

    dataset = polar.to_quantem()

    assert dataset.units[-2:] == ["mrad", "rad"]
    # The radial origin is the inner angle in mrad, the azimuthal origin the
    # rotation in rad.
    np.testing.assert_allclose(dataset.origin[-2:], (inner, rotation), rtol=1e-12)
    np.testing.assert_allclose(
        dataset.sampling[-2:],
        ((outer - inner) / nbins_radial, 2 * np.pi / nbins_azimuthal),
        rtol=1e-12,
    )
    # The ensemble axis (OrdinalAxis) has no offset, so only the polar axes.
    for axis in (-2, -1):
        coordinates = _abtem_coordinates(polar)[axis]
        np.testing.assert_allclose(
            _quantem_coordinates(dataset, axis), coordinates, rtol=1e-12
        )


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
def test_polar_measurements_to_diffraction_patterns_export_centred(lazy, device):
    # to_diffraction_patterns draws the polar bins on a centred grid, so it must
    # set fftshift=True: unflagged, the exports would shift the centred data a
    # second time and put the innermost bin at the corner.
    pytest.importorskip("xarray")
    nbins_radial, nbins_azimuthal = 5, 4
    array = np.repeat(
        10.0 * np.arange(1, nbins_radial + 1, dtype=np.float32)[:, None],
        nbins_azimuthal,
        axis=1,
    )
    polar = PolarMeasurements(
        da.from_array(array) if lazy else array,
        radial_sampling=5.0,
        azimuthal_sampling=2 * np.pi / nbins_azimuthal,
        metadata={"energy": 100e3, "label": "intensity", "units": "arb. unit"},
    ).copy_to_device(device)

    patterns = polar.to_diffraction_patterns(gpts=64)
    assert patterns.fftshift
    stored = _host(patterns.compute().array)
    assert stored[32, 32] == array[0, 0]

    data_array = patterns.to_data_array()
    # to_data_array keeps the device array, and CuPy refuses the implicit
    # conversion float() would ask for.
    assert float(_host(data_array.sel(kx=0, ky=0).data)) == array[0, 0]
    np.testing.assert_array_equal(_host(data_array.data), stored)
    if hyperspy is not None:
        np.testing.assert_array_equal(_host(patterns.to_hyperspy().data), stored)


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
