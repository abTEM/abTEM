import sys
import warnings
from types import SimpleNamespace

import dask.array as da
import hypothesis.strategies as st
import numpy as np
import pytest
import strategies as abtem_st
from hypothesis import given
from utils import gpu

import abtem.array
from abtem.core.axes import OrdinalAxis
from abtem.core.backend import asnumpy
from abtem.core.utils import get_dtype
from abtem.measurements import DiffractionPatterns, Images
from abtem.potentials.iam import PotentialArray
from abtem.waves import Waves

try:
    import hyperspy
except ImportError:
    hyperspy = None


@given(data=st.data())
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
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


GPTS = (16, 21)
SAMPLING = (0.1, 0.23)


@pytest.fixture
def quantem_records(monkeypatch):
    """Replace quantem by a stand-in whose Dataset.from_array records its kwargs,
    and the dataset's metadata under "metadata"."""
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


def _lazy_exports() -> dict:
    """A lazy object of each class that to_quantem handles."""
    rng = np.random.default_rng(0)
    real = rng.standard_normal((3, *GPTS)).astype(get_dtype())
    complex_data = (real + 1j * real[::-1]).astype(get_dtype(complex=True))
    members = [OrdinalAxis(values=(1, 2, 3))]
    return {
        "waves": Waves(
            da.from_array(complex_data, chunks=(1, -1, -1)),
            energy=100e3,
            sampling=SAMPLING,
            ensemble_axes_metadata=members,
        ),
        "images": Images(
            da.from_array(real, chunks=(1, -1, -1)),
            sampling=SAMPLING,
            ensemble_axes_metadata=members,
        ),
        "diffraction_patterns": DiffractionPatterns(
            da.from_array(np.abs(real), chunks=(1, -1, -1)),
            sampling=SAMPLING,
            ensemble_axes_metadata=members,
            metadata={"energy": 100e3},
        ),
        "potential_array": PotentialArray(
            da.from_array(real, chunks=(1, -1, -1)),
            sampling=SAMPLING,
            slice_thickness=1.0,
        ),
    }


@pytest.mark.parametrize("kind", list(_lazy_exports()))
def test_to_quantem_leaves_a_lazy_object_lazy(kind, quantem_records):
    lazy_object = _lazy_exports()[kind]
    assert lazy_object.is_lazy

    lazy_object.to_quantem()

    assert lazy_object.is_lazy
    assert isinstance(lazy_object.array, da.Array)
    assert type(quantem_records["array"]) is np.ndarray


@pytest.mark.parametrize(
    "metadata, signal_units",
    [
        ({"units": "eV / e"}, "eV / e"),
        ({"units": "%"}, "%"),
        ({"units": "electrons"}, "electrons"),
        ({"units": "Å"}, "Å"),
        # An empty unit is a dimensionless signal, not a missing one.
        ({"units": ""}, ""),
        ({"units": None}, "arb. unit"),
        ({}, "arb. unit"),
    ],
)
def test_to_quantem_signal_units_are_the_metadata_units(
    metadata, signal_units, quantem_records
):
    images = Images(
        np.zeros(GPTS, dtype=get_dtype()), sampling=SAMPLING, metadata=metadata
    )

    images.to_quantem()

    assert quantem_records["signal_units"] == signal_units


def test_to_quantem_does_not_share_nested_metadata(quantem_records):
    images = Images(
        np.zeros(GPTS, dtype=get_dtype()),
        sampling=SAMPLING,
        metadata={"label": "intensity", "units": "arb. unit", "nested": {"a": [1, 2]}},
    )

    dataset = images.to_quantem()

    assert dataset.metadata["nested"] == {"a": [1, 2]}
    dataset.metadata["nested"]["a"].append(3)
    dataset.metadata["nested"]["b"] = 0
    assert images.metadata["nested"] == {"a": [1, 2]}


def _images_with_two_ensemble_axes(lazy, device):
    # Every axis has its own size (3 and 4 members, 16 x 21 pixels), so an axis
    # list in an order other than the data's gives a wrong size or coordinates.
    shape = (3, 4, 16, 21)
    array = np.arange(np.prod(shape), dtype=get_dtype()).reshape(shape)
    if lazy:
        array = da.from_array(array, chunks=(1, 2, -1, -1))
    return Images(
        array,
        sampling=SAMPLING,
        ensemble_axes_metadata=[
            OrdinalAxis(label="a", values=(0.0, 1.0, 2.0)),
            OrdinalAxis(label="b", values=(0.0, 1.0, 2.0, 3.0)),
        ],
    ).copy_to_device(device)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("device", ["cpu", gpu])
def test_to_hyperspy_axes_describe_the_axes_of_the_data(transpose, lazy, device):
    pytest.importorskip("hyperspy")
    images = _images_with_two_ensemble_axes(lazy, device)

    signal = images.to_hyperspy(transpose=transpose)

    # transpose=True reverses the base axes only
    order = (0, 1, 3, 2) if transpose else (0, 1, 2, 3)
    # the signal holds the device array of the data, lazily or not
    data = signal.data.compute() if hasattr(signal.data, "compute") else signal.data
    data = asnumpy(data)
    np.testing.assert_array_equal(
        data, np.transpose(asnumpy(images.compute().array), order)
    )
    manager = signal.axes_manager
    axes = sorted(
        manager.navigation_axes + manager.signal_axes, key=lambda a: a.index_in_array
    )
    assert [axis.name for axis in axes] == [
        images.axes_metadata[i].label for i in order
    ]
    assert [axis.size for axis in axes] == list(data.shape)
    for axis, i in zip(axes, order):
        expected = images.axes_metadata[i].coordinates(images.shape[i])
        np.testing.assert_allclose(axis.axis, expected, rtol=1e-12)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("lazy", [True, False])
def test_to_hyperspy_passes_the_axes_in_the_order_of_the_data(
    transpose, lazy, monkeypatch
):
    """HyperSpy takes its axes in the order of the array's dimensions; a
    stand-in for HyperSpy records what to_hyperspy passes it."""
    captured = {}

    class Signal2D:
        def __init__(self, data, axes):
            captured["shape"] = data.shape
            captured["axes"] = axes

        def as_lazy(self):
            return self

    monkeypatch.setattr(
        abtem.array, "hs", SimpleNamespace(signals=SimpleNamespace(Signal2D=Signal2D))
    )
    images = _images_with_two_ensemble_axes(lazy, "cpu")

    images.to_hyperspy(transpose=transpose)

    order = (0, 1, 3, 2) if transpose else (0, 1, 2, 3)
    axes = captured["axes"]
    assert [axis["name"] for axis in axes] == [
        images.axes_metadata[i].label for i in order
    ]
    assert [axis["size"] for axis in axes] == list(captured["shape"])
    assert captured["shape"] == tuple(images.shape[i] for i in order)
    for axis, i in zip(axes[2:], order[2:]):
        assert axis["scale"] == images.axes_metadata[i].sampling


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_to_quantem_exports_a_host_array(device, quantem_records):
    # quantem's Dataset holds NumPy arrays only and rejects CuPy arrays.
    images = Images(
        np.arange(np.prod(GPTS), dtype=get_dtype()).reshape(GPTS), sampling=SAMPLING
    ).copy_to_device(device)

    images.to_quantem()

    assert type(quantem_records["array"]) is np.ndarray
    np.testing.assert_array_equal(quantem_records["array"], asnumpy(images.array))


def test_to_quantem_origin_is_the_offset_of_scan_axes(quantem_records):
    # the origin of every axis was written as 0
    from abtem.core.axes import ScanAxis

    scan_axes = [
        ScanAxis(label="x", sampling=0.5, offset=1.0, units="Å"),
        ScanAxis(label="y", sampling=0.4, offset=-2.0, units="Å"),
    ]
    images = Images(
        np.zeros((4, 5, *GPTS), dtype=np.float32),
        sampling=SAMPLING,
        ensemble_axes_metadata=scan_axes,
    )

    images.to_quantem()

    assert quantem_records["origin"][:2] == (1.0, -2.0)
    assert quantem_records["origin"][2:] == (0.0, 0.0)


def test_to_quantem_origin_puts_zero_frequency_at_k_zero(quantem_records):
    patterns = DiffractionPatterns(
        np.ones((*GPTS,), dtype=np.float32),
        sampling=SAMPLING,
        fftshift=True,
        metadata={"energy": 100e3},
    )

    patterns.to_quantem()

    for o, d, n in zip(quantem_records["origin"], quantem_records["sampling"], GPTS):
        assert abs(o + (n // 2) * d) <= 1e-12 * d
