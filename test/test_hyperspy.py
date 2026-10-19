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


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_to_quantem_exports_a_host_array(device, quantem_records):
    # quantem's Dataset holds NumPy arrays only and rejects CuPy arrays.
    images = Images(
        np.arange(np.prod(GPTS), dtype=get_dtype()).reshape(GPTS), sampling=SAMPLING
    ).copy_to_device(device)

    images.to_quantem()

    assert type(quantem_records["array"]) is np.ndarray
    np.testing.assert_array_equal(quantem_records["array"], asnumpy(images.array))
