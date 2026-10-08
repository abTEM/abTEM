"""build(first_slice, last_slice) of every potential builder, lazy and eager."""

import numpy as np
import pytest
from ase import Atoms
from utils import gpu, to_host_array

import abtem
from abtem.potentials.charge_density import ChargeDensityPotential

GPTS = (32, 28)
ATOMS = Atoms(
    "CO",
    positions=[(0.6, 0.8, 1.0), (1.9, 1.5, 2.4)],
    cell=(3.2, 2.8, 3.6),
    pbc=True,
)


@pytest.fixture(autouse=True)
def _config():
    with abtem.config.set({"precision": "float64", "fft": "numpy"}):
        yield


def _potentials(exit_planes=None):
    unit = abtem.Potential(
        ATOMS, gpts=GPTS, slice_thickness=0.4, exit_planes=exit_planes
    )
    return {
        "Potential": unit,
        "FrozenPhonons": abtem.Potential(
            abtem.FrozenPhonons(ATOMS, 2, 0.05, seed=(1, 2)),
            gpts=GPTS,
            slice_thickness=0.4,
            exit_planes=exit_planes,
        ),
        "CrystalPotential": abtem.CrystalPotential(
            unit, repetitions=(1, 1, 2), exit_planes=exit_planes
        ),
        "ChargeDensityPotential": ChargeDensityPotential(
            ATOMS,
            charge_density=np.zeros((16, 14, 18)),
            gpts=GPTS,
            slice_thickness=0.4,
            exit_planes=exit_planes,
        ),
    }


RANGES = [(0, 2), (0, 4), (2, 6), (5, None), (0, None)]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("name", list(_potentials()))
@pytest.mark.parametrize("first_slice, last_slice", RANGES)
@pytest.mark.parametrize("lazy", [False, True])
def test_slice_range_values(name, first_slice, last_slice, lazy, device):
    if device == "mps":
        pytest.skip("Metal is single precision; this test runs in float64")
    with abtem.config.set({"device": device}):
        potential = _potentials()[name]
        expected = to_host_array(potential.build(lazy=False))
        built = potential.build(first_slice, last_slice, lazy=lazy)
        array = to_host_array(built.compute() if lazy else built)
    expected = expected[
        (slice(None),) * len(potential.ensemble_shape)
        + (slice(first_slice, last_slice),)
    ]
    assert array.shape == expected.shape
    if device == "cpu":
        np.testing.assert_array_equal(array, expected)
    else:
        # CuPy's FFT is not bitwise reproducible across batch sizes, and a range
        # build runs other batches than the whole build
        np.testing.assert_allclose(
            array, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
        )
    assert built.slice_thickness == potential.slice_thickness[first_slice:last_slice]


@pytest.mark.parametrize(
    "name", ["Potential", "CrystalPotential", "ChargeDensityPotential"]
)
def test_full_build_matches_slices_generated_one_by_one(name):
    potential = _potentials()[name]
    by_hand = np.concatenate([s.array for s in potential.generate_slices()])
    np.testing.assert_array_equal(potential.build(lazy=False).array, by_hand)


@pytest.mark.parametrize("exit_planes", [None, 3, (2,)])
@pytest.mark.parametrize("first_slice, last_slice", RANGES)
@pytest.mark.parametrize("lazy", [False, True])
def test_slice_range_exit_planes(exit_planes, first_slice, last_slice, lazy):
    potential = _potentials(exit_planes)["Potential"]
    built = potential.build(first_slice, last_slice, lazy=lazy)
    expected = potential.build(lazy=False)[first_slice:last_slice].exit_planes
    if first_slice == 0 and -1 in potential.exit_planes:
        expected = (-1,) + expected
    assert built.exit_planes == expected
    thickness = np.cumsum(potential.slice_thickness[first_slice:last_slice])
    np.testing.assert_allclose(
        built.exit_thicknesses,
        [0.0 if p == -1 else thickness[p] for p in expected],
    )
    if (first_slice, last_slice) == (0, None):
        assert built.exit_planes == potential.exit_planes


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("exit_planes", [None, 3, (2,)])
def test_multislice_of_a_lazy_slice_range(exit_planes, device):
    if device == "mps":
        pytest.skip("Metal is single precision; this test runs in float64")
    with abtem.config.set({"device": device}):
        potential = _potentials(exit_planes)["Potential"]
        wave = abtem.PlaneWave(energy=100e3)
        lazy_range = potential.build(2, 6, lazy=True)
        result = to_host_array(wave.multislice(lazy_range).compute())
        expected = to_host_array(
            wave.multislice(potential.build(lazy=False)[2:6]).compute()
        )
    np.testing.assert_allclose(
        result, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
