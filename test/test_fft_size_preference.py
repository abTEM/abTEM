"""The automatic grid sizes favour powers of two where the FFT does (Metal)."""

import numpy as np
import pytest
from ase.build import bulk

import abtem
from abtem.core import config
from abtem.core.fft import (
    next_fast_fft_size,
    next_preferred_fft_multiplier,
    next_preferred_fft_size,
    prefers_powers_of_two,
)
from abtem.slicing import commensurate_gpts


@pytest.mark.parametrize("n", range(1, 1200, 7))
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_other_devices_keep_the_next_fast_length(n, device):
    assert next_preferred_fft_size(n, device) == next_fast_fft_size(n)


@pytest.mark.parametrize("device", ["mps", "metal", "torch"])
@pytest.mark.parametrize(
    "n, expected",
    [(100, 128), (200, 256), (330, 512), (360, 512), (500, 512), (1000, 1024)],
)
def test_metal_rounds_up_to_a_power_of_two_when_the_fast_length_is_close(
    device, n, expected
):
    assert next_preferred_fft_size(n, device) == expected


@pytest.mark.parametrize(
    "n, expected", [(129, 135), (257, 270), (300, 300), (513, 525)]
)
def test_metal_keeps_a_fast_length_when_the_power_of_two_is_far(n, expected):
    assert next_preferred_fft_size(n, "mps") == expected


@pytest.mark.parametrize("n", range(1, 1500, 11))
def test_metal_never_rounds_below_the_requested_length(n):
    assert next_preferred_fft_size(n, "mps") >= n
    assert next_preferred_fft_size(n, "mps") >= next_fast_fft_size(n)


def test_the_configured_device_is_the_default():
    with config.set({"device": "mps"}):
        assert prefers_powers_of_two()
        assert next_preferred_fft_size(330) == 512
    assert not prefers_powers_of_two()
    assert next_preferred_fft_size(330) == 336


def test_emulated_float64_prefers_powers_of_two_too():
    with config.set({"torch.emulate-float64": True}):
        assert prefers_powers_of_two("cpu")
        assert next_preferred_fft_size(330, "cpu") == 512


def test_multiplier_matches_the_fast_one_unless_the_period_is_a_power_of_two():
    for start in range(1, 90):
        # A period that is not a power of two never makes the grid one.
        assert next_preferred_fft_multiplier(start, 6, "mps") == next_fast_fft_size(
            start
        )
        assert next_preferred_fft_multiplier(start, 8, "cpu") == next_fast_fft_size(
            start
        )


def test_multiplier_reaches_a_power_of_two_for_a_power_of_two_period():
    # 8 * 60 = 480 is fast, but 8 * 64 = 512 is much cheaper on Metal.
    assert next_preferred_fft_multiplier(60, 8, "cpu") == 60
    assert next_preferred_fft_multiplier(60, 8, "mps") == 64
    assert next_preferred_fft_multiplier(33, 8, "mps") * 8 == 280


def test_commensurate_gpts_prefers_a_power_of_two_on_metal():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True)
    extent = (atoms.cell[0, 0], atoms.cell[1, 1])

    cpu = commensurate_gpts(extent, atoms.positions, device="cpu")
    mps = commensurate_gpts(extent, atoms.positions, device="mps")

    assert all(n % 4 == 0 for n in cpu)  # commensurate with the atom planes
    assert mps == (128, 128)
    assert all(m >= c for m, c in zip(mps, cpu))


def test_potential_derives_a_power_of_two_grid_on_metal():
    atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, 1)

    cpu = abtem.Potential(atoms, sampling="auto", device="cpu")
    mps = abtem.Potential(atoms, sampling="auto", device="mps")

    assert np.all(np.array(cpu.gpts) == 216)
    assert mps.gpts == (256, 256)
