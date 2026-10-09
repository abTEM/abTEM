"""Waves given in reciprocal space measure the same as the same waves in real space."""

import dask.array as da
import numpy as np
import pytest
from utils import devices

from abtem.core.axes import OrdinalAxis
from abtem.core.backend import asnumpy
from abtem.detectors import AnnularDetector
from abtem.waves import Waves


def _shared_fft_diffraction_patterns(waves):
    with waves._share_diffraction_pattern_fft():
        return waves.diffraction_patterns(max_angle=None)


def _downsampled_in_real_space(waves):
    downsampled = waves.downsample(gpts=(8, 10))
    assert downsampled.reciprocal_space == waves.reciprocal_space
    return downsampled.ensure_real_space()


MEASUREMENTS = {
    "intensity": lambda waves: waves.intensity(),
    "phase": lambda waves: waves.phase(),
    "to_images": lambda waves: waves.to_images(),
    "diffraction_patterns": lambda waves: waves.diffraction_patterns(),
    "complex_unshifted_diffraction_patterns": lambda waves: waves.diffraction_patterns(
        max_angle=None, return_complex=True, fftshift=False
    ),
    "shared_fft_diffraction_patterns": _shared_fft_diffraction_patterns,
    "unnormalized_complex_diffraction_patterns": lambda waves: waves.diffraction_patterns(
        max_angle=None, return_complex=True, fftshift=False, renormalize=False
    ),
    "annular_detector": lambda waves: AnnularDetector(inner=0, outer=50).detect(waves),
    "downsample": _downsampled_in_real_space,
}


def _real_space_waves(lazy):
    # three waves on a 16 x 20 grid, so the batch and both grid axes differ in size;
    # amplitudes in [0.5, 1.5] and phases away from the branch cut keep the phase
    # comparison well conditioned; single precision, the only one Metal stores
    rng = np.random.default_rng(0)
    shape = (3, 16, 20)
    psi = rng.uniform(0.5, 1.5, shape) * np.exp(1j * rng.uniform(-2.5, 2.5, shape))
    psi = psi.astype(np.complex64)
    return Waves(
        da.from_array(psi, chunks=(1, -1, -1)) if lazy else psi,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
        metadata={"normalization": "values"},
    )


def test_unnormalized_diffraction_patterns_do_not_share_memory_with_the_waves():
    real = _real_space_waves(lazy=False)
    reciprocal = real.ensure_reciprocal_space()

    patterns = MEASUREMENTS["unnormalized_complex_diffraction_patterns"](reciprocal)
    expected = MEASUREMENTS["unnormalized_complex_diffraction_patterns"](real)

    assert not np.shares_memory(patterns.array, reciprocal.array)
    atol = 100 * np.finfo(patterns.array.dtype).eps * np.abs(expected.array).max()
    assert np.allclose(patterns.array, expected.array, rtol=0, atol=atol)


# the lazy path computes its patterns block by block, without the shared FFT
CASES = [
    (measurement, lazy)
    for measurement in MEASUREMENTS
    for lazy in (False, True)
    if not (lazy and measurement == "shared_fft_diffraction_patterns")
]


@devices
@pytest.mark.parametrize("measurement, lazy", CASES)
def test_reciprocal_space_waves_measure_as_real_space_waves(measurement, lazy, device):
    real = _real_space_waves(lazy).copy_to_device(device)
    reciprocal = real.ensure_reciprocal_space()
    assert reciprocal.reciprocal_space

    measure = MEASUREMENTS[measurement]
    expected = asnumpy(measure(real).compute().array)
    result = asnumpy(measure(reciprocal).compute().array)

    assert result.shape == expected.shape
    atol = 100 * np.finfo(result.dtype).eps * np.abs(expected).max()
    assert np.allclose(result, expected, rtol=0, atol=atol)
