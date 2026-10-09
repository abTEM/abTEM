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
    "annular_detector": lambda waves: AnnularDetector(inner=0, outer=50).detect(waves),
    "downsample": _downsampled_in_real_space,
}


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("measurement", list(MEASUREMENTS))
def test_reciprocal_space_waves_measure_as_real_space_waves(measurement, lazy, device):
    # three waves on a 16 x 20 grid, so the batch and both grid axes differ in size;
    # amplitudes in [0.5, 1.5] and phases away from the branch cut keep the phase
    # comparison well conditioned
    rng = np.random.default_rng(0)
    shape = (3, 16, 20)
    psi = rng.uniform(0.5, 1.5, shape) * np.exp(1j * rng.uniform(-2.5, 2.5, shape))
    array = da.from_array(psi, chunks=(1, -1, -1)) if lazy else psi
    real = Waves(
        array,
        energy=100e3,
        sampling=0.1,
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
        metadata={"normalization": "values"},
    ).copy_to_device(device)
    reciprocal = real.ensure_reciprocal_space()
    assert reciprocal.reciprocal_space

    measure = MEASUREMENTS[measurement]
    expected = asnumpy(measure(real).compute().array)
    result = asnumpy(measure(reciprocal).compute().array)

    assert result.shape == expected.shape
    atol = 100 * np.finfo(result.dtype).eps * np.abs(expected).max()
    assert np.allclose(result, expected, rtol=0, atol=atol)
