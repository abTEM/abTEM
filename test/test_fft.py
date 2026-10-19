import numpy as np

import abtem
from abtem.core.fft import fft_crop, fft_interpolate
from abtem.waves import Waves


class TestFftCropInterpolateEmptyNewShape:
    """`fft_crop`/`fft_interpolate` sliced off the batch dimensions with
    `array.shape[: -len(new_shape)]` / `array.shape[-len(new_shape) :]`, the
    same -0 bug as abtem/array.py's base-less ArrayObject sites: for
    `new_shape == ()` (every dimension is a batch dimension, none are being
    resized), `-len(new_shape)` is `-0`, which collapses to the wrong end.

    Reachable from public API: `WavesDetector(gpts=())` builds `new_shape =
    waves.shape[:-2] + gpts`, which is `()` for a plain 2D `Waves` with no
    ensemble axes, since `gpts` itself contributes nothing. Before the fix
    this crashed inside `fft_crop` with an opaque
    `TypeError: only length-1 arrays can be converted to Python scalars`,
    three frames below the `gpts=()` that caused it.
    """

    @staticmethod
    def _array():
        rng = np.random.default_rng(0)
        return (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))).astype(
            complex
        )

    def test_fft_crop_with_empty_new_shape_is_a_no_op(self):
        array = self._array()
        assert np.array_equal(fft_crop(array, ()), array)

    def test_fft_interpolate_with_empty_new_shape_is_a_no_op(self):
        array = self._array()
        out = fft_interpolate(array, ())
        assert out.shape == array.shape
        assert np.allclose(out, array)

    def test_wavesdetector_empty_gpts_matches_none(self):
        """The end-to-end case: `gpts=()` is falsy but `is not None`, so it
        reaches `_calculate_new_array`'s `if self._gpts is not None:` guard
        and used to crash there. It now degenerates to the same no-resample
        behaviour as the documented `gpts=None` default, rather than either
        crashing or silently returning something else."""
        with abtem.config.set({"fft": "numpy"}):
            array = self._array().astype("complex64")
            waves = Waves(array, energy=100e3, extent=(10, 10))

            none_out = abtem.WavesDetector(gpts=None).detect(waves)
            empty_out = abtem.WavesDetector(gpts=()).detect(waves)

        assert np.array_equal(np.asarray(none_out.array), np.asarray(empty_out.array))
        assert np.array_equal(np.asarray(empty_out.array), array)
