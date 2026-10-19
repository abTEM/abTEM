import numpy as np
import pytest

import abtem
from abtem.core import config
from abtem.core import fft as abtem_fft
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


pyfftw = abtem_fft.pyfftw
requires_pyfftw = pytest.mark.skipif(pyfftw is None, reason="needs pyfftw")


def _random_convolution_inputs(rng, batch=3, gpts=32, dtype=np.complex64):
    real = np.dtype(dtype).type(0).real.dtype
    array = (
        rng.random((batch, gpts, gpts), dtype=real)
        + 1j * rng.random((batch, gpts, gpts), dtype=real)
    ).astype(dtype)
    kernel = (
        rng.random((gpts, gpts), dtype=real) + 1j * rng.random((gpts, gpts), dtype=real)
    ).astype(dtype)
    return array, kernel


def _copy_at_byte_offset(array, offset):
    """A C-contiguous copy of ``array`` starting ``offset`` bytes past a 64-byte
    boundary. At ``offset=16`` it is aligned as malloc guarantees, but not for
    FFTW's SIMD codelets wherever ``pyfftw.simd_alignment`` is 32 (x86 AVX). On
    Apple Silicon ``simd_alignment`` is 4 and every buffer counts as aligned, so
    there the tests using this pass trivially; they bite on x86 (CI)."""
    raw = pyfftw.empty_aligned(array.nbytes + 64, dtype=np.uint8, n=64)
    copy = raw[offset : offset + array.nbytes].view(array.dtype).reshape(array.shape)
    copy[...] = array
    return copy


def _assert_misaligned_for_fftw(array):
    if pyfftw.simd_alignment > 16:
        assert not pyfftw.is_byte_aligned(array)


@requires_pyfftw
@pytest.mark.parametrize("overwrite_x", [True, False])
@pytest.mark.parametrize("func", ["fft2", "ifft2"])
def test_fftw_result_is_independent_of_buffer_alignment(func, overwrite_x):
    """The numbers must not depend on where the input buffer happens to land.

    Regression test: ``get_fftw_object`` planned for the buffer it was given, and
    pyfftw makes an ``FFTW_UNALIGNED`` plan, which rounds differently (~1e-7
    relative), for a buffer that is only 16-byte aligned. Two builds of the same
    frozen-phonon configuration then differed in the last bit depending on where
    malloc had placed a buffer, failing CI intermittently on x86.
    """
    rng = np.random.default_rng(7)
    array, _ = _random_convolution_inputs(rng, gpts=64)
    aligned = _copy_at_byte_offset(array, 0)
    offset = _copy_at_byte_offset(array, 16)
    _assert_misaligned_for_fftw(offset)

    transform = getattr(abtem_fft, func)
    with config.set({"fft": "fftw"}):
        expected = transform(aligned, overwrite_x=overwrite_x)
        result = transform(offset, overwrite_x=overwrite_x)

    assert np.array_equal(
        result, expected
    ), f"FFTW {func} result depends on input buffer alignment"
    reference = getattr(np.fft, func)(array.astype(np.complex128))
    assert np.allclose(result, reference, atol=1e-4)
    if not overwrite_x:
        assert np.array_equal(offset, array)


@requires_pyfftw
@pytest.mark.parametrize("name", ["fft2", "ifft2"])
def test_fftw_plans_for_an_unaligned_buffer_as_for_an_aligned_one(name):
    """An unaligned buffer is aligned by copying, never given its own plan."""
    rng = np.random.default_rng(8)
    array, _ = _random_convolution_inputs(rng, gpts=48)
    offset = _copy_at_byte_offset(array, 16)
    _assert_misaligned_for_fftw(offset)

    aligned_plan = abtem_fft.get_fftw_object(_copy_at_byte_offset(array, 0), name)
    offset_plan = abtem_fft.get_fftw_object(offset, name)

    assert offset_plan.flags == aligned_plan.flags
    assert offset_plan.simd_aligned == aligned_plan.simd_aligned


@requires_pyfftw
@pytest.mark.parametrize("overwrite_x", [True, False])
@pytest.mark.parametrize("gpts", [4, 6, 8, 12])
def test_cached_fftw_convolution_plans_for_an_aligned_buffer(gpts, overwrite_x):
    """The scratch array a plan is made with must be aligned too.

    Regression test: it came from ``np.zeros_like``, so whenever malloc placed it
    16- rather than 32-byte aligned (likely for the small arrays here) the
    plans were ``FFTW_UNALIGNED``, and every convolution in that thread rounded
    differently from a thread whose scratch array had landed aligned.
    """
    rng = np.random.default_rng(9)
    array, kernel = _random_convolution_inputs(rng, batch=1, gpts=gpts)
    reference = pyfftw.FFTW(
        *(2 * (_copy_at_byte_offset(array, 0),)),
        axes=(-2, -1),
        flags=(config.get("fftw.planning_effort"),),
    )

    convolution = abtem_fft.CachedFFTWConvolution()
    result = convolution(_copy_at_byte_offset(array, 16), kernel, overwrite_x)

    for plan in convolution._fftw_objects.values():
        assert plan.simd_aligned == reference.simd_aligned
    expected = np.fft.ifft2(np.fft.fft2(array.astype(np.complex128)) * kernel)
    assert np.allclose(result, expected, atol=1e-4)


@requires_pyfftw
@pytest.mark.parametrize("func", ["fft2", "ifft2"])
def test_fftw_transforms_a_non_contiguous_array_in_place(func):
    # A strided view is copied to the aligned C-contiguous layout the plans are
    # made for, so the caller's array is left alone and the result is returned.
    rng = np.random.default_rng(10)
    array, _ = _random_convolution_inputs(rng, gpts=64)
    view = array[:, ::2, :].transpose(0, 2, 1)
    before = view.copy()

    with config.set({"fft": "fftw"}):
        result = getattr(abtem_fft, func)(view, overwrite_x=True)

    reference = getattr(np.fft, func)(before.astype(np.complex128))
    assert np.allclose(result, reference, atol=1e-4)
    assert np.array_equal(view, before)


@requires_pyfftw
def test_fftw_fftn_over_three_axes():
    """Wisdom must be made for the axes being transformed.

    Regression test: it was always made for the last two axes, so an ``fftn``
    over three never found its plan and fell back to ``pyfftw.builders``, which
    was not passed the axes either and transformed the batch axis too.
    """
    rng = np.random.default_rng(11)
    array = (rng.random((2, 6, 8, 10)) + 1j * rng.random((2, 6, 8, 10))).astype(
        np.complex64
    )
    axes = (-3, -2, -1)

    with config.set({"fft": "fftw"}):
        result = abtem_fft.fftn(array, axes=axes)
        plan = abtem_fft.get_fftw_object(array, "fftn", axes=axes)

    assert isinstance(plan, pyfftw.FFTW)
    assert np.allclose(result, np.fft.fftn(array, axes=axes), atol=1e-4)


@requires_pyfftw
@pytest.mark.parametrize("allow_fallback", [True, False])
def test_fftw_without_wisdom_falls_back_or_raises(monkeypatch, allow_fallback):
    # With no wisdom for a plan and none allowed to be made, `get_fftw_object`
    # builds a plan with `pyfftw.builders` if fallback is allowed, and else
    # raises. The fallback must transform the requested axes.
    pyfftw.forget_wisdom()
    monkeypatch.setattr(abtem_fft, "_new_fftw_object", lambda *a, **k: None)

    rng = np.random.default_rng(12)
    array = (rng.random((2, 6, 8, 10)) + 1j * rng.random((2, 6, 8, 10))).astype(
        np.complex64
    )
    offset = _copy_at_byte_offset(array, 16)
    axes = (-3, -2, -1)

    with config.set({"fftw.allow_fallback": allow_fallback}):
        if not allow_fallback:
            with pytest.raises(RuntimeError, match="No FFTW wisdom"):
                abtem_fft.get_fftw_object(offset, "fftn", axes=axes)
            return
        result = abtem_fft.get_fftw_object(offset, "fftn", axes=axes)()

    assert np.allclose(result, np.fft.fftn(array, axes=axes), atol=1e-4)
