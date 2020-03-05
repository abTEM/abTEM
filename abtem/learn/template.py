import cupy as cp
import numpy as np
from scipy.signal import fftconvolve


def _window_sum_2d(image, window_shape):
    xp = cp.get_array_module()

    window_sum = xp.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1] - window_sum[:-window_shape[0] - 1])

    window_sum = xp.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1] - window_sum[:, :-window_shape[1] - 1])

    return window_sum


def _window_sum_3d(image, window_shape):
    xp = cp.get_array_module()

    window_sum = _window_sum_2d(image, window_shape)

    window_sum = xp.cumsum(window_sum, axis=2)
    window_sum = (window_sum[:, :, window_shape[2]:-1] - window_sum[:, :, :-window_shape[2] - 1])

    return window_sum


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.
    Returns the result sliced to the correct size for the given mode.
    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.
    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.
    """
    if mode == "full":
        return ret.copy()
    elif mode == "same":
        return _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
                       for a in range(ret.ndim)]
        return _centered(ret, shape_valid).copy()
    else:
        raise ValueError("acceptable mode flags are 'valid',"
                         " 'same', or 'full'")


def match_template(image, template, pad_input=False, mode='constant', constant_values=0):
    if image.ndim < template.ndim:
        raise ValueError("Dimensionality of template must be less than or "
                         "equal to the dimensionality of image.")
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape

    xp = cp.get_array_module(image)

    #image = xp.array(image, dtype=xp.float32)

    pad_width = tuple((width, width) for width in template.shape)
    if mode == 'constant':
        image = xp.pad(image, pad_width=pad_width, mode=mode,
                       constant_values=constant_values)
    else:
        image = xp.pad(image, pad_width=pad_width, mode=mode)

    # Use special case for 2-D images for much better performance in
    # computation of integral images

    image_window_sum = _window_sum_2d(image, template.shape)
    image_window_sum2 = _window_sum_2d(image ** 2, template.shape)
    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = xp.sum((template - template_mean) ** 2)

    from cupyx.scipy.fft import fft2
    from cupyx.scipy.fft import ifft2
    from scipy.fft import rfft2
    from scipy.fft import irfft2
    from scipy.signal import fftconvolve

    fft_template = rfft2(template[::-1, ::-1], image.shape)
    fft_image = rfft2(image, image.shape)
    xcorr = irfft2(fft_template * fft_image)
    xcorr = _apply_conv_mode(xcorr, image.shape, template.shape, mode='valid', axes=(0, 1))
    xcorr = xcorr[1:-1, 1:-1]

    #fft_template = rfft2(template[::-1, ::-1], image.shape)
    #fft_image = rfft2(image, image.shape)
    #xcorr = irfft2(fft_template * fft_image)
    #xcorr = _apply_conv_mode(xcorr, image.shape, template.shape, mode='valid', axes=(0, 1))
    #xcorr = fftconvolve(image, template, mode='valid')[1:-1, 1:-1]

    numerator = xcorr - image_window_sum * template_mean

    denominator = image_window_sum2
    xp.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    xp.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    denominator *= template_ssd
    xp.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    xp.sqrt(denominator, out=denominator)

    response = xp.zeros_like(xcorr, dtype=xp.float32)

    # avoid zero-division
    mask = denominator > xp.finfo(xp.float64).eps

    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        if pad_input:
            d0 = (template.shape[i] - 1) // 2
            d1 = d0 + image_shape[i]
        else:
            d0 = template.shape[i] - 1
            d1 = d0 + image_shape[i] - template.shape[i] + 1
        slices.append(slice(d0, d1))

    return response[tuple(slices)]
