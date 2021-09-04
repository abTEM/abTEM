import numpy as np
from scipy.interpolate import RegularGridInterpolator


# def interpolate_cube(array, old_cell, new_cell, new_gpts, origin=None):
#     if origin is None:
#         origin = (0., 0., 0.)
#
#     padded_array = np.zeros((array.shape[0] + 1, array.shape[1] + 1, array.shape[2] + 1))
#     padded_array[:-1, :-1, :-1] = array
#     padded_array[-1] = padded_array[0]
#     padded_array[:, -1] = padded_array[:, 0]
#     padded_array[:, :, -1] = padded_array[:, :, 0]
#
#     x = np.linspace(0, 1, padded_array.shape[0], endpoint=True)
#     y = np.linspace(0, 1, padded_array.shape[1], endpoint=True)
#     z = np.linspace(0, 1, padded_array.shape[2], endpoint=True)
#
#     interpolator = RegularGridInterpolator((x, y, z), padded_array)
#
#     x = np.linspace(origin[0], origin[0] + new_cell[0], new_gpts[0], endpoint=False)
#     y = np.linspace(origin[1], origin[1] + new_cell[1], new_gpts[1], endpoint=False)
#     z = np.linspace(origin[2], origin[2] + new_cell[2], new_gpts[2], endpoint=False)
#
#     x, y, z = np.meshgrid(x, y, z, indexing='xy')
#
#     points = np.array([x.ravel(), y.ravel(), z.ravel()]).T
#
#     P_inv = np.linalg.inv(np.array(old_cell))
#
#     scaled_points = np.dot(points, P_inv) % 1.0
#     interpolated = interpolator(scaled_points)
#
#     return interpolated.reshape(new_gpts)


def _infer_lines(B, H, W, out_H, out_W, kH, kW):
    target_size = 2 ** 17
    line_size = B * (H * W // out_H + kH * kW * out_W)
    target_lines = target_size // line_size

    if target_lines < out_H:
        lines = 1
        while True:
            next_lines = lines * 2
            if next_lines > target_lines:
                break
            lines = next_lines
    else:
        lines = out_H

    return lines


def interpolate_bilinear(x, v, u, vw, uw):
    B, H, W = x.shape
    out_H, out_W = v.shape

    # Interpolation is done by each output panel (i.e. multi lines)
    # in order to better utilize CPU cache memory.
    lines = _infer_lines(B, H, W, out_H, out_W, 2, 2)

    vcol = np.empty((2, lines, out_W), dtype=v.dtype)
    ucol = np.empty((2, lines, out_W), dtype=u.dtype)
    wcol = np.empty((2, 2, lines, out_W), dtype=x.dtype)

    y = np.empty((B, out_H * out_W), dtype=x.dtype)

    for i in range(0, out_H, lines):
        l = min(lines, out_H - i)
        vcol = vcol[:, :l]
        ucol = ucol[:, :l]
        wcol = wcol[:, :, :l]
        i_end = i + l

        # indices
        vcol[0] = v[i:i_end]
        ucol[0] = u[i:i_end]
        np.add(vcol[0], 1, out=vcol[1])
        np.add(ucol[0], 1, out=ucol[1])
        np.minimum(vcol[1], H - 1, out=vcol[1])
        np.minimum(ucol[1], W - 1, out=ucol[1])

        wcol[0, 1] = uw[i:i_end]
        np.subtract(1, wcol[0, 1], out=wcol[0, 0])
        np.multiply(wcol[0], vw[i:i_end], out=wcol[1])
        wcol[0] -= wcol[1]

        # packing to the panel whose shape is (B, C, 2, 2, l, out_W)
        panel = x[:, vcol[:, None], ucol[None, :]]

        # interpolation
        panel = panel.reshape((B, 4, l * out_W))
        weights = wcol.reshape((4, l * out_W))
        iout = i * out_W
        iout_end = i_end * out_W
        np.einsum('ijk,jk->ik', panel, weights, out=y[:, iout:iout_end])
        del panel, weights

    return y.reshape((B, out_H, out_W))
