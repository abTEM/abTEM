import numpy as np


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


def interpolate_bilinear_cpu(x, v, u, vw, uw):
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


def compute_indices_and_weights(out_size, in_size, align_corners, xp):
    out_H, out_W = out_size
    H, W = in_size

    if align_corners:
        v = xp.linspace(0, H - 1, num=out_H, dtype=np.float)
        u = xp.linspace(0, W - 1, num=out_W, dtype=np.float)
    else:
        y_scale = H / out_H
        x_scale = W / out_W
        v = (xp.arange(out_H, dtype=np.float) + 0.5) * y_scale - 0.5
        v = xp.maximum(v, 0)
        u = (xp.arange(out_W, dtype=np.float) + 0.5) * x_scale - 0.5
        u = xp.maximum(u, 0)
    vw, v = xp.modf(v)
    uw, u = xp.modf(u)

    return v, u, vw, uw
