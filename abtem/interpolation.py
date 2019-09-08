from numba import jit, prange
import numpy as np


@jit(nopython=True)
def check_square_overlap(corner_a, corner_b, size):
    return not ((corner_a[0] + size < corner_b[0]) | (corner_b[0] + size < corner_a[0]) |
                (corner_a[1] + size < corner_b[1]) | (corner_b[1] + size < corner_a[1]))


@jit(nopython=True)
def thread_safe_coloring(corners, size):
    colors = np.empty(len(corners), dtype=np.int32)
    colors[0] = 0
    palette_size = 1

    color_count = np.zeros(len(corners), dtype=np.int32)
    color_count[0] = 1

    not_adjacent = np.zeros(len(corners), dtype=np.bool_)

    for i in prange(1, len(corners)):
        not_adjacent[:] = True

        for j in range(0, i):
            overlap = check_square_overlap(corners[i], corners[j], size)
            if overlap:
                color = colors[j]
                not_adjacent[color] = False

        first_not_adjacent = np.searchsorted(not_adjacent, True)

        if first_not_adjacent >= palette_size:
            colors[i] = palette_size
            color_count[palette_size] += 1
            palette_size += 1

        else:
            available_choices = np.where(not_adjacent[:palette_size])[0]
            color_choice = available_choices[np.argmin(color_count[available_choices])]
            colors[i] = color_choice
            color_count[color_choice] += 1

    return colors


@jit(nopython=True, nogil=True, parallel=True)
def interpolation_kernel(v, r, vr, r_cut, corner_positions, block_positions, x, y, colors):
    diff_r = np.diff(r)

    for color in np.unique(colors):
        for i in prange(len(corner_positions)):
            if colors[i] == color:
                vr_i = vr[i]
                diff_vr_r = np.diff(vr_i) / diff_r

                for j in range(len(x)):
                    for k in range(len(y)):
                        r_interp = np.sqrt((x[j] - block_positions[i, 0]) ** np.float32(2.)
                                           + (y[k] - block_positions[i, 1]) ** np.float32(2.))

                        if r_interp < r_cut:
                            l = int(np.floor((r_interp - r[0]) / (r[-1] - r[0]) * (len(r) - 1)))

                            if l < 0:
                                v[corner_positions[i, 0] + j, corner_positions[i, 1] + k] += vr_i[0]
                            elif l < (len(r) - 1):
                                value = vr_i[l] + (r_interp - r[l]) * diff_vr_r[l]
                                v[corner_positions[i, 0] + j, corner_positions[i, 1] + k] += value
