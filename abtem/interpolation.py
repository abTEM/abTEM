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


@jit(nopython=True, nogil=True)
def interpolation_kernel(v, r, vr, corner_positions, block_positions, x, y):
    diff_r = np.diff(r)
    diff_vr_r = np.diff(vr) / diff_r

    if corner_positions[0] < 0:
        rJa = abs(corner_positions[0])
    else:
        rJa = 0

    if corner_positions[0] + len(x) > v.shape[0]:
        rJb = v.shape[0] - corner_positions[0]
    else:
        rJb = len(x)

    rJ = range(rJa, rJb)
    rj = range(max(corner_positions[0], 0), min(corner_positions[0] + len(x), v.shape[0]))

    if corner_positions[1] < 0:
        rKa = abs(corner_positions[1])
    else:
        rKa = 0

    if corner_positions[0] + len(y) > v.shape[1]:
        rKb = v.shape[1] - corner_positions[1]
    else:
        rKb = len(y)

    rK = range(rKa, rKb)
    rk = range(max(corner_positions[1], 0), min(corner_positions[1] + len(y), v.shape[1]))  #

    for j, J in zip(rj, rJ):
        for k, K in zip(rk, rK):
            r_interp = np.sqrt((x[J] - block_positions[0]) ** 2. + (y[K] - block_positions[1]) ** 2.)

            if r_interp < r[-1]:
                l = np.searchsorted(r, r_interp) - 1

                if l < 0:
                    v[j, k] += vr[0]
                elif l < (len(vr) - 1):
                    value = vr[l] + (r_interp - r[l]) * diff_vr_r[l]
                    v[j, k] += value


@jit(nopython=True, nogil=True, parallel=True)
def interpolation_kernel_parallel(v, r, vr, corner_positions, block_positions, x, y, thread_safe=True):
    if thread_safe:
        colors = thread_safe_coloring(corner_positions, len(x))

        for color in np.unique(colors):
            for i in prange(len(corner_positions)):
                if colors[i] == color:
                    interpolation_kernel(v, r, vr[i], corner_positions[i], block_positions[i], x, y)

    else:
        for i in prange(len(corner_positions)):
            interpolation_kernel(v, r, vr[i], corner_positions[i], block_positions[i], x, y)


def interpolate_radial_functions(array, r, values, positions, sampling, thread_safe=True):
    block_margin = int(r[-1] / min(sampling))
    block_size = 2 * block_margin + 1

    corner_positions = np.round(positions[:, :2] / sampling).astype(np.int) - block_margin
    block_positions = positions[:, :2] - sampling * corner_positions

    x = np.linspace(0., block_size * sampling[0], block_size, endpoint=False)
    y = np.linspace(0., block_size * sampling[1], block_size, endpoint=False)

    if values.shape == (len(r),):
        values = np.tile(values, (len(corner_positions), 1))

    interpolation_kernel_parallel(array, r, values, corner_positions, block_positions, x, y, thread_safe)
