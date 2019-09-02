import numpy as np
from numba import jit, prange


@jit(nopython=True)
def check_square_overlap(corner_a, corner_b, size):
    return not ((corner_a[0] + size < corner_b[0]) | (corner_b[0] + size < corner_a[0]) |
                (corner_a[1] + size < corner_b[1]) | (corner_b[1] + size < corner_a[1]))


@jit(nopython=True, parallel=True, nogil=True)
def get_overlap_graph(corners, size):
    graph = np.zeros((len(corners), len(corners)), dtype=np.bool_)
    for i in prange(len(corners)):
        for j in range(i, len(corners)):
            overlap = check_square_overlap(corners[i], corners[j], size)
            graph[i, j] = graph[j, i] = overlap

    return graph
