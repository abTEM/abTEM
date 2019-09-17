import numpy as np
from abtem.utils import ind2sub, sub2ind
from numba import njit


# @njit
def index_disc(distance):
    x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)

    x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
    y_disc = x_disc.copy().T
    x_disc -= distance
    y_disc -= distance
    r2 = x_disc ** 2 + y_disc ** 2

    x_disc = x_disc[r2 < distance ** 2]
    y_disc = y_disc[r2 < distance ** 2]

    return x_disc, y_disc


def non_maximum_suppresion(markers, distance, threshold=None):
    shape = markers.shape

    markers = markers.ravel()
    accepted = np.zeros(markers.shape, dtype=np.bool_)
    suppressed = np.zeros(markers.shape, dtype=np.bool_)

    y_disc, x_disc = index_disc(distance)

    if threshold is not None:
        suppressed[markers < threshold] = True

    for i in np.argsort(-markers.ravel()):
        if not suppressed[i]:
            accepted[i] = True

            y, x = ind2sub(shape, i)
            neighbors_x = x + x_disc
            neighbors_y = y + y_disc

            valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (neighbors_y < shape[1]))

            neighbors_x = neighbors_x[valid]
            neighbors_y = neighbors_y[valid]

            k = sub2ind(neighbors_y, neighbors_x, shape)
            suppressed[k] = True

    accepted = accepted.reshape(shape)
    return accepted
