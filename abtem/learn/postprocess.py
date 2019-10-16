import numpy as np
from abtem.utils import ind2sub, sub2ind
from numba import njit


# def index_disc(distance):
#     x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)
#
#     x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
#     y_disc = x_disc.copy().T
#     x_disc -= distance
#     y_disc -= distance
#     r2 = x_disc ** 2 + y_disc ** 2
#
#     x_disc = x_disc[r2 < distance ** 2]
#     y_disc = y_disc[r2 < distance ** 2]
#
#     return x_disc, y_disc
#
#
# def non_maximum_suppresion(markers, distance, threshold=None):
#     shape = markers.shape
#
#     markers = markers.ravel()
#     accepted = np.zeros(markers.shape, dtype=np.bool_)
#     suppressed = np.zeros(markers.shape, dtype=np.bool_)
#
#     y_disc, x_disc = index_disc(distance)
#
#     if threshold is not None:
#         suppressed[markers < threshold] = True
#
#     for i in np.argsort(-markers.ravel()):
#         if not suppressed[i]:
#             accepted[i] = True
#
#             y, x = ind2sub(shape, i)
#             neighbors_x = x + x_disc
#             neighbors_y = y + y_disc
#
#             valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (neighbors_y < shape[1]))
#
#             neighbors_x = neighbors_x[valid]
#             neighbors_y = neighbors_y[valid]
#
#             k = sub2ind(neighbors_y, neighbors_x, shape)
#             suppressed[k] = True
#
#     accepted = accepted.reshape(shape)
#     return accepted


def non_maximum_suppresion(density, classes, distance, threshold):
    shape = density.shape[2:]

    density = density.reshape((density.shape[0], -1))

    # if class_indicators is not None:
    classes = classes.reshape(classes.shape[:2] + (-1,))
    probabilities = np.zeros(classes.shape, dtype=classes.dtype)

    accepted = np.zeros(density.shape, dtype=np.bool_)
    suppressed = np.zeros(density.shape, dtype=np.bool_)

    x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)

    x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
    y_disc = x_disc.copy().T
    x_disc -= distance
    y_disc -= distance
    x_disc = x_disc.ravel()
    y_disc = y_disc.ravel()

    r2 = x_disc ** 2 + y_disc ** 2

    x_disc = x_disc[r2 < distance ** 2]
    y_disc = y_disc[r2 < distance ** 2]

    weights = np.exp(-r2 / (2 * (distance / 3) ** 2))
    weights = np.reshape(weights[r2 < distance ** 2], (-1, 1))

    for i in range(density.shape[0]):
        suppressed[i][density[i] < threshold] = True
        for j in np.argsort(-density[i].ravel()):
            if not suppressed[i, j]:
                accepted[i, j] = True

                x, y = ind2sub(shape, j)
                neighbors_x = x + x_disc
                neighbors_y = y + y_disc

                valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (
                        neighbors_y < shape[1]))

                neighbors_x = neighbors_x[valid]
                neighbors_y = neighbors_y[valid]

                k = sub2ind(neighbors_x, neighbors_y, shape)
                suppressed[i][k] = True

                tmp = np.sum(classes[i, :, k] * weights[valid], axis=0)
                probabilities[i, :, j] = tmp / np.sum(tmp)

    accepted = accepted.reshape((classes.shape[0],) + shape)

    probabilities = probabilities.reshape(classes.shape[:2] + shape)

    return accepted, probabilities
