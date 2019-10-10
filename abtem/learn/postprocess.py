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


def non_maximum_suppresion(markers, class_indicators, distance, threshold):
    shape = markers.shape[2:]

    markers = markers.reshape((markers.shape[0], -1))

    # if class_indicators is not None:
    class_indicators = class_indicators.reshape(class_indicators.shape[:2] + (-1,))
    class_probabilities = np.zeros(class_indicators.shape, dtype=class_indicators.dtype)

    accepted = np.zeros(markers.shape, dtype=np.bool_)
    suppressed = np.zeros(markers.shape, dtype=np.bool_)

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

    for i in range(markers.shape[0]):
        suppressed[i][markers[i] < threshold] = True
        for j in np.argsort(-markers[i].ravel()):
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

                tmp = np.sum(class_indicators[i, :, k] * weights[valid], axis=0)
                class_probabilities[i, :, j] = tmp / np.sum(tmp)

    accepted = accepted.reshape((markers.shape[0],) + shape)

    class_probabilities = class_probabilities.reshape(class_indicators.shape[:2] + shape)

    return accepted, class_probabilities
