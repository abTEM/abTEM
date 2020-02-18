import numpy as np
from abtem.utils import coordinates_in_disc, ind2sub2d


class NonMaximumSuppression:

    def __init__(self, distance, threshold, max_num_maxima):
        self._threshold = threshold
        self._max_num_maxima = max_num_maxima
        self._disc = coordinates_in_disc(distance)

    def predict(self, density, segmentation=None):

        shape = density.shape
        density = density.reshape((-1,))
        accepted = np.zeros(density.shape, dtype=np.bool_)

        suppressed = np.zeros(np.prod(shape), dtype=np.bool_)

        if segmentation is not None:
            segmentation = segmentation.reshape((segmentation.shape[0],) + (-1,))
            probabilities = np.zeros(segmentation.shape, dtype=segmentation.dtype)
        else:
            probabilities = None

        disc_row, disc_col = self._disc
        disc_flat = disc_col + disc_row * shape[1]

        suppressed[density < self._threshold] = True

        num_maxima = 0
        for i in np.argsort(-density):
            if num_maxima == self._max_num_maxima:
                break

            if not suppressed[i]:
                accepted[i] = True

                row, col = np.unravel_index(i, shape)

                neighbors_row = row + disc_row
                neighbors_col = col + disc_col

                valid = ((neighbors_row > -1) & (neighbors_col > -1) &
                         (neighbors_row < shape[0]) & (neighbors_col < shape[1]))

                neighbors = i + disc_flat
                neighbors = neighbors[valid]
                suppressed[neighbors] = True

                if probabilities is not None:
                    probabilities[:, i] = np.sum(segmentation[:, neighbors] * density[neighbors], axis=1)
                    probabilities[:, i] /= np.sum(probabilities[:, i])

                num_maxima += 1

        positions = np.array(np.where(accepted.reshape(shape))).T
        if segmentation is None:
            return positions

        else:
            return positions, probabilities.reshape((-1,) + shape)[:, positions[:, 0], positions[:, 1]]
