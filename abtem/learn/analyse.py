import numpy as np
from abtem.points import inside_cell


def find_pairs(positions_1, positions_2, cutoff):
    norms_1 = (positions_1 ** 2).sum(1)[:, None]
    norms_2 = (positions_2 ** 2).sum(1)[None, :]
    distances = norms_1 + norms_2 - 2. * np.dot(positions_1, positions_2.T)
    shape = distances.shape
    # max_pairs = min(len(positions_1), len(positions_2))

    distances = distances.ravel()
    pairs = np.zeros(shape, dtype=np.bool)
    paired_1 = np.zeros(len(positions_1), dtype=np.bool)
    paired_2 = np.zeros(len(positions_2), dtype=np.bool)
    indices = np.argsort(distances)
    distances = distances[indices]

    for i, (raveled_idx, distance) in enumerate(zip(indices, distances)):
        if distance > cutoff:
            break

        idx_1, idx_2 = np.unravel_index(raveled_idx, shape)
        if paired_1[idx_1] or paired_2[idx_2]:
            continue

        pairs[idx_1, idx_2] = 1
        paired_1[idx_1] = 1
        paired_2[idx_2] = 1

    return pairs  # , np.where(paired_1 == 0)[0], np.where(paired_2 == 0)[0]


class Evaluator:

    def __init__(self, distance_threshold):
        self._distance_threshold = distance_threshold ** 2

    @property
    def distance_threshold(self):
        return np.sqrt(self._distance_threshold)

    def get_precission_and_recall(self, beta=1, filter_by_attribute=None):
        tp = self.num_true_positives(filter_by_attribute=filter_by_attribute)
        fp = self.num_false_positives(filter_by_attribute=filter_by_attribute)
        fn = self.num_false_negatives(filter_by_attribute=filter_by_attribute)
        precission = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precission + recall > 0:
            f_score = (1 + beta ** 2) * precission * recall / (precission + recall)
        else:
            f_score = 0.
        return precission, recall, f_score

    def num_true_positives(self, filter_by_attribute=None):
        return len(self.get_true_positives(filter_by_attribute=filter_by_attribute))

    def num_false_positives(self, filter_by_attribute=None):
        return len(self.get_false_positives(filter_by_attribute=filter_by_attribute))

    def num_false_negatives(self, filter_by_attribute=None):
        return len(self.get_false_negatives(filter_by_attribute=filter_by_attribute))


    def get_true_positives(self, filter_by_attribute=None):
        pairs = self._pairs

        if filter_by_attribute is not None:
            name, value = filter_by_attribute
            pairs = pairs * (self._true_points.get_attributes(name) == value)[:, None]
            pairs *= (self._detected_points.get_attributes(name) == value)[None]

        if self._mask_true is not None:
            pairs = pairs * self._mask_true[:, None]

        if self._mask_detected is not None:
            pairs = pairs * self._mask_detected[None]

        paired_detected = np.any(pairs, axis=0)
        return self._detected_points[paired_detected]

    def get_false_positives(self, filter_by_attribute=None):
        unpaired_detected = (np.any(self._pairs, axis=0) == 0)
        if self._mask_detected is not None:
             unpaired_detected *= self._mask_detected

        if filter_by_attribute is None:
            return self._detected_points[unpaired_detected]
        else:
            return self._detected_points[unpaired_detected].filter_by_attribute(*filter_by_attribute)

    def get_false_negatives(self, filter_by_attribute=None):
        unpaired_true = np.any(self._pairs, axis=1) == 0
        if self._mask_true is not None:
             unpaired_true *= self._mask_true

        if filter_by_attribute is None:
            return self._true_points[unpaired_true]
        else:
            return self._true_points[unpaired_true].filter_by_attribute(*filter_by_attribute)

    def find_pairs(self, true_points, detected_points, mask_true=None, mask_detected=None):
        self._pairs = find_pairs(true_points.positions,
                                 detected_points.positions,
                                 self._distance_threshold)
        self._true_points = true_points
        self._detected_points = detected_points

        self._mask_true = mask_true
        self._mask_detected = mask_detected

        # self._paired_true = np.any(self._pairs, axis=1)
        # self._paired_detected = np.any(self._pairs, axis=0)
        # self._unpaired_true = self._paired_true == 0
        # self._unpaired_detected = self._paired_detected == 0
        #
        # if mask_true is not None:
        #     self._unpaired_true *= mask_true
        #
        # if mask_detected is not None:
        #     self._unpaired_detected *= mask_detected
