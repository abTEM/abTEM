import numpy as np


def label_to_index_generator(labels: np.ndarray, first_label: int = 0):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(first_label, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side="left")
    hi = np.searchsorted(sorted_labels, index, side="right")
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield indices[l:h]
