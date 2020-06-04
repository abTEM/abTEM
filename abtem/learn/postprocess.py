import numpy as np
import skimage.measure
from abtem.utils import label_to_index_generator
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def mask_outside_points(points, shape, margin=0):
    mask = ((points[:, 0] >= margin) & (points[:, 1] >= margin) &
            (points[:, 0] < shape[0] - margin) & (points[:, 1] < shape[1] - margin))
    return mask


def merge_close_points(points, distance):
    if len(points) < 2:
        return points, np.arange(len(points))

    clusters = fcluster(linkage(pdist(points), method='complete'), distance, criterion='distance')
    new_points = np.zeros_like(points)
    indices = np.zeros(len(points), dtype=np.int)
    k = 0
    for i, cluster in enumerate(label_to_index_generator(clusters, 1)):
        new_points[i] = np.mean(points[cluster], axis=0)
        indices[i] = np.min(indices)
        k += 1
    return new_points[:k], indices[:k]


def markers_to_points(markers, threshold=.5, merge_distance=0.1):
    points = np.array(np.where(markers > threshold)).T
    if len(points) > 1:
        points, _ = merge_close_points(points, merge_distance)
    return points


def index_array_with_points(points, array, outside_value=0):
    values = np.full(len(points), outside_value, dtype=array.dtype)
    rounded = np.round(points).astype(np.int)
    inside = mask_outside_points(rounded, array.shape)
    inside_points = rounded[inside]
    values[inside] = array[inside_points[:, 0], inside_points[:, 1]]
    return values


def disc_indices(radius):
    X = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
    x = np.linspace(0, 2 * radius, 2 * radius + 1)
    X[:] = np.linspace(0, 2 * radius, 2 * radius + 1)
    X -= radius

    Y = X.copy().T.ravel()
    X = X.ravel()

    x = x - radius
    r2 = (x[:, None] ** 2 + x[None] ** 2).ravel()
    return X[r2 < radius ** 2], Y[r2 < radius ** 2], r2[r2 < radius ** 2]


def integrate_discs(points, array, radius):
    points = np.round(points).astype(np.int)
    X, Y, r2 = disc_indices(radius)
    weights = np.exp(-r2 / (2 * (radius / 3) ** 2))

    probabilities = np.zeros((len(points), 3))
    for i, point in enumerate(points):
        X_ = point[0] + X
        Y_ = point[1] + Y
        inside = ((X_ > 0) & (X_ < array.shape[1]) & (Y_ > 0) & (Y_ < array.shape[2]))

        X_ = X_[inside]
        Y_ = Y_[inside]
        probabilities[i] = np.sum(array[:, X_, Y_] * weights[None, inside], axis=1)

    return probabilities


def merge_dopants_into_contamination(segmentation):
    binary = segmentation != 0
    labels, n = skimage.measure.label(binary, return_num=True)

    new_segmentation = np.zeros_like(segmentation)
    for label in range(1, n + 1):
        in_segment = labels == label
        if np.sum(segmentation[in_segment] == 1) > np.sum(segmentation[in_segment] == 2):
            new_segmentation[in_segment] = 1
        else:
            new_segmentation[in_segment] = 2

    return new_segmentation
