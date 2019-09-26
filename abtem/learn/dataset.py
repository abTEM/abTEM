import numpy as np
from skimage.morphology import watershed

from abtem.interpolation import interpolate_radial_functions


def gaussian_marker_labels(points, width, gpts):
    gpts = np.array(gpts)
    extent = np.diag(points.cell)
    sampling = extent / gpts
    markers = np.zeros(gpts)

    r = np.linspace(0, 4 * width, 100)
    values = np.exp(-r ** 2 / (2 * width ** 2))

    interpolate_radial_functions(markers, r, values, points.positions, sampling)
    return markers


def voronoi_labels(points, gpts):
    gpts = np.array(gpts)
    margin = np.ceil(
        np.max((np.abs(np.min(points.scaled_positions * gpts)),
                np.max(points.scaled_positions * gpts - gpts)))).astype(np.int)
    markers = np.zeros(gpts + 2 * margin, dtype=np.int)
    indices = (points.scaled_positions * gpts + margin).astype(int)
    markers[indices[:, 0], indices[:, 1]] = 1 + points.labels
    labels = watershed(np.zeros_like(markers), markers, compactness=1000)
    labels = labels[margin:-margin, margin:-margin]
    return labels - 1


def generate_indices(labels):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(1, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def labels_to_masks(labels, n_classes):
    masks = np.zeros((np.prod(labels.shape),) + (n_classes,), dtype=bool)

    for i, indices in enumerate(generate_indices(labels)):
        masks[indices, i] = True

    return masks.reshape(labels.shape + (-1,))


def data_generator(images, markers, classes, batch_size=32, augmentations=None):
    if augmentations is None:
        augmentations = []

    num_iter = len(images) // batch_size
    while True:
        for i in range(num_iter):
            if i == 0:
                indices = np.arange(len(images))
                np.random.shuffle(indices)

            batch_images = []
            batch_markers = []
            batch_classes = []

            for j, k in enumerate(indices[i * batch_size:(i + 1) * batch_size]):
                batch_images.append(images[k])
                batch_markers.append(markers[k])
                batch_classes.append(classes[k])

                for augmentation in augmentations:
                    augmentation.randomize()
                    batch_images[j] = augmentation(batch_images[j])

                    if augmentation.apply_to_label:
                        batch_markers[j] = augmentation(batch_markers[j])
                        batch_classes[j] = augmentation(batch_classes[j])

            batch_images = np.array(batch_images)
            batch_images = batch_images.reshape((batch_images.shape[0], 1,) +
                                                batch_images.shape[1:]).astype(np.float32)

            batch_markers = np.array(batch_markers)
            batch_markers = batch_markers.reshape((batch_markers.shape[0], 1,) +
                                                  batch_markers.shape[1:]).astype(np.float32)

            batch_classes = np.array(batch_classes).astype(np.int)

            yield batch_images, batch_markers, batch_classes
