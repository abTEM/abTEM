import numpy as np
import torch
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
    markers = np.zeros(gpts, dtype=np.int)
    scaled_positions = points.scaled_positions
    inside = ((scaled_positions[:, 0] > 0) & (scaled_positions[:, 1] > 0) &
              (scaled_positions[:, 0] < 1) & (scaled_positions[:, 1] < 1))

    indices = (scaled_positions[inside] * gpts).astype(int)
    markers[indices[:, 0], indices[:, 1]] = 1 + points.labels[inside]
    labels = watershed(np.zeros_like(markers), markers, compactness=1000)
    return labels


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


def data_generator(images, markers, classes=None, batch_size=32, augmentations=None):
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
            if classes is not None:
                batch_classes = []
            else:
                batch_classes = None

            for j, k in enumerate(indices[i * batch_size:(i + 1) * batch_size]):
                batch_images.append(images[k])
                batch_markers.append(markers[k])
                if batch_classes is not None:
                    batch_classes.append(classes[k])

                for augmentation in augmentations:
                    augmentation.randomize()
                    batch_images[j] = augmentation(batch_images[j])

                    if augmentation.apply_to_label:
                        batch_markers[j] = augmentation(batch_markers[j])

                        if batch_classes is not None:
                            batch_classes[j] = augmentation(batch_classes[j])

            batch_images = np.array(batch_images)
            batch_images = batch_images.reshape((batch_images.shape[0], 1,) +
                                                batch_images.shape[1:]).astype(np.float32)
            batch_images = torch.from_numpy(batch_images)

            batch_markers = np.array(batch_markers)
            batch_markers = batch_markers.reshape((batch_markers.shape[0], 1,) +
                                                  batch_markers.shape[1:]).astype(np.float32)
            batch_markers = torch.from_numpy(batch_markers)

            if batch_classes is None:
                yield batch_images, batch_markers

            else:
                batch_classes = np.array(batch_classes)
                yield batch_images, batch_markers, batch_classes
