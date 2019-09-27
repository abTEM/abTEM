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


def safe_assign(assignee, assignment, index):
    try:
        assignee[index] = assignment
    except:
        assignee = np.zeros((assignee.shape[0],) + assignment.shape)
        assignee[index] = assignment

    return assignee


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
            batch_density = []
            batch_classes = []

            for j, k in enumerate(indices[i * batch_size:(i + 1) * batch_size]):
                batch_images.append(images[k].copy())
                batch_density.append(markers[k].copy())
                batch_classes.append(classes[k].copy())

                for augmentation in augmentations:
                    augmentation.randomize()

                    original = batch_images[j].copy()

                    if augmentation.channels is None:
                        channels = range(batch_images[j].shape[0])
                    else:
                        channels = augmentation.channels

                    for channel in channels:
                        augmented = augmentation(original[channel])
                        batch_images[j] = safe_assign(batch_images[j], augmented, channel)

                    if augmentation.apply_to_label:
                        batch_density[j] = augmentation(batch_density[j])
                        batch_classes[j] = augmentation(batch_classes[j])

                batch_images = np.array(batch_images).astype(np.float32)

                batch_density = np.array(batch_density).astype(np.float32)

                batch_classes = np.array(batch_classes).astype(np.int)

                yield batch_images, batch_density, batch_classes
