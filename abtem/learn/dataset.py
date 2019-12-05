import numpy as np
from skimage.morphology import watershed

from abtem.interpolation import interpolation_kernel_parallel


def interpolate_radial_functions(array, r, values, positions, sampling, thread_safe=True):
    block_margin = int(r[-1] / min(sampling))
    block_size = 2 * block_margin + 1

    corner_positions = np.round(positions[:, :2] / sampling).astype(np.int) - block_margin
    block_positions = positions[:, :2] - sampling * corner_positions

    x = np.linspace(0., block_size * sampling[0], block_size, endpoint=False)
    y = np.linspace(0., block_size * sampling[1], block_size, endpoint=False)

    if values.shape == (len(r),):
        values = np.tile(values, (len(corner_positions), 1))

    interpolation_kernel_parallel(array, r, values, corner_positions, block_positions, x, y, thread_safe)


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
    if len(points) == 0:
        return np.zeros(gpts, dtype=np.int)

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
    index = np.arange(0, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def labels_to_masks(labels, n_classes):
    masks = np.zeros((n_classes,) + (np.prod(labels.shape),), dtype=bool)

    for i, indices in enumerate(generate_indices(labels)):
        masks[i, indices] = True

    return masks.reshape((-1,) + labels.shape)


def safe_assign(assignee, assignment, index):
    try:
        assignee[index] = assignment
    except:
        assignee = np.zeros((assignee.shape[0],) + assignment.shape)
        assignee[index] = assignment

    return assignee


class DataGenerator:

    def __init__(self, images, labels, batch_size=8, augmentations=None):
        assert len(images.shape) == 4

        for label in labels:
            assert len(label) == len(images)
            assert len(label.shape) == 3

        self._images = images
        self._labels = labels

        self._image_dtype = images.dtype
        self._label_dtypes = [label.dtype for label in labels]

        if augmentations is None:
            augmentations = []

        self._augmentations = augmentations

        self._num_iter = len(images) // batch_size
        self._batch_size = batch_size

        self._indices = np.arange(self.num_examples)
        self._i = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def labels_per_example(self):
        return len(self._labels)

    @property
    def num_examples(self):
        return len(self._images)

    def __iter__(self):
        return self

    def __next__(self):
        i = self._i % self.num_examples

        if i == 0:
            np.random.shuffle(self._indices)

        batch_images = []
        batch_labels = [[] for _ in range(self.labels_per_example)]

        for j, k in enumerate(self._indices[i * self.batch_size:(i + 1) * self.batch_size]):
            batch_images.append(self._images[k].copy())
            for l in range(self.labels_per_example):
                batch_labels[l].append(self._labels[l][k].copy())

            for augmentation in self._augmentations:
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
                    try:
                        for l in augmentation.apply_to_label:
                            batch_labels[l][j] = augmentation(batch_labels[l][j])
                    except:
                        for l in range(self.labels_per_example):
                            batch_labels[l][j] = augmentation(batch_labels[l][j])

        batch_images = np.array(batch_images).astype(self._image_dtype)

        for j in range(self.labels_per_example):
            batch_labels[j] = np.array(batch_labels[j]).astype(self._label_dtypes[j])

        i += 1

        return batch_images, batch_labels
