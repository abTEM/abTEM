import numpy as np
from skimage.morphology import watershed

from abtem.interpolation import interpolation_kernel_parallel
from abtem.learn.augment import RandomCropStack


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
    masks = np.zeros((labels.shape[0], n_classes,) + (np.prod(labels.shape[1:]),), dtype=bool)

    for i in range(labels.shape[0]):
        for j, indices in enumerate(generate_indices(labels[i])):
            masks[i, j, indices] = True

    return masks.reshape((labels.shape[0], n_classes,) + labels.shape[1:])


class DataGenerator:

    def __init__(self, images, labels, crop, batch_size=8, augmentations=None):

        self._num_examples = len(images)
        for label in labels:
            assert len(label) == self._num_examples

        self._images = images
        self._labels = labels

        if augmentations is None:
            augmentations = []

        self._augmentations = augmentations

        self._num_iter = self._num_examples // batch_size
        self._batch_size = batch_size

        self._indices = np.arange(self.num_examples)
        self._global_iteration = 280

        self._crop = RandomCropStack(out_shape=crop)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_examples(self):
        return len(self._images)

    def __iter__(self):
        return self

    def __next__(self):
        epoch_iter = (self._global_iteration) % (self.num_examples // self.batch_size)

        if epoch_iter == 0:
            np.random.shuffle(self._indices)

        self._crop.randomize()

        batch_indices = self._indices[epoch_iter * self.batch_size:(epoch_iter + 1) * self.batch_size]

        batch_images = self._crop(self._images[batch_indices].copy())
        batch_labels = [self._crop(label[batch_indices].copy()) for label in self._labels]

        for augmentation in self._augmentations:
            augmentation.randomize()
            for i in range(len(batch_indices)):
                augmented = augmentation(batch_images[i])
                batch_images[i, :, :augmented.shape[1], :augmented.shape[2]] = augmented

                if not augmentation.apply_to_label:
                    continue

                for j in range(len(batch_labels)):
                    augmented = augmentation(batch_labels[j][i])
                    batch_labels[j][i, :, :augmented.shape[1], :augmented.shape[2]] = augmented

        self._global_iteration += 1

        batch_images = batch_images.astype(np.float32)
        batch_labels = [bl.astype(np.float32) for bl in batch_labels]

        return batch_images, batch_labels
