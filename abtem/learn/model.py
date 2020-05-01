import cupy as cp
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.util
import torch
import torch.nn as nn
from abtem.learn.scale import add_margin_to_image
from abtem.learn.unet import GaussianFilter2d, PeakEnhancementFilter
from abtem.learn.utils import pytorch_to_cupy, cupy_to_pytorch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from cupyx.scipy.ndimage import zoom
from abtem.learn.utils import pad_to_size


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


def weighted_normalization(image, mask):
    weighted_means = cp.sum(image * mask) / cp.sum(mask)
    weighted_stds = cp.sqrt(cp.sum(mask * (image - weighted_means) ** 2) / cp.sum(mask ** 2))
    return (image - weighted_means) / weighted_stds


def merge_close(points, labels, distance):
    clusters = fcluster(linkage(pdist(points), method='complete'), distance, criterion='distance')
    cluster_labels, counts = np.unique(clusters, return_counts=True)

    to_delete = []
    for cluster_label in cluster_labels[counts > 1]:
        cluster_members = np.where(clusters == cluster_label)[0][1:]
        to_delete.append(cluster_members)

    if len(to_delete) > 0:
        points = np.delete(points, np.concatenate(to_delete), axis=0)
        labels = np.delete(labels, np.concatenate(to_delete), axis=0)

    return points, labels

class AtomRecognitionModel:

    def __init__(self, model, marker_head, segmentation_head, training_sampling, scale_model=None, margin=0):
        self._scale_model = scale_model
        self._model = model
        self._segmentation_head = segmentation_head
        self._marker_head = marker_head
        self._training_sampling = training_sampling
        self._margin = int(np.ceil(margin / self._training_sampling))

    def prepare_image(self, image, weights=None):
        image = cp.asarray(image)
        #sampling = self._scale_model(image)
        sampling = .028

        if weights is None:
            image = (image - image.mean()) / image.std()
        else:
            image = weighted_normalization(image, weights)

        image = zoom(image, zoom=sampling / self._training_sampling)
        image, padding = pad_to_size(image, image.shape[0] + 8,
                                             image.shape[1] + 8, mode='reflect', n=16)

        #image, padding = add_margin_to_image(image, self._margin)
        #image = cp.stack((image, cp.zeros_like(image)))
        #image[1, :padding[0][0]] = 1
        #image[1, -padding[0][1]:] = 1
        #image[1, :, :padding[1][0]] = 1
        #image[1, :, -padding[1][1]:] = 1

        image = cupy_to_pytorch(image)
        image = image[None,None]
        return image, sampling, padding

    def __call__(self, image):
        return self.predict(image)

    def predict(self, image):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        peak_enhancement_filter = PeakEnhancementFilter(1.8, 6, 25).to(device)

        with torch.no_grad():

            preprocessed, sampling, padding = self.prepare_image(image.copy())

            #features = self._model(preprocessed)
            #segmentation = nn.Softmax(1)(self._segmentation_head(features))

            #weights = pytorch_to_cupy(segmentation[0, 0])
            #weights = weights[padding[0][0]:-padding[0][1],padding[1][0]:-padding[1][1]]

            #preprocessed, sampling, padding = self.prepare_image(image, weights)

            #markers = torch.zeros((2,1) + preprocessed.shape[2:], device=device)
            #segmentation = torch.zeros((2,) + (3,) + preprocessed.shape[2:], device=device)

            features = self._model(preprocessed)
            segmentation = nn.Softmax(1)(self._segmentation_head(features))
            markers = self._marker_head(features)

            segmentation[:, 1] = 0

            markers = peak_enhancement_filter(markers)

            markers = pytorch_to_cupy(markers)
            segmentation = pytorch_to_cupy(segmentation)

        points = cp.array(cp.where(markers[0,0] > .2)).T

        segmentation = cp.argmax(segmentation[0], axis=0)

        points = cp.asnumpy(points)
        segmentation = cp.asnumpy(segmentation)

        labels = segmentation[points[:, 0], points[:, 1]]

        points, labels = merge_close(points, labels, 4)

        segmentation = merge_dopants_into_contamination(segmentation)
        contamination = segmentation == 1

        not_contaminated = contamination[points[:, 0], points[:, 1]] == 0
        points = points[not_contaminated]
        labels = labels[not_contaminated]

        scale_factor = 16
        contamination = skimage.util.view_as_blocks(contamination, (scale_factor,) * 2).sum((-2, -1)) > (
                scale_factor ** 2 / 2)
        contamination = (np.array(np.where(contamination)).T) * 16 + 8

        points = np.vstack((points, contamination))
        labels = np.concatenate((labels, np.ones(len(contamination))))

        points = (points - padding[0]) * self._training_sampling / sampling
        return points[:, ::-1], labels