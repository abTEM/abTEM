import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils import dlpack

from abtem.learn.scale import add_margin_to_image
from abtem.learn.unet import GaussianFilter2d, PeakEnhancementFilter
from abtem.learn.utils import pytorch_to_cupy, cupy_to_pytorch
from cupyx.scipy.ndimage import zoom




def weighted_normalization(image, mask):
    weighted_means = cp.sum(image * mask) / cp.sum(mask)
    weighted_stds = cp.sqrt(cp.sum(mask * (image - weighted_means) ** 2) / cp.sum(mask ** 2))
    return (image - weighted_means) / weighted_stds


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
        sampling = .05  # self._scale_model(image)
        if weights is None:
            image = (image - image.mean()) / image.std()
        else:
            image = weighted_normalization(image, weights)

        # image = zoom(image, zoom=sampling / self._training_sampling)
        image, padding = add_margin_to_image(image, self._margin)
        image = cp.stack((image, cp.zeros_like(image)))
        image[1, :padding[0][0]] = 1
        image[1, -padding[0][1]:] = 1
        image[1, :, :padding[1][0]] = 1
        image[1, :, -padding[1][1]:] = 1

        image = cupy_to_pytorch(image)
        image = image[None]
        return image, sampling, padding

    def __call__(self, image):
        return self.predict(image)

    def predict(self, image):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            preprocessed, sampling, padding = self.prepare_image(image.copy())

            features = self._model(preprocessed)
            segmentation = nn.Softmax(1)(self._segmentation_head(features))

            weights = pytorch_to_cupy(segmentation[0, 0])
            weights = weights[padding[0][0]:-padding[0][1],padding[1][0]:-padding[1][1]]

            preprocessed, sampling, padding = self.prepare_image(image, weights)

            markers = torch.zeros((5,1) + preprocessed.shape[2:], device=device)
            segmentation = torch.zeros((5,) + (3,) + preprocessed.shape[2:], device=device)

            for i in range(5):
                features = self._model(preprocessed)
                segmentation[i] = nn.Softmax(1)(self._segmentation_head(features))
                markers[i] = self._marker_head(features)

            segmentation = segmentation.mean(0)
            markers_mean = markers.mean(0)

            base_filter = GaussianFilter2d(7)
            base_filter = base_filter.to(device)
            peak_enhancement_filter = PeakEnhancementFilter(base_filter, 20, 1.4)
            markers_mean = peak_enhancement_filter(markers_mean[None])[0]

            for i in range(markers.shape[0]):
                markers[i] = base_filter(markers[i][None])

            markers_variance = markers.std(0)

            markers_mean = pytorch_to_cupy(markers_mean)
            markers_variance = pytorch_to_cupy(markers_variance)
            segmentation = pytorch_to_cupy(segmentation)

        points = cp.array(cp.where(markers_mean[0] > .3)).T

        segmentation = cp.argmax(segmentation, axis=0)
        labels = segmentation[points[:, 0], points[:, 1]]

        points = cp.asnumpy(points) #.astype(np.float)
        labels = cp.asnumpy(labels)

        #points, labels = merge_close(points, labels, 8)
        #contamination_points = np.array(np.where(np.random.poisson((cp.asnumpy(segmentation) == 1) * .0025))).T

        #points = np.vstack((points, contamination_points))
        #labels = np.concatenate((labels, np.ones(len(contamination_points), dtype=np.int)))

        #points = (points - padding[0])
        return points[:, ::-1], labels, markers_mean[0], segmentation, padding