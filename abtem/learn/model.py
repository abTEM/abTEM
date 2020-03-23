import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils import dlpack

from abtem.learn.scale import add_margin_to_image
from abtem.learn.unet import GaussianFilter2d, PeakEnhancementFilter
from abtem.learn.utils import pytorch_to_cupy, cupy_to_pytorch
from cupyx.scipy.ndimage import zoom


class AtomRecognitionModel:

    def __init__(self, training_sampling, scale_model, model, marker_head, segmentation_head, margin=0):
        self._scale_model = scale_model
        self._model = model
        self._segmentation_head = segmentation_head
        self._marker_head = marker_head
        self._training_sampling = training_sampling
        self._margin = int(np.ceil(margin / self._training_sampling))

    def prepare_image(self, image):
        image = cp.asarray(image)
        sampling = self._scale_model(image)
        image = (image - image.mean()) / image.std()
        image = zoom(image, zoom=sampling / self._training_sampling)
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

        preprocessed, sampling, padding = self.prepare_image(image)

        with torch.no_grad():
            features = self._model(preprocessed)
            markers = self._marker_head(features)
            segmentation = nn.LogSoftmax(1)(self._segmentation_head(features))

            base_filter = GaussianFilter2d(4)
            base_filter = base_filter.to(device)
            peak_enhancement_filter = PeakEnhancementFilter(base_filter, 20, 1.4)
            markers = peak_enhancement_filter(markers)

            markers = pytorch_to_cupy(markers)
            segmentation = pytorch_to_cupy(segmentation)

        points = cp.array(cp.where(markers[0, 0] > .3)).T

        segmentation = cp.argmax(segmentation[0], axis=0)
        labels = segmentation[points[:, 0], points[:, 1]]

        points = cp.asnumpy(points).astype(np.float)
        labels = cp.asnumpy(labels)

        # labels = np.zeros(len(points), dtype=np.int)
        points = (points - padding[0]) * self._training_sampling / sampling
        return points[:, ::-1], labels  # labels

        # segmentation = self._segmentation_model(preprocessed)

        # preprocessed = weighted_normalization(preprocessed, segmentation[:, 0][None])
        # print(preprocessed.shape)
        # density = self._density_model(preprocessed)
        #
        # segmentation = segmentation[0, :].detach().cpu().numpy()
        #
        # density = density[0, 0].detach().cpu().numpy()
        #
        # points, labels = self._discretization_model(density, segmentation)
        # points = (points - padding[0]) * self._training_sampling / sampling
        #
        # return points[:, ::-1], np.argmax(labels, axis=0)

    # def predict(self, image):
    #     preprocessed, sampling, padding = self.prepare_image(image)
    #     segmentation = self._segmentation_model(preprocessed)
    #
    #     preprocessed = weighted_normalization(preprocessed, segmentation[:, 0][None])
    #     print(preprocessed.shape)
    #     density = self._density_model(preprocessed)
    #
    #     segmentation = segmentation[0, :].detach().cpu().numpy()
    #
    #     density = density[0, 0].detach().cpu().numpy()
    #
    #     points, labels = self._discretization_model(density, segmentation)
    #     points = (points - padding[0]) * self._training_sampling / sampling
    #
    #     return points[:, ::-1], np.argmax(labels, axis=0)  # {'species': np.argmax(labels, axis=0),
    #     # 'distortion': np.zeros(len(labels), dtype=np.int)}
