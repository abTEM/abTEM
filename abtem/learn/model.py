import json
import os

import cupy as cp
import numpy as np
import skimage.util
import torch
import torch.nn as nn
from abtem.learn.filters import PeakEnhancementFilter
from abtem.learn.postprocess import merge_close_points, integrate_discs, merge_dopants_into_contamination
from abtem.learn.r2unet import R2UNet, ConvHead
from abtem.learn.utils import pad_to_size, pytorch_to_cupy, cupy_to_pytorch, weighted_normalization
from cupyx.scipy.ndimage import zoom


class AtomRecognitionModel:

    def __init__(self, backbone, density_head, segmentation_head, training_sampling, density_sigma, threshold,
                 enhancement_filter_kwargs):
        self._backbone = backbone
        self._segmentation_head = segmentation_head
        self._density_head = density_head
        self._training_sampling = training_sampling
        self._density_sigma = density_sigma
        self._threshold = threshold
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._enhancement_filter = PeakEnhancementFilter(**enhancement_filter_kwargs).to(device)

    def to(self, *args, **kwargs):
        self._backbone.to(*args, **kwargs)
        self._segmentation_head.to(*args, **kwargs)
        self._density_head.to(*args, **kwargs)
        self._enhancement_filter.to(*args, **kwargs)
        return self

    @classmethod
    def load(cls, path):

        with open(path, 'r') as fp:
            state = json.load(fp)

        folder = os.path.dirname(path)

        backbone = R2UNet(1, 8)
        density_head = ConvHead(backbone.out_type, 1)
        segmentation_head = ConvHead(backbone.out_type, 3)

        backbone.load_state_dict(torch.load(os.path.join(folder, state['backbone']['weights_file'])))
        density_head.load_state_dict(torch.load(os.path.join(folder, state['density_head']['weights_file'])))
        segmentation_head.load_state_dict(torch.load(os.path.join(folder, state['segmentation_head']['weights_file'])))

        return cls(backbone=backbone,
                   density_head=density_head,
                   segmentation_head=segmentation_head,
                   training_sampling=state['training_sampling'],
                   density_sigma=state['density_sigma'],
                   threshold=state['threshold'],
                   enhancement_filter_kwargs=state['enhancement_filter'])

    @property
    def training_sampling(self):
        return self._training_sampling

    def prepare_image(self, image, sampling=None, mask=None):
        image = image.astype(np.float32)
        sampling=.1
        image = cp.asarray(image)

        if sampling is not None:
            image = zoom(image, zoom=sampling / self._training_sampling)

        image, padding = pad_to_size(image, image.shape[0], image.shape[1], n=16)

        image = cupy_to_pytorch(image)[None, None]

        image = weighted_normalization(image, mask)

        return image, sampling, padding

    def __call__(self, image, sampling=None):
        try:
            return self.predict(image, sampling=sampling)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: ran out of memory for image of size {}'.format(image.shape))
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def predict(self, image, sampling=None):
        with torch.no_grad():
            preprocessed, sampling, padding = self.prepare_image(image.copy(), sampling)

            features = self._backbone(preprocessed)
            segmentation = nn.Softmax(1)(self._segmentation_head(features))

            mask = (segmentation[0, 0] + segmentation[0, 2])[None, None]

            #import matplotlib.pyplot as plt
            #plt.imshow(mask.cpu().numpy()[0,0],vmin=0,vmax=1)
            #plt.show()
            #plt.show()

            preprocessed, sampling, padding = self.prepare_image(image.copy(), sampling, mask)
            features = self._backbone(preprocessed)

            segmentation = nn.Softmax(1)(self._segmentation_head(features))
            density = nn.Sigmoid()(self._density_head(features))

            #segmentation[:, 1] = 0

            markers = self._enhancement_filter(density)

            markers = pytorch_to_cupy(markers)
            segmentation = pytorch_to_cupy(segmentation)

        torch.cuda.empty_cache()

        points = cp.array(cp.where(markers[0, 0] > self._threshold)).T

        points = cp.asnumpy(points)
        segmentation = cp.asnumpy(segmentation)[0]

        points, indices = merge_close_points(points, self._density_sigma)

        label_probabilities = integrate_discs(points, segmentation, self._density_sigma)
        labels = np.argmax(label_probabilities, axis=-1)

        points = points[labels != 1]
        labels = labels[labels != 1]

        contamination = merge_dopants_into_contamination(np.argmax(segmentation, axis=0)) == 1

        scale_factor = 16
        contamination = skimage.util.view_as_blocks(contamination, (scale_factor,) * 2).sum((-2, -1)) > (
                scale_factor ** 2 / 2)
        contamination = (np.array(np.where(contamination)).T) * 16 + 8

        points = np.vstack((points, contamination))
        labels = np.concatenate((labels, np.ones(len(contamination))))

        points = points.astype(np.float)
        points = points - padding[0]

        if sampling is not None:
            points *= self._training_sampling / sampling

        output = {'points': points[:, ::-1],
                  'labels': labels,
                  'density': density[0, 0].detach().cpu().numpy(),
                  'segmentation': segmentation}

        return output
