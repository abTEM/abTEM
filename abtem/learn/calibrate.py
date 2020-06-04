import numpy as np
from psm.geometry import regular_polygon, polygon_area
from psm.graph import stable_delaunay_faces
from psm.rmsd import pairwise_rmsd
from scipy.ndimage import zoom


class RealSpaceCalibrator:

    def __init__(self, model, template, alpha, rmsd_max, min_sampling, max_sampling, step_size=.01):
        self._model = model
        self._template = template
        self._alpha = alpha
        self._rmsd_max = rmsd_max
        self._min_sampling = min_sampling
        self._max_sampling = max_sampling
        self._steps = int(np.ceil((max_sampling - min_sampling) / step_size))

    def __call__(self, image):
        reference_area = polygon_area(self._template)
        max_valid = 0
        min_rmsd = np.inf
        pred_sampling = None

        for sampling in np.linspace(self._min_sampling, self._max_sampling, self._steps):

            scale_factor = sampling / self._model.training_sampling
            rescaled = zoom(image, scale_factor, order=1)

            output = self._model(rescaled)

            points = output['points']

            if len(points) < 3:
                continue

            faces = stable_delaunay_faces(points, self._alpha)

            segments = [points[face] for face in faces]
            rmsd = pairwise_rmsd([self._template], segments).ravel()

            valid = rmsd < self._rmsd_max
            num_valid = np.sum(valid)
            if num_valid == 0:
                continue

            if num_valid >= max_valid:
                if rmsd[valid].mean() < min_rmsd:
                    area = np.mean([polygon_area(segment) for i, segment in enumerate(segments) if valid[i]])
                    pred_sampling = scale_factor * self._model.training_sampling / np.sqrt(reference_area / area)

                    max_valid = num_valid
                    min_rmsd = rmsd[valid].mean()

        return pred_sampling


class GrapheneCalibrator(RealSpaceCalibrator):

    def __init__(self, model, min_sampling, max_sampling, step_size=.005):
        template = regular_polygon(1.42 / model.training_sampling, 6)
        super().__init__(model=model, template=template, alpha=2, rmsd_max=.05, min_sampling=min_sampling,
                         max_sampling=max_sampling, step_size=step_size)
