import numpy as np

from abtem.learn.dataset import gaussian_marker_labels
from abtem.learn.scale import find_ring
from abtem.learn.structures import graphene_like
from abtem.points import fill_rectangle, rotate


def test_scale_detection():
    points = graphene_like()
    points = rotate(points, 15, rotate_cell=True)

    points = fill_rectangle(points, origin=[0, 0], extent=[80, 80], margin=4)
    image = gaussian_marker_labels(points, .4, (512, 512))

    assert np.isclose(find_ring(image, 2.46), 80. / 512, 80. / 512 * .1)
