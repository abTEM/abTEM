import numpy as np
from scipy import ndimage


def center_of_mass(data):
    com = np.zeros(data.shape[:2] + (2,))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            com[i, j] = ndimage.measurements.center_of_mass(data[i, j])

    return com
