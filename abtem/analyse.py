from itertools import combinations

import numpy as np
from ase.symbols import Symbols
from ase.symbols import string2symbols, symbols2numbers
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize_scalar


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


def cluster_columns(atoms, tol=1e-6, longitudinal_ordering=False):
    xy = atoms.get_positions()[:, :2]
    z = atoms.get_positions()[:, 2]
    column_labels = fcluster(linkage(xy), tol, criterion='distance')

    n_columns = len(np.unique(column_labels))
    positions = np.zeros((n_columns, 2), dtype=np.float)
    labels = np.zeros(n_columns, dtype=np.int)
    column_types = []

    for i, indices in enumerate(generate_indices(column_labels)):
        numbers = atoms.get_atomic_numbers()[indices]

        if longitudinal_ordering:
            order = np.argsort(z[indices])
            numbers = numbers[order]
        else:
            numbers = np.sort(numbers)

        key = Symbols(numbers).get_chemical_formula()

        positions[i] = np.mean(atoms.get_positions()[indices, :2], axis=0)

        try:
            labels[i] = column_types.index(key)

        except ValueError:
            column_types.append(key)

            labels[i] = len(column_types) - 1

    return positions, labels, column_types


def intensity_ratios(intensities):
    ratios = {}
    for combination in combinations(list(intensities.keys()), 2):
        intensity_1 = np.mean(intensities[combination[0]])
        intensity_2 = np.mean(intensities[combination[1]])

        order = np.argsort((intensity_1, intensity_2))
        ratio = (intensity_1, intensity_2)[order[0]] / (intensity_1, intensity_2)[order[1]]
        ratios['/'.join([combination[i] for i in order])] = ratio

    return ratios


def fit_powerlaw(ratios):
    powerlaw = {}
    for ratio_name, ratio in ratios.items():
        Z_1 = sum(symbols2numbers(string2symbols(ratio_name.split('/')[0])))
        Z_2 = sum(symbols2numbers(string2symbols(ratio_name.split('/')[1])))

        eq = lambda n: ((Z_1 / Z_2) ** n - ratio) ** 2

        powerlaw[ratio_name] = minimize_scalar(eq).x

    return powerlaw


def center_of_mass(data):
    shape = data.shape[2:]
    center = np.array(shape) / 2 - [.5 * (shape[0] % 2), .5 * (shape[1] % 2)]
    com = np.zeros(data.shape[:2] + (2,))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            com[i, j] = ndimage.measurements.center_of_mass(data[i, j])

    return com - center[None, None]


def fwhm(probe):
    array = probe.build().array
    y = array[0, array.shape[1] // 2]
    peak_idx = np.argmax(y)
    peak_value = y[peak_idx]
    left = np.argmin(np.abs(y[:peak_idx] - peak_value / 2))
    right = peak_idx + np.argmin(np.abs(y[peak_idx:] - peak_value / 2))
    return (right - left) * probe.sampling[0]

# def spectrogram(image):
