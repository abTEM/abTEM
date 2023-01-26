from numbers import Number
from typing import Union, TYPE_CHECKING, Tuple

import numpy as np
from ase.cell import Cell
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from abtem.core.utils import label_to_index
from abtem.atoms import (
    is_cell_hexagonal,
    is_cell_orthogonal,
)

if TYPE_CHECKING:
    from abtem.measurements import DiffractionPatterns


def _frequency_bin_indices(shape):
    x = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1 / shape[0])).astype(int)
    y = np.fft.fftshift(np.fft.fftfreq(shape[1], d=1 / shape[1])).astype(int)
    x, y = np.meshgrid(x, y, indexing="ij")
    return np.array([x.ravel(), y.ravel()]).T


def _find_linearly_independent_row(array, row, tol: float = 1e-6):
    for other_row in array:
        A = np.row_stack([row, other_row])
        U, s, V = np.linalg.svd(A)
        if np.all(np.abs(s) > tol):
            break
    else:
        raise RuntimeError()

    return other_row


def _find_independent_spots(array):
    spots = array > array.max() * 1e-2
    half = array.shape[0] // 2, array.shape[1] // 2
    spots = spots[half[0]:, half[1]:]

    spots = np.array(np.where(spots)).T
    spot_0 = spots[0]
    spot_1 = _find_linearly_independent_row(spots[1:], spot_0)
    return spot_0, spot_1


def _planar_spacing_from_bin_index(index, sampling):
    d = 1 / np.sqrt((index[0] * sampling[0]) ** 2 + (index[1] * sampling[1]) ** 2)
    return d


def _planar_angle_from_bin_indices(index1, index2, sampling):
    v1 = index1[0] * sampling[0], index1[1] * sampling[1]
    v2 = index2[0] * sampling[0], index2[1] * sampling[1]
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def _orthorhombic_spacings(indices, d):
    g = (
            indices[:, None, None] ** 2 / d[0] ** 2
            + indices[None, :, None] ** 2 / d[1] ** 2
            + indices[None, None] ** 2 / d[2] ** 2
    )

    planes = np.zeros_like(g)
    planes[g > 0.0] = 1 / np.sqrt(g[g > 0.0])
    return planes


def _closest_indices(array, value):
    difference = np.abs(array - value)
    return np.array(np.where(difference == np.min(difference))).T


def _spacing_consistent_miller_indices(spacing, cell_edges):
    max_index = int(np.ceil(max(cell_edges) / spacing)) + 1

    indices = np.arange(-max_index, max_index + 1, 1)
    d = _orthorhombic_spacings(indices, cell_edges)
    return np.array([indices[i] for i in _closest_indices(d, spacing)])


def _planar_angle(hkl1, hkl2, cell_edges):
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    a, b, c = cell_edges
    d1 = 1 / a ** 2 * h1 ** 2 + 1 / b ** 2 * k1 ** 2 + 1 / c ** 2 * l1 ** 2
    d2 = 1 / a ** 2 * h2 ** 2 + 1 / b ** 2 * k2 ** 2 + 1 / c ** 2 * l2 ** 2
    d3 = 1 / a ** 2 * h1 * h2 + 1 / b ** 2 * k1 * k2 + 1 / c ** 2 * l1 * l2
    return np.arccos(d3 / np.sqrt(d1 * d2))


def _find_consistent_miller_index_pair(spacing_1, spacing_2, angle, cell_edges):
    hkl1 = _spacing_consistent_miller_indices(spacing_1, cell_edges)[0]
    hkl2 = _spacing_consistent_miller_indices(spacing_2, cell_edges)
    angles = np.array([_planar_angle(hkl1, x, cell_edges) for x in hkl2])
    return hkl1, hkl2[np.argmin(np.abs(angles - angle))]


def _bin_index_to_orthorhombic_miller(array, sampling, cell_edges):
    bin1, bin2 = _find_independent_spots(array)
    spacing1 = _planar_spacing_from_bin_index(bin1, sampling)
    spacing2 = _planar_spacing_from_bin_index(bin2, sampling)
    angle = _planar_angle_from_bin_indices(bin1, bin2, sampling)

    hkl1, hkl2 = _find_consistent_miller_index_pair(
        spacing1, spacing2, angle, cell_edges
    )
    return (bin1, bin2), (hkl1, hkl2)


def _validate_cell_edges(cell):
    if isinstance(cell, Number):
        cell = [cell]

    if not isinstance(cell, Cell):
        if len(cell) == 1:
            cell_edges = cell * 3
        elif len(cell) == 2:
            cell_edges = [cell[0]] * 2 + [cell[1]]
        elif len(cell) != 3:
            raise RuntimeError()
        else:
            raise RuntimeError()

        hexagonal = False

    elif is_cell_hexagonal(cell):
        lengths = cell.lengths()
        cell_edges = [lengths[0], np.sqrt(3) * lengths[1], lengths[2]]
        hexagonal = True
    elif is_cell_orthogonal(cell):
        cell_edges = list(np.diag(cell))
        hexagonal = False
    else:
        raise RuntimeError()

    return cell_edges, hexagonal


def _miller_to_miller_bravais(hkl):
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    HKIL = np.zeros((len(hkl), 4), dtype=int)
    HKIL[:, 0] = 2 * h - k
    HKIL[:, 1] = 2 * k - h
    HKIL[:, 2] = -HKIL[:, 0] - HKIL[:, 1]
    HKIL[:, 3] = l
    return HKIL


def _remap_vector_space(vector_space1, vector_space2, vectors):
    A = np.linalg.inv(np.array(vector_space1))
    bins_v = np.dot(vectors, A)
    return np.dot(bins_v, vector_space2)


def _map_all_bin_indices_to_miller_indices(array, sampling, cell, tolerance=1e-6):
    cell_edges, hexagonal = _validate_cell_edges(cell)

    (v1, v2), (u1, u2) = _bin_index_to_orthorhombic_miller(array, sampling, cell_edges)

    bins = _frequency_bin_indices(array.shape)
    hkl = _remap_vector_space((v1, v2), (u1, u2), bins)

    # remove fractional planes
    mask = np.all(np.abs(hkl - np.round(hkl)) < tolerance, axis=1)
    hkl = hkl[mask].astype(int)
    bins = bins[mask]

    if hexagonal:
        hkl[:, 1] = hkl[:, :-1].sum(axis=1) / 2
        hkl = _miller_to_miller_bravais(hkl)

    return bins, hkl


def _equivalent_miller_indices(hkl):
    is_negation = np.zeros((len(hkl), len(hkl)), dtype=bool)

    for i in range(hkl.shape[1]):
        negated = hkl.copy()
        negated[:, i] = -negated[:, i]
        is_negation += np.all(hkl[:, None] == negated[None], axis=2)

    is_negation += np.all(hkl[:, None] == -hkl[None], axis=2)

    sorted = np.sort(hkl, axis=1)
    is_permutation = np.all(sorted[:, None] == sorted[None], axis=-1)

    is_connected = is_negation + is_permutation

    n, labels = connected_components(csr_matrix(is_connected))

    return labels


def _split_at_threshold(values, threshold):
    order = np.argsort(values)
    max_value = values.max()

    split = (np.diff(values[order]) > (max_value * threshold)) * (
            np.diff(values[order]) > 1e-6
    )

    split = np.insert(split, 0, False)
    return np.cumsum(split)[np.argsort(order)]


def _find_equivalent_spots(hkl, intensities, intensity_split: float = 1.0):
    labels = _equivalent_miller_indices(hkl)

    spots = np.zeros(len(hkl), dtype=bool)
    for indices in label_to_index(labels):
        sub_labels = _split_at_threshold(intensities[indices], intensity_split)
        for sub_indices in label_to_index(sub_labels):
            order = np.lexsort(np.rot90(hkl[indices][sub_indices]))
            spots[indices[sub_indices[order][-1]]] = True

    return spots


def _index_diffraction_patterns(diffraction_patterns, cell, tol: float = 1e-6):
    if len(diffraction_patterns.shape) > 3:
        raise NotImplementedError
    elif len(diffraction_patterns.shape) == 3:
        array = diffraction_patterns.array.sum(-3)
        ensemble_shape = diffraction_patterns.array.shape[-3]
    else:
        array = diffraction_patterns.array
        ensemble_shape = 1

    bins, miller_indices = _map_all_bin_indices_to_miller_indices(
        array, diffraction_patterns.sampling, cell, tolerance=tol * ensemble_shape
    )

    unique, indices = np.unique(miller_indices, axis=0, return_index=True)
    miller_indices = miller_indices[indices]
    bins = bins[indices]

    all_intensities = diffraction_patterns._select_frequency_bin(bins)

    spots = {
        tuple(hkl): intensities
        for hkl, intensities in zip(miller_indices, all_intensities.T)
    }

    assert len(spots) == len(bins)
    return bins, spots


def tabulate_diffraction_pattern(
        diffraction_pattern,
        cell,
        return_data_frame: bool = False,
        normalize: bool = True,
        spot_threshold: float = 0.01,
        intensity_split: float = 1.0,
):
    # if len(diffraction_pattern.ensemble_shape) > 0:
    # raise NotImplementedError("tabulating not implemented for ensembles, select a single pattern by indexing")

    bins, hkl = _map_all_bin_indices_to_miller_indices(
        diffraction_pattern.array, diffraction_pattern.sampling, cell
    )

    intensities = diffraction_pattern._select_frequency_bin(bins)

    _, hexagonal = _validate_cell_edges(cell)
    include = _find_equivalent_spots(
        hkl,
        intensities=intensities,
        hexagonal=hexagonal,
        intensity_split=intensity_split,
    )
    hkl, intensities = hkl[include], intensities[include]

    if hexagonal:
        hkl = _miller_to_miller_bravais(hkl)

    order = np.lexsort(np.rot90(hkl))
    hkl, intensities = hkl[order], intensities[order]

    table = {
        "".join(map(str, list(hkli))): intensity
        for intensity, hkli in zip(intensities, hkl)
    }

    max_intensity = max(table.values())

    if normalize is True:
        normalization = max_intensity
    elif isinstance(normalize, str):
        normalization = table[normalize]
    else:
        normalization = 1.0

    table = {
        key: [value / normalization]
        for key, value in table.items()
        if value / max_intensity > spot_threshold
    }

    if return_data_frame:
        import pandas as pd

        return pd.DataFrame.from_dict(table)

    return table


class IndexedDiffractionPatterns:
    def __init__(self, spots: dict, vectors: np.ndarray):
        """
        Diffraction patterns indexed by their Miller indices.

        Parameters
        ----------
        spots : dict
            Dictionary mapping Miller (or Miller-Bravais) indices to diffraction spot intensities.
        vectors : np.ndarray
            The reciprocal space coordinates of the diffraction spots [1/Å].
        """

        self._spots = spots
        self._vectors = vectors

    @property
    def intensities(self) -> np.ndarray:
        return self._dict_to_arrays(self._spots)[1]

    @property
    def miller_indices(self) -> np.ndarray:
        return self._dict_to_arrays(self._spots)[0]

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def remove_equivalent(self, inequivalency_threshold: float = 1.0) -> "IndexedDiffractionPatterns":
        """
        Remove symmetry equivalent diffraction spots.

        Parameters
        ----------
        inequivalency_threshold : float
            Relative intensity difference to determine whether two symmetry-equivalent diffraction spots should be
            independently labeled (e.g. due to a unit cell with a basis of more than one element).
        """
        miller_indices, intensities = self._dict_to_arrays(self._spots)

        if len(intensities.shape) > 1:
            summed_intensities = intensities.sum(-1)
        else:
            summed_intensities = intensities

        include = _find_equivalent_spots(
            miller_indices,
            intensities=summed_intensities,
            intensity_split=inequivalency_threshold,
        )

        miller_indices, intensities = miller_indices[include], intensities[include]
        spots = self._arrays_to_dict(miller_indices, intensities)
        vectors = self._vectors[include]
        return self.__class__(spots, vectors)

    @property
    def ensemble_shape(self) -> tuple:
        return self.intensities.shape[:-1]

    def __getitem__(self, item):
        if not self.ensemble_shape:
            raise IndexError(
                "indexing not available for indexed diffraction pattern without ensemble dimension"
            )

        new_spots = {hkl: intensity[item] for hkl, intensity in self._spots.items()}

        return self.__class__(new_spots, self._vectors.copy())

    def remove_low_intensity(self, threshold: float = 1e-3):
        """
        Remove diffraction spots with intensity below a threshold.

        Parameters
        ----------
        threshold : float
            Relative intensity threshold for removing diffraction spots.

        """

        miller_indices, intensities = self._dict_to_arrays(self._spots)

        if len(intensities.shape) > 1:
            summed_intensities = intensities.sum(-1, keepdims=True)
        else:
            summed_intensities = intensities

        include = summed_intensities > threshold * summed_intensities.max()
        include = np.squeeze(include)

        vectors = self._vectors[include]
        miller_indices, intensities = miller_indices[include], intensities[include]
        spots = self._arrays_to_dict(miller_indices, intensities)
        return self.__class__(spots, vectors)

    @staticmethod
    def _dict_to_arrays(spots):
        miller_indices, intensities = zip(*spots.items())
        return np.array(miller_indices), np.array(intensities)

    @staticmethod
    def _arrays_to_dict(miller_indices, intensities):
        return dict(zip([tuple(hkl) for hkl in miller_indices], tuple(intensities)))

    @classmethod
    def index_diffraction_patterns(
            cls,
            diffraction_patterns: "DiffractionPatterns",
            cell: Union[Cell, float, Tuple[float, float, float]],
            tol: float = 1e-6,
    ) -> "IndexedDiffractionPatterns":

        bins, spots = _index_diffraction_patterns(diffraction_patterns, cell, tol=tol)

        vectors = bins * diffraction_patterns.sampling
        return cls(spots, vectors)

    def normalize_intensity(self, spot: tuple = None):
        """
        Normalize the intensity of the diffraction spots.

        Parameters
        ----------
        spot :

        Returns
        -------

        """

        if spot is None:
            c = self.intensities.max()
        else:
            c = self._spots[spot]

        spots = {hkl: intensity / c for hkl, intensity in self._spots.items()}

        return self.__class__(spots, self._vectors)

    def to_dataframe(
            self,
            intensity_threshold: float = 1e-3,
            inequivalency_threshold: float = 1.0,
            normalize: bool = False,
    ):
        """
        Convert the indexed diffraction to pandas dataframe.

        Parameters
        ----------
        intensity_threshold : float
            Relative intensity threshold for removing diffraction spots from the dataframe.
        inequivalency_threshold : float
            Relative intensity difference to determine whether two symmetry-equivalent diffraction spots should be
            independently labeled (e.g. due to a unit cell with a basis of more than one element).
        normalize : bool
            If True, normalize

        Returns
        -------
        dataframe_with_spots : pd.DataFrame

        """

        import pandas as pd

        indexed = self.remove_equivalent(inequivalency_threshold=inequivalency_threshold)

        indexed = indexed.remove_low_intensity(intensity_threshold)

        if normalize is True:
            indexed = indexed.normalize_intensity(spot=None)
        elif normalize is not False:
            indexed = indexed.normalize_intensity(spot=normalize)

        spots = {
            "".join(map(str, list(hkl))): intensity
            for hkl, intensity in indexed._spots.items()
        }
        spots = dict(sorted(spots.items()))

        try:
            return pd.DataFrame(spots)
        except ValueError:
            return pd.DataFrame(spots, index=[0])

    def show(self, **kwargs):
        """


        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        from abtem.visualize import _show_indexed_diffraction_pattern

        indexed_diffraction_patterns = self
        if self.ensemble_shape:
            indexed_diffraction_patterns = indexed_diffraction_patterns[
                (0,) * len(self.ensemble_shape)
                ]

        return _show_indexed_diffraction_pattern(indexed_diffraction_patterns, **kwargs)
