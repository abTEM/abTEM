from __future__ import annotations

from abc import abstractmethod, ABCMeta

import numpy as np

from abtem.visualize.artists import (
    ImageArtist,
    DomainColoringArtist,
    LinesArtist,
    ScatterArtist,
)
from abtem.visualize.widgets import ImageGUI, BaseGUI


def reduce_complex(array, method):
    if method in ("domain_coloring", "none", None):
        return array

    if method in ("phase", "angle"):
        array = np.angle(array)
    elif method in ("amplitude", "abs"):
        array = np.abs(array)
    elif method in ("intensity", "abs2"):
        array = np.abs(array) ** 2
    elif method in ("real",):
        array = np.real(array)
    elif method in ("imaginary", "imag"):
        array = np.imag(array)
    else:
        raise ValueError(f"{method} is not")

    return array


def _validate_axes_types(overlay, explode, ensemble_dims):
    if explode is True:
        explode = tuple(range(ensemble_dims))
    elif explode is False:
        explode = ()

    if overlay is True:
        overlay = tuple(range(ensemble_dims))
    elif overlay is False:
        overlay = ()

    if len(overlay + explode) > ensemble_dims:
        raise ValueError

    if len(set(explode) & set(overlay)) > 0:
        raise ValueError("An axis cannot be both exploded and overlaid.")

    return overlay, explode


class VisualizationData(metaclass=ABCMeta):
    def __init__(self, shape, complex_conversion, base_dims, overlay=(), explode=()):
        ensemble_dims = len(shape) - base_dims
        ensemble_shape = shape[:ensemble_dims]

        overlay, explode = _validate_axes_types(overlay, explode, ensemble_dims)

        self._overlay = overlay
        self._explode = explode
        self._ensemble_shape = ensemble_shape
        self._complex_conversion = complex_conversion

        base_axes = set(range(ensemble_dims, len(shape)))
        ensemble_axes = set(range(ensemble_dims))

        self._indexing_axes = tuple(ensemble_axes - set(overlay) - set(explode))

        if set(explode + overlay) & base_axes:
            raise ValueError("")

    @property
    def axes_shape(self):
        return tuple(
            n
            for i, n in enumerate(self._ensemble_shape)
            if not i in self._indexing_axes and i not in self.overlay
        )

    def _reduce_arrays(
        self, *arrays, indices: tuple[int | tuple[int, int], ...], axis_indices
    ) -> tuple:
        assert len(indices) <= len(self._indexing_axes)
        assert len(axis_indices) == 2

        validated_indices = ()
        summed_axes = ()
        removed_axes = 0
        j = 0
        k = 0
        for i in range(len(self.ensemble_shape)):
            if i in self._indexing_axes:
                if j >= len(indices):
                    validated_indices += (0,)
                elif isinstance(indices[j], int):
                    validated_indices += (indices[j],)
                    removed_axes += 1
                elif isinstance(indices[j], tuple):
                    validated_indices += (slice(*indices[j]),)
                    summed_axes += (i - removed_axes,)
                j += 1
            elif i in self.explode:
                validated_indices += (axis_indices[k],)
                k += 1
                removed_axes += 1
            elif i not in self.overlay:
                validated_indices += (0,)

        output = ()
        for array in arrays:
            array = array[validated_indices]
            if len(summed_axes) > 0:
                array = np.sum(array, axis=summed_axes)

            output += (array,)
        return output

    @property
    def overlay(self):
        return self._overlay

    @property
    def explode(self):
        return self._explode

    @property
    def ensemble_shape(self):
        return self._ensemble_shape

    @property
    def index_axes(self):
        return self._indexing_axes

    @property
    @abstractmethod
    def is_complex(self):
        pass

    @property
    def complex_out(self):
        return self.is_complex and self.complex_conversion == "none"

    @property
    def complex_conversion(self):
        return self._complex_conversion

    @complex_conversion.setter
    def complex_conversion(self, value):
        self._complex_conversion = value

    @abstractmethod
    def get_data_for_indices(self, index, axis_indices):
        pass

    @abstractmethod
    def get_artist(self):
        pass

    @abstractmethod
    def get_gui(self):
        pass


class LinesData(VisualizationData):
    def __init__(self, x, y, complex_conversion: str = "none", overlay=(), explode=()):
        self._x = x
        self._y = y
        super().__init__(
            shape=y.shape,
            complex_conversion=complex_conversion,
            base_dims=1,
            overlay=overlay,
            explode=explode,
        )

    @property
    def is_complex(self):
        return np.iscomplexobj(self._y)

    def get_artist(self):
        if self.complex_out:
            raise NotImplementedError
        else:
            return LinesArtist

    def get_gui(self):
        return BaseGUI

    def get_data_for_indices(self, indices, axis_indices):
        (y,) = self._reduce_arrays(self._y, indices=indices, axis_indices=axis_indices)
        y = reduce_complex(y, self.complex_conversion)
        return self.__class__(
            x=self._x, y=y, complex_conversion=self.complex_conversion
        )


class ImageData(VisualizationData):
    def __init__(
        self,
        array,
        complex_conversion: str = "none",
        overlay=(),
        explode=(),
        value_label: str = None,
    ):
        self._array = array
        super().__init__(
            shape=array.shape,
            complex_conversion=complex_conversion,
            base_dims=2,
            overlay=overlay,
            explode=explode,
        )

    @property
    def array(self):
        return self._array

    @property
    def is_complex(self):
        return np.iscomplexobj(self._array)

    def get_value_label(self):
        pass

    def get_artist(self):
        if self.complex_out:
            return DomainColoringArtist
        else:
            return ImageArtist

    def get_gui(self):
        return ImageGUI

    def get_data_for_indices(self, indices, axis_indices):
        (array,) = self._reduce_arrays(
            self._array, indices=indices, axis_indices=axis_indices
        )
        array = reduce_complex(array, self.complex_conversion)
        return self.__class__(array, complex_conversion=self.complex_conversion)


class PointsData(VisualizationData):
    def __init__(
        self,
        points,
        array,
        complex_conversion: str = "none",
        overlay=(),
        explode=(),
    ):
        self._points = points
        self._array = array

        super().__init__(
            shape=array.shape,
            complex_conversion=complex_conversion,
            base_dims=1,
            overlay=overlay,
            explode=explode,
        )

    @property
    def is_complex(self):
        return np.iscomplexobj(self._array)

    @property
    def array(self):
        return self._array

    @property
    def points(self):
        return self._points

    def get_artist(self):
        if self.complex_out:
            raise NotImplementedError
        else:
            return ScatterArtist

    @property
    def ensemble_shape(self):
        return self._array.shape[:-1]

    def get_data_for_indices(self, indices, axis_indices):
        points, array = self._reduce_arrays(
            self._points, self._array, indices=indices, axis_indices=axis_indices
        )

        array = reduce_complex(self._array, self.complex_conversion)
        return self.__class__(points, array, complex_conversion=self.complex_conversion)
