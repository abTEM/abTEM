from numbers import Number


class HasAxesMetadata:
    _axes_metadata: list

    @property
    def axes_metadata(self):
        return self._axes_metadata

    def _type_indices(self, key):
        axes_indices = ()
        for axes_index, axes_metadata in enumerate(self._axes_metadata):
            if axes_metadata['type'] == key:
                axes_indices += (axes_index,)

        return axes_indices

    @property
    def scan_axes(self):
        return self._type_indices('scan')

    @property
    def num_scan_axes(self):
        return len(self._type_indices('scan'))

    def _remove_axes_metadata(self, axes):
        if isinstance(axes, Number):
            axes = (axes,)

        return [element for i, element in enumerate(self._axes_metadata) if not i in axes]

    def _axis_by_key(self):
        pass
