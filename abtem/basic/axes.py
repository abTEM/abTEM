from numbers import Number


class HasAxesMetadata:
    axes_metadata: list

    def _type_indices(self, keys):
        axes_indices = ()
        for axes_index, axes_metadata in enumerate(self.axes_metadata):
            if axes_metadata['type'] in keys:
                axes_indices += (axes_index,)

        return axes_indices

    @property
    def scan_axes(self):
        return self._type_indices(('scan', 'linescan', 'gridscan'))

    @property
    def num_scan_axes(self):
        return len(self.scan_axes)

    @property
    def ensemble_axes(self):
        return self._type_indices(('ensemble',))

    @property
    def num_ensemble_axes(self):
        return len(self.ensemble_axes)

    def _remove_axes_metadata(self, axes):
        if isinstance(axes, Number):
            axes = (axes,)

        return [element for i, element in enumerate(self.axes_metadata) if not i in axes]

    def _axis_by_key(self):
        pass
