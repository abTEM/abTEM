from numbers import Number

frozen_phonons_axes_metadata = {'label': 'frozen_phonons', 'type': 'ensemble'}


class HasAxesMetadata:
    _extra_axes_metadata: list
    _base_axes_metadata: list
    num_base_axes: int
    num_axes: int

    def _type_indices(self, keys):
        axes_indices = ()
        for axes_index, axes_metadata in enumerate(self.axes_metadata):
            if axes_metadata['type'] in keys:
                axes_indices += (axes_index,)

        return axes_indices

    def _validate_extra_axes_metadata(self, extra_axes_metadata):
        if extra_axes_metadata is None:
            extra_axes_metadata = []

        missing_extra_axes_metadata = self.num_axes - len(extra_axes_metadata) - self.num_base_axes

        extra_axes_metadata = [{'type': 'unknown'} for _ in range(missing_extra_axes_metadata)] + extra_axes_metadata

        if (self.num_base_axes + len(extra_axes_metadata)) != self.num_axes:
            raise RuntimeError()

        return extra_axes_metadata

    @property
    def num_extra_axes(self):
        return len(self._extra_axes_metadata)

    @property
    def axes_metadata(self):
        return self._extra_axes_metadata + self._base_axes_metadata

    @property
    def real_space_axes(self):
        return self._type_indices(('scan', 'linescan', 'gridscan', 'positions'))

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
