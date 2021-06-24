class HasDeviceMixin:
    _device: str

    @property
    def device(self):
        return self._device

    @property
    def calculation_device(self):
        return self._device
