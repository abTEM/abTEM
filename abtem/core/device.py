class HasDeviceMixin:
    _device: str

    @property
    def device(self) -> str:
        """

        """
        return self._device