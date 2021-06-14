import xarray as xr
import matplotlib.pyplot as plt

@xr.register_dataarray_accessor("abtem")
class abTEMAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, **kwargs):

        self._obj.plot(**kwargs)

        plt.gca().set_aspect('equal')
