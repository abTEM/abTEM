from dask.utils import Dispatch
import scipy


def get_type(array):
    """Return type of arrays contained within the dask array chunks."""
    try:
        datatype = type(array._meta)  # Check chunk type backing dask array
    except AttributeError:
        datatype = type(array)  # For all non-dask arrays
    return datatype


class Dispatcher(Dispatch):
    """Simple single dispatch for different dask array types."""

    def __call__(self, arg, *args, **kwargs):
        """
        Call the corresponding method based on type of dask array.
        """
        datatype = get_type(arg)
        meth = self.dispatch(datatype)
        return meth(arg, *args, **kwargs)


dispatch_gaussian_filter = Dispatcher(name="dispatch_gaussian_filter")


@dispatch_gaussian_filter.register(np.ndarray)
def numpy_gaussian_filter(*args, **kwargs):
    return scipy.ndimage.filters.gaussian_filter


@dispatch_gaussian_filter.register_lazy("cupy")
def register_cupy_gaussian_filter():
    import cupy
    import cupyx.scipy.ndimage

    @dispatch_gaussian_filter.register(cupy.ndarray)
    def cupy_gaussian_filter(*args, **kwargs):
        return cupyx.scipy.ndimage.filters.gaussian_filter


image = np.random.rand(10, 10)
image = cp.random.rand(10, 10)
image = da.from_array(image)

dispatch_gaussian_filter(image)(image, 2)