"""Module for applying Modulation Transfer function."""
import numpy as np
from abtem.utils import spatial_frequencies
from abtem.measure import Measurement

def default_mtf_func(k: np.ndarray, c0: float, c1: float, c2: float, c3: float):
    """
    A default MTF function

    Parameters
    ----------
    k : float
        Spatial frequency
    c : float
        Coefficients for function

    Returns
    -------
    float
        Result from MTF
    """
    return (c0 - c1) / (1 + (k / (2 * c2)) ** np.abs(c3)) + c1


class MTF:
    """
    MTF Object for applying a specified MTF function with specified parameters
    Parameters
    ----------
    func: function
        The Modulation Transfer Function
    kwargs:
        Provide the MTF parameters as keyword arguments.
    measurement: Measurement object
        The measurement to apply MTF to

    Returns
    -------
    measurement: Measurement object
        The MFT applied measurement
    """

    def __init__(self, func: callable = None, **kwargs):
        if func is None:
            self.f = default_mtf_func
        else:
            self.f = func
        self.params = kwargs

    def __call__(self, measurement) -> Measurement:
        """
        Apply a modulation transfer function to the image.

        Parameters
        ----------
        measurement : Measurement object
            The original Measurement

        Returns
        -------
        measurement: Measurement object
            The MFT applied measurement
        """
        # Get sampling from measurement
        sampling = []
        for calibration in measurement.calibrations:
            if calibration is not None:
                if calibration.units.lower() in ('angstrom', 'å'):
                    sampling.append(calibration.sampling)

        # Get number of gridpoints from measurement
        if len(measurement.array.shape) == 2:
            gpts = (measurement.array.shape[0], measurement.array.shape[1])
        else:
            gpts = (measurement.array.shape[1], measurement.array.shape[2])

        # Get measurement array   
        measurement = measurement.copy()
        img = measurement.array

        # Get spatial frequencies
        kx, ky = spatial_frequencies(gpts, sampling)
        
        # Create 2D grid
        Ky, Kx = np.meshgrid(ky,kx)
        K = np.sqrt(Kx ** 2 + Ky ** 2)

        # Compute MTF
        mtf = self.f(K, **self.params)

        # Apply MTF
        img = np.fft.ifft2(np.fft.fft2(img) * mtf)
        measurement.array[:] = img.real

        return measurement
