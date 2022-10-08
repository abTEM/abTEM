"""Module describing the detector modulation transfer function (MTF)."""
import numpy as np

def default_mtf_func(k: np.ndarray, c0: float, c1: float, c2: float, c3: float):
    """
    A default modulation transfer function (MTF).

    Parameters
    ----------
    k : float or np.ndarray
        Spatial frequencies at which to evaluate the MTF.
    c : float
        Coefficients for the MTF.

    Returns
    -------
    mtf : float or np.ndarray
        Modulation transfer function for given spatial frequencies.
    """
    return (c0 - c1) / (1 + (k / (2 * c2)) ** np.abs(c3)) + c1


class MTF:
    """
    Apply a modulation transfer function (MTF) with specified parameters.

    Parameters
    ----------
    func : function, optional
        The modulation transfer function. If not specified, uses the default MTF.
    kwargs :
        Provide the MTF parameters as keyword arguments.
    """
    def __init__(self, func: callable = None, **kwargs):
        if func is None:
            self.f = default_mtf_func
        else:
            self.f = func
        self.params = kwargs

    def __call__(self, measurement):
        """
        Apply a modulation transfer function to a given measurement.

        Parameters
        ----------
        measurement : BaseMeasurement
            The original measurement.

        Returns
        -------
        modulated_measurement : BaseMeasurement
            The measurement with the MTF applied.
        """
        # Get sampling from measurement
        sampling = []
        for calibration in measurement.calibrations:
            if calibration is not None:
                if calibration.units.lower() in ('angstrom', 'å'):
                    sampling.append(calibration.sampling)

        # Get number of grid points from measurement
        if len(measurement.array.shape) == 2:
            gpts = (measurement.array.shape[0], measurement.array.shape[1])
        else:
            gpts = (measurement.array.shape[1], measurement.array.shape[2])

        # Get measurement array   
        measurement = measurement.copy()
        img = measurement.array

        # Get spatial frequencies
        kx, ky = spatial_frequencies(gpts, sampling)
        k = np.sqrt(kx ** 2 + ky ** 2)

        # Compute MTF
        mtf = self.f(k, **self.params)

        # Apply MTF
        img = np.fft.ifft2(np.fft.fft2(img) * np.sqrt(mtf))
        measurement.array[:] = (img.real + img.imag) / 2

        return measurement
