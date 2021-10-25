import numpy as np

from abtem.device import get_array_module, get_scipy_module
from abtem.measure import Measurement, calibrations_from_grid
from abtem.utils import ProgressBar, fft_shift
from abtem.waves import Probe


def _run_epie(object,
              probe: np.ndarray,
              diffraction_patterns: np.ndarray,
              positions: np.ndarray,
              maxiter: int,
              alpha: float = 1.,
              beta: float = 1.,
              fix_probe: bool = False,
              fix_com: bool = False,
              return_iterations: bool = False,
              seed=None):
    xp = get_array_module(probe)

    object = xp.array(object)
    probe = xp.array(probe)

    if len(diffraction_patterns.shape) != 3:
        raise ValueError()

    if len(diffraction_patterns) != len(positions):
        raise ValueError()

    if object.shape == (2,):
        object = xp.ones((int(object[0]), int(object[1])), dtype=xp.complex64)
    elif len(object.shape) != 2:
        raise ValueError()

    if probe.shape != diffraction_patterns.shape[1:]:
        raise ValueError()

    if probe.shape != object.shape:
        raise ValueError()

    if return_iterations:
        object_iterations = []
        probe_iterations = []
        SSE_iterations = []

    if seed is not None:
        np.random.seed(seed)

    diffraction_patterns = np.fft.ifftshift(np.sqrt(diffraction_patterns), axes=(-2, -1))

    SSE = 0.
    k = 0
    outer_pbar = ProgressBar(total=maxiter)
    inner_pbar = ProgressBar(total=len(positions))

    center_of_mass = get_scipy_module(xp).ndimage.center_of_mass

    while k < maxiter:
        indices = np.arange(len(positions))
        np.random.shuffle(indices)

        old_position = xp.array((0., 0.))
        inner_pbar.reset()
        SSE = 0.
        for j in indices:
            position = xp.array(positions[j])

            diffraction_pattern = xp.array(diffraction_patterns[j])
            illuminated_object = fft_shift(object, old_position - position)

            g = illuminated_object * probe
            gprime = xp.fft.ifft2(diffraction_pattern * xp.exp(1j * xp.angle(xp.fft.fft2(g))))

            object = illuminated_object + alpha * (gprime - g) * xp.conj(probe) / (xp.max(xp.abs(probe)) ** 2)
            old_position = position

            if not fix_probe:
                probe = probe + beta * (gprime - g) * xp.conj(illuminated_object) / (
                        xp.max(xp.abs(illuminated_object)) ** 2)

            # SSE += xp.sum(xp.abs(G) ** 2 - diffraction_pattern) ** 2

            inner_pbar.update(1)

        object = fft_shift(object, position)

        if fix_com:
            com = center_of_mass(xp.fft.fftshift(xp.abs(probe) ** 2))
            probe = xp.fft.ifftshift(fft_shift(probe, - xp.array(com)))

        # SSE = SSE / np.prod(diffraction_patterns.shape)

        # print(SSE)

        if return_iterations:
            object_iterations.append(object)
            probe_iterations.append(probe)
            SSE_iterations.append(SSE)

        outer_pbar.update(1)
        # if verbose:
        #    print(f'Iteration {k:<{len(str(maxiter))}}, SSE = {float(SSE):.3e}')

        k += 1

    inner_pbar.close()
    outer_pbar.close()

    if return_iterations:
        return object_iterations, probe_iterations, SSE_iterations
    else:
        return object, probe, SSE


def epie(measurement: Measurement,
         probe_guess: Probe,
         maxiter: int = 5,
         alpha: float = 1.,
         beta: float = 1.,
         fix_probe: bool = False,
         fix_com: bool = False,
         return_iterations: bool = False,
         max_angle: float = None,
         crop_to_valid: bool = False,
         seed: int = None,
         device: str = 'cpu', ):
    """
    Reconstruct the phase of a 4D-STEM measurement using the extended Ptychographical Iterative Engine.

    See https://doi.org/10.1016/j.ultramic.2009.05.012

    Parameters
    ----------
    measurement : Measurement object
        4D-STEM measurement.
    probe_guess : Probe object
        The initial guess for the probe.
    maxiter : int
        Run the algorithm for this many iterations.
    alpha : float
        Controls the size of the iterative updates for the object. See reference.
    beta : float
        Controls the size of the iterative updates for the probe. See reference.
    fix_probe : bool
        If True, the probe will not be updated by the algorithm. Default is False.
    fix_com : bool
        If True, the center of mass of the probe will be centered. Default is True.
    return_iterations : bool
        If True, return the reconstruction after every iteration. Default is False.
    max_angle : float, optional
        The maximum reconstructed scattering angle. If this is larger than the input data, the data will be zero-padded.
    crop_to_valid : bool
        If true, the output is cropped to the scan area.
    seed : int, optional
        Seed the random number generator.
    device : str
        Set the calculation device.

    Returns
    -------
    List of Measurement objects

    """

    diffraction_patterns = measurement.array.reshape((-1,) + measurement.array.shape[2:])

    if max_angle:
        padding_x = int((max_angle / abs(measurement.calibrations[-2].offset) *
                         diffraction_patterns.shape[-2]) // 2) - diffraction_patterns.shape[-2] // 2
        padding_y = int((max_angle / abs(measurement.calibrations[-1].offset) *
                         diffraction_patterns.shape[-1]) // 2) - diffraction_patterns.shape[-1] // 2
        diffraction_patterns = np.pad(diffraction_patterns, ((0,) * 2, (padding_x,) * 2, (padding_y,) * 2))

    extent = (probe_guess.wavelength * 1e3 / measurement.calibrations[2].sampling,
              probe_guess.wavelength * 1e3 / measurement.calibrations[3].sampling)

    sampling = (extent[0] / diffraction_patterns.shape[-2],
                extent[1] / diffraction_patterns.shape[-1])

    x = measurement.calibrations[0].coordinates(measurement.shape[0]) / sampling[0]
    y = measurement.calibrations[1].coordinates(measurement.shape[1]) / sampling[1]
    x, y = np.meshgrid(x, y, indexing='ij')
    positions = np.array([x.ravel(), y.ravel()]).T

    probe_guess.extent = extent
    probe_guess.gpts = diffraction_patterns.shape[-2:]

    calibrations = calibrations_from_grid(probe_guess.gpts, probe_guess.sampling, names=['x', 'y'], units='Å')

    probe_guess._device = device
    probe_guess = probe_guess.build(np.array([0, 0])).array[0]

    result = _run_epie(diffraction_patterns.shape[-2:],
                       probe_guess,
                       diffraction_patterns,
                       positions,
                       maxiter=maxiter,
                       alpha=alpha,
                       beta=beta,
                       return_iterations=return_iterations,
                       fix_probe=fix_probe,
                       fix_com=fix_com,
                       seed=seed)

    valid_extent = (measurement.calibration_limits[0][1] - measurement.calibration_limits[0][0],
                    measurement.calibration_limits[1][1] - measurement.calibration_limits[1][0])

    if return_iterations:
        object_iterations = [Measurement(obj, calibrations=calibrations) for obj in result[0]]
        probe_iterations = [Measurement(np.fft.fftshift(probe), calibrations=calibrations) for probe in result[1]]

        if crop_to_valid:
            object_iterations = [object_iteration.crop(valid_extent) for object_iteration in object_iterations]

        return object_iterations, probe_iterations, result[2]
    else:
        object = Measurement(result[0], calibrations=calibrations)

        if crop_to_valid:
            result = object.crop(valid_extent)

        return (object,
                Measurement(np.fft.fftshift(result[1]), calibrations=calibrations),
                result[2])
