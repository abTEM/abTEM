import matplotlib.pyplot as plt
import numpy as np

from abtem.measure import Measurement, calibrations_from_grid
from abtem.utils import fourier_translation_operator
from abtem.waves import Probe
from scipy.ndimage import center_of_mass


def fft_shift(array, positions):
    return np.fft.ifft2(np.fft.fft2(array) * fourier_translation_operator(positions, array.shape))


def _run_epie(object, probe, diffraction_patterns, positions, maxiter, alpha=1., beta=1., fix_probe=False,
              fix_com=False,
              return_iterations=False,
              verbose=True):
    object = np.array(object)
    probe = np.array(probe)
    diffraction_patterns = np.array(diffraction_patterns)

    if len(diffraction_patterns.shape) != 3:
        raise ValueError()

    if len(diffraction_patterns) != len(positions):
        raise ValueError()

    if object.shape == (2,):
        object = np.ones(object, dtype=np.complex64)
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

    k = 0
    SSE = 0.
    while k < maxiter:
        indices = np.arange(len(positions))
        np.random.shuffle(indices)

        SSE = 0.
        for j in indices:
            position = positions[j]
            illuminated_object = fft_shift(object, - position)

            g = illuminated_object * probe
            G = np.fft.fftshift(np.fft.fft2(g))
            Gprime = np.sqrt(diffraction_patterns[j]) * np.exp(1j * np.angle(G))
            gprime = np.fft.ifft2(np.fft.ifftshift(Gprime))

            shifted_object = illuminated_object + alpha * (gprime - g) * np.conj(probe) / (np.max(np.abs(probe)) ** 2)
            object = fft_shift(shifted_object, position)

            if not fix_probe:
                probe = probe + beta * (gprime - g) * np.conj(illuminated_object) / (
                        np.max(np.abs(illuminated_object)) ** 2)

            SSE += np.sum(np.abs(G) ** 2 - diffraction_patterns[j]) ** 2

        if fix_com:
            com = center_of_mass(np.fft.fftshift(np.abs(probe) ** 2))
            probe = np.fft.ifftshift(fft_shift(probe, - np.array(com)))

        SSE = SSE / np.prod(diffraction_patterns.shape)

        if return_iterations:
            object_iterations.append(object)
            probe_iterations.append(probe)
            SSE_iterations.append(SSE)

        if verbose:
            print(f'Iteration {k:<{len(str(maxiter))}}, SSE = {SSE:.3e}')

        k += 1

    if return_iterations:
        return object_iterations, probe_iterations, SSE_iterations
    else:
        return object, probe, SSE


def epie(measurement: Measurement, probe_guess: Probe, maxiter: int = 5, alpha: float = 1., beta: float = 1.,
         fix_probe=False,
         fix_com=False,
         return_iterations: bool = False, verbose=True):
    diffraction_patterns = measurement.array.reshape((-1,) + measurement.array.shape[2:])

    extent = (probe_guess.wavelength * 1e3 / measurement.calibrations[2].sampling,
              probe_guess.wavelength * 1e3 / measurement.calibrations[3].sampling)

    sampling = (extent[0] / measurement.shape[2],
                extent[0] / measurement.shape[3])

    x = measurement.calibrations[0].coordinates(measurement.shape[0]) / sampling[0]
    y = measurement.calibrations[1].coordinates(measurement.shape[1]) / sampling[1]
    x, y = np.meshgrid(x, y, indexing='ij')
    positions = np.array([x.ravel(), y.ravel()]).T

    probe_guess.extent = extent
    probe_guess.gpts = measurement.shape[2:]
    calibrations = calibrations_from_grid(probe_guess.gpts, probe_guess.sampling, names=['x', 'y'], units='Å')

    probe_guess = probe_guess.build((0, 0)).array[0]

    result = _run_epie(measurement.shape[2:],
                       probe_guess,
                       diffraction_patterns,
                       positions,
                       maxiter=maxiter,
                       alpha=alpha,
                       beta=beta,
                       return_iterations=return_iterations,
                       fix_probe=fix_probe,
                       fix_com=fix_com,
                       verbose=verbose)

    if return_iterations:
        object_iterations = [Measurement(object, calibrations=calibrations) for object in result[0]]
        probe_iterations = [Measurement(np.fft.fftshift(probe), calibrations=calibrations) for probe in result[1]]
        return object_iterations, probe_iterations, result[2]
    else:
        return (Measurement(result[0], calibrations=calibrations),
                Measurement(np.fft.fftshift(result[1]), calibrations=calibrations),
                result[2])
