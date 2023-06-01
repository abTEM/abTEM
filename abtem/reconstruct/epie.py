from typing import Tuple

import numpy as np
from scipy.spatial.distance import squareform

from abtem.core.backend import get_array_module, get_ndimage_module
from abtem.core.chunks import validate_chunks, iterate_chunk_ranges
from abtem.core.fft import fft_shift
from abtem.measurements import DiffractionPatterns, Images
from abtem.waves import Probe


# # max_batch = 8
# # chunks = validate_chunks((len(positions),), (max_batch,))
# # import matplotlib.pyplot as plt
# # import cupy as cp
# k = 0
# while k < maxiter:
#     indices = np.arange(len(positions))
#     np.random.shuffle(indices)
#
#     old_position = xp.array((0., 0.))
#     inner_pbar.reset()
#
#     # for _, slic in iterate_chunk_ranges(chunks):
#     #
#     #     ind = indices[slic]
#     #
#     #     batch_positions = xp.asarray(positions[indices[slic]])
#     #
#     #     diffraction_pattern = xp.array(diffraction_patterns[ind])
#     #     illuminated_object = fft_shift(object, - batch_positions)
#     #
#     #     g = illuminated_object * probe
#     #     gprime = xp.fft.ifft2(diffraction_pattern * xp.exp(1j * xp.angle(xp.fft.fft2(g))))
#     #
#     #     object = illuminated_object + alpha * (gprime - g) * xp.conj(probe) / (xp.max(xp.abs(probe)) ** 2)
#     #
#     #
#     #
#     #     if not fix_probe:
#     #         probe = probe + beta * (gprime - g) * xp.conj(illuminated_object) / (
#     #                 xp.max(xp.abs(illuminated_object)) ** 2)
#     #
#     #     illuminated_object = fft_shift(object, - batch_positions)
#     #
#     #
#     #     plt.imshow(cp.asnumpy(xp.abs(illuminated_object))[0])
#     #     plt.show()
#     #     #print(illuminated_object.shape)
#     #     sss


def _run_epie(
    object,
    probe: np.ndarray,
    diffraction_patterns: np.ndarray,
    positions: np.ndarray,
    maxiter: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    fix_probe: bool = False,
    fix_com: bool = False,
    return_iterations: bool = False,
    max_batch: int = 8,
    seed=None,
):
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
    else:
        object_iterations = None
        probe_iterations = None

    if seed is not None:
        np.random.seed(seed)

    diffraction_patterns = np.fft.ifftshift(
        np.sqrt(diffraction_patterns), axes=(-2, -1)
    )

    center_of_mass = get_ndimage_module(xp).center_of_mass

    chunks = validate_chunks((len(positions),), (max_batch,))
    outer_pbar = ProgressBar(total=maxiter)
    inner_pbar = ProgressBar(total=len(positions))

    k = 0
    while k < maxiter:
        indices = np.arange(len(positions))
        np.random.shuffle(indices)

        inner_pbar.reset()

        for _, slic in iterate_chunk_ranges(chunks):
            ind = indices[slic]

            batch_positions = xp.asarray(positions[ind])

            diffraction_pattern = xp.array(diffraction_patterns[ind])
            illuminated_object = fft_shift(object, -batch_positions)

            g = illuminated_object * probe
            gprime = xp.fft.ifft2(
                diffraction_pattern * xp.exp(1j * xp.angle(xp.fft.fft2(g)))
            )

            object = illuminated_object + alpha * (gprime - g) * xp.conj(probe) / (
                xp.max(xp.abs(probe)) ** 2
            )

            if not fix_probe:
                probe = probe + beta * (gprime - g) * xp.conj(illuminated_object) / (
                    xp.max(xp.abs(illuminated_object)) ** 2
                )

            object = fft_shift(object, batch_positions)

            object = object.mean(0)

            inner_pbar.update(len(batch_positions))

        if fix_com:
            com = center_of_mass(xp.fft.fftshift(xp.abs(probe) ** 2))
            probe = xp.fft.ifftshift(fft_shift(probe, -xp.array(com)))

        if object_iterations is not None and probe_iterations is not None:
            object_iterations.append(object)
            probe_iterations.append(probe)

        outer_pbar.update(1)
        k += 1

    inner_pbar.close()
    outer_pbar.close()

    if object_iterations is not None and probe_iterations is not None:
        return object_iterations, probe_iterations
    else:
        return object, probe


def periodic_distances(positions, bounds, square=False):
    difference = positions[:, None] - positions[None]
    difference = difference % bounds[None, None]
    upper = difference[np.triu_indices(len(positions), k=1)]
    lower = np.swapaxes(difference, 1, 0)[np.triu_indices(len(positions), k=1)]
    distances = np.sqrt(np.sum(np.minimum(upper, lower) ** 2, axis=1))
    if square:
        distances = squareform(distances)

    return distances


def _equivalent_real_space_extent(diffraction_patterns):
    return 1 / diffraction_patterns.sampling[0], 1 / diffraction_patterns.sampling[1]


def _equivalent_real_space_sampling(diffraction_patterns):
    return (
        1 / diffraction_patterns.sampling[0] / diffraction_patterns.base_shape[0],
        1 / diffraction_patterns.sampling[1] / diffraction_patterns.base_shape[1],
    )



def scan_positions(self) -> Tuple[np.ndarray, ...]:
    positions = ()
    for n, metadata in zip(_scan_shape(self), _scan_axes_metadata(self)):
        positions += (
            np.linspace(
                metadata.offset,
                metadata.offset + metadata.sampling * n,
                n,
                endpoint=metadata.endpoint,
            ),
        )
    return positions

def epie(
    diffraction_patterns: DiffractionPatterns,
    probe_guess: Probe,
    max_iter: int = 5,
    max_batch: int = 8,
    alpha: float = 1.0,
    beta: float = 1.0,
    fix_probe: bool = False,
    fix_com: bool = False,
    return_iterations: bool = False,
    seed: int = None,
):
    """
    Reconstruct the phase of a 4D-STEM measurement using the extended Ptychographical Iterative Engine.

    See https://doi.org/10.1016/j.ultramic.2009.05.012

    Parameters
    ----------
    measurement : Measurement object
        4D-STEM measurement.
    probe_guess : Probe object
        The initial guess for the probe.
    max_iter : int
        Run the algorithm for this many iterations.
    alpha : float
        Step size of the iterative updates for the object. See reference.
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

    if diffraction_patterns.is_lazy:
        diffraction_patterns = diffraction_patterns.compute()

    probe_guess = probe_guess.copy()
    probe_guess.extent = diffraction_patterns.equivalent_real_space_extent
    probe_guess.gpts = diffraction_patterns.base_shape

    if len(diffraction_patterns.scan_shape) != 2:
        raise ValueError()

    x, y = diffraction_patterns.scan_positions()
    x, y = np.meshgrid(x, y, indexing="ij")
    scan_positions = (
        np.array([x.ravel(), y.ravel()]).T
        / diffraction_patterns.equivalent_real_space_sampling
    )

    probe_guess = probe_guess.build((0.0, 0.0), lazy=False).array
    diffraction_patterns_array = diffraction_patterns.array.reshape(
        (-1,) + diffraction_patterns.array.shape[-2:]
    )

    result = _run_epie(
        diffraction_patterns.shape[-2:],
        probe_guess,
        diffraction_patterns_array,
        scan_positions,
        maxiter=max_iter,
        alpha=alpha,
        beta=beta,
        return_iterations=return_iterations,
        fix_probe=fix_probe,
        fix_com=fix_com,
        max_batch=max_batch,
        seed=seed,
    )

    if return_iterations:
        object_iterations = [
            Images(obj, sampling=diffraction_patterns.equivalent_real_space_sampling)
            for obj in result[0]
        ]
        # probe_iterations = [Measurement(np.fft.fftshift(probe), calibrations=calibrations) for probe in result[1]]

        # if crop_to_valid:
        #    object_iterations = [object_iteration.crop(valid_extent) for object_iteration in object_iterations]

        return object_iterations  # object_iterations, probe_iterations, result[2]
    else:
        object = Images(
            result[0], sampling=diffraction_patterns.equivalent_real_space_sampling
        )

        # if crop_to_valid:
        #    object = object.crop(valid_extent)

        return object

        # return (object,
        #        Measurement(np.fft.fftshift(result[1]), calibrations=calibrations),
        #        result[2])
