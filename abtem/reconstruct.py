import numpy as np

from abtem.device import get_array_module, get_scipy_module
from abtem.measure import Measurement, Calibration
from abtem.utils import ProgressBar, fft_shift
from abtem.waves import Probe
from typing import Union, Sequence
from abtem.potentials import TransmissionFunction

def _epie_update_function(f: np.ndarray,
                          g: np.ndarray,
                          delta_psi:np.ndarray,
                          alpha: float = 1.):
    
    xp = get_array_module(f)
    return f + alpha * delta_psi * xp.conj(g) / (xp.max(xp.abs(g)) ** 2)

def _epie_simultaneous_update_function(f: np.ndarray,
                                       g: np.ndarray,
                                       delta_psi:np.ndarray,
                                       alpha: float = 1.,
                                       beta: float = 1.):
    
    return _epie_update_function(f,g,delta_psi,alpha), _epie_update_function(g,f,delta_psi,beta)


def _epie_regularization(tf: PotentialArray, gamma: float = 1.):
    
    array     = tf.array
    xp        = get_array_module(array)
    nz,nx,ny  = array.shape
    sampling  = (tf.slice_thicknesses[0],)+tf.sampling
    
    ikz       = xp.fft.fftfreq(nz, d=sampling[0])
    ikx       = xp.fft.fftfreq(nx, d=sampling[1])
    iky       = xp.fft.fftfreq(ny, d=sampling[2])
    grid_ikz, grid_ikx, grid_iky = xp.meshgrid(ikz, ikx, iky, indexing='ij')
    
    kz        = grid_ikz**2 * gamma**2
    kxy       = grid_ikx**2 + grid_iky**2
    
    weight    = 1-2/xp.pi*xp.arctan2(kz,kxy)
    tf._array = xp.fft.ifftn(xp.fft.fftn(array)*weight)
    
    return tf

def _run_epie(object_dims: Sequence[int],
              probe: np.ndarray,
              diffraction_patterns: np.ndarray,
              positions: np.ndarray,
              maxiter: int,
              alpha: float = 1.,
              beta: float = 1.,
              verbose: bool = False,
              fix_probe: bool = False,
              fix_com: bool = False,
              return_iterations: bool = False,
              seed=None):
    xp = get_array_module(probe)

    object_dims = xp.array(object_dims)
    probe = xp.array(probe)

    if len(diffraction_patterns.shape) != 3:
        raise ValueError()

    if len(diffraction_patterns) != len(positions):
        raise ValueError()

    if object_dims.shape == (2,):
        object = xp.ones((int(object_dims[0]), int(object_dims[1])), dtype=xp.complex64)
    elif len(object_dims.shape) != 2:
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

    diffraction_patterns     = np.fft.ifftshift(np.sqrt(diffraction_patterns), axes=(-2, -1))

    SSE = 0.
    k = 0
    outer_pbar = ProgressBar(total=maxiter)
    inner_pbar = ProgressBar(total=len(positions),leave=False)

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

            
            SSE += xp.sum(xp.abs(xp.fft.fft2(g)) - diffraction_pattern)**2

            inner_pbar.update(1)

        object = fft_shift(object, position)

        if fix_com:
            com = center_of_mass(xp.fft.fftshift(xp.abs(probe) ** 2))
            probe = xp.fft.ifftshift(fft_shift(probe, - xp.array(com)))

        #SSE = SSE / np.prod(diffraction_patterns.shape)
        SSE = SSE / len(positions)


        if return_iterations:
            object_iterations.append(object)
            probe_iterations.append(probe)
            SSE_iterations.append(SSE)
            
        if verbose:
            print(f'Iteration {k:<{len(str(maxiter))}}, SSE = {float(SSE):.3e}')

        outer_pbar.update(1)

        k += 1

    inner_pbar.close()
    outer_pbar.close()

    if return_iterations:
        return object_iterations, probe_iterations, SSE_iterations
    else:
        return object, probe, SSE

def _run_epie_ms(object_dims: Sequence[int],
                 probe: np.ndarray,
                 diffraction_patterns: np.ndarray,
                 positions: np.ndarray,
                 maxiter: int,
                 num_slices: int = 1,
                 slice_thicknesses: Union[float, Sequence[float]] = None,
                 energy=None,
                 extent=None,
                 sampling=None,
                 alpha: float = 1.,
                 beta: float = 1.,
                 gamma: float = None,
                 verbose: bool = False,
                 fix_probe: bool = False,
                 fix_com: bool = False,
                 return_iterations: bool = False,
                 seed=None):
    
    if num_slices == 1:
        return _run_epie(object_dims,
                        probe,
                        diffraction_patterns,
                        positions,
                        maxiter=maxiter,
                        alpha=alpha,
                        beta=beta,
                        verbose=verbose,
                        fix_probe=fix_probe,
                        fix_com=fix_com,
                        return_iterations=return_iterations,
                        seed=seed)
    
    if slice_thicknesses is None: 
        raise ValueError()

    xp          = get_array_module(probe)
        
    object_dims = xp.array(object_dims)
    
    probe       = xp.array(probe)

    if len(diffraction_patterns.shape) != 3:
        raise ValueError()

    if len(diffraction_patterns) != len(positions):
        raise ValueError()

    if object_dims.shape == (2,):
        object_tf = TransmissionFunction(
            xp.ones((num_slices,) + (int(object_dims[0]), int(object_dims[1])), dtype=xp.complex64),
            slice_thicknesses=slice_thicknesses,
            extent=extent,
            sampling=sampling,
            energy=energy
        )
    elif len(object_dims.shape) != 2:
        raise ValueError()

    if probe.shape != diffraction_patterns.shape[1:]:
        raise ValueError()

    if probe.shape != object_tf.array.shape[-2:]:
        raise ValueError()

    if return_iterations:
        object_iterations = []
        probe_iterations  = []
        SSE_iterations    = []

    if seed is not None:
        np.random.seed(seed)
    
    propagator               = FresnelPropagator()
    diffraction_patterns     = np.fft.ifftshift(np.sqrt(diffraction_patterns), axes=(-2, -1))
    
        
    SSE            = 0.
    k              = 0
    outer_pbar     = ProgressBar(total=maxiter)
    inner_pbar     = ProgressBar(total=len(positions),leave=False)

    center_of_mass = get_scipy_module(xp).ndimage.center_of_mass
    
    probe          = [Waves(probe,
                            energy=energy,
                            extent=extent,
                            sampling=sampling) for slice_id in range(num_slices)]
    
    exit_waves     = [Waves(xp.ones((int(object_dims[0]), int(object_dims[1])), dtype=xp.complex64),
                            energy=energy,
                            extent=extent,
                            sampling=sampling) for slice_id in range(num_slices)]
    
    while k < maxiter:
        indices      = np.arange(len(positions))
        np.random.shuffle(indices)

        old_position = xp.array((0., 0.))
        inner_pbar.reset()
        SSE          = 0.
        
        for j in indices:
            position                 = xp.array(positions[j])

            diffraction_pattern      = xp.array(diffraction_patterns[j])
            object_tf._array         = fft_shift(object_tf.array, old_position - position)
            
            # Forward Multislice
            for t_index, _, t in object_tf.generate_transmission_functions(energy=energy,max_batch=1):
                
                wave                 = probe[t_index].copy()
                exit_waves[t_index]  = t.transmit(wave)
                
                if t_index + 1 < num_slices:
                    probe[t_index+1] = propagator.propagate(wave, t.thickness,in_place=False)
            
            # Correct Modulus
            g       = exit_waves[-1].array
            g_prime = xp.fft.ifft2(diffraction_pattern * xp.exp(1j * xp.angle(xp.fft.fft2(g))))
            
            SSE    += xp.sum(xp.abs(xp.fft.fft2(g)) - diffraction_pattern)**2
            
            # Last Slice Update
            object_tf._array[-1],probe[-1]._array = _epie_simultaneous_update_function(
                object_tf.array[-1],
                probe[-1].array,
                g_prime-g,
                alpha,
                beta)
            
            # Backward Propagation & Update
            for t_index in reversed(range(1,num_slices)):
                
                wave        = probe[t_index].copy()
                g           = exit_waves[t_index-1].array
                g_prime     = propagator.propagate(wave,-slice_thicknesses[t_index-1],in_place=False).array
                
                # Don't update probe at final iteration if fix_probe is True
                # alternatively, we could just set beta = 0.
                if t_index == 1 and fix_probe:
                    object_tf._array[t_index-1] = _epie_update_function(
                        object_tf.array[t_index-1],
                        probe[t_index-1].array,
                        g_prime-g,
                        alpha)
                    
                # Otherwise update both probe and object
                else:
                    object_tf._array[t_index-1],probe[t_index-1]._array = _epie_simultaneous_update_function(
                        object_tf.array[t_index-1],
                        probe[t_index-1].array,
                        g_prime-g,
                        alpha,
                        beta)

            old_position    = position
            
            # Regularization
            if gamma is not None:
                object_tf   = _epie_regularization(object_tf,gamma=gamma)
            
            inner_pbar.update(1)
            
        object_tf._array    = fft_shift(object_tf.array, position)

        if fix_com:
            com             = center_of_mass(xp.fft.fftshift(xp.abs(probe[0].array) ** 2))
            probe[0]._array = xp.fft.ifftshift(fft_shift(probe[0].array, - xp.array(com)))
        
        #SSE = SSE / np.prod(diffraction_patterns.shape)
        SSE = SSE / len(positions)
        
        if return_iterations:
            object_iterations.append(object_tf.array)
            probe_iterations.append(probe[0].array)
            SSE_iterations.append(SSE)

        if verbose:
            print(f'Iteration {k:<{len(str(maxiter))}}, SSE = {float(SSE):.3e}')
        
        outer_pbar.update(1)

        k += 1

    inner_pbar.close()
    outer_pbar.close()

    if return_iterations:
        return object_iterations, probe_iterations, SSE_iterations
    else:
        return object_tf.array, probe[0].array, SSE
    
    
def epie(measurement: Measurement,
         probe_guess: Probe,
         maxiter: int = 5,
         num_slices: int = 1,
         slice_thicknesses: Union[float, Sequence[float]] = None,
         energy: float = None,
         alpha: float = 1.,
         beta: float = 1.,
         gamma: float  = None,
         verbose: bool = False,
         fix_probe: bool = False,
         fix_com: bool = False,
         return_iterations: bool = False,
         max_angle: float = None,
         crop_to_valid: bool = False,
         seed: int = None,
         device: str = 'cpu'):
    """
    Reconstruct the phase of a 4D-STEM measurement using the multislice extended Ptychographical Iterative Engine.
    See:
    - https://doi.org/10.1016/j.ultramic.2009.05.012
    - https://doi.org/10.1364/JOSAA.29.001606
    Parameters
    ----------
    measurement : Measurement object
        4D-STEM measurement.
    probe_guess : Probe object
        The initial guess for the probe.
    maxiter : int
        Run the algorithm for this many iterations.
    num_slices: int
        If num_slices > 1, the multislice ePIE algorithm will be used. See reference.
    slice_thicknesses: float
        The thicknesses of object slices in Å. If a float, the thickness is the same for all slices.
        If a sequence, the length must equal num_slices.
    alpha : float
        Controls the size of the iterative updates for the object. See reference.
    beta : float
        Controls the size of the iterative updates for the probe. See reference.
    gamma : float
        Controls the size of the out-of-plane regularization. Default is None.
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

    #calibrations = calibrations_from_grid(probe_guess.gpts, probe_guess.sampling, names=['x', 'y'], units='Å')
    calibrations_probe = ()
    for name, d in zip(['x','y'], sampling):
        calibrations_probe += (Calibration(0., d, units='Å', name=name, endpoint=False),)
        
    if num_slices > 1:
        slice_thicknesses = np.array(slice_thicknesses)
        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, num_slices)
            
        calibrations_object = (Calibration(0.,slice_thicknesses[0],
                                           units='Å',name='z',endpoint=False) ,)+calibrations_probe
    else:
        calibrations_object = calibrations_probe

    probe_guess._device = device
    probe_guess = probe_guess.build(np.array([0, 0])).array
    
    result = _run_epie_ms(diffraction_patterns.shape[-2:],
                          probe_guess,
                          diffraction_patterns,
                          positions,
                          maxiter=maxiter,
                          num_slices=num_slices,
                          slice_thicknesses=slice_thicknesses,
                          energy=energy,
                          extent=extent,
                          sampling=sampling,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma,
                          verbose=verbose,
                          return_iterations=return_iterations,
                          fix_probe=fix_probe,
                          fix_com=fix_com,
                          seed=seed)

    valid_extent = (measurement.calibration_limits[0][1] - measurement.calibration_limits[0][0],
                    measurement.calibration_limits[1][1] - measurement.calibration_limits[1][0])

    if return_iterations:
        object_iterations = [Measurement(obj, calibrations = calibrations_object) for obj in result[0]]
        probe_iterations = [Measurement(np.fft.fftshift(probe), calibrations=calibrations_probe) for probe in result[1]]

        if crop_to_valid:
            object_iterations = [object_iteration.crop(valid_extent) for object_iteration in object_iterations]

        return object_iterations, probe_iterations, result[2]
    else:
        object = Measurement(result[0], calibrations=calibrations_object)

        if crop_to_valid:
            result = object.crop(valid_extent)

        return (object,
                Measurement(np.fft.fftshift(result[1]), calibrations=calibrations_probe),
                result[2])
