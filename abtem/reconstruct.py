import numpy as np
from abtem.device import get_scipy_module, get_array_module_from_device, asnumpy
from abtem.measure import Measurement, Calibration
from abtem.utils import ProgressBar, fft_shift
from abtem.waves import FresnelPropagator, Probe, Waves
from typing import Union, Sequence, Iterable
from abtem.base_classes import AntialiasFilter

def ptychographic_reconstruction(diffraction_measurements: Union[Measurement, Sequence[Measurement]],
                                 probe_guesses           : Union[Probe,np.ndarray,Sequence[np.ndarray]],
                                 object_guesses          : Union[np.ndarray,Sequence[np.ndarray]] = None,
                                 max_angle               : float                                  = None,
                                 crop_to_valid           : bool                                   = False,
                                 energy                  : float                                  = None,
                                 num_probes              : int                                    = None,
                                 num_objects             : int                                    = None,
                                 num_slices              : int                                    = None,
                                 slice_thicknesses       : Union[float,Sequence[float]]           = None,
                                 maxiter                 : int                                    = 5,
                                 alpha                   : float                                  = 1.,
                                 beta                    : float                                  = 1.,
                                 gamma                   : float                                  = 1.,
                                 pre_pos_correction_iter : int                                    = 1,
                                 damping                 : float                                  = 1.,
                                 damping_rate            : float                                  = 0.995,
                                 fix_probe               : bool                                   = False, 
                                 fix_com                 : bool                                   = True,
                                 fix_pos                 : bool                                   = True,
                                 return_iterations       : bool                                   = False,
                                 verbose                 : bool                                   = False,
                                 seed                    : int                                    = None,
                                 functions_dictionary    : dict                                   = None,
                                 device                  : str                                    = 'cpu'):
   
    """
    Reconstruct the complex objects of 4D-STEM measurements using the Ptychographical Iterative Engine (PIE).
    
    The current implementation automatically handles the following common PIE extensions:
    - regularized  PIE: See https://doi.org/10.1364/OPTICA.4.000736
    - mixed-state  PIE: See https://doi.org/10.1038/s41467-020-16688-6
    - multislice   PIE: See https://doi.org/10.1364/JOSAA.29.001606 
    - simultaneous PIE: See example notebook

    Optionally, it also allows for probe-position correction using steepest descent. See 10.1016/j.ultramic.2018.04.004 
    
    Further, it allows the user to define custom functions for the four main steps of the algorithm.
    See example notebook for a walkthrough.
    
    Parameters
    ----------
    diffraction_measurements : Measurement object(s)
        4D-STEM measurement(s).
    probe_guesses  : Probe object(s) or np.ndarray(s)
        The initial guess(es) for the probe(s).
        If passed a tuple of Probe/np.ndarray objects, this will triger the mixed-state PIE algorithm with that many probes.
    object_guesses : np.ndarray(s)
        The initial guess(es) for the object(s).
    max_angle : float, optional
        The maximum reconstructed scattering angle. If this is larger than the input data, the data will be zero-padded.
    crop_to_valid : bool
        If true, the output is cropped to the scan area.
    energy : float
        The probe energy. Only necessary is probe_guesses is not a Probe object.
    num_probes : int
        If not None, this will trigger the mixed-state PIE algorithm with num_probes probes.
    num_objects : int
        If not None, this will trigger the mixed-state PIE algorithm with num_objects objects.
    num_slices : int
        If not None, this will trigger the multislice PIE algorithm with num_slices slices of thickness slice_thicknesses.
    slice_thicknesses : float(s)
        If not None, this will trigger the multislice PIE algorithm with num_slices slices of thickness slice_thicknesses.
    maxiter : int
        Run the algorithm for this many iterations.
    alpha : float
        Controls the regularization of the iterative updates for the object. See regularized PIE reference.
        alpha=1, corresponds to ePIE with alpha=1. alpha < 1 regularizes towards true gradient descent.
    beta : float
        Controls the regularization of the iterative updates for the probe. See regularized PIE reference.
        beta=1, corresponds to ePIE with beta=1. beta < 1 regularizes towards true gradient descent.
    gamma : float
        Controls the probe-position update step. See position-correction reference.
    pre_pos_correction_iter : int
        Run the algorithm for this many iterations before applying position correction.
    damping : float
        Backwards-compatible parameter if user wants to run ePIE with alpha/beta != 1.
    damping_rate: float
        Per-iteration damping parameter to speed-up convergence if close to minimum.
        If you want to run ePIE with alpha/beta != 1, set this to 1.0 and set damping appropriately.
    fix_probe : bool
        If True, the probe will not be updated by the algorithm. Default is False.
    fix_com : bool
        If True, the center of mass of the probe will be centered. Default is True.
    fix_pos : bool
        If True, the positions will not be updated by the algorithm. Default is True.
    return_iterations : bool
        If True, return the reconstruction after every iteration. Default is False.
    verbose : bool
        If True, prints the current error after every iteration. Default is False.
    seed : int, optional
        Seed the random number generator.
    functions_dictionary : dict
        If not None, it allows the user to specify custom functions for the algorithm steps. See example notebook.
    device : str
        Set the calculation device.
        
    Returns
    -------
    Measurements of object(s), probe(s), probe positions, and error
    """

    if num_slices is not None:
        if slice_thicknesses is None:
            raise ValueError()
        else:
            slice_thicknesses           = np.array(slice_thicknesses)
            if slice_thicknesses.shape == ():
                slice_thicknesses       = np.tile(slice_thicknesses, num_slices)
                
    
    flat_dps, probes, objects, positions, calibrations, props = _prepare_pie_inputs(diffraction_measurements,
                                                                                    probe_guesses,
                                                                                    object_guesses,
                                                                                    max_angle         = max_angle,
                                                                                    energy            = energy,
                                                                                    num_probes        = num_probes,
                                                                                    num_objects       = num_objects,
                                                                                    num_slices        = num_slices,
                                                                                    slice_thicknesses = slice_thicknesses,
                                                                                    device            = device)
   

    if functions_dictionary is None:
    
        if not isinstance(diffraction_measurements,Iterable):

            if isinstance(probes,tuple):
                # Mixed-state PIE
                _exit_wave_func              = _mixed_state_exit_wave_func
                _amplitude_modification_func = _mixed_state_amplitude_modification_func
                _update_func                 = _mixed_state_update_func
                _update_positions_func       = _update_positions_multiple_probes_and_objects
                _update_center_of_mass_func  = _update_multiple_probes_center_of_mass_func

            elif len(objects.shape) == 3:
                # Multislice PIE
                _exit_wave_func              = _multislice_exit_wave_func
                _amplitude_modification_func = _multislice_amplitude_modification_func
                _update_func                 = _multislice_update_func
                _update_positions_func       = _update_positions_multislice_object
                _update_center_of_mass_func  = _update_single_probe_first_index_center_of_mass_func

            else:
                # Extended PIE
                _exit_wave_func              = _epie_exit_wave_func
                _amplitude_modification_func = _epie_amplitude_modification_func
                _update_func                 = _epie_update_func
                _update_positions_func       = _update_positions_single_object
                _update_center_of_mass_func  = _update_single_probe_center_of_mass_func

        else:
            # Simultaneous PIE
            _exit_wave_func                  = _simultaneous_exit_wave_func
            _amplitude_modification_func     = _simultaneous_amplitude_modification_func
            _update_func                     = _simultaneous_update_func
            _update_positions_func           = _update_positions_simultaneous_objects
            _update_center_of_mass_func      = _update_single_probe_center_of_mass_func
        
    else:
        _exit_wave_func                      = functions_dictionary['exit_wave_func']
        _amplitude_modification_func         = functions_dictionary['amplitude_modification_func']
        _update_func                         = functions_dictionary['update_func']
        _update_positions_func               = functions_dictionary['update_positions_func']
        _update_center_of_mass_func          = functions_dictionary['update_center_of_mass_func']
        
    xp = get_array_module_from_device(device)
    
    if return_iterations:
        objects_iterations   = []
        probes_iterations    = []
        positions_iterations = []
        SSE_iterations       = []

    if seed is not None:
        np.random.seed(seed)
        
    flat_dps               = np.fft.ifftshift(np.sqrt(flat_dps),axes=(-2,-1))
    
    iteration              = 0
    outer_pbar             = ProgressBar(total=maxiter)
    inner_pbar             = ProgressBar(total=len(positions),leave=False)

    center_of_mass         = get_scipy_module(xp).ndimage.center_of_mass
    
    propagator             = FresnelPropagator()
    antialias_filter       = AntialiasFilter()

    if not isinstance(diffraction_measurements,Iterable):
        pixel_size_y             = positions[1,1]
        pixel_size_x             = positions[diffraction_measurements.shape[1],0]
        real_space_sampling_x    = diffraction_measurements.calibrations[0].sampling
        real_space_sampling_y    = diffraction_measurements.calibrations[1].sampling
        positions_scaling_factor = np.array([[pixel_size_x/real_space_sampling_x,pixel_size_y/real_space_sampling_y]])
    else:
        pixel_size_y             = positions[1,1]
        pixel_size_x             = positions[diffraction_measurements[0].shape[1],0]
        real_space_sampling_x    = diffraction_measurements[0].calibrations[0].sampling
        real_space_sampling_y    = diffraction_measurements[0].calibrations[1].sampling
        positions_scaling_factor = np.array([[pixel_size_x/real_space_sampling_x,pixel_size_y/real_space_sampling_y]])

    while iteration < maxiter:
        indices            = np.arange(len(positions))
        np.random.shuffle(indices)
        
        old_position       = xp.array((0.,0.))
        inner_pbar.reset()
        SSE                = 0.0
        
        for j in indices:
            
            position       = xp.array(positions[j])
            
            if isinstance(diffraction_measurements,Iterable):
                diffraction_patterns = tuple(xp.array(flat_dp[j]) for flat_dp in flat_dps)
            else:
                diffraction_patterns = xp.array(flat_dps[j])
                
            if isinstance(objects,tuple):
                objects  = tuple(fft_shift(obj,old_position - position) for obj in objects)
            else:
                objects  = fft_shift(objects,old_position - position)

            exit_waves, objects, probes = _exit_wave_func(objects,
                                                          probes,
                                                          xp                = xp,
                                                          propagator        = propagator,
                                                          antialias_filter  = antialias_filter,
                                                          slice_thicknesses = slice_thicknesses,
                                                          wave_properties   = props)

        
            modified_exit_waves, SSE    = _amplitude_modification_func(exit_waves,
                                                                       diffraction_patterns,
                                                                       SSE,
                                                                       xp  = xp)

            
            objects, probes             = _update_func(exit_waves,
                                                       modified_exit_waves,
                                                       objects,
                                                       probes,
                                                       fix_probe          = fix_probe,
                                                       alpha              = alpha,
                                                       beta               = beta,
                                                       damping            = damping,
                                                       xp                 = xp,
                                                       propagator         = propagator,
                                                       slice_thicknesses  = slice_thicknesses,
                                                       wave_properties    = props)
            
            old_position = position
        
            if not fix_pos and iteration >= pre_pos_correction_iter:
                positions[j]            = _update_positions_func(objects,
                                                                 probes,
                                                                 diffraction_patterns,
                                                                 position,
                                                                 pixel_sizes     = (pixel_size_x,pixel_size_y),
                                                                 gamma           = gamma,
                                                                 damping         = damping,
                                                                 xp              = xp)
            

            inner_pbar.update(1)
        
        if isinstance(objects,tuple):
            objects  = tuple(fft_shift(obj,position) for obj in objects)
        else:
            objects  = fft_shift(objects,position)
        
        if fix_com:
            probes   = _update_center_of_mass_func(probes,
                                                   center_of_mass,
                                                   xp=xp)
            
        SSE        /= len(positions)
        
        if return_iterations:
            objects_iterations.append(objects)
            probes_iterations.append(probes)
            positions_iterations.append(positions/positions_scaling_factor)
            SSE_iterations.append(SSE)
            
        if verbose:
            print(f'Iteration {iteration:<{len(str(maxiter))}}, SSE = {float(SSE):.3e}')
            
        damping   *= damping_rate
        outer_pbar.update(1)
        
        iteration += 1
        
    inner_pbar.close()
    outer_pbar.close()
    
    if isinstance(diffraction_measurements,Iterable):
        valid_extent = (diffraction_measurements[0].calibration_limits[0][1] - diffraction_measurements[0].calibration_limits[0][0],
                        diffraction_measurements[0].calibration_limits[1][1] - diffraction_measurements[0].calibration_limits[1][0])
        
    else:
        valid_extent = (diffraction_measurements.calibration_limits[0][1] - diffraction_measurements.calibration_limits[0][0],
                        diffraction_measurements.calibration_limits[1][1] - diffraction_measurements.calibration_limits[1][0])        
    
    if return_iterations:
        
        if isinstance(objects,tuple):
            _objects_iterations    = []
            for obj_iter in objects_iterations:
                if crop_to_valid:
                    _objects_iterations.append(tuple(Measurement(obj, calibrations).crop(valid_extent) for obj in obj_iter))
                else:
                    _objects_iterations.append(tuple(Measurement(obj, calibrations) for obj in obj_iter))
            objects_iterations     = _objects_iterations
        else:
            if crop_to_valid:
                objects_iterations = [Measurement(obj,calibrations).crop(valid_extent) for obj in objects_iterations]
            else:
                objects_iterations = [Measurement(obj,calibrations) for obj in objects_iterations]
        
        if isinstance(probes,tuple):
            _probes_iterations     = []
            for probe_iter in probes_iterations:
                _probes_iterations.append(tuple(
                    Measurement(np.fft.fftshift(probe),calibrations) for probe in probe_iter))
            probes_iterations      = _probes_iterations
        else:
            probes_iterations      = [Measurement(np.fft.fftshift(probe),calibrations) for probe in probes_iterations]
        
        return objects_iterations, probes_iterations, positions_iterations, SSE_iterations
    
    else:
        if isinstance(objects,tuple):
            if crop_to_valid:
                objects            = tuple(Measurement(obj, calibrations).crop(valid_extent) for obj in objects)
            else:
                objects            = tuple(Measurement(obj, calibrations) for obj in objects)
        else:
            if crop_to_valid:
                objects            = Measurement(objects, calibrations).crop(valid_extent)
            else:
                objects            = Measurement(objects, calibrations)
        
        if isinstance(probes,tuple):
            probes                 = tuple(Measurement(np.fft.fftshift(probe),calibrations) for probe in probes)
        else:
            probes                 = Measurement(np.fft.fftshift(probes),calibrations)
        
        return objects, probes, positions/positions_scaling_factor, SSE


def _prepare_pie_inputs(diffraction_measurements : Union[Measurement, Sequence[Measurement]],
                        probe_guesses            : Union[Probe,np.ndarray,Sequence[np.ndarray]],
                        object_guesses           : Union[np.ndarray,Sequence[np.ndarray]] = None,
                        max_angle                : float                                  = None,
                        energy                   : float                                  = None,
                        num_probes               : int                                    = None,
                        num_objects              : int                                    = None,
                        num_slices               : int                                    = None,
                        slice_thicknesses        : np.ndarray                             = None,
                        device                   : str                                    = 'cpu'):

    xp = get_array_module_from_device(device)

    # Flatten Diffraction Patterns
    if isinstance(diffraction_measurements,Iterable):
        diffraction_measurements_shape        = diffraction_measurements[0].array.shape
        diffraction_measurements_calibrations = diffraction_measurements[0].calibrations

        flat_diffraction_patterns             = tuple(
            diffraction_measurement.array.reshape((-1,) + diffraction_measurements_shape[2:]) for
            diffraction_measurement in diffraction_measurements)
    else:
        diffraction_measurements_shape        = diffraction_measurements.array.shape
        diffraction_measurements_calibrations = diffraction_measurements.calibrations

        flat_diffraction_patterns             = diffraction_measurements.array.reshape(
            (-1,) + diffraction_measurements_shape[2:])

    # Pad Diffraction Patterns
    if max_angle:
        padding_x = max(
            int((max_angle / abs(diffraction_measurements_calibrations[-2].offset) *
                 diffraction_measurements_shape[-2]) // 2) - diffraction_measurements_shape[-2] // 2, 0)
        padding_y = max(
            int((max_angle / abs(diffraction_measurements_calibrations[-1].offset) *
                 diffraction_measurements_shape[-1]) // 2) - diffraction_measurements_shape[-1] // 2, 0)

        if isinstance(diffraction_measurements,Iterable):
            flat_diffraction_patterns      = tuple(
                np.pad(flat_diffraction_pattern, ((0,0),(padding_x,padding_x),(padding_y,padding_y))) for
                flat_diffraction_pattern in flat_diffraction_patterns)
        else:
            flat_diffraction_patterns = np.pad(flat_diffraction_patterns,
                                               ((0,0),(padding_x,padding_x),(padding_y,padding_y)))


    # Prepare Probes First Pass
    if isinstance(probe_guesses, tuple):
        num_probes    = len(probe_guesses)

        if isinstance(probe_guesses[0],np.ndarray):

            if energy is None:
                raise ValueError()

            _probe_guesses = []
            for probe in probe_guesses:
                _probe = Waves(np.fft.ifftshift(probe),energy=energy)
                _probe._grid._lock_gpts = False
                _probe_guesses.append(_probe)

            probe_guesses = tuple(_probe_guesses)

        else:
            raise ValueError()

        probe_wavelength  = probe_guesses[0].wavelength
        energy            = probe_guesses[0].energy

    else:

        if isinstance(probe_guesses,np.ndarray):

            if energy is None:
                raise ValueError()

            _probe = Waves(np.fft.ifftshift(probe_guesses),energy=energy)
            _probe._grid._lock_gpts = False

            probe_guesses = _probe

        probe_wavelength  = probe_guesses.wavelength
        energy            = probe_guesses.energy


    # Prepare Positions, Extent, and Sampling
    extent    = (probe_wavelength * 1e3 / diffraction_measurements_calibrations[2].sampling,
              probe_wavelength * 1e3 / diffraction_measurements_calibrations[3].sampling)

    sampling  = (extent[0] / diffraction_measurements_shape[-2],
                extent[1] / diffraction_measurements_shape[-1])

    x         = diffraction_measurements_calibrations[0].coordinates(diffraction_measurements_shape[0]) / sampling[0]
    y         = diffraction_measurements_calibrations[1].coordinates(diffraction_measurements_shape[1]) / sampling[1]
    x, y      = np.meshgrid(x, y, indexing='ij')
    positions = np.array([x.ravel(), y.ravel()]).T

    calibrations = ()
    for name, d in zip(['x','y'], sampling):
        calibrations += (Calibration(0., d, units='Å', name=name, endpoint=False),)

    if num_slices is not None:
        calibrations  = (Calibration(0.,slice_thicknesses[0],units='Å', name='z',endpoint=False),) + calibrations


    # Prepare Probes 2nd Pass
    if isinstance(probe_guesses, tuple):

        _probe_guesses    = []
        for probe in probe_guesses:
            probe.extent  = extent
            probe.gpts    = diffraction_measurements_shape[-2:]
            probe._device = device
            _probe_guesses.append(xp.array(probe.array))

        probe_guesses     = tuple(_probe_guesses)

    else:

        probe_guesses.extent  = extent
        probe_guesses.gpts    = diffraction_measurements_shape[-2:]
        probe_guesses._device = device

        if isinstance(probe_guesses,Probe):
            probe_guesses      = probe_guesses.build(np.array([0,0])).array
        else:
            probe_guesses      = xp.array(probe_guesses.array)

        # SVD Decomposition
        if num_probes is not None:

            u, s, v               = xp.linalg.svd(probe_guesses,full_matrices=True)

            for i in range(num_probes):
                probe_guesses     = tuple(xp.array(s[i] *xp.outer(u.T[i], v[i])) for i in range(num_probes))
        else:
            if num_objects is not None:
                probe_guesses     = (probe_guesses,)

            if num_slices is not None:
                _probe_guesses    = xp.zeros((num_slices,)+diffraction_measurements_shape[-2:],dtype=xp.complex64)
                _probe_guesses[0] = probe_guesses
                probe_guesses     = _probe_guesses


    # Prepare Objects
    if object_guesses is None:

        object_shape  = diffraction_measurements_shape[-2:]
        if num_slices is not None:
            object_shape = (num_slices,) + object_shape
        if num_objects is None:

            if isinstance(diffraction_measurements,Iterable):
                num_objects    = len(diffraction_measurements)
                object_guesses = tuple(xp.ones(object_shape,dtype=xp.complex64) for obj in range(num_objects))

            elif num_probes is None:
                object_guesses = xp.ones(object_shape,dtype=xp.complex64)
            else:
                object_guesses = (xp.ones(object_shape,dtype=xp.complex64),)
        else:
            object_guesses = tuple(xp.ones(object_shape,dtype=xp.complex64) for obj in range(num_objects))

    else:

        if isinstance(object_guesses,tuple):
            num_objects    = len(object_guesses)
            object_guesses = tuple(xp.array(obj) for obj in object_guesses)

        else:
            object_guesses = xp.array(object_guesses)


    return flat_diffraction_patterns, probe_guesses, object_guesses, positions, calibrations, (energy,extent,sampling)

### CoM update functions

def _update_single_probe_center_of_mass_func(probe_array:np.ndarray,
                                             center_of_mass,
                                             xp=np,
                                             **kwargs):

    com         = center_of_mass(xp.fft.fftshift(xp.abs(probe_array) ** 2))
    probe_array = xp.fft.ifftshift(fft_shift(probe_array, - xp.array(com)))

    return probe_array

def _update_single_probe_first_index_center_of_mass_func(probe_array:np.ndarray,
                                                         center_of_mass,
                                                         xp=np,
                                                         **kwargs):

    com            = center_of_mass(xp.fft.fftshift(xp.abs(probe_array[0]) ** 2))
    probe_array[0] = xp.fft.ifftshift(fft_shift(probe_array[0], - xp.array(com)))

    return probe_array

def _update_multiple_probes_center_of_mass_func(probes:Sequence[np.ndarray],
                                                center_of_mass,
                                                xp=np,
                                                **kwargs):

    _probes = []
    for probe_array in probes:
        com           = center_of_mass(xp.fft.fftshift(xp.abs(probe_array) ** 2))
        _probes.append(xp.fft.ifftshift(fft_shift(probe_array, - xp.array(com))))

    return tuple(_probes)

### e-PIE functions

def _epie_exit_wave_func(object_array:np.ndarray,
                         probe_array:np.ndarray,
                         xp = np,
                         **kwargs):
    return object_array * probe_array, object_array, probe_array


def _epie_amplitude_modification_func(exit_wave_array:np.ndarray,
                                      diffraction_pattern:np.ndarray,
                                      sse:float,
                                      xp = np,
                                      **kwargs):
    exit_wave_array_fft = xp.fft.fft2(exit_wave_array)
    sse                += xp.mean(xp.abs(xp.abs(exit_wave_array_fft) - diffraction_pattern)**2)
    modified_exit_wave  = xp.fft.ifft2(diffraction_pattern * xp.exp(1j * xp.angle(exit_wave_array_fft)))
    return modified_exit_wave, sse

def _epie_update_func(exit_wave_array:np.ndarray,
                      modified_exit_wave_array:np.ndarray,
                      object_array:np.ndarray,
                      probe_array:np.ndarray,
                      fix_probe: bool = False,
                      alpha: float = 1.,
                      beta: float = 1.,
                      damping: float = 1.,
                      xp = np,
                      **kwargs):

    exit_wave_diff    = modified_exit_wave_array - exit_wave_array
    probe_conj        = xp.conj(probe_array)
    obj_conj          = xp.conj(object_array)
    probe_abs_squared = xp.abs(probe_array)**2
    obj_abs_squared   = xp.abs(object_array)**2

    object_array     += damping * probe_conj*exit_wave_diff / (
                        (1-alpha)*probe_abs_squared + alpha*xp.max(probe_abs_squared))

    if not fix_probe:
        probe_array  += damping * obj_conj*exit_wave_diff / (
                        (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared))

    return object_array, probe_array

### multislice-PIE functions

def _multislice_exit_wave_func(object_array:np.ndarray,
                               probe_array:np.ndarray,
                               xp = np,
                               propagator:FresnelPropagator = None,
                               antialias_filter:AntialiasFilter = None,
                               slice_thicknesses:np.ndarray=None,
                               wave_properties:tuple = None,
                               **kwargs):

    energy,extent,sampling = wave_properties
    num_slices             = slice_thicknesses.shape[0]
    exit_waves_array       = xp.empty_like(object_array)

    object_array = antialias_filter._bandlimit(object_array)

    for s in range(num_slices):
        exit_waves_array[s]  = object_array[s]*probe_array[s]

        if s+1 < num_slices:
            exit_waves_waves = Waves(exit_waves_array[s],energy=energy,extent=extent,sampling=sampling)
            probe_array[s+1] = propagator.propagate(exit_waves_waves,slice_thicknesses[s],in_place=False).array

    return exit_waves_array, object_array, probe_array

def _multislice_amplitude_modification_func(exit_wave_array:np.ndarray,
                                            diffraction_pattern:np.ndarray,
                                            sse:float,
                                            xp = np,
                                            **kwargs):

    modified_exit_wave_array     = xp.empty_like(exit_wave_array)

    exit_wave_array_fft          = xp.fft.fft2(exit_wave_array[-1])
    sse                         += xp.mean(xp.abs(xp.abs(exit_wave_array_fft) - diffraction_pattern)**2)
    modified_exit_wave_array[-1] = xp.fft.ifft2(diffraction_pattern * xp.exp(1j * xp.angle(exit_wave_array_fft)))

    return modified_exit_wave_array, sse

def _multislice_update_func(exit_waves_array:np.ndarray,
                            modified_exit_waves_array:np.ndarray,
                            object_array:np.ndarray,
                            probe_array:np.ndarray,
                            fix_probe: bool = False,
                            alpha: float = 1.,
                            beta: float = 1.,
                            damping: float = 1.,
                            xp = np,
                            propagator:FresnelPropagator=None,
                            slice_thicknesses:np.ndarray = None,
                            wave_properties:tuple = None,
                            **kwargs):

    energy,extent,sampling   = wave_properties
    num_slices               = slice_thicknesses.shape[0]

    for s in reversed(range(num_slices)):

        exit_wave            = exit_waves_array[s]
        modified_exit_wave   = modified_exit_waves_array[s]

        exit_wave_diff       = modified_exit_wave - exit_wave
        probe_conj           = xp.conj(probe_array[s])
        obj_conj             = xp.conj(object_array[s])
        probe_abs_squared    = xp.abs(probe_array[s])**2
        obj_abs_squared      = xp.abs(object_array[s])**2

        object_array[s]     += damping * probe_conj*exit_wave_diff / (
                        (1-alpha)*probe_abs_squared + alpha*xp.max(probe_abs_squared))

        if not fix_probe or s > 0:
            probe_array[s]  += damping * obj_conj*exit_wave_diff / (
                            (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared))

        if s > 0:
            probe_wave = Waves(probe_array[s],energy=energy,extent=extent,sampling=sampling)
            modified_exit_waves_array[s-1] = propagator.propagate(probe_wave,-slice_thicknesses[s-1],in_place=False).array

    return object_array, probe_array

### mix-PIE functions

def _mixed_state_exit_wave_func(objects :Sequence[np.ndarray],
                                probes  :Sequence[np.ndarray],
                                xp = np,
                                **kwargs):

    num_objects      = len(objects)
    num_probes       = len(probes)
    shape            = objects[0].shape

    exit_waves_array = xp.empty((num_objects,num_probes)+shape,dtype=xp.complex64)

    for l in range(num_objects):
        for k in range(num_probes):
            exit_waves_array[l,k] = objects[l]*probes[k]

    return exit_waves_array, objects, probes

def _mixed_state_amplitude_modification_func(exit_wave_arrays:np.ndarray,
                                             diffraction_pattern:np.ndarray,
                                             sse:float,
                                             xp = np,
                                             **kwargs):

    num_objects, num_probes   = exit_wave_arrays.shape[:2]
    modified_exit_wave_arrays = xp.empty_like(exit_wave_arrays)

    exit_wave_arrays_fft      = xp.fft.fft2(exit_wave_arrays,axes=(-2,-1))
    intensity_norm            = xp.sqrt(xp.sum(xp.abs(exit_wave_arrays_fft)**2,axis=(0,1)))
    sse                      += xp.mean(xp.abs(intensity_norm - diffraction_pattern)**2)

    for l in range(num_objects):
        for k in range(num_probes):
            exit_wave_fft     = exit_wave_arrays_fft[l,k]
            modified_exit_wave_arrays[l,k] = xp.fft.ifft2(diffraction_pattern*xp.exp(1j*xp.angle(exit_wave_fft)))

    return modified_exit_wave_arrays, sse

def _mixed_state_update_func(exit_wave_arrays:np.ndarray,
                             modified_exit_wave_arrays:np.ndarray,
                             objects:Sequence[np.ndarray],
                             probes:Sequence[np.ndarray],
                             fix_probe: bool = False,
                             alpha: float = 1.,
                             beta: float = 1.,
                             damping: float = 1.,
                             xp = np,
                             **kwargs):

    num_objects, num_probes  = exit_wave_arrays.shape[:2]
    exit_wave_differences   = modified_exit_wave_arrays - exit_wave_arrays

    probes_array            = xp.array(probes)
    probes_array_conj       = xp.conj(probes_array)
    probes_squared_norm     = xp.sum(xp.abs(probes_array)**2,axis=0)
    objects_array           = xp.array(objects)
    objects_array_conj      = xp.conj(objects_array)
    objects_squared_norm    = xp.sum(xp.abs(objects_array)**2,axis=0)


    objects = tuple(objects[l] + damping*xp.sum(probes_array_conj*exit_wave_differences[l],axis=0)/(
                    (1-alpha)*probes_squared_norm + alpha*xp.max(probes_squared_norm))
          for l in range(num_objects))

    if not fix_probe:
        probes = tuple(probes[k] + damping*xp.sum(objects_array_conj*exit_wave_differences[:,k],axis=0)/(
                        (1-beta)*objects_squared_norm + beta*xp.max(objects_squared_norm))
              for k in range(num_probes))

    return objects, probes

### sim-PIE functions
def _simultaneous_exit_wave_func(objects:Sequence[np.ndarray],
                                 probe_array:np.ndarray,
                                 xp = np,
                                 **kwargs):

    electrostatic_object, magnetic_object = objects
    exit_wave_forward = electrostatic_object*magnetic_object*probe_array
    exit_wave_reverse = electrostatic_object*xp.conj(magnetic_object)*probe_array

    return (exit_wave_forward,exit_wave_reverse), objects, probe_array

def _simultaneous_amplitude_modification_func(exit_waves:Sequence[np.ndarray],
                                              diffraction_patterns:Sequence[np.ndarray],
                                              sse:float,
                                              xp = np,
                                              **kwargs):

    exit_wave_forward, exit_wave_reverse     = exit_waves
    diffraction_forward, diffraction_reverse = diffraction_patterns

    exit_wave_forward_fft = xp.fft.fft2(exit_wave_forward)
    exit_wave_reverse_fft = xp.fft.fft2(exit_wave_reverse)

    sse                  += xp.mean(xp.abs(xp.abs(exit_wave_forward_fft) - diffraction_forward)**2)/2
    sse                  += xp.mean(xp.abs(xp.abs(exit_wave_reverse_fft) - diffraction_reverse)**2)/2

    modified_exit_wave_forward = xp.fft.ifft2(diffraction_forward * xp.exp(1j * xp.angle(exit_wave_forward_fft)))
    modified_exit_wave_reverse = xp.fft.ifft2(diffraction_reverse * xp.exp(1j * xp.angle(exit_wave_reverse_fft)))

    return (modified_exit_wave_forward,modified_exit_wave_reverse), sse

def _simultaneous_update_func(exit_waves:Sequence[np.ndarray],
                              modified_exit_waves:Sequence[np.ndarray],
                              objects:Sequence[np.ndarray],
                              probe_array:np.ndarray,
                              fix_probe: bool = False,
                              alpha: float = 1.,
                              beta: float = 1.,
                              damping:float=1.,
                              xp = np,
                              **kwargs):

    exit_wave_forward,exit_wave_reverse                   = exit_waves
    modified_exit_wave_forward,modified_exit_wave_reverse = modified_exit_waves
    electrostatic_object, magnetic_object                 = objects

    exit_wave_diff_forward   = modified_exit_wave_forward - exit_wave_forward
    exit_wave_diff_reverse   = modified_exit_wave_reverse - exit_wave_reverse

    probe_conj                                            = xp.conj(probe_array)
    electrostatic_conj                                    = xp.conj(electrostatic_object)
    magnetic_conj                                         = xp.conj(magnetic_object)

    probe_magnetic_abs_squared                            = xp.abs(probe_array*magnetic_object)**2
    probe_electrostatic_abs_squared                       = xp.abs(probe_array*electrostatic_object)**2
    electrostatic_magnetic_abs_squared                    = xp.abs(electrostatic_object*magnetic_object)**2

    _electrostatic_object = electrostatic_object + damping*((probe_conj*magnetic_conj*exit_wave_diff_forward +
                                                             probe_conj*magnetic_object*exit_wave_diff_reverse)/(
            (1-alpha)*probe_magnetic_abs_squared + alpha*xp.max(probe_magnetic_abs_squared)))/2

    _magnetic_object      = magnetic_object + damping*((probe_conj*electrostatic_conj*exit_wave_diff_forward -
                                                        probe_conj*electrostatic_conj*exit_wave_diff_reverse)/(
            (1-alpha)*probe_electrostatic_abs_squared + alpha*xp.max(probe_electrostatic_abs_squared)))/2

    if not fix_probe:
        probe_array       = probe_array + damping*((electrostatic_conj*magnetic_conj*exit_wave_diff_forward +
                                                    electrostatic_conj*magnetic_object*exit_wave_diff_reverse)/(
            (1-beta)*electrostatic_magnetic_abs_squared + beta*xp.max(electrostatic_magnetic_abs_squared)))/2

    electrostatic_object  = _electrostatic_object
    magnetic_object       = _magnetic_object

    return (electrostatic_object,magnetic_object), probe_array

def _update_positions_single_object(objects: np.ndarray,
                                   probes : np.ndarray,
                                   diffraction_pattern: np.ndarray,
                                   position: np.ndarray,
                                   pixel_sizes: tuple = None,
                                   gamma:       float = 1.0,
                                   damping:     float = 1.0,
                                   xp=np,
                                   **kwargs):

    pixel_size_x, pixel_size_y = pixel_sizes

    # Note: `objects` is already shifted by position
    exit_waves_fft             = xp.fft.fft2(objects*probes)
    estimated_intensities      = xp.abs(exit_waves_fft)**2
    actual_intensities         = diffraction_pattern**2
    diff_intensities           = (actual_intensities - estimated_intensities).ravel()

    dx                         = xp.array([pixel_size_x,0.])
    exit_waves_fft_dx          = xp.fft.fft2(fft_shift(objects,-dx)*probes)
    d_exit_waves_fft_dx        = (exit_waves_fft -exit_waves_fft_dx)/pixel_size_x

    dy                         = xp.array([0.,pixel_size_y])
    exit_waves_fft_dy          = xp.fft.fft2(fft_shift(objects,-dy)*probes)
    d_exit_waves_fft_dy        = (exit_waves_fft - exit_waves_fft_dy)/pixel_size_y

    exit_waves_fft_conj        = xp.conj(exit_waves_fft)

    partial_intensities_dx     = 2*xp.real(d_exit_waves_fft_dx*exit_waves_fft_conj).ravel()
    partial_intensities_dy     = 2*xp.real(d_exit_waves_fft_dy*exit_waves_fft_conj).ravel()

    coefficients_matrix        = xp.column_stack((partial_intensities_dx,partial_intensities_dy))
    displacements              = xp.linalg.lstsq(coefficients_matrix,diff_intensities,rcond=None)[0]

    return asnumpy(position - damping*gamma*displacements)
            

def _update_positions_simultaneous_objects(objects: np.ndarray,
                                           probes : np.ndarray,
                                           diffraction_pattern: np.ndarray,
                                           position: np.ndarray,
                                           pixel_sizes: tuple = None,
                                           gamma:       float = 1.0,
                                           damping:     float = 1.0,
                                           xp=np,
                                           **kwargs):

    raise NotImplementedError()

def _update_positions_multiple_probes_and_objects(objects: np.ndarray,
                                                  probes : np.ndarray,
                                                  diffraction_pattern: np.ndarray,
                                                  position: np.ndarray,
                                                  pixel_sizes: tuple = None,
                                                  gamma:       float = 1.0,
                                                  damping:     float = 1.0,
                                                  xp=np,
                                                  **kwargs):

    raise NotImplementedError()

def _update_positions_multislice_object(objects: np.ndarray,
                                        probes : np.ndarray,
                                        diffraction_pattern: np.ndarray,
                                        position: np.ndarray,
                                        pixel_sizes: tuple = None,
                                        gamma:       float = 1.0,
                                        damping:     float = 1.0,
                                        xp=np,
                                        **kwargs):

    raise NotImplementedError()

