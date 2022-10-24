from typing import Union, Sequence, Mapping, Callable, Iterable
from abc import ABCMeta, abstractmethod
from functools import partial
from copy import copy
import numpy as np

from abtem.measure import Measurement, Calibration
from abtem.waves import Probe, FresnelPropagator
from abtem.base_classes import AntialiasFilter
from abtem.transfer import CTF, polar_symbols, polar_aliases
from abtem.utils import fft_shift,energy2wavelength, ProgressBar
from abtem.device import copy_to_device, get_array_module, get_array_module_from_device, get_scipy_module, asnumpy, get_device_function

experimental_symbols   = ('rotation_angle',
                          'scan_step_sizes',
                          'angular_sampling',
                          'background_counts_cutoff',
                          'counts_scaling_factor',
                          'grid_scan_shape',
                          'object_px_padding')

reconstruction_symbols = {'alpha': 1.0, 'beta': 1.0, # object/probe update regularization parameter
                          'object_step_size': 1.0, 'probe_step_size': 1.0, 'position_step_size': 1.0, # step-sizes
                          'step_size_damping_rate': 0.995, # step-size damping rate
                          'pre_position_correction_update_steps': None,
                          'pre_probe_correction_update_steps': None,
                         }

def _wrapped_indices_2D_window(center_position: np.ndarray,
                               window_shape: Sequence[int],
                               array_shape: Sequence[int]):
    '''
    Computes periodic indices for window_shaped probe centered at center_position, in object of size array_shape
    '''

    sx, sy     = array_shape
    nx, ny     = window_shape

    cx, cy     = np.round(asnumpy(center_position)).astype(int)
    ox, oy     = (cx - nx//2,cy - ny//2)

    return np.ix_(np.arange(ox,ox+nx)%sx, np.arange(oy,oy+ny)%sy)

def _projection(u,v):
    return u * np.vdot(u,v) / np.vdot(u,u)

def _orthogonalize(V):
    '''
    Non-normalized QR decomposition using repeated projections.
    Compare with np.linalg.qr(V.T)[0].T
    '''
    U = V.copy()
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i,:] -= _projection(U[j,:], V[i,:])
    return U

def _propagate_array(propagator:FresnelPropagator,
                     waves_array: np.ndarray,
                     sampling: Sequence[float],
                     wavelength: float,
                     dz: float,
                     fft2_convolve: Callable = None,
                     overwrite: bool = False,
                     xp = np):
    '''
    Simplified re-write of abtem.FresnelPropagator.propagate() to operate on arrays directly
    '''
    propagator_array = propagator._evaluate_propagator_array(waves_array.shape,sampling,wavelength,dz,None,xp)
    return fft2_convolve(waves_array,propagator_array, overwrite_x = overwrite)


class AbstractPtychographicOperator(metaclass=ABCMeta):
    '''
    Abstract base class for all ptychographic operators.
    Each subclass defines its own (static) reconstruction methods.
      Note: The reason to have these methods be static is to allow the user to specify arbirtary functions.
    Additionally, base class defines various common functions and properties for all subclasses to inherit.
    '''

    @abstractmethod
    def preprocess(self):
        '''
        All subclasses must imlement a .preprocess() method which does the following:
        - Pads CBED patterns to ROI dimensions
        - Prepares scanning positions in real-space resolution pixels
        - Initializes probes & objects
        '''
        pass

    @staticmethod
    @abstractmethod
    def _overlap_projection(objects,probes,position,old_position,**kwargs):
        '''
        All subclasses must implement a static ._overlap_projection() method.
        This typically gets called inside the .reconstruct() method.
        Additionally, subclasses may define a static ._warmup_overlap_projection() method.
        '''
        pass

    @staticmethod
    @abstractmethod
    def _fourier_projection(exit_waves,diffraction_patterns,sse,**kwargs):
        '''
        All subclasses must implement a static ._fourier_projection() method.
        This typically gets called inside the .reconstruct() method.
        Additionally, subclasses may define a static ._warmup_fourier_projection() method.
        '''
        pass

    @staticmethod
    @abstractmethod
    def _update_function(objects,probes,position,exit_waves,modified_exit_waves,**kwargs):
        '''
        All subclasses must implement a static ._update_function() method.
        This typically gets called inside the .reconstruct() method.
        Additionally, subclasses may define a static ._warmup_update_function() method.
        '''
        pass

    @staticmethod
    @abstractmethod
    def _position_correction(objects, probes, position,**kwargs):
        '''
        All subclasses must implement a static ._position_correction() method.
        This typically gets called inside the .reconstruct() method.
        Additionally, subclasses may define a static ._warmup_position_correction() method.
        '''
        pass

    @staticmethod
    @abstractmethod
    def _fix_probe_center_of_mass(probes, center_of_mass,**kwargs):
        '''
        All subclasses must implement a static ._fix_probe_center_of_mass() method.
        This typically gets called inside the .reconstruct() method.
        '''
        pass

    @abstractmethod
    def _prepare_functions_queue(self, max_iterations, **kwargs):
        '''
        All subclasses must implement a ._prepare_functions_queue() method.
        This typically gets called inside the .reconstruct() method.
        '''
        pass

    @abstractmethod
    def reconstruct(self,
                    max_iterations,
                    return_iterations,
                    fix_com,
                    random_seed,
                    verbose,
                    parameters,
                    measurement_output_view,
                    functions_queue,
                    **kwargs):
        '''
        All subclasses must implement a .reconstruct() method which does the following:
        - Precomputes functions queue using ._prepare_functions_queue()
        - Performs reconstruction
        - Passes reconstruction outputs to ._prepare_measurement_outputs()
        '''
        pass

    @abstractmethod
    def _prepare_measurement_outputs(self, objects, probes, positions, sse):
        '''
        All subclasses must implement a ._prepare_measurement_outputs() method which formats the reconstruction outputs.
        This typically gets called at the end of the .reconstruct() method.
        '''
        pass

    @staticmethod
    def _update_parameters(parameters: dict,
                           polar_parameters: dict = {},
                           experimental_parameters:dict = {}):
        '''
        Common static method to update polar and experimental parameters during child class initialization.
        '''
        for symbol, value in parameters.items():
            if symbol in polar_symbols:
                polar_parameters[symbol] = value
            elif symbol == 'defocus':
                polar_parameters[polar_aliases[symbol]] = -value
            elif symbol in polar_aliases.keys():
                polar_parameters[polar_aliases[symbol]] = value
            elif symbol in experimental_symbols:
                experimental_parameters[symbol] = value
            else:
                raise ValueError('{} not a recognized parameter'.format(symbol))

        return polar_parameters, experimental_parameters

    @staticmethod
    def _pad_diffraction_patterns(diffraction_patterns: np.ndarray,
                                  region_of_interest_shape: Sequence[int]):
        '''
        Common static method to zero-pad diffraction patterns to match region_of_interest_shape.
        Assumes diffraction patterns dimensions (J,M,N), i.e. flat - list
        '''

        diffraction_patterns_size = diffraction_patterns.shape[-2:]
        xp                        = get_array_module(diffraction_patterns)

        if any(dp_shape > roi_shape for dp_shape, roi_shape
                                    in zip(diffraction_patterns_size,region_of_interest_shape)):
            raise ValueError()

        if diffraction_patterns_size != region_of_interest_shape:
            padding_list = [(0,0)]  # No padding along first dimension
            for current_dim, target_dim in zip(diffraction_patterns_size, region_of_interest_shape):
                pad_value = (target_dim - current_dim)
                pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
                padding_list.append(pad_tuple)

            diffraction_patterns = xp.pad(diffraction_patterns, tuple(padding_list), mode='constant')

        return diffraction_patterns

    @staticmethod
    def _extract_calibrations_from_measurement_object(measurement: Measurement,
                                                      energy: float):
        '''
        Common static method to extract calibrations from measurement object.
        '''
        calibrations               = measurement.calibrations
        calibration_units          = measurement.calibration_units
        diffraction_patterns       = measurement.array

        if any(unit != 'Å' for unit in calibration_units[:-2]):
            raise ValueError()

        if any(unit != 'mrad' and unit != '1/Å' for unit in calibration_units[-2:]):
            raise ValueError()

        angular_sampling  = []
        for cal, cal_unit in zip(calibrations[-2:],calibration_units[-2:]):
            scale_factor  = 1. if cal_unit == 'mrad' else energy2wavelength(energy)*1e3
            angular_sampling.append(cal.sampling * scale_factor)
        angular_sampling  = tuple(angular_sampling)

        step_sizes     = None
        if len(diffraction_patterns.shape) == 4:
            step_sizes = tuple(cal.sampling for cal in calibrations[:2])

        return diffraction_patterns, angular_sampling, step_sizes

    @staticmethod
    def _calculate_scan_positions_in_pixels(positions:np.ndarray,
                                            sampling: Sequence[float],
                                            region_of_interest_shape: Sequence[int],
                                            experimental_parameters: dict):
        '''
        Common static method to calculate the scan positions in pixels
        '''

        grid_scan_shape   = experimental_parameters['grid_scan_shape']
        step_sizes        = experimental_parameters['scan_step_sizes']
        rotation_angle    = experimental_parameters['rotation_angle']
        object_px_padding = experimental_parameters['object_px_padding']

        if positions is None:
            if grid_scan_shape is not None:
                nx, ny = grid_scan_shape

                if step_sizes is not None:
                    sx, sy = step_sizes
                    x      = np.arange(nx)*sx
                    y      = np.arange(ny)*sy
                else:
                    raise ValueError()
            else:
                raise ValueError()

        else:
            x = positions[:,0]
            y = positions[:,1]

        x     = (x-np.ptp(x)/2) / sampling[0]
        y     = (y-np.ptp(y)/2) / sampling[1]
        x, y  = np.meshgrid(x, y, indexing='ij')

        if rotation_angle is not None:
            x, y       = x*np.cos(rotation_angle) + y*np.sin(rotation_angle), -x*np.sin(rotation_angle) + y*np.cos(rotation_angle)

        positions      = np.array([x.ravel(),y.ravel()]).T
        positions     -= np.min(positions,axis=0)

        if object_px_padding is None:
            object_px_padding = np.array(region_of_interest_shape)/2
        else:
            object_px_padding = np.array(object_px_padding)

        positions     += object_px_padding

        experimental_parameters['object_px_padding'] = object_px_padding
        return positions, experimental_parameters

    @property
    def angular_sampling(self):
        '''Angular sampling in mrad'''
        if not self._preprocessed:
            return None

        return self._experimental_parameters['angular_sampling']

    @property
    def sampling(self):
        '''Sampling in Å'''
        if not self._preprocessed:
            return None

        return tuple(energy2wavelength(self._energy)*1e3/dk/n
                         for dk,n in zip(self.angular_sampling,self._region_of_interest_shape))


class RegularizedPtychographicOperator(AbstractPtychographicOperator):
    '''
    Regularized PIE Operator.
    
    diffraction_patterns dimensions   : (J,M,N)
    objects dimensions                : (P,Q)
    probes dimensions                 : (R,S)
    '''
    def __init__(self,
                 diffraction_patterns:Union[np.ndarray,Measurement],
                 energy: float,
                 region_of_interest_shape: Sequence[int] = None,
                 objects: np.ndarray = None,
                 probes: Union[np.ndarray,Probe] = None,
                 positions: np.ndarray = None,
                 semiangle_cutoff: float = None,
                 preprocess: bool = False,
                 device: str = 'cpu',
                 parameters: Mapping[str,float] = None,
                 **kwargs):
        
        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()) and (key not in experimental_symbols):
                raise ValueError('{} not a recognized parameter'.format(key))
    
        self._polar_parameters        = dict(zip(polar_symbols, [0.] * len(polar_symbols)))
        self._experimental_parameters = dict(zip(experimental_symbols, [None] * len(experimental_symbols)))
        
        if parameters is None:
            parameters = {}
        
        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(parameters,
                                                                                        self._polar_parameters,
                                                                                        self._experimental_parameters)
        
        self._region_of_interest_shape   = region_of_interest_shape
        self._energy                     = energy
        self._semiangle_cutoff           = semiangle_cutoff
        self._positions                  = positions
        self._device                     = device
        self._objects                    = objects
        self._probes                     = probes
        self._diffraction_patterns       = diffraction_patterns
    
        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False
            
        
    def preprocess(self):
        '''
        Regularized PIE preprocessing method, to do the following:
        - Pads CBED patterns (J,M,N) to ROI dimensions -> (J,R,S)
        - Prepares scanning positions in real-space resolution pixels
        - Initializes probes (R,S) & objects (P,Q)
        '''
        
        self._preprocessed = True
        
        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            self._diffraction_patterns, angular_sampling, step_sizes = self._extract_calibrations_from_measurement_object(
                                                                                              self._diffraction_patterns,
                                                                                              self._energy)
            self._experimental_parameters['angular_sampling'] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters['scan_step_sizes']   = step_sizes
                
        # Preprocess Diffraction Patterns
        xp                             = get_array_module_from_device(self._device)
        self._diffraction_patterns     = copy_to_device(self._diffraction_patterns,self._device)
        
        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters['grid_scan_shape'] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns                       = self._diffraction_patterns.reshape(
                                                                    (-1,)+self._diffraction_patterns.shape[-2:])
            
        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]
        
        self._diffraction_patterns         = self._pad_diffraction_patterns(self._diffraction_patterns,
                                                                            self._region_of_interest_shape)
        self._num_diffraction_patterns     = self._diffraction_patterns.shape[0]
        
        if self._experimental_parameters['background_counts_cutoff'] is not None:
            self._diffraction_patterns[self._diffraction_patterns < self._experimental_parameters['background_counts_cutoff']] = 0.0

        if self._experimental_parameters['counts_scaling_factor'] is not None:
            self._diffraction_patterns /= self._experimental_parameters['counts_scaling_factor']   

        self._diffraction_patterns      = xp.fft.ifftshift(xp.sqrt(self._diffraction_patterns),axes=(-2,-1))
        
        
        # Scan Positions Initialization
        positions_px, self._experimental_parameters = self._calculate_scan_positions_in_pixels(self._positions,
                                                                                               self.sampling,
                                                                                               self._region_of_interest_shape,
                                                                                               self._experimental_parameters)
        
        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y  = self._experimental_parameters['object_px_padding']
            p, q          = np.max(positions_px,axis=0)
            p             = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(int)
            q             = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(int)
            self._objects = xp.ones((p,q),dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects,self._device)
                
        self._positions_px                                = copy_to_device(positions_px,self._device)
        self._positions_px_com                            = xp.mean(self._positions_px,axis=0)
        
        # Probes Initialization
        if self._probes is None:
            ctf           = CTF(energy=self._energy,
                                semiangle_cutoff=self._semiangle_cutoff,
                                parameters= self._polar_parameters)
            self._probes = Probe(semiangle_cutoff = self._semiangle_cutoff,
                                 energy           = self._energy,
                                 gpts             = self._region_of_interest_shape,
                                 sampling         = self.sampling,
                                 ctf              = ctf,
                                 device           = self._device).build().array
        else:
            if isinstance(self._probes,Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = copy_to_device(self._probes.build().array,self._device)
            else:
                self._probes = copy_to_device(self._probes,self._device)
        
        return self
        
        
    @staticmethod
    def _overlap_projection(objects:np.ndarray,
                            probes:np.ndarray,
                            position:np.ndarray,
                            old_position:np.ndarray,
                            xp = np,
                            **kwargs):
        '''
        r-PIE overlap projection:
        \psi_{R_j}(r) = O_{R_j}(r) * P(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probes                   = fft_shift(probes,fractional_position - old_fractional_position)
        object_indices           = _wrapped_indices_2D_window(position,probes.shape,objects.shape)
        object_roi               = objects[object_indices]
        exit_wave                = object_roi * probes
        
        return probes, exit_wave
     
    @staticmethod
    def _fourier_projection(exit_waves:np.ndarray,
                            diffraction_patterns:np.ndarray,
                            sse:float,
                            xp = np,
                            **kwargs):
        '''
        r-PIE Fourier-amplitude modification projection:
        \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]
        '''
        exit_wave_fft       = xp.fft.fft2(exit_waves)
        sse                += xp.mean(xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns)**2)/xp.sum(diffraction_patterns**2)
        modified_exit_wave  = xp.fft.ifft2(diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft)))
        
        return modified_exit_wave, sse
        
    @staticmethod   
    def _update_function(objects:np.ndarray,
                         probes:np.ndarray,
                         position:np.ndarray,
                         exit_waves:np.ndarray,
                         modified_exit_waves:np.ndarray,
                         diffraction_patterns:np.ndarray,
                         fix_probe: bool = False,
                         position_correction: Callable = None,
                         sobel: Callable = None,
                         reconstruction_parameters: Mapping[str,float] = None,
                         xp = np,
                         **kwargs):
        '''
        r-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        
        object_indices           = _wrapped_indices_2D_window(position,probes.shape,objects.shape)
        object_roi               = objects[object_indices]
        
        exit_wave_diff           = modified_exit_waves - exit_waves
        
        probe_conj               = xp.conj(probes)
        probe_abs_squared        = xp.abs(probes)**2
        obj_conj                 = xp.conj(object_roi)
        obj_abs_squared          = xp.abs(object_roi)**2
        
        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes,position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)
            
        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        objects[object_indices] += object_step_size * probe_conj*exit_wave_diff / (
                                    (1-alpha)*probe_abs_squared + alpha*xp.max(probe_abs_squared))
        
        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            probes              += probe_step_size * obj_conj*exit_wave_diff / (
                                    (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared))
            
        return objects, probes, position
    
        
    """
    @staticmethod
    def _position_correction(objects: np.ndarray,
                             probes: np.ndarray,
                             position:np.ndarray,
                             exit_wave: np.ndarray,
                             modified_exit_wave: np.ndarray,
                             diffraction_pattern:np.ndarray,
                             sobel: Callable,
                             position_step_size: float = 1.0,
                             xp=np,
                             **kwargs):
        '''
        r-PIE position correction function.
        '''
        
        object_dx                  = sobel(objects,axis=0,mode='wrap')
        object_dy                  = sobel(objects,axis=1,mode='wrap')
        
        object_indices             = _wrapped_indices_2D_window(position,probes.shape,objects.shape)
        d_exit_wave_fft_dx         = xp.fft.fft2(object_dx[object_indices]*probes)
        d_exit_wave_fft_dy         = xp.fft.fft2(object_dy[object_indices]*probes)
        
        exit_wave_fft              = xp.fft.fft2(exit_wave)
        estimated_intensity        = xp.abs(exit_wave_fft)**2
        intensity                  = diffraction_pattern**2
        difference_intensity       = (intensity - estimated_intensity).ravel()

        exit_wave_fft_conj         = xp.conj(exit_wave_fft)

        partial_intensity_dx       = 2*xp.real(d_exit_wave_fft_dx*exit_wave_fft_conj).ravel()
        partial_intensity_dy       = 2*xp.real(d_exit_wave_fft_dy*exit_wave_fft_conj).ravel()

        coefficients_matrix        = xp.column_stack((partial_intensity_dx,partial_intensity_dy))
        displacements              = xp.linalg.lstsq(coefficients_matrix,difference_intensity,rcond=None)[0]
        
        return position - position_step_size*displacements
    """
    
    @staticmethod
    def _position_correction(objects: np.ndarray,
                             probes: np.ndarray,
                             position:np.ndarray,
                             exit_wave:np.ndarray,
                             modified_exit_wave: np.ndarray,
                             diffraction_pattern:np.ndarray,
                             sobel:Callable,
                             position_step_size: float = 1.0,
                             xp=np,
                             **kwargs):
        '''
        r-PIE position correction function.
        '''

        object_dx       = sobel(objects,axis=0,mode='wrap')
        object_dy       = sobel(objects,axis=1,mode='wrap')
        
        object_indices  = _wrapped_indices_2D_window(position,probes.shape,objects.shape)
        exit_wave_dx    = object_dx[object_indices]*probes
        exit_wave_dy    = object_dy[object_indices]*probes
        
        exit_wave_diff  = modified_exit_wave - exit_wave
        displacement_x  = xp.sum(xp.real(xp.conj(exit_wave_dx)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dx)**2)
        displacement_y  = xp.sum(xp.real(xp.conj(exit_wave_dy)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dy)**2)
        
        return position + position_step_size*xp.array([displacement_x,displacement_y])
    
    
    @staticmethod
    def _fix_probe_center_of_mass(probes:np.ndarray,
                                  center_of_mass:Callable,
                                  xp = np,
                                  **kwargs):
        '''
        Fix probes CoM to array center. 
        '''
        
        probe_center = xp.array(probes.shape)/2
        com          = center_of_mass(xp.abs(probes) ** 2)
        probes       = fft_shift(probes, probe_center - xp.array(com))
        
        return probes
        
    def _prepare_functions_queue(self,
                                 max_iterations: int,
                                 pre_position_correction_update_steps: int = None,
                                 pre_probe_correction_update_steps: int = None,
                                 **kwargs):
        '''
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting. 
        '''
        total_update_steps   = max_iterations*self._num_diffraction_patterns
        queue_summary        = "Ptychographic reconstruction will perform the following steps:"
    
        functions_tuple      = (self._overlap_projection,self._fourier_projection, self._update_function, None)
        functions_queue      = [functions_tuple]
        if pre_position_correction_update_steps is None:
            functions_queue *= total_update_steps
            queue_summary   += f"\n--Regularized PIE for {total_update_steps} steps"
        else:
            functions_queue *= pre_position_correction_update_steps
            queue_summary   += f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"

            functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, self._position_correction)

            remaining_update_steps = total_update_steps - pre_position_correction_update_steps
            functions_queue += [functions_tuple]*remaining_update_steps
            queue_summary   += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"
        
        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"
        
        functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
        
        return functions_queue, queue_summary
    
    def reconstruct(self,
                    max_iterations: int = 5,
                    return_iterations: bool = False,
                    fix_com: bool = True,
                    random_seed = None,
                    verbose: bool = False,
                    parameters: Mapping[str,float] = None,
                    measurement_output_view: str = 'padded',
                    functions_queue: Iterable = None,
                    **kwargs):
        '''
        '''
        for key in kwargs.keys():
            if (key not in reconstruction_symbols.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))
                
        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)
                 
        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                                        max_iterations,
                                        pre_position_correction_update_steps = self._reconstruction_parameters['pre_position_correction_update_steps'],
                                        pre_probe_correction_update_steps    = self._reconstruction_parameters['pre_probe_correction_update_steps'])
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [[function_tuples]*self._num_diffraction_patterns for function_tuples in functions_queue]
            elif len(functions_queue) == max_iterations*self._num_diffraction_patterns:
                functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
            else:
                raise ValueError()
        
        self._functions_queue = functions_queue
        
        ### Main Loop
        xp                  = get_array_module_from_device(self._device)
        outer_pbar          = ProgressBar(total=max_iterations,leave=False)
        inner_pbar          = ProgressBar(total=self._num_diffraction_patterns,leave=False)
        indices             = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(self._experimental_parameters['object_px_padding'])
        center_of_mass      = get_scipy_module(xp).ndimage.center_of_mass
        sobel               = get_scipy_module(xp).ndimage.sobel
        
        if return_iterations:
            objects_iterations   = []
            probes_iterations    = []
            positions_iterations = []
            sse_iterations       = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        for iteration_index, iteration_step in enumerate(self._functions_queue):
            
            inner_pbar.reset()
            
            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse    = 0.0
            
            for update_index, update_step in enumerate(iteration_step):
                
                index               = indices[update_index]
                position            = self._positions_px[index]
                
                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i  = iteration_index*self._num_diffraction_patterns + update_index
                
                if self._reconstruction_parameters['pre_probe_correction_update_steps'] is None:
                    fix_probe       = False
                else:
                    fix_probe       = global_iteration_i < self._reconstruction_parameters['pre_probe_correction_update_steps']

                _overlap_projection,_fourier_projection,_update_function,_position_correction = update_step
            
                self._probes, exit_wave                                = _overlap_projection(self._objects,
                                                                                             self._probes,
                                                                                             position,
                                                                                             old_position,
                                                                                             xp=xp)
                
                modified_exit_wave, self._sse                          = _fourier_projection(exit_wave,
                                                                                             diffraction_pattern,
                                                                                             self._sse,
                                                                                             xp=xp)

                self._objects, self._probes, self._positions_px[index] = _update_function(self._objects,
                                                                                          self._probes,
                                                                                          position,
                                                                                          exit_wave,
                                                                                          modified_exit_wave,
                                                                                          diffraction_pattern,
                                                                                          fix_probe = fix_probe,
                                                                                          position_correction = _position_correction,
                                                                                          sobel = sobel,
                                                                                          reconstruction_parameters = self._reconstruction_parameters,
                                                                                          xp = xp)
                    
                old_position    = position
                inner_pbar.update(1)
              
            # Shift probe back to origin
            self._probes     = fft_shift(self._probes,xp.round(position) - position)
            
            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(self._probes,center_of_mass,xp=xp)
                 
            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (xp.mean(self._positions_px,axis=0) - self._positions_px_com)
                self._reconstruction_parameters['position_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            
            # Update Parameters
            self._reconstruction_parameters['object_step_size'] *= self._reconstruction_parameters['step_size_damping_rate']
            self._reconstruction_parameters['probe_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            self._sse                                           /= self._num_diffraction_patterns
            
            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(self._positions_px.copy() * xp.array(self.sampling))
                sse_iterations.append(self._sse)
            
            if verbose:
                print(f'----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}')
            
            outer_pbar.update(1)
        
        inner_pbar.close()
        outer_pbar.close()
        
        #  Return Results
        if return_iterations:
            results = map(self._prepare_measurement_outputs,
                          objects_iterations,
                          probes_iterations,
                          positions_iterations,
                          sse_iterations)
            
            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(self._objects,
                                                        self._probes,
                                                        self._positions_px * xp.array(self.sampling),
                                                        self._sse)
            return results
        
        
    def _prepare_measurement_outputs(self,
                                     objects:np.ndarray,
                                     probes: np.ndarray,
                                     positions: np.ndarray,
                                     sse: np.ndarray):
        '''
        Base measurement outputs function operating on a single iteration's outputs.
        Called using map if more than one iteration required.
        '''
        
        calibrations = tuple(Calibration(0, s, units='Å', name = n, endpoint=False) for s,n in zip(self.sampling,('x','y')))
        
        measurement_objects = Measurement(asnumpy(objects),calibrations)
        measurement_probes  = Measurement(asnumpy(probes),calibrations)
        
        return measurement_objects, measurement_probes, asnumpy(positions), sse

class SimultaneousPtychographicOperator(AbstractPtychographicOperator):
    '''
    Simultaneous PIE Operator.
    
    diffraction_patterns dimensions   : tuple of length I, each with dimensions (J,M,N)
    objects dimensions                : tuple of length I, each with dimensions (P,Q)
    probes dimensions                 : tuple of length I, each with dimensions (R,S)
    
    We specialize for the case of I=2, where user supplies two sets of diffraction patterns,
    one collected with the sign of the magnetic phase contribution reversed 
    (e.g. by flipping the sample 180 degrees, or by reversing the sign of current-biasing)
    '''
    def __init__(self,
                 diffraction_patterns:Union[Sequence[np.ndarray],Sequence[Measurement]],
                 energy: float,
                 region_of_interest_shape: Sequence[int] = None,
                 objects: np.ndarray = None,
                 probes: Union[np.ndarray,Probe] = None,
                 positions: np.ndarray = None,
                 semiangle_cutoff: float = None,
                 preprocess: bool = False,
                 device: str = 'cpu',
                 parameters: Mapping[str,float] = None,
                 **kwargs):
        
        if len(diffraction_patterns) != 2:
            raise NotImplementedError('Simultaneous ptychographic reconstruction is currently only implemented for two sets of diffraction patterns'
                                      'allowing reconstruction of the electrostatic and magnetic phase contributions.'
                                      'See the documentation for AbstractPtychographicOperator to implement your own class to handle more cases.')
        
        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()) and (key not in experimental_symbols):
                raise ValueError('{} not a recognized parameter'.format(key))
    
        self._polar_parameters        = dict(zip(polar_symbols, [0.] * len(polar_symbols)))
        self._experimental_parameters = dict(zip(experimental_symbols, [None] * len(experimental_symbols)))
        
        if parameters is None:
            parameters = {}
        
        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(parameters,
                                                                                        self._polar_parameters,
                                                                                        self._experimental_parameters)
        
        self._region_of_interest_shape   = region_of_interest_shape
        self._energy                     = energy
        self._semiangle_cutoff           = semiangle_cutoff
        self._positions                  = positions
        self._device                     = device
        self._objects                    = objects
        self._probes                     = probes
        self._diffraction_patterns       = diffraction_patterns
    
        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False
            
    def preprocess(self):
        '''
        Simultaneous PIE preprocessing method, to do the following:
        - Pads each set of CBED patterns (J,M,N) to ROI dimensions -> (J,R,S)
        - Prepares scanning positions in real-space resolution pixels
        - Initializes each set of probes (R,S) & objects (P,Q)
        '''
        
        self._preprocessed    = True
        
        xp                    = get_array_module_from_device(self._device)
        _diffraction_patterns = []
        for dp in self._diffraction_patterns:
            
            # Convert Measurement Objects    
            if isinstance(dp, Measurement):
                _dp, angular_sampling, step_sizes = self._extract_calibrations_from_measurement_object(dp, self._energy)
                
            # Preprocess Diffraction Patterns
            _dp     = copy_to_device(_dp,self._device)

            if len(_dp.shape) == 4:
                self._experimental_parameters['grid_scan_shape']   = _dp.shape[:2]
                _dp                                                = _dp.reshape((-1,)+_dp.shape[-2:])
                
            if self._region_of_interest_shape is None:
                self._region_of_interest_shape = _dp.shape[-2:]
            _dp     = self._pad_diffraction_patterns(_dp,self._region_of_interest_shape)
            
            if self._experimental_parameters['background_counts_cutoff'] is not None:
                _dp[_dp < self._experimental_parameters['background_counts_cutoff']] = 0.0

            if self._experimental_parameters['counts_scaling_factor'] is not None:
                _dp    /= self._experimental_parameters['counts_scaling_factor'] 
                
            _dp         = xp.fft.ifftshift(xp.sqrt(_dp),axes=(-2,-1))
            _diffraction_patterns.append(_dp)
        
        
        self._diffraction_patterns                              = tuple(_diffraction_patterns)
        self._experimental_parameters['angular_sampling']       = angular_sampling
        self._num_diffraction_patterns                          = self._diffraction_patterns[0].shape[0]
        if step_sizes is not None:
            self._experimental_parameters['scan_step_sizes']    = step_sizes
            
        # Scan Positions Initialization
        positions_px, self._experimental_parameters = self._calculate_scan_positions_in_pixels(self._positions,
                                                                                               self.sampling,
                                                                                               self._region_of_interest_shape,
                                                                                               self._experimental_parameters)
        
        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y  = self._experimental_parameters['object_px_padding']
            p, q          = np.max(positions_px,axis=0)
            p             = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(int)
            q             = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(int)
            self._objects = tuple(xp.ones((p,q),dtype=xp.complex64) for _obj_i in range(2))
        else:
            if len(self._objects) != 2:
                raise ValueError()
            self._objects = tuple(copy_to_device(_obj,self._device) for _obj in self._objects)
                
        self._positions_px                                = copy_to_device(positions_px,self._device)
        self._positions_px_com                            = xp.mean(self._positions_px,axis=0)
        
        # Probes Initialization
        if self._probes is None:
            ctf          = CTF(energy=self._energy,
                               semiangle_cutoff=self._semiangle_cutoff,
                               parameters= self._polar_parameters)
            self._probes = Probe(semiangle_cutoff = self._semiangle_cutoff,
                                 energy           = self._energy,
                                 gpts             = self._region_of_interest_shape,
                                 sampling         = self.sampling,
                                 ctf              = ctf,
                                 device           = self._device).build().array
                                  
            self._probes = (self._probes, self._probes.copy())
        else:
            if len(self._probes) != 2:
                raise ValueError()
            if isinstance(self._probes[0],Probe):
                if self._probes[0].gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = tuple(copy_to_device(_probe.build().array,self._device) for _probe in self._probes)
            else:
                self._probes = tuple(copy_to_device(_probe,self._device) for _probe in self._probes)
        
        return self
        
    @staticmethod
    def _warmup_overlap_projection(objects:Sequence[np.ndarray],
                                   probes:Sequence[np.ndarray],
                                   position:np.ndarray,
                                   old_position:np.ndarray,
                                   xp = np,
                                   **kwargs):
        '''
        r-PIE overlap projection:
        \psi_{R_j}(r) = V_{R_j}(r) * P(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probe_forward, probe_reverse = probes
        probe_forward                = fft_shift(probe_forward,fractional_position - old_fractional_position)

        electrostatic_object, magnetic_object = objects

        object_indices           = _wrapped_indices_2D_window(position,probe_forward.shape,electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]

        exit_wave_forward        = electrostatic_roi*probe_forward

        return (probe_forward,probe_reverse), (exit_wave_forward, None)
        
    @staticmethod
    def _overlap_projection(objects:Sequence[np.ndarray],
                            probes:Sequence[np.ndarray],
                            position:np.ndarray,
                            old_position:np.ndarray,
                            xp = np,
                            **kwargs):
        '''
        sim-PIE overlap projection:
        \psi_{R_j}(r) = V_{R_j}(r)* M_{R_j}(r) * P_I(r)
        \phi_{R_j}(r) = V_{R_j}(r)* M*_{R_j}(r) * P_{\Omega}(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probe_forward, probe_reverse = probes
        probe_forward                = fft_shift(probe_forward,fractional_position - old_fractional_position)
        probe_reverse                = fft_shift(probe_reverse,fractional_position - old_fractional_position)

        electrostatic_object, magnetic_object = objects

        object_indices           = _wrapped_indices_2D_window(position,probe_forward.shape,electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]
        magnetic_roi             = magnetic_object[object_indices]

        exit_wave_forward        = electrostatic_roi*magnetic_roi*probe_forward
        exit_wave_reverse        = electrostatic_roi*xp.conj(magnetic_roi)*probe_reverse

        return (probe_forward,probe_reverse), (exit_wave_forward,exit_wave_reverse)
    
    @staticmethod
    def _alternative_overlap_projection(objects:Sequence[np.ndarray],
                                        probes:Sequence[np.ndarray],
                                        position:np.ndarray,
                                        old_position:np.ndarray,
                                        xp = np,
                                        **kwargs):
        '''
        sim-PIE overlap projection:
        \psi_{R_j}(r) = V_{R_j}(r)* M_{R_j}(r) * P(r)
        \phi_{R_j}(r) = V_{R_j}(r)* M*_{R_j}(r) * P(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probe_forward, probe_reverse = probes
        probe_forward                = fft_shift(probe_forward,fractional_position - old_fractional_position)

        electrostatic_object, magnetic_object = objects

        object_indices           = _wrapped_indices_2D_window(position,probe_forward.shape,electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]
        magnetic_roi             = magnetic_object[object_indices]

        exit_wave_forward        = electrostatic_roi*magnetic_roi*probe_forward
        exit_wave_reverse        = electrostatic_roi*xp.conj(magnetic_roi)*probe_forward
        
        # return dummy probe_reverse to avoid complicating unpacking logic
        return (probe_forward,probe_reverse), (exit_wave_forward,exit_wave_reverse)
    
    @staticmethod
    def _warmup_fourier_projection(exit_waves:np.ndarray,
                                   diffraction_patterns:Sequence[np.ndarray],
                                   sse:float,
                                   xp = np,
                                   **kwargs):
        '''
        r-PIE Fourier-amplitude modification projection:
        \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]
        '''
        exit_wave_forward  , exit_wave_reverse   = exit_waves
        diffraction_forward, diffraction_reverse = diffraction_patterns

        exit_wave_forward_fft                    = xp.fft.fft2(exit_wave_forward)
        sse                                     += xp.mean(xp.abs(xp.abs(exit_wave_forward_fft) - diffraction_forward)**2)/xp.sum(diffraction_forward**2)
        modified_exit_wave_forward               = xp.fft.ifft2(diffraction_forward * xp.exp(1j * xp.angle(exit_wave_forward_fft)))

        return (modified_exit_wave_forward,None), sse
        
    @staticmethod
    def _fourier_projection(exit_waves:Sequence[np.ndarray],
                            diffraction_patterns:Sequence[np.ndarray],
                            sse:float,
                            xp = np,
                            **kwargs):
        '''
        sim-PIE Fourier-amplitude modification projection:
        \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]
        \phi'_{R_j}(r) = F^{-1}[\sqrt{\Omega_j(u)} F[\phi_{R_j}(u)] / |F[\phi_{R_j}(u)]|]
        '''
        exit_wave_forward  , exit_wave_reverse   = exit_waves
        diffraction_forward, diffraction_reverse = diffraction_patterns

        exit_wave_forward_fft                    = xp.fft.fft2(exit_wave_forward)
        exit_wave_reverse_fft                    = xp.fft.fft2(exit_wave_reverse)

        sse                                     += xp.mean(xp.abs(xp.abs(exit_wave_forward_fft) - diffraction_forward)**2)/xp.sum(diffraction_forward**2)/2
        sse                                     += xp.mean(xp.abs(xp.abs(exit_wave_reverse_fft) - diffraction_reverse)**2)/xp.sum(diffraction_reverse**2)/2

        modified_exit_wave_forward               = xp.fft.ifft2(diffraction_forward * xp.exp(1j * xp.angle(exit_wave_forward_fft)))
        modified_exit_wave_reverse               = xp.fft.ifft2(diffraction_reverse * xp.exp(1j * xp.angle(exit_wave_reverse_fft)))

        return (modified_exit_wave_forward,modified_exit_wave_reverse), sse

    @staticmethod   
    def _warmup_update_function(objects:Sequence[np.ndarray],
                                probes:Sequence[np.ndarray],
                                position:np.ndarray,
                                exit_waves:Sequence[np.ndarray],
                                modified_exit_waves:Sequence[np.ndarray],
                                diffraction_patterns:Sequence[np.ndarray],
                                fix_probe: bool = False,
                                position_correction: Callable = None,
                                sobel: Callable = None,
                                reconstruction_parameters: Mapping[str,float] = None,
                                xp = np,
                                **kwargs):
        '''
        r-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        exit_wave_forward,exit_wave_reverse                   = exit_waves
        modified_exit_wave_forward,modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object                 = objects
        probe_forward, probe_reverse                          = probes
        
        exit_wave_diff_forward   = modified_exit_wave_forward - exit_wave_forward
        
        object_indices           = _wrapped_indices_2D_window(position, probe_forward.shape, electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]

        probe_forward_conj                                   = xp.conj(probe_forward)
        electrostatic_conj                                   = xp.conj(electrostatic_roi)

        probe_forward_abs_squared                            = xp.abs(probe_forward)**2
        electrostatic_abs_squared                            = xp.abs(electrostatic_roi)**2

        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes, position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)

        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            probe_forward                    += probe_step_size*electrostatic_conj*exit_wave_diff_forward/(
                                                                (1-beta)*electrostatic_abs_squared + beta*xp.max(electrostatic_abs_squared))

        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        electrostatic_object[object_indices] += object_step_size*probe_forward_conj*exit_wave_diff_forward/ (
                                                                 (1-alpha)*probe_forward_abs_squared + alpha*xp.max(probe_forward_abs_squared))

        return (electrostatic_object, magnetic_object), (probe_forward,probe_reverse), position
    
    
    @staticmethod   
    def _update_function(objects:Sequence[np.ndarray],
                         probes:Sequence[np.ndarray],
                         position:np.ndarray,
                         exit_waves:Sequence[np.ndarray],
                         modified_exit_waves:Sequence[np.ndarray],
                         diffraction_patterns:Sequence[np.ndarray],
                         fix_probe: bool = False,
                         position_correction: Callable = None,
                         sobel: Callable = None,
                         reconstruction_parameters: Mapping[str,float] = None,
                         xp = np,
                         **kwargs):
        '''
        sim-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        exit_wave_forward,exit_wave_reverse                   = exit_waves
        modified_exit_wave_forward,modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object                 = objects
        probe_forward, probe_reverse                          = probes
        
        exit_wave_diff_forward   = modified_exit_wave_forward - exit_wave_forward
        exit_wave_diff_reverse   = modified_exit_wave_reverse - exit_wave_reverse
        
        object_indices           = _wrapped_indices_2D_window(position, probe_forward.shape, electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]
        magnetic_roi             = magnetic_object[object_indices]

        probe_forward_conj                                   = xp.conj(probe_forward)
        probe_reverse_conj                                   = xp.conj(probe_reverse)
        electrostatic_conj                                   = xp.conj(electrostatic_roi)
        magnetic_conj                                        = xp.conj(magnetic_roi)

        probe_forward_magnetic_abs_squared                   = xp.abs(probe_forward*magnetic_roi)**2
        probe_reverse_magnetic_abs_squared                   = xp.abs(probe_reverse*magnetic_roi)**2
        probe_forward_electrostatic_abs_squared              = xp.abs(probe_forward*electrostatic_roi)**2
        probe_reverse_electrostatic_abs_squared              = xp.abs(probe_reverse*electrostatic_roi)**2
        electrostatic_magnetic_abs_squared                   = xp.abs(electrostatic_roi*magnetic_roi)**2

        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes, position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)

        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            probe_forward                    += probe_step_size*electrostatic_conj*magnetic_conj*exit_wave_diff_forward/(
                                                                (1-beta)*electrostatic_magnetic_abs_squared + beta*xp.max(electrostatic_magnetic_abs_squared))/2
            probe_reverse                    += probe_step_size*electrostatic_conj*magnetic_roi*exit_wave_diff_reverse/(
                                                                (1-beta)*electrostatic_magnetic_abs_squared + beta*xp.max(electrostatic_magnetic_abs_squared))/2

        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        electrostatic_object[object_indices] += object_step_size*probe_forward_conj*magnetic_conj*exit_wave_diff_forward/ (
                                                                 (1-alpha)*probe_forward_magnetic_abs_squared + alpha*xp.max(probe_forward_magnetic_abs_squared))/2
        electrostatic_object[object_indices] += object_step_size*probe_reverse_conj*magnetic_roi*exit_wave_diff_reverse/ (
                                                                 (1-alpha)*probe_reverse_magnetic_abs_squared + alpha*xp.max(probe_reverse_magnetic_abs_squared))/2

        magnetic_object[object_indices]      += object_step_size*probe_forward_conj*electrostatic_conj*exit_wave_diff_forward/(
                                                                 (1-alpha)*probe_forward_electrostatic_abs_squared + alpha*xp.max(probe_forward_electrostatic_abs_squared))/2
        magnetic_object[object_indices]      -= object_step_size*probe_reverse_conj*electrostatic_conj*exit_wave_diff_reverse/(
                                                                 (1-alpha)*probe_reverse_electrostatic_abs_squared + alpha*xp.max(probe_reverse_electrostatic_abs_squared))/2

        return (electrostatic_object, magnetic_object), (probe_forward,probe_reverse), position
        
    @staticmethod   
    def _alternative_update_function(objects:Sequence[np.ndarray],
                                     probes:Sequence[np.ndarray],
                                     position:np.ndarray,
                                     exit_waves:Sequence[np.ndarray],
                                     modified_exit_waves:Sequence[np.ndarray],
                                     diffraction_patterns:Sequence[np.ndarray],
                                     fix_probe: bool = False,
                                     position_correction: Callable = None,
                                     sobel: Callable = None,
                                     reconstruction_parameters: Mapping[str,float] = None,
                                     xp = np,
                                     **kwargs):
        '''
        sim-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        exit_wave_forward,exit_wave_reverse                   = exit_waves
        modified_exit_wave_forward,modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object                 = objects
        probe_forward, probe_reverse                          = probes
        
        exit_wave_diff_forward   = modified_exit_wave_forward - exit_wave_forward
        exit_wave_diff_reverse   = modified_exit_wave_reverse - exit_wave_reverse
        
        object_indices           = _wrapped_indices_2D_window(position, probe_forward.shape, electrostatic_object.shape)
        electrostatic_roi        = electrostatic_object[object_indices]
        magnetic_roi             = magnetic_object[object_indices]

        probe_forward_conj                                   = xp.conj(probe_forward)
        electrostatic_conj                                   = xp.conj(electrostatic_roi)
        magnetic_conj                                        = xp.conj(magnetic_roi)

        probe_forward_magnetic_abs_squared                   = xp.abs(probe_forward*magnetic_roi)**2
        probe_forward_electrostatic_abs_squared              = xp.abs(probe_forward*electrostatic_roi)**2
        electrostatic_magnetic_abs_squared                   = xp.abs(electrostatic_roi*magnetic_roi)**2

        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes, position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)

        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            probe_forward                    += probe_step_size*electrostatic_conj*magnetic_conj*exit_wave_diff_forward/(
                                                                (1-beta)*electrostatic_magnetic_abs_squared + beta*xp.max(electrostatic_magnetic_abs_squared))/2
            probe_forward                    += probe_step_size*electrostatic_conj*magnetic_roi*exit_wave_diff_reverse/(
                                                                (1-beta)*electrostatic_magnetic_abs_squared + beta*xp.max(electrostatic_magnetic_abs_squared))/2

        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        electrostatic_object[object_indices] += object_step_size*probe_forward_conj*magnetic_conj*exit_wave_diff_forward/ (
                                                                 (1-alpha)*probe_forward_magnetic_abs_squared + alpha*xp.max(probe_forward_magnetic_abs_squared))/2
        electrostatic_object[object_indices] += object_step_size*probe_forward_conj*magnetic_roi*exit_wave_diff_reverse/ (
                                                                 (1-alpha)*probe_forward_magnetic_abs_squared + alpha*xp.max(probe_forward_magnetic_abs_squared))/2

        magnetic_object[object_indices]      += object_step_size*probe_forward_conj*electrostatic_conj*exit_wave_diff_forward/(
                                                                 (1-alpha)*probe_forward_electrostatic_abs_squared + alpha*xp.max(probe_forward_electrostatic_abs_squared))/2
        magnetic_object[object_indices]      -= object_step_size*probe_forward_conj*electrostatic_conj*exit_wave_diff_reverse/(
                                                                 (1-alpha)*probe_forward_electrostatic_abs_squared + alpha*xp.max(probe_forward_electrostatic_abs_squared))/2

        # return dummy probe_reverse to avoid complicating unpacking logic
        return (electrostatic_object, magnetic_object), (probe_forward,probe_reverse), position
    
    @staticmethod
    def _position_correction(objects: Sequence[np.ndarray],
                             probes: Sequence[np.ndarray],
                             position: np.ndarray,
                             exit_wave: Sequence[np.ndarray],
                             modified_exit_wave: Sequence[np.ndarray],
                             diffraction_pattern: Sequence[np.ndarray],
                             sobel:Callable,
                             position_step_size: float = 1.0,
                             xp=np,
                             **kwargs):
        '''
        sim-PIE position correction function.
        '''
        
        electrostatic_object, magnetic_object                 = objects
        probe_forward, probe_reverse                          = probes
        exit_wave_forward,exit_wave_reverse                   = exit_waves
        modified_exit_wave_forward,modified_exit_wave_reverse = modified_exit_waves
        exit_wave_diff_forward   = modified_exit_wave_forward - exit_wave_forward
        
        object_dx       = sobel(electrostatic_object,axis=0,mode='wrap')
        object_dy       = sobel(electrostatic_object,axis=1,mode='wrap')
        
        object_indices  = _wrapped_indices_2D_window(position,probe_forward.shape,electrostatic_object.shape)
        exit_wave_dx    = object_dx[object_indices]*probe_forward
        exit_wave_dy    = object_dy[object_indices]*probe_forward
        
        exit_wave_diff  = modified_exit_wave - exit_wave
        displacement_x  = xp.sum(xp.real(xp.conj(exit_wave_dx)*exit_wave_diff_forward))/xp.sum(xp.abs(exit_wave_dx)**2)
        displacement_y  = xp.sum(xp.real(xp.conj(exit_wave_dy)*exit_wave_diff_forward))/xp.sum(xp.abs(exit_wave_dy)**2)
        
        return position + position_step_size*xp.array([displacement_x,displacement_y])
    
    
    @staticmethod
    def _fix_probe_center_of_mass(probes: Sequence[np.ndarray],
                                  center_of_mass:Callable,
                                  xp = np,
                                  **kwargs):
        '''
        Fix probes CoM to array center. 
        '''
        
        probe_center = xp.array(probes[0].shape)/2
        
        _probes      = []
        for k in range(len(probes)):
            com          = center_of_mass(xp.abs(probes[k]) ** 2)
            _probes.append(fft_shift(probes[k], probe_center - xp.array(com)))
        
        return tuple(_probes)
        
    def _prepare_functions_queue(self,
                                 max_iterations: int,
                                 warmup_update_steps: int = 0,
                                 common_probe:bool = False,
                                 pre_position_correction_update_steps: int = None,
                                 pre_probe_correction_update_steps: int = None,
                                 **kwargs):
        '''
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting. 
        '''
        _overlap_projection  = self._alternative_overlap_projection if common_probe else self._overlap_projection
        _update_function     = self._alternative_update_function    if common_probe else self._update_function
        
        total_update_steps   = max_iterations*self._num_diffraction_patterns
        queue_summary        = "Ptychographic reconstruction will perform the following steps:"
        
        functions_tuple      = (self._warmup_overlap_projection,self._warmup_fourier_projection, self._warmup_update_function, None)
        functions_queue      = [functions_tuple]
        
        if pre_position_correction_update_steps is None:
            functions_queue *= warmup_update_steps
            queue_summary   += f"\n--Regularized PIE for {warmup_update_steps} steps"

            functions_tuple = (_overlap_projection,self._fourier_projection, _update_function, None)
            remaining_update_steps = total_update_steps - warmup_update_steps
            functions_queue += [functions_tuple]*remaining_update_steps
            queue_summary   += f"\n--Simultaneous PIE for {remaining_update_steps} steps"
        else:
            if warmup_update_steps <= pre_position_correction_update_steps:
                functions_queue *= warmup_update_steps
                queue_summary   += f"\n--Regularized PIE for {warmup_update_steps} steps"

                functions_tuple = (_overlap_projection,self._fourier_projection, _update_function, None)
                remaining_update_steps = pre_position_correction_update_steps - warmup_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Simultaneous PIE for {remaining_update_steps} steps"
                
                functions_tuple = (_overlap_projection,self._fourier_projection, _update_function, self._position_correction)
                remaining_update_steps = total_update_steps - pre_position_correction_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Simultaneous PIE with position correction for {remaining_update_steps} steps"
            else:
                functions_queue *= pre_position_correction_update_steps
                queue_summary   += f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"

                functions_tuple = (self._warmup_overlap_projection,self._warmup_fourier_projection, self._warmup_update_function, self._position_correction)
                remaining_update_steps = warmup_update_steps - pre_position_correction_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"
                
                functions_tuple = (_overlap_projection,self._fourier_projection, _update_function, self._position_correction)
                remaining_update_steps = total_update_steps - warmup_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Simultaneous PIE with position correction for {remaining_update_steps} steps"
        
        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"
            
        if common_probe:
            queue_summary += f"\n--Using the first probe as a common probe for both objects"
        
        functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
        
        return functions_queue, queue_summary
    
    def reconstruct(self,
                    max_iterations: int = 5,
                    return_iterations: bool = False,
                    warmup_update_steps: int = 0,
                    common_probe: bool = False,
                    fix_com: bool = True,
                    random_seed = None,
                    verbose: bool = False,
                    parameters: Mapping[str,float] = None,
                    measurement_output_view: str = 'padded',
                    functions_queue: Iterable = None,
                    **kwargs):
        '''
        '''
        for key in kwargs.keys():
            if (key not in reconstruction_symbols.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))
                
        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)
                 
        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                                        max_iterations,
                                        warmup_update_steps= warmup_update_steps,
                                        common_probe= common_probe,
                                        pre_position_correction_update_steps = self._reconstruction_parameters['pre_position_correction_update_steps'],
                                        pre_probe_correction_update_steps    = self._reconstruction_parameters['pre_probe_correction_update_steps'])
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [[function_tuples]*self._num_diffraction_patterns for function_tuples in functions_queue]
            elif len(functions_queue) == max_iterations*self._num_diffraction_patterns:
                functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
            else:
                raise ValueError()
        
        self._functions_queue = functions_queue
        
        ### Main Loop
        xp                  = get_array_module_from_device(self._device)
        outer_pbar          = ProgressBar(total=max_iterations,leave=False)
        inner_pbar          = ProgressBar(total=self._num_diffraction_patterns,leave=False)
        indices             = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(self._experimental_parameters['object_px_padding'])
        center_of_mass      = get_scipy_module(xp).ndimage.center_of_mass
        sobel               = get_scipy_module(xp).ndimage.sobel
        
        if return_iterations:
            objects_iterations   = []
            probes_iterations    = []
            positions_iterations = []
            sse_iterations       = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        for iteration_index, iteration_step in enumerate(self._functions_queue):
            
            inner_pbar.reset()
            
            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse    = 0.0
            
            for update_index, update_step in enumerate(iteration_step):
                
                index               = indices[update_index]
                position            = self._positions_px[index]
                
                # Skip empty diffraction patterns
                diffraction_pattern = tuple(dp[index] for dp in self._diffraction_patterns)
                
                if any(tuple(xp.sum(dp) == 0.0 for dp in diffraction_pattern)):
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i  = iteration_index*self._num_diffraction_patterns + update_index
                
                if self._reconstruction_parameters['pre_probe_correction_update_steps'] is None:
                    fix_probe       = False
                else:
                    fix_probe       = global_iteration_i < self._reconstruction_parameters['pre_probe_correction_update_steps']
                    
                if warmup_update_steps != 0 and global_iteration_i == (warmup_update_steps + 1):
                    self._probes = (self._probes[0], self._probes[0].copy())

                _overlap_projection,_fourier_projection,_update_function,_position_correction = update_step
            
                self._probes, exit_wave                                = _overlap_projection(self._objects,
                                                                                             self._probes,
                                                                                             position,
                                                                                             old_position,
                                                                                             xp=xp)
                
                modified_exit_wave, self._sse                          = _fourier_projection(exit_wave,
                                                                                             diffraction_pattern,
                                                                                             self._sse,
                                                                                             xp=xp)

                self._objects, self._probes, self._positions_px[index] = _update_function(self._objects,
                                                                                          self._probes,
                                                                                          position,
                                                                                          exit_wave,
                                                                                          modified_exit_wave,
                                                                                          diffraction_pattern,
                                                                                          fix_probe = fix_probe,
                                                                                          position_correction = _position_correction,
                                                                                          sobel = sobel,
                                                                                          reconstruction_parameters = self._reconstruction_parameters,
                                                                                          xp = xp)
                    
                old_position    = position
                inner_pbar.update(1)
              
            # Shift probe back to origin
            self._probes     = tuple(fft_shift(_probe,xp.round(position) - position) for _probe in self._probes)
            
            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(self._probes,center_of_mass,xp=xp)
                 
            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (xp.mean(self._positions_px,axis=0) - self._positions_px_com)
                self._reconstruction_parameters['position_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            
            # Update Parameters
            self._reconstruction_parameters['object_step_size'] *= self._reconstruction_parameters['step_size_damping_rate']
            self._reconstruction_parameters['probe_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            self._sse                                           /= self._num_diffraction_patterns
            
            if return_iterations:
                objects_iterations.append(copy(self._objects))
                probes_iterations.append(copy(self._probes))
                positions_iterations.append(self._positions_px.copy() * xp.array(self.sampling))
                sse_iterations.append(self._sse)
            
            if verbose:
                print(f'----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}')
            
            outer_pbar.update(1)
        
        inner_pbar.close()
        outer_pbar.close()
        
        #  Return Results
        if return_iterations:
            results = map(self._prepare_measurement_outputs,
                          objects_iterations,
                          probes_iterations,
                          positions_iterations,
                          sse_iterations)
            
            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(self._objects,
                                                        self._probes,
                                                        self._positions_px * xp.array(self.sampling),
                                                        self._sse)
            return results
        
        
    def _prepare_measurement_outputs(self,
                                     objects:Sequence[np.ndarray],
                                     probes: Sequence[np.ndarray],
                                     positions: np.ndarray,
                                     sse: np.ndarray):
        '''
        Base measurement outputs function operating on a single iteration's outputs.
        Called using map if more than one iteration required.
        '''
        
        calibrations = tuple(Calibration(0, s, units='Å', name = n, endpoint=False) for s,n in zip(self.sampling,('x','y')))
        
        measurement_objects = tuple(Measurement(asnumpy(_object),calibrations) for _object in objects)
        measurement_probes  = tuple(Measurement(asnumpy(_probe),calibrations) for _probe in probes)
        
        return measurement_objects, measurement_probes, asnumpy(positions), sse

class MixedStatePtychographicOperator(AbstractPtychographicOperator):
    '''
    Mixed-State PIE Operator.
    
    diffraction_patterns dimensions   : (J,M,N)
    objects dimensions                : (P,Q)
    probes dimensions                 : (K,R,S)
    '''
    def __init__(self,
                 diffraction_patterns:Union[np.ndarray,Measurement],
                 energy: float,
                 num_probes:int,
                 region_of_interest_shape: Sequence[int] = None,
                 objects: np.ndarray = None,
                 probes: Union[np.ndarray,Probe] = None,
                 positions: np.ndarray = None,
                 semiangle_cutoff: float = None,
                 preprocess: bool = False,
                 device: str = 'cpu',
                 parameters: Mapping[str,float] = None,
                 **kwargs):
        
        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()) and (key not in experimental_symbols):
                raise ValueError('{} not a recognized parameter'.format(key))
    
        self._polar_parameters        = dict(zip(polar_symbols, [0.] * len(polar_symbols)))
        self._experimental_parameters = dict(zip(experimental_symbols, [None] * len(experimental_symbols)))
        
        if parameters is None:
            parameters = {}
        
        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(parameters,
                                                                                        self._polar_parameters,
                                                                                        self._experimental_parameters)
        
        self._region_of_interest_shape          = region_of_interest_shape
        self._energy                            = energy
        self._semiangle_cutoff                  = semiangle_cutoff
        self._positions                         = positions
        self._device                            = device
        self._objects                           = objects
        self._probes                            = probes
        self._num_probes                        = num_probes
        self._diffraction_patterns              = diffraction_patterns
    
        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False
            
        
    def preprocess(self):
        '''
        Mixed PIE preprocessing method, to do the following:
        - Pads CBED patterns (J,M,N) to ROI dimensions -> (J,R,S)
        - Prepares scanning positions in real-space resolution pixels
        - Initializes probes (K,R,S) & objects (P,Q)
        '''
        
        self._preprocessed = True
        
        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            self._diffraction_patterns, angular_sampling, step_sizes = self._extract_calibrations_from_measurement_object(
                                                                                              self._diffraction_patterns,
                                                                                              self._energy)
            self._experimental_parameters['angular_sampling'] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters['scan_step_sizes']   = step_sizes
                
        # Preprocess Diffraction Patterns
        xp                             = get_array_module_from_device(self._device)
        self._diffraction_patterns     = copy_to_device(self._diffraction_patterns,self._device)
        
        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters['grid_scan_shape'] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns                       = self._diffraction_patterns.reshape(
                                                                    (-1,)+self._diffraction_patterns.shape[-2:])
            
        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]
        
        self._diffraction_patterns         = self._pad_diffraction_patterns(self._diffraction_patterns,
                                                                            self._region_of_interest_shape)
        self._num_diffraction_patterns     = self._diffraction_patterns.shape[0]
        
        if self._experimental_parameters['background_counts_cutoff'] is not None:
            self._diffraction_patterns[self._diffraction_patterns < self._experimental_parameters['background_counts_cutoff']] = 0.0

        if self._experimental_parameters['counts_scaling_factor'] is not None:
            self._diffraction_patterns /= self._experimental_parameters['counts_scaling_factor']   

        self._diffraction_patterns      = xp.fft.ifftshift(xp.sqrt(self._diffraction_patterns),axes=(-2,-1))
        
        
        # Scan Positions Initialization
        positions_px, self._experimental_parameters = self._calculate_scan_positions_in_pixels(self._positions,
                                                                                               self.sampling,
                                                                                               self._region_of_interest_shape,
                                                                                               self._experimental_parameters)
        
        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y  = self._experimental_parameters['object_px_padding']
            p, q          = np.max(positions_px,axis=0)
            p             = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(int)
            q             = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(int)
            self._objects = xp.ones((p,q),dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects,self._device)
                
        self._positions_px                                = copy_to_device(positions_px,self._device)
        self._positions_px_com                            = xp.mean(self._positions_px,axis=0)
        
        # Probes Initialization
        if self._probes is None:
            ctf           = CTF(energy=self._energy,
                                semiangle_cutoff=self._semiangle_cutoff,
                                parameters= self._polar_parameters)
            self._probes  = Probe(semiangle_cutoff = self._semiangle_cutoff,
                                 energy           = self._energy,
                                 gpts             = self._region_of_interest_shape,
                                 sampling         = self.sampling,
                                 ctf              = ctf,
                                 device           = self._device).build().array
        else:
            if isinstance(self._probes,Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = copy_to_device(self._probes.build().array,self._device)
            else:
                self._probes = copy_to_device(self._probes,self._device)
                
                
        self._probes  = xp.tile(self._probes,(self._num_probes,1,1))
        self._probes /= (xp.arange(self._num_probes)[:,None,None]+1)
        
        return self
        
        
    @staticmethod
    def _warmup_overlap_projection(objects:np.ndarray,
                                   probes:np.ndarray,
                                   position:np.ndarray,
                                   old_position:np.ndarray,
                                   xp = np,
                                   **kwargs):
        '''
        r-PIE overlap projection:
        \psi_{R_j}(r) = O_{R_j}(r) * P(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probes[0]                = fft_shift(probes[0],fractional_position - old_fractional_position)
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape)
        object_roi               = objects[object_indices]
        exit_wave                = object_roi * probes[0]
        
        return probes, exit_wave
    
    @staticmethod
    def _overlap_projection(objects:np.ndarray,
                            probes:np.ndarray,
                            position:np.ndarray,
                            old_position:np.ndarray,
                            xp = np,
                            **kwargs):
        '''
        mix-PIE overlap projection:
        \psi^k_{R_j}(r) = O_{R_j}(r) * P^k(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probes                   = fft_shift(probes,fractional_position - old_fractional_position)
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape)
        object_roi               = objects[object_indices]
        
        exit_waves               = xp.empty_like(probes)
        for k in range(probes.shape[0]):
            exit_waves[k]        = object_roi * probes[k]
        
        return probes, exit_waves
     
    @staticmethod
    def _warmup_fourier_projection(exit_waves:np.ndarray,
                                   diffraction_patterns:np.ndarray,
                                   sse:float,
                                   xp = np,
                                   **kwargs):
        '''
        r-PIE Fourier-amplitude modification projection:
        \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]
        '''
        exit_wave_fft       = xp.fft.fft2(exit_waves)
        sse                += xp.mean(xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns)**2)/xp.sum(diffraction_patterns**2)
        modified_exit_wave  = xp.fft.ifft2(diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft)))
        
        return modified_exit_wave, sse
    
    @staticmethod
    def _fourier_projection(exit_waves:np.ndarray,
                            diffraction_patterns:np.ndarray,
                            sse:float,
                            xp = np,
                            **kwargs):
        '''
        mix-PIE Fourier-amplitude modification projection:
        \psi'^k_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^k_{R_j}(u)] / \sqrt{\sum_k|F[\psi^k_{R_j}(u)]|^2}]
        '''
        exit_waves_fft            = xp.fft.fft2(exit_waves,axes=(-2,-1))
        intensity_norm            = xp.sqrt(xp.sum(xp.abs(exit_waves_fft)**2,axis=0))
        amplitude_modification    = diffraction_patterns/intensity_norm
        sse                      += xp.mean(xp.abs(intensity_norm - diffraction_patterns)**2)/xp.sum(diffraction_patterns**2)

        modified_exit_wave        = xp.fft.ifft2(amplitude_modification[None]*exit_waves_fft,axes=(-2,-1))
        
        return modified_exit_wave, sse

    @staticmethod   
    def _warmup_update_function(objects:np.ndarray,
                                probes:np.ndarray,
                                position:np.ndarray,
                                exit_waves:np.ndarray,
                                modified_exit_waves:np.ndarray,
                                diffraction_patterns:np.ndarray,
                                fix_probe: bool = False,
                                position_correction: Callable = None,
                                sobel: Callable = None,
                                reconstruction_parameters: Mapping[str,float] = None,
                                xp = np,
                                **kwargs):
        '''
        r-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape)
        object_roi               = objects[object_indices]
        
        exit_wave_diff           = modified_exit_waves - exit_waves
        
        probe_conj               = xp.conj(probes[0])
        probe_abs_squared        = xp.abs(probes[0])**2
        obj_conj                 = xp.conj(object_roi)
        obj_abs_squared          = xp.abs(object_roi)**2
        
        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes, position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)
            
        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        objects[object_indices] += object_step_size * probe_conj*exit_wave_diff / (
                                    (1-alpha)*probe_abs_squared + alpha*xp.max(probe_abs_squared))
        
        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            probes[0]           += probe_step_size * obj_conj*exit_wave_diff / (
                                    (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared))
            
        return objects, probes, position
    
    @staticmethod   
    def _update_function(objects:np.ndarray,
                         probes:np.ndarray,
                         position:np.ndarray,
                         exit_waves:np.ndarray,
                         modified_exit_waves:np.ndarray,
                         diffraction_patterns:np.ndarray,
                         fix_probe: bool = False,
                         orthogonalize_probes: bool = False,
                         position_correction: Callable = None,
                         sobel: Callable = None,
                         reconstruction_parameters: Mapping[str,float] = None,
                         xp = np,
                         **kwargs):
        '''
        mix-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape)
        object_roi               = objects[object_indices]
        
        exit_wave_diff           = modified_exit_waves - exit_waves
        
        probe_conj               = xp.conj(probes)
        probe_abs_squared_norm   = xp.sum(xp.abs(probes)**2,axis=0)
        obj_conj                 = xp.conj(object_roi)
        obj_abs_squared          = xp.abs(object_roi)**2
        
        if position_correction is not None:
            position_step_size   = reconstruction_parameters['position_step_size']
            position             = position_correction(objects, probes,position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)
            
        alpha                    = reconstruction_parameters['alpha']
        object_step_size         = reconstruction_parameters['object_step_size']
        objects[object_indices] += object_step_size * xp.sum(probe_conj*exit_wave_diff,axis=0) / (
                                    (1-alpha)*probe_abs_squared_norm + alpha*xp.max(probe_abs_squared_norm))
        
        if not fix_probe:
            beta                 = reconstruction_parameters['beta']
            probe_step_size      = reconstruction_parameters['probe_step_size']
            update_numerator     = probe_step_size * obj_conj[None]*exit_wave_diff
            update_denominator   = (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared)
            probes              += update_numerator/update_denominator[None]
            
            if orthogonalize_probes:
                probes           = _orthogonalize(probes.reshape((probes.shape[0],-1))).reshape(probes.shape)
            
        return objects, probes, position
    
    @staticmethod
    def _position_correction(objects: np.ndarray,
                             probes: np.ndarray,
                             position:np.ndarray,
                             exit_wave:np.ndarray,
                             modified_exit_wave: np.ndarray,
                             diffraction_pattern:np.ndarray,
                             sobel:Callable,
                             position_step_size: float = 1.0,
                             xp=np,
                             **kwargs):
        '''
        r-PIE position correction function.
        '''

        object_dx       = sobel(objects,axis=0,mode='wrap')
        object_dy       = sobel(objects,axis=1,mode='wrap')
        
        object_indices  = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape)
        exit_wave_dx    = object_dx[object_indices]*probes[0]
        exit_wave_dy    = object_dy[object_indices]*probes[0]
        
        exit_wave_diff  = modified_exit_wave[0] - exit_wave[0]
        displacement_x  = xp.sum(xp.real(xp.conj(exit_wave_dx)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dx)**2)
        displacement_y  = xp.sum(xp.real(xp.conj(exit_wave_dy)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dy)**2)
        
        return position + position_step_size*xp.array([displacement_x,displacement_y])
    
    
    @staticmethod
    def _fix_probe_center_of_mass(probes:np.ndarray,
                                  center_of_mass:Callable,
                                  xp = np,
                                  **kwargs):
        '''
        Fix probes CoM to array center. 
        '''
        
        probe_center = xp.array(probes.shape[-2:])/2
        for k in range(probes.shape[0]):
            com          = center_of_mass(xp.abs(probes[k]) ** 2)
            probes[k]    = fft_shift(probes[k], probe_center - xp.array(com))
        
        return probes
        
    def _prepare_functions_queue(self,
                                 max_iterations: int,
                                 warmup_update_steps: int = 0,
                                 pre_position_correction_update_steps: int = None,
                                 pre_probe_correction_update_steps: int = None,
                                 **kwargs):
        '''
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting. 
        '''
        total_update_steps   = max_iterations*self._num_diffraction_patterns
        queue_summary        = "Ptychographic reconstruction will perform the following steps:"
    
        functions_tuple      = (self._warmup_overlap_projection,self._warmup_fourier_projection, self._warmup_update_function, None)
        functions_queue      = [functions_tuple]
        
        if pre_position_correction_update_steps is None:
            functions_queue *= warmup_update_steps
            queue_summary   += f"\n--Regularized PIE for {warmup_update_steps} steps"

            functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, None)
            remaining_update_steps = total_update_steps - warmup_update_steps
            functions_queue += [functions_tuple]*remaining_update_steps
            queue_summary   += f"\n--Mixed-State PIE for {remaining_update_steps} steps"
        else:
            if warmup_update_steps <= pre_position_correction_update_steps:
                functions_queue *= warmup_update_steps
                queue_summary   += f"\n--Regularized PIE for {warmup_update_steps} steps"

                functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, None)
                remaining_update_steps = pre_position_correction_update_steps - warmup_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Mixed-State PIE for {remaining_update_steps} steps"
                
                functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, self._position_correction)
                remaining_update_steps = total_update_steps - pre_position_correction_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Mixed-State PIE with position correction for {remaining_update_steps} steps"
            else:
                functions_queue *= pre_position_correction_update_steps
                queue_summary   += f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"

                functions_tuple = (self._warmup_overlap_projection,self._warmup_fourier_projection, self._warmup_update_function, self._position_correction)
                remaining_update_steps = warmup_update_steps - pre_position_correction_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"
                
                functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, self._position_correction)
                remaining_update_steps = total_update_steps - warmup_update_steps
                functions_queue += [functions_tuple]*remaining_update_steps
                queue_summary   += f"\n--Mixed-State PIE with position correction for {remaining_update_steps} steps"
        
        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"
        
        functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
        
        return functions_queue, queue_summary
    
    def reconstruct(self,
                    max_iterations: int = 5,
                    return_iterations: bool = False,
                    probe_orthogonalization_frequency: int = None,
                    warmup_update_steps: int = 0,
                    fix_com: bool = True,
                    random_seed = None,
                    verbose: bool = False,
                    parameters: Mapping[str,float] = None,
                    measurement_output_view: str = 'padded',
                    functions_queue: Iterable = None,
                    **kwargs):
        '''
        '''
        for key in kwargs.keys():
            if (key not in reconstruction_symbols.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))
                
        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)
                 
        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                                        max_iterations,
                                        warmup_update_steps=warmup_update_steps,
                                        pre_position_correction_update_steps = self._reconstruction_parameters['pre_position_correction_update_steps'],
                                        pre_probe_correction_update_steps    = self._reconstruction_parameters['pre_probe_correction_update_steps'])
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [[function_tuples]*self._num_diffraction_patterns for function_tuples in functions_queue]
            elif len(functions_queue) == max_iterations*self._num_diffraction_patterns:
                functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
            else:
                raise ValueError()
        
        self._functions_queue = functions_queue
        
        ### Main Loop
        xp                  = get_array_module_from_device(self._device)
        outer_pbar          = ProgressBar(total=max_iterations,leave=False)
        inner_pbar          = ProgressBar(total=self._num_diffraction_patterns,leave=False)
        indices             = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(self._experimental_parameters['object_px_padding'])
        center_of_mass      = get_scipy_module(xp).ndimage.center_of_mass
        sobel               = get_scipy_module(xp).ndimage.sobel
        
        if return_iterations:
            objects_iterations   = []
            probes_iterations    = []
            positions_iterations = []
            sse_iterations       = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        for iteration_index, iteration_step in enumerate(self._functions_queue):
            
            inner_pbar.reset()
            
            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse    = 0.0
            
            for update_index, update_step in enumerate(iteration_step):
                
                index               = indices[update_index]
                position            = self._positions_px[index]
                
                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i  = iteration_index*self._num_diffraction_patterns + update_index
                
                if self._reconstruction_parameters['pre_probe_correction_update_steps'] is None:
                    fix_probe       = False
                else:
                    fix_probe       = global_iteration_i < self._reconstruction_parameters['pre_probe_correction_update_steps']
                    
                if probe_orthogonalization_frequency is None:
                    orthogonalize_probes = False
                else:
                    orthogonalize_probes = not (global_iteration_i % probe_orthogonalization_frequency)

                _overlap_projection,_fourier_projection,_update_function,_position_correction = update_step
            
                self._probes, exit_wave                                = _overlap_projection(self._objects,
                                                                                             self._probes,
                                                                                             position,
                                                                                             old_position,
                                                                                             xp=xp)
                
                modified_exit_wave, self._sse                          = _fourier_projection(exit_wave,
                                                                                             diffraction_pattern,
                                                                                             self._sse,
                                                                                             xp=xp)

                self._objects, self._probes, self._positions_px[index] = _update_function(self._objects,
                                                                                          self._probes,
                                                                                          position,
                                                                                          exit_wave,
                                                                                          modified_exit_wave,
                                                                                          diffraction_pattern,
                                                                                          fix_probe = fix_probe,
                                                                                          orthogonalize_probes = orthogonalize_probes,
                                                                                          position_correction = _position_correction,
                                                                                          sobel = sobel,
                                                                                          reconstruction_parameters = self._reconstruction_parameters,
                                                                                          xp = xp)
                    
                old_position    = position
                inner_pbar.update(1)
              
            # Shift probe back to origin
            self._probes     = fft_shift(self._probes,xp.round(position) - position)
            
            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(self._probes,center_of_mass,xp=xp)
                
            # Probe Orthogonalization
            if probe_orthogonalization_frequency is not None:
                self._probes    = _orthogonalize(self._probes.reshape((self._num_probes,-1))).reshape(self._probes.shape)  
                 
            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (xp.mean(self._positions_px,axis=0) - self._positions_px_com)
                self._reconstruction_parameters['position_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            
            # Update Parameters
            self._reconstruction_parameters['object_step_size'] *= self._reconstruction_parameters['step_size_damping_rate']
            self._reconstruction_parameters['probe_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            self._sse                                           /= self._num_diffraction_patterns
            
            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(self._positions_px.copy() * xp.array(self.sampling))
                sse_iterations.append(self._sse)
            
            if verbose:
                print(f'----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}')
            
            outer_pbar.update(1)
        
        inner_pbar.close()
        outer_pbar.close()
        
        #  Return Results
        if return_iterations:
            results = map(self._prepare_measurement_outputs,
                          objects_iterations,
                          probes_iterations,
                          positions_iterations,
                          sse_iterations)
            
            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(self._objects,
                                                        self._probes,
                                                        self._positions_px * xp.array(self.sampling),
                                                        self._sse)
            return results
        
        
    def _prepare_measurement_outputs(self,
                                     objects:np.ndarray,
                                     probes: np.ndarray,
                                     positions: np.ndarray,
                                     sse: np.ndarray):
        '''
        Base measurement outputs function operating on a single iteration's outputs.
        Called using map if more than one iteration required.
        '''
        
        calibrations = tuple(Calibration(0, s, units='Å', name = n, endpoint=False) for s,n in zip(self.sampling,('x','y')))
        
        measurement_objects = Measurement(asnumpy(objects),calibrations)
        measurement_probes  = [Measurement(asnumpy(probe),calibrations) for probe in probes]
        
        return measurement_objects, measurement_probes, asnumpy(positions), sse

class MultislicePtychographicOperator(AbstractPtychographicOperator):
    '''
    Multislice PIE Operator.
    
    diffraction_patterns dimensions   : (J,M,N)
    objects dimensions                : (T,P,Q)
    probes dimensions                 : (T,R,S)
    '''
    def __init__(self,
                 diffraction_patterns:Union[np.ndarray,Measurement],
                 energy: float,
                 num_slices: int,
                 slice_thicknesses: Union[float,Sequence[float]],
                 region_of_interest_shape: Sequence[int] = None,
                 objects: np.ndarray = None,
                 probes: Union[np.ndarray,Probe] = None,
                 positions: np.ndarray = None,
                 semiangle_cutoff: float = None,
                 preprocess: bool = False,
                 device: str = 'cpu',
                 parameters: Mapping[str,float] = None,
                 **kwargs):
        
        for key in kwargs.keys():
            if (key not in polar_symbols) and (key not in polar_aliases.keys()) and (key not in experimental_symbols):
                raise ValueError('{} not a recognized parameter'.format(key))
    
        self._polar_parameters        = dict(zip(polar_symbols, [0.] * len(polar_symbols)))
        self._experimental_parameters = dict(zip(experimental_symbols, [None] * len(experimental_symbols)))
        
        if parameters is None:
            parameters = {}
        
        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(parameters,
                                                                                        self._polar_parameters,
                                                                                        self._experimental_parameters)
        
        slice_thicknesses      = np.array(slice_thicknesses)
        if slice_thicknesses.shape == ():
            slice_thicknesses  = np.tile(slice_thicknesses,num_slices)
        
        self._region_of_interest_shape   = region_of_interest_shape
        self._energy                     = energy
        self._semiangle_cutoff           = semiangle_cutoff
        self._positions                  = positions
        self._device                     = device
        self._objects                    = objects
        self._probes                     = probes
        self._num_slices                 = num_slices
        self._slice_thicknesses          = slice_thicknesses
        self._diffraction_patterns       = diffraction_patterns
    
        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False
            
        
    def preprocess(self):
        '''
        Multislice PIE preprocessing method, to do the following:
        - Pads CBED patterns (J,M,N) to ROI dimensions -> (J,R,S)
        - Prepares scanning positions in real-space resolution pixels
        - Initializes probes (T,R,S) & objects (T,P,Q)
        '''
        
        self._preprocessed = True
        
        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            self._diffraction_patterns, angular_sampling, step_sizes = self._extract_calibrations_from_measurement_object(
                                                                                              self._diffraction_patterns,
                                                                                              self._energy)
            self._experimental_parameters['angular_sampling'] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters['scan_step_sizes']   = step_sizes
                
        # Preprocess Diffraction Patterns
        xp                             = get_array_module_from_device(self._device)
        self._diffraction_patterns     = copy_to_device(self._diffraction_patterns,self._device)
        
        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters['grid_scan_shape'] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns                       = self._diffraction_patterns.reshape(
                                                                    (-1,)+self._diffraction_patterns.shape[-2:])
            
        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]
        
        self._diffraction_patterns         = self._pad_diffraction_patterns(self._diffraction_patterns,
                                                                            self._region_of_interest_shape)
        self._num_diffraction_patterns     = self._diffraction_patterns.shape[0]
        
        if self._experimental_parameters['background_counts_cutoff'] is not None:
            self._diffraction_patterns[self._diffraction_patterns < self._experimental_parameters['background_counts_cutoff']] = 0.0

        if self._experimental_parameters['counts_scaling_factor'] is not None:
            self._diffraction_patterns /= self._experimental_parameters['counts_scaling_factor']   

        self._diffraction_patterns      = xp.fft.ifftshift(xp.sqrt(self._diffraction_patterns),axes=(-2,-1))
        
        
        # Scan Positions Initialization
        positions_px, self._experimental_parameters = self._calculate_scan_positions_in_pixels(self._positions,
                                                                                               self.sampling,
                                                                                               self._region_of_interest_shape,
                                                                                               self._experimental_parameters)
        
        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y  = self._experimental_parameters['object_px_padding']
            p, q          = np.max(positions_px,axis=0)
            p             = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(int)
            q             = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(int)
            self._objects = xp.ones((self._num_slices,p,q),dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects,self._device)
                
        self._positions_px                                = copy_to_device(positions_px,self._device)
        self._positions_px_com                            = xp.mean(self._positions_px,axis=0)
        
        # Probes Initialization
        if self._probes is None:
            ctf           = CTF(energy=self._energy,
                                semiangle_cutoff=self._semiangle_cutoff,
                                parameters= self._polar_parameters)
            _probes       = Probe(semiangle_cutoff = self._semiangle_cutoff,
                                 energy           = self._energy,
                                 gpts             = self._region_of_interest_shape,
                                 sampling         = self.sampling,
                                 ctf              = ctf,
                                 device           = self._device).build().array
            
        else:
            if isinstance(self._probes,Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                _probes = copy_to_device(self._probes.build().array,self._device)
            else:
                _probes = copy_to_device(self._probes,self._device)
        
        self._probes = xp.zeros((self._num_slices,) + _probes.shape, dtype=xp.complex64)
        self._probes[0] = _probes
        
        return self
        
    @staticmethod
    def _overlap_projection(objects:np.ndarray,
                            probes:np.ndarray,
                            position:np.ndarray,
                            old_position:np.ndarray,
                            propagator: FresnelPropagator = None,
                            slice_thicknesses: Sequence[float] = None,
                            sampling: Sequence[float] = None,
                            wavelength: float = None,
                            fft2_convolve: Callable = None,
                            xp = np,
                            **kwargs):
        '''
        MS-PIE overlap projection:
        \psi^n_{R_j}(r) = O^n_{R_j}(r) * P^n(r)
        '''
        
        fractional_position      = position - xp.round(position)
        old_fractional_position  = old_position - xp.round(old_position)
        
        probes[0]                = fft_shift(probes[0],fractional_position - old_fractional_position)
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape[-2:])
        exit_waves               = xp.empty_like(probes)
        
        # Removed antialiasing - didn't seem to add much, and more consistent w/o modifying self._objects here
        #objects                  = antialias_filter._bandlimit(objects)
        
        num_slices = slice_thicknesses.shape[0]
        for s in range(num_slices):
            exit_waves[s] = objects[s][object_indices]*probes[s]
            if s+1 < num_slices:
                probes[s+1] = _propagate_array(propagator,
                                               exit_waves[s],
                                               sampling,
                                               wavelength,
                                               slice_thicknesses[s],
                                               fft2_convolve = fft2_convolve,
                                               overwrite = False,
                                               xp = xp)
            
        return probes, exit_waves
     
    @staticmethod
    def _fourier_projection(exit_waves:np.ndarray,
                            diffraction_patterns:np.ndarray,
                            sse:float,
                            xp = np,
                            **kwargs):
        '''
        MS-PIE Fourier-amplitude modification projection:
        \psi'^N_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^N_{R_j}(u)] / |F[\psi^N_{R_j}(u)]|]
        '''
        
        modified_exit_waves     = xp.empty_like(exit_waves)
        exit_wave_fft           = xp.fft.fft2(exit_waves[-1])
        sse                    += xp.mean(xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns)**2)/xp.sum(diffraction_patterns**2)
        modified_exit_waves[-1] = xp.fft.ifft2(diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft)))
        
        return modified_exit_waves, sse
        
    @staticmethod   
    def _update_function(objects:np.ndarray,
                         probes:np.ndarray,
                         position:np.ndarray,
                         exit_waves:np.ndarray,
                         modified_exit_waves:np.ndarray,
                         diffraction_patterns:np.ndarray,
                         fix_probe: bool = False,
                         position_correction: Callable = None,
                         sobel: Callable = None,
                         reconstruction_parameters: Mapping[str,float] = None,
                         propagator: FresnelPropagator = None,
                         slice_thicknesses: Sequence[float] = None,
                         sampling: Sequence[float] = None,
                         wavelength: float = None,
                         fft2_convolve: Callable = None,
                         xp = np,
                         **kwargs):
        '''
        MS-PIE objects and probes update function.
        Optionally performs position correction too.
        '''
        
        object_indices           = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape[-2:])
        num_slices               = slice_thicknesses.shape[0]
        
        if position_correction is not None:
            position             = position_correction(objects, probes, position, exit_waves, modified_exit_waves, diffraction_patterns,
                                                       sobel=sobel,position_step_size=position_step_size, xp=xp)
        
        for s in reversed(range(num_slices)):
            exit_wave                   = exit_waves[s]
            modified_exit_wave          = modified_exit_waves[s]
            exit_wave_diff              = modified_exit_wave - exit_wave
        
            probe_conj                  = xp.conj(probes[s])
            probe_abs_squared           = xp.abs(probes[s])**2
            obj_conj                    = xp.conj(objects[s][object_indices])
            obj_abs_squared             = xp.abs(objects[s][object_indices])**2
            
            alpha                       = reconstruction_parameters['alpha']
            object_step_size            = reconstruction_parameters['object_step_size']
            objects[s][object_indices] += object_step_size * probe_conj*exit_wave_diff / (
                                           (1-alpha)*probe_abs_squared + alpha*xp.max(probe_abs_squared))
        
            if not fix_probe or s > 0:
                beta                 = reconstruction_parameters['beta']
                probe_step_size      = reconstruction_parameters['probe_step_size']
                probes[s]           += probe_step_size * obj_conj*exit_wave_diff / (
                                        (1-beta)*obj_abs_squared + beta*xp.max(obj_abs_squared))
                
            if s > 0:
                modified_exit_waves[s-1] = _propagate_array(propagator,
                                                            probes[s],
                                                            sampling,
                                                            wavelength,
                                                            -slice_thicknesses[s-1],
                                                            fft2_convolve = fft2_convolve,
                                                            overwrite = False,
                                                            xp = xp)
            
        return objects, probes, position
    
    
    @staticmethod
    def _position_correction(objects: np.ndarray,
                             probes: np.ndarray,
                             position:np.ndarray,
                             exit_wave:np.ndarray,
                             modified_exit_wave: np.ndarray,
                             diffraction_pattern:np.ndarray,
                             sobel:Callable,
                             position_step_size: float = 1.0,
                             xp=np,
                             **kwargs):
        '''
        r-PIE position correction function.
        '''

        object_dx       = sobel(objects[-1],axis=0,mode='wrap')
        object_dy       = sobel(objects[-1],axis=1,mode='wrap')
        
        object_indices  = _wrapped_indices_2D_window(position,probes.shape[-2:],objects.shape[-2:])
        exit_wave_dx    = object_dx[object_indices]*probes[-1]
        exit_wave_dy    = object_dy[object_indices]*probes[-1]
        
        exit_wave_diff  = modified_exit_wave[-1] - exit_wave[-1]
        displacement_x  = xp.sum(xp.real(xp.conj(exit_wave_dx)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dx)**2)
        displacement_y  = xp.sum(xp.real(xp.conj(exit_wave_dy)*exit_wave_diff))/xp.sum(xp.abs(exit_wave_dy)**2)
        
        return position + position_step_size*xp.array([displacement_x,displacement_y])
    
    @staticmethod
    def _fix_probe_center_of_mass(probes:np.ndarray,
                                  center_of_mass:Callable,
                                  xp = np,
                                  **kwargs):
        '''
        Fix probes CoM to array center. 
        '''
        
        probe_center = xp.array(probes.shape[-2:])/2
        com          = center_of_mass(xp.abs(probes[0]) ** 2)
        probes[0]    = fft_shift(probes[0], probe_center - xp.array(com))
        
        return probes
        
    def _prepare_functions_queue(self,
                                 max_iterations: int,
                                 pre_position_correction_update_steps: int = None,
                                 pre_probe_correction_update_steps: int = None,
                                 **kwargs):
        '''
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting. 
        '''
        total_update_steps   = max_iterations*self._num_diffraction_patterns
        queue_summary        = "Ptychographic reconstruction will perform the following steps:"
    
        functions_tuple      = (self._overlap_projection,self._fourier_projection, self._update_function, None)
        functions_queue      = [functions_tuple]
        if pre_position_correction_update_steps is None:
            functions_queue *= total_update_steps
            queue_summary   += f"\n--Multislice PIE for {total_update_steps} steps"
        else:
            functions_queue *= pre_position_correction_update_steps
            queue_summary   += f"\n--Multislice PIE for {pre_position_correction_update_steps} steps"

            functions_tuple = (self._overlap_projection,self._fourier_projection, self._update_function, self._position_correction)

            remaining_update_steps = total_update_steps - pre_position_correction_update_steps
            functions_queue += [functions_tuple]*remaining_update_steps
            queue_summary   += f"\n--Multislice PIE with position correction for {remaining_update_steps} steps"
        
        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"
        
        functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
        
        return functions_queue, queue_summary
    
    def reconstruct(self,
                    max_iterations: int = 5,
                    return_iterations: bool = False,
                    fix_com: bool = True,
                    random_seed = None,
                    verbose: bool = False,
                    parameters: Mapping[str,float] = None,
                    measurement_output_view: str = 'padded',
                    functions_queue: Iterable = None,
                    **kwargs):
        '''
        '''
        for key in kwargs.keys():
            if (key not in reconstruction_symbols.keys()):
                raise ValueError('{} not a recognized parameter'.format(key))
                
        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)
                 
        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                                        max_iterations,
                                        pre_position_correction_update_steps = self._reconstruction_parameters['pre_position_correction_update_steps'],
                                        pre_probe_correction_update_steps    = self._reconstruction_parameters['pre_probe_correction_update_steps'])
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [[function_tuples]*self._num_diffraction_patterns for function_tuples in functions_queue]
            elif len(functions_queue) == max_iterations*self._num_diffraction_patterns:
                functions_queue = [functions_queue[x:x+self._num_diffraction_patterns] for x in range(0, total_update_steps, self._num_diffraction_patterns)]
            else:
                raise ValueError()
        
        self._functions_queue = functions_queue
        
        ### Main Loop
        xp                  = get_array_module_from_device(self._device)
        outer_pbar          = ProgressBar(total=max_iterations,leave=False)
        inner_pbar          = ProgressBar(total=self._num_diffraction_patterns,leave=False)
        indices             = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(self._experimental_parameters['object_px_padding'])
        center_of_mass      = get_scipy_module(xp).ndimage.center_of_mass
        sobel               = get_scipy_module(xp).ndimage.sobel
        fft2_convolve       = get_device_function(xp, 'fft2_convolve')
        propagator          = FresnelPropagator()
        wavelength          = energy2wavelength(self._energy)
        
        if return_iterations:
            objects_iterations   = []
            probes_iterations    = []
            positions_iterations = []
            sse_iterations       = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        for iteration_index, iteration_step in enumerate(self._functions_queue):
            
            inner_pbar.reset()
            
            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse    = 0.0
            
            for update_index, update_step in enumerate(iteration_step):
                
                index               = indices[update_index]
                position            = self._positions_px[index]
                
                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i  = iteration_index*self._num_diffraction_patterns + update_index
                
                if self._reconstruction_parameters['pre_probe_correction_update_steps'] is None:
                    fix_probe       = False
                else:
                    fix_probe       = global_iteration_i < self._reconstruction_parameters['pre_probe_correction_update_steps']

                _overlap_projection,_fourier_projection,_update_function,_position_correction = update_step
            
                self._probes, exit_wave                                = _overlap_projection(self._objects,
                                                                                             self._probes,
                                                                                             position,
                                                                                             old_position,
                                                                                             propagator = propagator,
                                                                                             slice_thicknesses = self._slice_thicknesses,
                                                                                             sampling = self.sampling,
                                                                                             wavelength = wavelength,
                                                                                             fft2_convolve = fft2_convolve,
                                                                                             xp=xp)
                
                modified_exit_wave, self._sse                          = _fourier_projection(exit_wave,
                                                                                             diffraction_pattern,
                                                                                             self._sse,
                                                                                             xp=xp)

                self._objects, self._probes, self._positions_px[index] = _update_function(self._objects,
                                                                                          self._probes,
                                                                                          position,
                                                                                          exit_wave,
                                                                                          modified_exit_wave,
                                                                                          diffraction_pattern,
                                                                                          fix_probe = fix_probe,
                                                                                          position_correction = _position_correction,
                                                                                          sobel = sobel,
                                                                                          reconstruction_parameters = self._reconstruction_parameters,
                                                                                          propagator = propagator,
                                                                                          slice_thicknesses = self._slice_thicknesses,
                                                                                          sampling = self.sampling,
                                                                                          wavelength = wavelength,
                                                                                          fft2_convolve = fft2_convolve,
                                                                                          xp = xp)
                    
                old_position    = position
                inner_pbar.update(1)
              
            # Shift probe back to origin
            self._probes     = fft_shift(self._probes,xp.round(position) - position)
            
            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(self._probes,center_of_mass,xp=xp)
                 
            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (xp.mean(self._positions_px,axis=0) - self._positions_px_com)
                self._reconstruction_parameters['position_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            
            # Update Parameters
            self._reconstruction_parameters['object_step_size'] *= self._reconstruction_parameters['step_size_damping_rate']
            self._reconstruction_parameters['probe_step_size']  *= self._reconstruction_parameters['step_size_damping_rate']
            self._sse                                           /= self._num_diffraction_patterns
            
            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(self._positions_px.copy() * xp.array(self.sampling))
                sse_iterations.append(self._sse)
            
            if verbose:
                print(f'----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}')
            
            outer_pbar.update(1)
        
        inner_pbar.close()
        outer_pbar.close()
        
        #  Return Results
        if return_iterations:
            mapfunc = partial(self._prepare_measurement_outputs, slice_thicknesses = self._slice_thicknesses)
            results = map(mapfunc,
                          objects_iterations,
                          probes_iterations,
                          positions_iterations,
                          sse_iterations)
            
            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(self._objects,
                                                        self._probes,
                                                        self._positions_px * xp.array(self.sampling),
                                                        self._sse,
                                                        slice_thicknesses = self._slice_thicknesses)
            return results
        
        
    def _prepare_measurement_outputs(self,
                                     objects:np.ndarray,
                                     probes: np.ndarray,
                                     positions: np.ndarray,
                                     sse: np.ndarray,
                                     slice_thicknesses: Sequence[float] = None):
        '''
        Base measurement outputs function operating on a single iteration's outputs.
        Called using map if more than one iteration required.
        '''
        
        calibrations  = tuple(Calibration(0, s, units='Å', name = n, endpoint=False) for s,n in zip(self.sampling,('x','y')))
        calibrations  = (Calibration(0,slice_thicknesses[0],units='Å', name = 'z', endpoint=False),) + calibrations
        
        measurement_objects = Measurement(asnumpy(objects),calibrations)
        measurement_probes  = Measurement(asnumpy(probes),calibrations)
        
        return measurement_objects, measurement_probes, asnumpy(positions), sse
