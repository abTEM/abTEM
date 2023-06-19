"""Module for reconstructing phase objects from far-field intensity measurements using iterative ptychography."""
from typing import Union, Sequence, Mapping, Callable, Iterable
from abc import ABCMeta, abstractmethod
from functools import partial
from copy import copy
import numpy as np

from abtem.measure import Measurement, Calibration
from abtem.waves import Probe, FresnelPropagator
from abtem.base_classes import AntialiasFilter
from abtem.transfer import CTF, polar_symbols, polar_aliases
from abtem.utils import fft_shift, energy2wavelength, ProgressBar
from abtem.device import (
    copy_to_device,
    get_array_module,
    get_array_module_from_device,
    get_scipy_module,
    asnumpy,
    get_device_function,
)

experimental_symbols = (
    "rotation_angle",
    "scan_step_sizes",
    "angular_sampling",
    "background_counts_cutoff",
    "counts_scaling_factor",
    "grid_scan_shape",
    "object_px_padding",
)

reconstruction_symbols = {
    "alpha": 1.0,
    "beta": 1.0,
    "object_step_size": 1.0,
    "probe_step_size": 1.0,
    "position_step_size": 1.0,
    "step_size_damping_rate": 0.995,
    "pre_position_correction_update_steps": None,
    "pre_probe_correction_update_steps": None,
    "pure_phase_object_update_steps": None,
}


def _wrapped_indices_2D_window(
    center_position: np.ndarray, window_shape: Sequence[int], array_shape: Sequence[int]
):
    """
    Computes periodic indices for a window_shape probe centered at center_position, in object of size array_shape.

    Parameters
    ----------
    center_position: (2,) np.ndarray
        The window center positions in pixels
    window_shape: (2,) Sequence[int]
        The pixel dimensions of the window
    array_shape: (2,) Sequence[int]
        The pixel dimensions of the array the window will be embedded in

    Returns
    -------
    window_indices: length-2 tuple of
        The 2D indices of the window
    """

    sx, sy = array_shape
    nx, ny = window_shape

    cx, cy = np.round(asnumpy(center_position)).astype(int)
    ox, oy = (cx - nx // 2, cy - ny // 2)

    return np.ix_(np.arange(ox, ox + nx) % sx, np.arange(oy, oy + ny) % sy)


def _projection(u: np.ndarray, v: np.ndarray):
    """Projection of vector u onto vector v."""
    return u * np.vdot(u, v) / np.vdot(u, u)


def _orthogonalize(V):
    """Non-normalized QR decomposition using repeated projections."""
    U = V.copy()
    for i in range(1, V.shape[0]):
        for j in range(i):
            U[i, :] -= _projection(U[j, :], V[i, :])
    return U


def _propagate_array(
    propagator: FresnelPropagator,
    waves_array: np.ndarray,
    sampling: Sequence[float],
    wavelength: float,
    thickness: float,
    fft2_convolve: Callable = None,
    overwrite: bool = False,
    xp=np,
):
    """
    Propagates complex wave function array through free space distance dz.

    Simplified re-write of abtem.FresnelPropagator.propagate() to operate on arrays directly.

    Parameters
    ----------
    propagator: FresnelPropagator
        Near-field (Fresnel diffraction) propagation operator
    waves_array: np.ndarray
        The wavefunction array to propagate
    wavelength: float
        The relativistic electron wavelength [Å]
    thickness: float
        Distance [Å] in free space to propagate
    fft2_convolve: Callable
        Device-specific fft2_convolve function
    overwrite: bool
        If true, the wave array may be overwritten
    xp
        Numpy/Cupy module to use

    Returns
    -------
    propagated_array: np.ndarray
        Propagated array
    """
    propagator_array = propagator._evaluate_propagator_array(
        waves_array.shape, sampling, wavelength, thickness, None, xp
    )
    return fft2_convolve(waves_array, propagator_array, overwrite_x=overwrite)


class AbstractPtychographicOperator(metaclass=ABCMeta):
    """
    Base ptychographic operator class.

    Defines various common functions and properties for all subclasses to inherit,
    as well as setup various abstract methods each subclass must define.
    """

    @abstractmethod
    def preprocess(self):
        """
        Abstract method all subclasses must define which does the following:
        - Pads CBED patterns to region of interest dimensions
        - Prepares initial guess for scanning positions
        - Prepares initial guesses for the objects and probes arrays
        """
        pass

    @staticmethod
    @abstractmethod
    def _overlap_projection(objects, probes, position, old_position, **kwargs):
        """Abstract method all subclasses must define to perform overlap projection."""
        pass

    @staticmethod
    @abstractmethod
    def _fourier_projection(exit_waves, diffraction_patterns, sse, **kwargs):
        """Abstract method all subclasses must define to perform fourier projection."""
        pass

    @staticmethod
    @abstractmethod
    def _update_function(
        objects, probes, position, exit_waves, modified_exit_waves, **kwargs
    ):
        """Abstract method all subclasses must define to update the current probes, objects, and position estimates."""
        pass

    @staticmethod
    @abstractmethod
    def _constraints_function(objects, probes, **kwargs):
        """Abstract method all subclasses must define to enforce constraints on the objects and probes."""
        pass

    @staticmethod
    @abstractmethod
    def _position_correction(objects, probes, position, **kwargs):
        """
        Abstract method all subclasses must define to perform position correction.
        Typically gets called inside the subclass _update_function() method.
        """
        pass

    @staticmethod
    @abstractmethod
    def _fix_probe_center_of_mass(probes, center_of_mass, **kwargs):
        """Abstract method all subclasses must define to fix the center of mass of the probes."""
        pass

    @abstractmethod
    def _prepare_functions_queue(self, max_iterations, **kwargs):
        """Abstract method all subclasses must define to precompute the order of function calls during reconstruction."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        max_iterations,
        return_iterations,
        fix_com,
        random_seed,
        verbose,
        functions_queue,
        parameters,
        **kwargs,
    ):
        """
        Abstract method all subclasses must define which does the following:
        - Precomputes the order of function calls using the subclass _prepare_functions_queue() method
        - Performs main reconstruction loop
        - Passes reconstruction outputs to the subclass _prepare_measurement_outputs() method
        """
        pass

    @abstractmethod
    def _prepare_measurement_outputs(self, objects, probes, positions, sse):
        """Abstract method all subclasses must define to postprocess reconstruction outputs to Measurement objects."""
        pass

    @staticmethod
    def _update_parameters(
        parameters: dict,
        polar_parameters: dict = {},
        experimental_parameters: dict = {},
    ):
        """
        Common static method to update polar and experimental parameters during initialization.

        Parameters
        ----------
        parameters: dict
            Input dictionary used to update the default polar_parameters and experimental_parameters dictionaries
        polar_parameters: dict, optional
            Default polar parameters dictionary to be updated
        experimental_parameters: dict, optional
            Default experimental parameters dictionary to be updated

        Returns
        -------
        updated_polar_parameters: dict
            Updated polar parameters dictionary
        updated_experimental_parameters: dict
            Updated experimental parameters dictionary
        """
        for symbol, value in parameters.items():
            if symbol in polar_symbols:
                polar_parameters[symbol] = value
            elif symbol == "defocus":
                polar_parameters[polar_aliases[symbol]] = -value
            elif symbol in polar_aliases.keys():
                polar_parameters[polar_aliases[symbol]] = value
            elif symbol in experimental_symbols:
                experimental_parameters[symbol] = value
            else:
                raise ValueError("{} not a recognized parameter".format(symbol))

        return polar_parameters, experimental_parameters

    @staticmethod
    def _pad_diffraction_patterns(
        diffraction_patterns: np.ndarray, region_of_interest_shape: Sequence[int]
    ):
        """
        Common static method to zero-pad CBED patterns to a certain region of interest shape.

        Parameters
        ----------
        diffraction_patterns: (J,M,N) np.ndarray
            Flat array of CBED patterns to be zero-padded
        region_of_interest_shape: (2,) Sequence[int]
            Pixel dimensions (R,S) the CBED patterns will be padded to


        Returns
        -------
        padded_diffraction_patterns: (J,R,S) np.ndarray
            Zero-padded CBED patterns
        """

        diffraction_patterns_size = diffraction_patterns.shape[-2:]
        xp = get_array_module(diffraction_patterns)

        if any(
            dp_shape > roi_shape
            for dp_shape, roi_shape in zip(
                diffraction_patterns_size, region_of_interest_shape
            )
        ):
            raise ValueError()

        if diffraction_patterns_size != region_of_interest_shape:
            padding_list = [(0, 0)]  # No padding along first dimension
            for current_dim, target_dim in zip(
                diffraction_patterns_size, region_of_interest_shape
            ):
                pad_value = target_dim - current_dim
                pad_tuple = (pad_value // 2, pad_value // 2 + pad_value % 2)
                padding_list.append(pad_tuple)

            diffraction_patterns = xp.pad(
                diffraction_patterns, tuple(padding_list), mode="constant"
            )

        return diffraction_patterns

    @staticmethod
    def _extract_calibrations_from_measurement_object(
        measurement: Measurement, energy: float = None
    ):
        """
        Common static method to extract angular sampling and scan step sizes from Measurement object.

        Parameters
        ----------
        measurement: Measurement
            Measurement object to exract calibrations from
        energy: float, optional
            Electron energy [eV] used to convert 1/Å sampling to angular sampling in mrad

        Returns
        -------
        diffraction_patterns: np.ndarray
            CBED patterns from Measurement object
        angular_sampling: (2,) Sequence[float]
            Measurement angular sampling in mrad
        step_sizes: (2,) Sequence[float] or None
            Measurement scan step sizes in Å if Measurement holds a 4D array
            None if Measurement holds a 3D array
        """
        calibrations = measurement.calibrations
        calibration_units = measurement.calibration_units
        diffraction_patterns = measurement.array

        if any(unit != "mrad" and unit != "1/Å" for unit in calibration_units[-2:]):
            raise ValueError()

        angular_sampling = []
        for cal, cal_unit in zip(calibrations[-2:], calibration_units[-2:]):
            scale_factor = (
                1.0 if cal_unit == "mrad" else energy2wavelength(energy) * 1e3
            )
            angular_sampling.append(cal.sampling * scale_factor)
        angular_sampling = tuple(angular_sampling)

        step_sizes = None
        if len(diffraction_patterns.shape) == 4:
            step_sizes = tuple(cal.sampling for cal in calibrations[:2])

        return diffraction_patterns, angular_sampling, step_sizes

    @staticmethod
    def _calculate_scan_positions_in_pixels(
        positions: np.ndarray,
        sampling: Sequence[float],
        region_of_interest_shape: Sequence[int],
        experimental_parameters: dict,
    ):
        """
        Common static method to compute the initial guess of scan positions in pixels.

        Parameters
        ----------
        positions: (J,2) np.ndarray or None
            Input experimental positions [Å].
            If None, a raster scan using experimental parameters is constructed.
        sampling: (2,) Sequence[float]
            Real-space sampling [Å] to convert positions to pixels
        region_of_interest_shape: (2,) Sequence[int]
            Pixel dimensions of the region of interest
        experimental_parameters: dict
            Dictionary with relevant experimental parameters
        Returns
        -------

        positions_in_px: (J,2) np.ndarray
            Initial guess of scan positions in pixels
        updated_experimental_parameters:
            Updated experimental parameters dataset
        """

        grid_scan_shape = experimental_parameters["grid_scan_shape"]
        step_sizes = experimental_parameters["scan_step_sizes"]
        rotation_angle = experimental_parameters["rotation_angle"]
        object_px_padding = experimental_parameters["object_px_padding"]

        if positions is None:
            if grid_scan_shape is not None:
                nx, ny = grid_scan_shape

                if step_sizes is not None:
                    sx, sy = step_sizes
                    x = np.arange(nx) * sx
                    y = np.arange(ny) * sy
                else:
                    raise ValueError()
            else:
                raise ValueError()

        else:
            x = positions[:, 0]
            y = positions[:, 1]

        x = (x - np.ptp(x) / 2) / sampling[0]
        y = (y - np.ptp(y) / 2) / sampling[1]
        x, y = np.meshgrid(x, y, indexing="ij")

        if rotation_angle is not None:
            x, y = x * np.cos(rotation_angle) + y * np.sin(rotation_angle), -x * np.sin(
                rotation_angle
            ) + y * np.cos(rotation_angle)

        positions = np.array([x.ravel(), y.ravel()]).T
        positions -= np.min(positions, axis=0)

        if object_px_padding is None:
            object_px_padding = np.array(region_of_interest_shape) / 2
        else:
            object_px_padding = np.array(object_px_padding)

        positions += object_px_padding

        experimental_parameters["object_px_padding"] = object_px_padding
        return positions, experimental_parameters

    @property
    def angular_sampling(self):
        """Angular sampling [mrad]"""
        if not self._preprocessed:
            return None

        return self._experimental_parameters["angular_sampling"]

    @property
    def sampling(self):
        """Sampling [Å]"""
        if not self._preprocessed:
            return None

        return tuple(
            energy2wavelength(self._energy) * 1e3 / dk / n
            for dk, n in zip(self.angular_sampling, self._region_of_interest_shape)
        )


class RegularizedPtychographicOperator(AbstractPtychographicOperator):
    """
    Regularized Ptychographic Iterative Engine (r-PIE).
    Used to reconstruct weak-phase objects using a set of measured far-field CBED patterns with the following array dimensions:

    CBED pattern dimensions     : (J,M,N)
    objects dimensions          : (P,Q)
    probes dimensions           : (R,S)

    Parameters
    ----------

    diffraction_patterns: np.ndarray or Measurement
        Input 3D or 4D CBED pattern intensities with dimensions (M,N)
    energy: float,
        Electron energy [eV]
    region_of_interest_shape: (2,) Sequence[int], optional
        Pixel dimensions (R,S) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the CBED dimensions (M,N)
    objects: np.ndarray, optional
        Initial objects guess with dimensions (P,Q) - Useful for restarting reconstructions
        If None, an array with 1.0j is initialized
    probes: np.ndarray or Probe, optional
        Initial probes guess with dimensions/gpts (R,S) - Useful for restarting reconstructions
        If None, a Probe with CTF given by the polar_parameters dictionary is initialized
    positions: np.ndarray, optional
        Initial positions guess [Å]
        If None, a raster scan with step sizes given by the experimental_parameters dictionary is initialized
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial Probe guess
    preprocess: bool, optional
        If True, it runs the preprocess method after initialization
    device: str, optional
        Device to perform Fourier-based reconstructrions - Either 'cpu' or 'gpu'
    parameters: dict, optional
       Dictionary specifying any of the abtem.transfer.polar_symbols or abtem.reconstruct.experimental_symbols parameters
       Additionally, these can also be specified using kwargs
    """

    def __init__(
        self,
        diffraction_patterns: Union[np.ndarray, Measurement],
        energy: float,
        region_of_interest_shape: Sequence[int] = None,
        objects: np.ndarray = None,
        probes: Union[np.ndarray, Probe] = None,
        positions: np.ndarray = None,
        semiangle_cutoff: float = None,
        preprocess: bool = False,
        device: str = "cpu",
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        for key in kwargs.keys():
            if (
                (key not in polar_symbols)
                and (key not in polar_aliases.keys())
                and (key not in experimental_symbols)
            ):
                raise ValueError("{} not a recognized parameter".format(key))

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))
        self._experimental_parameters = dict(
            zip(experimental_symbols, [None] * len(experimental_symbols))
        )

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(
            parameters, self._polar_parameters, self._experimental_parameters
        )

        self._region_of_interest_shape = region_of_interest_shape
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._positions = positions
        self._device = device
        self._objects = objects
        self._probes = probes
        self._diffraction_patterns = diffraction_patterns

        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False

    def preprocess(self):
        """
        Preprocess method to do the following:
        - Pads CBED patterns to region of interest dimensions
        - Prepares initial guess for scanning positions
        - Prepares initial guesses for the objects and probes arrays


        Returns
        -------
        preprocessed_ptychographic_operator: RegularizedPtychographicOperator
        """

        self._preprocessed = True

        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            (
                self._diffraction_patterns,
                angular_sampling,
                step_sizes,
            ) = self._extract_calibrations_from_measurement_object(
                self._diffraction_patterns, self._energy
            )
            self._experimental_parameters["angular_sampling"] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters["scan_step_sizes"] = step_sizes

        # Preprocess Diffraction Patterns
        xp = get_array_module_from_device(self._device)
        self._diffraction_patterns = copy_to_device(
            self._diffraction_patterns, self._device
        )

        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters[
                "grid_scan_shape"
            ] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns = self._diffraction_patterns.reshape(
                (-1,) + self._diffraction_patterns.shape[-2:]
            )

        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]

        self._diffraction_patterns = self._pad_diffraction_patterns(
            self._diffraction_patterns, self._region_of_interest_shape
        )
        self._num_diffraction_patterns = self._diffraction_patterns.shape[0]

        if self._experimental_parameters["background_counts_cutoff"] is not None:
            self._diffraction_patterns[
                self._diffraction_patterns
                < self._experimental_parameters["background_counts_cutoff"]
            ] = 0.0

        if self._experimental_parameters["counts_scaling_factor"] is not None:
            self._diffraction_patterns /= self._experimental_parameters[
                "counts_scaling_factor"
            ]

        self._mean_diffraction_intensity = (
            xp.sum(self._diffraction_patterns) / self._num_diffraction_patterns
        )
        self._diffraction_patterns = xp.fft.ifftshift(
            xp.sqrt(self._diffraction_patterns), axes=(-2, -1)
        )

        # Scan Positions Initialization
        (
            positions_px,
            self._experimental_parameters,
        ) = self._calculate_scan_positions_in_pixels(
            self._positions,
            self.sampling,
            self._region_of_interest_shape,
            self._experimental_parameters,
        )

        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y = self._experimental_parameters["object_px_padding"]
            p, q = np.max(positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._objects = xp.ones((p, q), dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects, self._device)

        self._positions_px = copy_to_device(positions_px, self._device)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)

        # Probes Initialization
        if self._probes is None:
            ctf = CTF(
                energy=self._energy,
                semiangle_cutoff=self._semiangle_cutoff,
                parameters=self._polar_parameters,
            )
            self._probes = (
                Probe(
                    semiangle_cutoff=self._semiangle_cutoff,
                    energy=self._energy,
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    ctf=ctf,
                    device=self._device,
                )
                .build()
                .array
            )
        else:
            if isinstance(self._probes, Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = copy_to_device(self._probes.build().array, self._device)
            else:
                self._probes = copy_to_device(self._probes, self._device)

        probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probes)) ** 2)
        self._probes *= np.sqrt(self._mean_diffraction_intensity / probe_intensity)

        return self

    @staticmethod
    def _overlap_projection(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE overlap projection static method:
        .. math::
            \psi_{R_j}(r) = O_{R_j}(r) * P(r)


        Parameters
        ----------
        objects: np.ndarray
            Object array to be illuminated
        probes: np.ndarray
            Probe window array to illuminate object with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Fractionally shifted probe window array
        exit_wave: np.ndarray
            Overlap projection of illuminated probe
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probes = fft_shift(probes, fractional_position - old_fractional_position)
        object_indices = _wrapped_indices_2D_window(
            position, probes.shape, objects.shape
        )
        object_roi = objects[object_indices]
        exit_wave = object_roi * probes

        return probes, exit_wave

    @staticmethod
    def _fourier_projection(
        exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE fourier projection static method:
        .. math::
            \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]


        Parameters
        ----------
        exit_waves: np.ndarray
            Exit waves array given by RegularizedPtychographicOperator._overlap_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: np.ndarray
            Fourier projection of illuminated probe
        sse: float
            Updated sum of squares error estimate
        """
        exit_wave_fft = xp.fft.fft2(exit_waves)
        sse += xp.mean(
            xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns) ** 2
        ) / xp.sum(diffraction_patterns**2)
        modified_exit_wave = xp.fft.ifft2(
            diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft))
        )

        return modified_exit_wave, sse

    @staticmethod
    def _update_function(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_waves: np.ndarray,
        modified_exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE objects and probes update static method:
        .. math::
            O'_{R_j}(r)    &= O_{R_j}(r) + \frac{P^*(r)}{\left(1-\alpha\right)|P(r)|^2 + \alpha|P(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right) \\
            P'(r)          &= P(r) + \frac{O^*_{R_j}(r)}{\left(1-\beta\right)|O_{R_j}(r)|^2 + \beta|O_{R_j}(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right)


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: np.ndarray
            Exit waves array given by RegularizedPtychographicOperator._overlap_projection method
        modified_exit_waves: np.ndarray
            Modified exit waves array given by RegularizedPtychographicOperator._fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Updated objects array estimate
        probes: np.ndarray
            Updated probes array estimate
        position: np.ndarray
            Updated probe position estimate
        """

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape, objects.shape
        )
        object_roi = objects[object_indices]

        exit_wave_diff = modified_exit_waves - exit_waves

        probe_conj = xp.conj(probes)
        probe_abs_squared = xp.abs(probes) ** 2
        obj_conj = xp.conj(object_roi)
        obj_abs_squared = xp.abs(object_roi) ** 2

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        objects[object_indices] += (
            object_step_size
            * probe_conj
            * exit_wave_diff
            / ((1 - alpha) * probe_abs_squared + alpha * xp.max(probe_abs_squared))
        )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            probes += (
                probe_step_size
                * obj_conj
                * exit_wave_diff
                / ((1 - beta) * obj_abs_squared + beta * xp.max(obj_abs_squared))
            )

        return objects, probes, position

    @staticmethod
    def _constraints_function(
        objects: np.ndarray,
        probes: np.ndarray,
        pure_phase_object: bool,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE constraints static method:

        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        pure_phase_object:bool
            If True, constraints object to being a pure phase object, i.e. with unit amplitude
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Constrained objects array
        probes: np.ndarray
            Constrained probes array
        """
        phase = xp.exp(1.0j * xp.angle(objects))
        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(objects), 1.0)
        return amplitude * phase, probes

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
    def _position_correction(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_wave: np.ndarray,
        modified_exit_wave: np.ndarray,
        diffraction_pattern: np.ndarray,
        sobel: Callable,
        position_step_size: float = 1.0,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE probe position correction method.


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_wave: np.ndarray
            Exit wave array given by RegularizedPtychographicOperator._overlap_projection method
        modified_exit_wave: np.ndarray
            Modified exit wave array given by RegularizedPtychographicOperator._fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        position_step_size: float, optional
            Gradient step size for position update step
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        position: np.ndarray
            Updated probe position estimate
        """

        object_dx = sobel(objects, axis=0, mode="wrap")
        object_dy = sobel(objects, axis=1, mode="wrap")

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape, objects.shape
        )
        exit_wave_dx = object_dx[object_indices] * probes
        exit_wave_dy = object_dy[object_indices] * probes

        exit_wave_diff = modified_exit_wave - exit_wave
        displacement_x = xp.sum(
            xp.real(xp.conj(exit_wave_dx) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dx) ** 2)
        displacement_y = xp.sum(
            xp.real(xp.conj(exit_wave_dy) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dy) ** 2)

        return position + position_step_size * xp.array(
            [displacement_x, displacement_y]
        )

    @staticmethod
    def _fix_probe_center_of_mass(
        probes: np.ndarray, center_of_mass: Callable, xp=np, **kwargs
    ):
        """
        Regularized-PIE probe center correction method.


        Parameters
        ----------
        probes: np.ndarray
            Current probes array estimate
        center_of_mass: Callable
            The scipy.ndimage module used to compute the array center of mass
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Center-of-mass corrected probes array
        """

        probe_center = xp.array(probes.shape) / 2
        com = center_of_mass(xp.abs(probes) ** 2)
        probes = fft_shift(probes, probe_center - xp.array(com))

        return probes

    def _prepare_functions_queue(
        self,
        max_iterations: int,
        pre_position_correction_update_steps: int = None,
        pre_probe_correction_update_steps: int = None,
        pure_phase_object_update_steps: int = None,
        **kwargs,
    ):
        """
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting.

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        pre_position_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling position correction
        pre_probe_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling probe correction

        Returns
        -------
        functions_queue: (max_iterations,J) list
            List of function calls
        queue_summary: str
            Summary of function calls the reconstruction loop will perform
        """
        total_update_steps = max_iterations * self._num_diffraction_patterns
        queue_summary = "Ptychographic reconstruction will perform the following steps:"

        functions_tuple = (
            self._overlap_projection,
            self._fourier_projection,
            self._update_function,
            self._constraints_function,
            None,
        )
        functions_queue = [functions_tuple]
        if pre_position_correction_update_steps is None:
            functions_queue *= total_update_steps
            queue_summary += f"\n--Regularized PIE for {total_update_steps} steps"
        else:
            functions_queue *= pre_position_correction_update_steps
            queue_summary += (
                f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"
            )

            functions_tuple = (
                self._overlap_projection,
                self._fourier_projection,
                self._update_function,
                self._constraints_function,
                self._position_correction,
            )

            remaining_update_steps = (
                total_update_steps - pre_position_correction_update_steps
            )
            functions_queue += [functions_tuple] * remaining_update_steps
            queue_summary += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"

        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"

        if pure_phase_object_update_steps is not None:
            queue_summary += f"\n--Reconstructed object will be constrained to a pure-phase object for the first {pure_phase_object_update_steps} steps"

        functions_queue = [
            functions_queue[x : x + self._num_diffraction_patterns]
            for x in range(0, total_update_steps, self._num_diffraction_patterns)
        ]

        return functions_queue, queue_summary

    def reconstruct(
        self,
        max_iterations: int = 5,
        return_iterations: bool = False,
        fix_com: bool = True,
        random_seed=None,
        verbose: bool = False,
        functions_queue: Iterable = None,
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):
        """
        Main reconstruction loop method to do the following:
        - Precompute the order of function calls using the RegularizedPtychographicOperator._prepare_functions_queue method
        - Iterate through function calls in queue
        - Pass reconstruction outputs to the RegularizedPtychographicOperator._prepare_measurement_outputs method

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        return_iterations: bool, optional
            If True, method will return a list of current objects, probes, and positions estimates for each iteration
        fix_com: bool, optional
            If True, the center of mass of the probes array will be corrected at the end of the iteration
        random_seed
            If not None, used to seed the numpy random number generator
        verbose: bool, optional
            If True, prints functions queue and current iteration error
        functions_queue: (max_iterations, J) Iterable, optional
            If not None, the reconstruction algorithm will use the input functions queue instead
        parameters: dict, optional
            Dictionary specifying any of the abtem.reconstruct.recontruction_symbols parameters
            Additionally, these can also be specified using kwargs

        Returns
        -------
        reconstructed_object_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the objects estimate at each iteration is returned
        reconstructed_probes_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the probes estimate at each iteration is returned
        reconstructed_position_measurement: np.ndarray or Sequence[np.ndarray]
            If return_iterations, a list of position estimates at each iteration is returned
        reconstruction_error: float or Sequence[float]
            If return_iterations, a list of the reconstruction error at each iteration is returned
        """
        for key in kwargs.keys():
            if key not in reconstruction_symbols.keys():
                raise ValueError("{} not a recognized parameter".format(key))

        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)

        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                max_iterations,
                pre_position_correction_update_steps=self._reconstruction_parameters[
                    "pre_position_correction_update_steps"
                ],
                pre_probe_correction_update_steps=self._reconstruction_parameters[
                    "pre_probe_correction_update_steps"
                ],
                pure_phase_object_update_steps=self._reconstruction_parameters[
                    "pure_phase_object_update_steps"
                ],
            )
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [
                        [function_tuples] * self._num_diffraction_patterns
                        for function_tuples in functions_queue
                    ]
            elif (
                len(functions_queue) == max_iterations * self._num_diffraction_patterns
            ):
                functions_queue = [
                    functions_queue[x : x + self._num_diffraction_patterns]
                    for x in range(
                        0, total_update_steps, self._num_diffraction_patterns
                    )
                ]
            else:
                raise ValueError()

        self._functions_queue = functions_queue

        ### Main Loop
        xp = get_array_module_from_device(self._device)
        outer_pbar = ProgressBar(total=max_iterations, leave=False)
        inner_pbar = ProgressBar(total=self._num_diffraction_patterns, leave=False)
        indices = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(
            self._experimental_parameters["object_px_padding"]
        )
        center_of_mass = get_scipy_module(xp).ndimage.center_of_mass
        sobel = get_scipy_module(xp).ndimage.sobel

        if return_iterations:
            objects_iterations = []
            probes_iterations = []
            positions_iterations = []
            sse_iterations = []

        if random_seed is not None:
            np.random.seed(random_seed)

        for iteration_index, iteration_step in enumerate(self._functions_queue):

            inner_pbar.reset()

            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse = 0.0

            for update_index, update_step in enumerate(iteration_step):

                index = indices[update_index]
                position = self._positions_px[index]

                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i = (
                    iteration_index * self._num_diffraction_patterns + update_index
                )

                if (
                    self._reconstruction_parameters["pre_probe_correction_update_steps"]
                    is None
                ):
                    fix_probe = False
                else:
                    fix_probe = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pre_probe_correction_update_steps"
                        ]
                    )

                if (
                    self._reconstruction_parameters["pure_phase_object_update_steps"]
                    is None
                ):
                    pure_phase_object = False
                else:
                    pure_phase_object = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pure_phase_object_update_steps"
                        ]
                    )

                (
                    _overlap_projection,
                    _fourier_projection,
                    _update_function,
                    _constraints_function,
                    _position_correction,
                ) = update_step

                self._probes, exit_wave = _overlap_projection(
                    self._objects, self._probes, position, old_position, xp=xp
                )

                modified_exit_wave, self._sse = _fourier_projection(
                    exit_wave, diffraction_pattern, self._sse, xp=xp
                )

                (
                    self._objects,
                    self._probes,
                    self._positions_px[index],
                ) = _update_function(
                    self._objects,
                    self._probes,
                    position,
                    exit_wave,
                    modified_exit_wave,
                    diffraction_pattern,
                    fix_probe=fix_probe,
                    position_correction=_position_correction,
                    sobel=sobel,
                    reconstruction_parameters=self._reconstruction_parameters,
                    xp=xp,
                )

                self._objects, self._probes = _constraints_function(
                    self._objects, self._probes, pure_phase_object, xp=xp
                )

                old_position = position
                inner_pbar.update(1)

            # Shift probe back to origin
            self._probes = fft_shift(self._probes, xp.round(position) - position)

            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(
                    self._probes, center_of_mass, xp=xp
                )

            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (
                    xp.mean(self._positions_px, axis=0) - self._positions_px_com
                )
                self._reconstruction_parameters[
                    "position_step_size"
                ] *= self._reconstruction_parameters["step_size_damping_rate"]

            # Update Parameters
            self._reconstruction_parameters[
                "object_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._reconstruction_parameters[
                "probe_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._sse /= self._num_diffraction_patterns

            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(
                    self._positions_px.copy() * xp.array(self.sampling)
                )
                sse_iterations.append(self._sse)

            if verbose:
                print(
                    f"----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}"
                )

            outer_pbar.update(1)

        inner_pbar.close()
        outer_pbar.close()

        #  Return Results
        if return_iterations:
            results = map(
                self._prepare_measurement_outputs,
                objects_iterations,
                probes_iterations,
                positions_iterations,
                sse_iterations,
            )

            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(
                self._objects,
                self._probes,
                self._positions_px * xp.array(self.sampling),
                self._sse,
            )
            return results

    def _prepare_measurement_outputs(
        self,
        objects: np.ndarray,
        probes: np.ndarray,
        positions: np.ndarray,
        sse: np.ndarray,
    ):
        """
        Method to format the reconstruction outputs as Measurement objects.

        Parameters
        ----------
        objects: np.ndarray
            Reconstructed objects array
        probes: np.ndarray
            Reconstructed probes array
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error

        Returns
        -------
        objects_measurement: Measurement
            Reconstructed objects Measurement
        probes_measurement: Measurement
            Reconstructed probes Measurement
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error
        """

        calibrations = tuple(
            Calibration(0, s, units="Å", name=n, endpoint=False)
            for s, n in zip(self.sampling, ("x", "y"))
        )

        measurement_objects = Measurement(asnumpy(objects), calibrations)
        measurement_probes = Measurement(asnumpy(probes), calibrations)

        return measurement_objects, measurement_probes, asnumpy(positions), sse


class SimultaneousPtychographicOperator(AbstractPtychographicOperator):
    """
    Simultaneous Ptychographic Iterative Engine (sim-PIE).
    Used to reconstruct the electrostatic phase and magnetic phase objects simultaneously using two set of measured far-field CBED patterns with the following array dimensions:

    CBED pattern dimensions     : (2,) Sequence of dimensions (J,M,N) each
    objects dimensions          : (2,) Sequence of dimensions (P,Q) each
    probes dimensions           : (2,) Sequence of dimensions (R,S) each

    Parameters
    ----------

    diffraction_patterns: (2,) Sequence[np.ndarray] or (2,) Sequence[Measurement]
        Two sets of 3D or 4D CBED pattern intensities with dimensions (M,N)
    energy: float,
        Electron energy [eV]
    region_of_interest_shape: (2,) Sequence[int], optional
        Pixel dimensions (R,S) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the CBED dimensions (M,N)
    objects: np.ndarray, optional
        Initial objects guess with dimensions (P,Q) - Useful for restarting reconstructions
        If None, an array with 1.0j is initialized
    probes: np.ndarray or Probe, optional
        Initial probes guess with dimensions/gpts (R,S) - Useful for restarting reconstructions
        If None, a Probe with CTF given by the polar_parameters dictionary is initialized
    positions: np.ndarray, optional
        Initial positions guess [Å]
        If None, a raster scan with step sizes given by the experimental_parameters dictionary is initialized
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial Probe guess
    preprocess: bool, optional
        If True, it runs the preprocess method after initialization
    device: str, optional
        Device to perform Fourier-based reconstructrions - Either 'cpu' or 'gpu'
    parameters: dict, optional
       Dictionary specifying any of the abtem.transfer.polar_symbols or abtem.reconstruct.experimental_symbols parameters
       Additionally, these can also be specified using kwargs
    """

    def __init__(
        self,
        diffraction_patterns: Union[Sequence[np.ndarray], Sequence[Measurement]],
        energy: float,
        region_of_interest_shape: Sequence[int] = None,
        objects: np.ndarray = None,
        probes: Union[np.ndarray, Probe] = None,
        positions: np.ndarray = None,
        semiangle_cutoff: float = None,
        preprocess: bool = False,
        device: str = "cpu",
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        if len(diffraction_patterns) != 2:
            raise NotImplementedError(
                "Simultaneous ptychographic reconstruction is currently only implemented for two sets of diffraction patterns"
                "allowing reconstruction of the electrostatic and magnetic phase contributions."
                "See the documentation for AbstractPtychographicOperator to implement your own class to handle more cases."
            )

        for key in kwargs.keys():
            if (
                (key not in polar_symbols)
                and (key not in polar_aliases.keys())
                and (key not in experimental_symbols)
            ):
                raise ValueError("{} not a recognized parameter".format(key))

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))
        self._experimental_parameters = dict(
            zip(experimental_symbols, [None] * len(experimental_symbols))
        )

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(
            parameters, self._polar_parameters, self._experimental_parameters
        )

        self._region_of_interest_shape = region_of_interest_shape
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._positions = positions
        self._device = device
        self._objects = objects
        self._probes = probes
        self._diffraction_patterns = diffraction_patterns

        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False

    def preprocess(self):
        """
        Preprocess method to do the following:
        - Pads CBED patterns to region of interest dimensions
        - Prepares initial guess for scanning positions
        - Prepares initial guesses for the objects and probes arrays


        Returns
        -------
        preprocessed_ptychographic_operator: SimultaneousPtychographicOperator
        """

        self._preprocessed = True

        xp = get_array_module_from_device(self._device)
        _diffraction_patterns = []
        self._mean_diffraction_intensity = 0
        for dp in self._diffraction_patterns:

            # Convert Measurement Objects
            if isinstance(dp, Measurement):
                (
                    _dp,
                    angular_sampling,
                    step_sizes,
                ) = self._extract_calibrations_from_measurement_object(dp, self._energy)

            # Preprocess Diffraction Patterns
            _dp = copy_to_device(_dp, self._device)

            if len(_dp.shape) == 4:
                self._experimental_parameters["grid_scan_shape"] = _dp.shape[:2]
                _dp = _dp.reshape((-1,) + _dp.shape[-2:])

            if self._region_of_interest_shape is None:
                self._region_of_interest_shape = _dp.shape[-2:]
            _dp = self._pad_diffraction_patterns(_dp, self._region_of_interest_shape)

            if self._experimental_parameters["background_counts_cutoff"] is not None:
                _dp[
                    _dp < self._experimental_parameters["background_counts_cutoff"]
                ] = 0.0

            if self._experimental_parameters["counts_scaling_factor"] is not None:
                _dp /= self._experimental_parameters["counts_scaling_factor"]

            self._mean_diffraction_intensity += xp.sum(_dp)
            _dp = xp.fft.ifftshift(xp.sqrt(_dp), axes=(-2, -1))
            _diffraction_patterns.append(_dp)

        self._diffraction_patterns = tuple(_diffraction_patterns)
        self._experimental_parameters["angular_sampling"] = angular_sampling
        self._num_diffraction_patterns = self._diffraction_patterns[0].shape[0]

        self._mean_diffraction_intensity /= 2 * self._num_diffraction_patterns

        if step_sizes is not None:
            self._experimental_parameters["scan_step_sizes"] = step_sizes

        # Scan Positions Initialization
        (
            positions_px,
            self._experimental_parameters,
        ) = self._calculate_scan_positions_in_pixels(
            self._positions,
            self.sampling,
            self._region_of_interest_shape,
            self._experimental_parameters,
        )

        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y = self._experimental_parameters["object_px_padding"]
            p, q = np.max(positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._objects = tuple(
                xp.ones((p, q), dtype=xp.complex64) for _obj_i in range(2)
            )
        else:
            if len(self._objects) != 2:
                raise ValueError()
            self._objects = tuple(
                copy_to_device(_obj, self._device) for _obj in self._objects
            )

        self._positions_px = copy_to_device(positions_px, self._device)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)

        # Probes Initialization
        if self._probes is None:
            ctf = CTF(
                energy=self._energy,
                semiangle_cutoff=self._semiangle_cutoff,
                parameters=self._polar_parameters,
            )
            self._probes = (
                Probe(
                    semiangle_cutoff=self._semiangle_cutoff,
                    energy=self._energy,
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    ctf=ctf,
                    device=self._device,
                )
                .build()
                .array
            )

            self._probes = (self._probes, self._probes.copy())
        else:
            if len(self._probes) != 2:
                raise ValueError()
            if isinstance(self._probes[0], Probe):
                if self._probes[0].gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = tuple(
                    copy_to_device(_probe.build().array, self._device)
                    for _probe in self._probes
                )
            else:
                self._probes = tuple(
                    copy_to_device(_probe, self._device) for _probe in self._probes
                )

        return self

    @staticmethod
    def _warmup_overlap_projection(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE overlap projection static method using the forward probe and electrostatic object
        .. math::
            \psi_{R_j}(r) = V_{R_j}(r) * P^{\mathrm{forward}}(r)


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Electrostatic and magnetic object arrays to be illuminated
        probes: Sequence[np.ndarray]
            Forward and reverse probe window array to illuminate objects with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: Sequence[np.ndarray]
            Fractionally shifted forward probe window array, reverse probe array
        exit_waves: Sequence[np.ndarray]
            Overlap projection of electrostatic object with forward probe, dummy reverse exit wave
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probe_forward, probe_reverse = probes
        probe_forward = fft_shift(
            probe_forward, fractional_position - old_fractional_position
        )

        electrostatic_object, magnetic_object = objects

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]

        exit_wave_forward = electrostatic_roi * probe_forward

        return (probe_forward, probe_reverse), (exit_wave_forward, None)

    @staticmethod
    def _overlap_projection(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE overlap projection static method:
        .. math:: 
            \psi_{R_j}(r) &= V_{R_j}(r) M_{R_j}(r)* P^{\mathrm{forward}}(r) \\
            \phi_{R_j}(r) &= V_{R_j}(r) M^*_{R_j}(r)* P^{\mathrm{reverse}}(r)


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Electrostatic and magnetic object arrays to be illuminated
        probes: Sequence[np.ndarray]
            Forward and reverse probe window array to illuminate objects with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: Sequence[np.ndarray]
            Fractionally shifted probe window arrays
        exit_waves: Sequence[np.ndarray]
            Overlap projection of objects with forward and reverse probes
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probe_forward, probe_reverse = probes
        probe_forward = fft_shift(
            probe_forward, fractional_position - old_fractional_position
        )
        probe_reverse = fft_shift(
            probe_reverse, fractional_position - old_fractional_position
        )

        electrostatic_object, magnetic_object = objects

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]
        magnetic_roi = magnetic_object[object_indices]

        exit_wave_forward = electrostatic_roi * magnetic_roi * probe_forward
        exit_wave_reverse = electrostatic_roi * xp.conj(magnetic_roi) * probe_reverse

        return (probe_forward, probe_reverse), (exit_wave_forward, exit_wave_reverse)

    @staticmethod
    def _alternative_overlap_projection(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE overlap projection static method using a common probe
        .. math:: 
            \psi_{R_j}(r) &= V_{R_j}(r) M_{R_j}(r)* P(r) \\
            \phi_{R_j}(r) &= V_{R_j}(r) M^*_{R_j}(r)* P(r)


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Electrostatic and magnetic object arrays to be illuminated
        probes: Sequence[np.ndarray]
            Forward and reverse probe window array to illuminate objects with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: Sequence[np.ndarray]
            Fractionally shifted forward probe window array, reverse probe array
        exit_waves: Sequence[np.ndarray]
            Overlap projection of objects with forward probe
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probe_forward, probe_reverse = probes
        probe_forward = fft_shift(
            probe_forward, fractional_position - old_fractional_position
        )

        electrostatic_object, magnetic_object = objects

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]
        magnetic_roi = magnetic_object[object_indices]

        exit_wave_forward = electrostatic_roi * magnetic_roi * probe_forward
        exit_wave_reverse = electrostatic_roi * xp.conj(magnetic_roi) * probe_forward

        return (probe_forward, probe_reverse), (exit_wave_forward, exit_wave_reverse)

    @staticmethod
    def _warmup_fourier_projection(
        exit_waves: Sequence[np.ndarray],
        diffraction_patterns: Sequence[np.ndarray],
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE fourier projection static method:
        .. math::
            \psi'_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|]


        Parameters
        ----------
        exit_waves: Sequence[np.ndarray]
            Exit waves array given by SimultaneousPtychographicOperator._warmup_overlap_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of forward and reverse CBED intensities arrays used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: Sequence[np.ndarray]
            Fourier projection of forward illuminated probe, dummy projection of reverse illuminated probe
        sse: float
            Updated sum of squares error estimate
        """
        exit_wave_forward, exit_wave_reverse = exit_waves
        diffraction_forward, diffraction_reverse = diffraction_patterns

        exit_wave_forward_fft = xp.fft.fft2(exit_wave_forward)
        sse += xp.mean(
            xp.abs(xp.abs(exit_wave_forward_fft) - diffraction_forward) ** 2
        ) / xp.sum(diffraction_forward**2)
        modified_exit_wave_forward = xp.fft.ifft2(
            diffraction_forward * xp.exp(1j * xp.angle(exit_wave_forward_fft))
        )

        return (modified_exit_wave_forward, None), sse

    @staticmethod
    def _fourier_projection(
        exit_waves: Sequence[np.ndarray],
        diffraction_patterns: Sequence[np.ndarray],
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE fourier projection static method:
        .. math:: 
            \psi'_{R_j}(r) &= F^{-1}[\sqrt{I_j(u)} F[\psi_{R_j}(u)] / |F[\psi_{R_j}(u)]|] \\
            \phi'_{R_j}(r) &= F^{-1}[\sqrt{\Omega_j(u)} F[\phi_{R_j}(u)] / |F[\phi_{R_j}(u)]|]


        Parameters
        ----------
        exit_waves: Sequence[np.ndarray]
            Exit waves array given by SimultaneousPtychographicOperator._overlap_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of forward and reverse CBED intensities arrays used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: Sequence[np.ndarray]
            Fourier projection of forward and reverse illuminated probes
        sse: float
            Updated sum of squares error estimate
        """
        exit_wave_forward, exit_wave_reverse = exit_waves
        diffraction_forward, diffraction_reverse = diffraction_patterns

        exit_wave_forward_fft = xp.fft.fft2(exit_wave_forward)
        exit_wave_reverse_fft = xp.fft.fft2(exit_wave_reverse)

        sse += (
            xp.mean(xp.abs(xp.abs(exit_wave_forward_fft) - diffraction_forward) ** 2)
            / xp.sum(diffraction_forward**2)
            / 2
        )
        sse += (
            xp.mean(xp.abs(xp.abs(exit_wave_reverse_fft) - diffraction_reverse) ** 2)
            / xp.sum(diffraction_reverse**2)
            / 2
        )

        modified_exit_wave_forward = xp.fft.ifft2(
            diffraction_forward * xp.exp(1j * xp.angle(exit_wave_forward_fft))
        )
        modified_exit_wave_reverse = xp.fft.ifft2(
            diffraction_reverse * xp.exp(1j * xp.angle(exit_wave_reverse_fft))
        )

        return (modified_exit_wave_forward, modified_exit_wave_reverse), sse

    @staticmethod
    def _warmup_update_function(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        exit_waves: Sequence[np.ndarray],
        modified_exit_waves: Sequence[np.ndarray],
        diffraction_patterns: Sequence[np.ndarray],
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE objects and probes update static method:
        .. math::
            O'_{R_j}(r)    &= O_{R_j}(r) + \frac{P^*(r)}{\left(1-\alpha\right)|P(r)|^2 + \alpha|P(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right) \\
            P'(r)          &= P(r) + \frac{O^*_{R_j}(r)}{\left(1-\beta\right)|O_{R_j}(r)|^2 + \beta|O_{R_j}(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right)


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Current objects array estimate
        probes: Sequence[np.ndarray]
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: Sequence[np.ndarray]
            Exit waves array given by SimultaneousPtychographicOperator._warmup_overlap_projection method
        modified_exit_waves: Sequence[np.ndarray]
            Modified exit waves array given by SimultaneousPtychographicOperator._fourier_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: Sequence[np.ndarray]
            Updated electrostatic object array estimate, dummy magnetic object array estimate
        probes: Sequence[np.ndarray]
            Updated forward probe array estimate, dummy reverse probe array estimate
        position: np.ndarray
            Updated probe position estimate
        """
        exit_wave_forward, exit_wave_reverse = exit_waves
        modified_exit_wave_forward, modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object = objects
        probe_forward, probe_reverse = probes

        exit_wave_diff_forward = modified_exit_wave_forward - exit_wave_forward

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]

        probe_forward_conj = xp.conj(probe_forward)
        electrostatic_conj = xp.conj(electrostatic_roi)

        probe_forward_abs_squared = xp.abs(probe_forward) ** 2
        electrostatic_abs_squared = xp.abs(electrostatic_roi) ** 2

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            probe_forward += (
                probe_step_size
                * electrostatic_conj
                * exit_wave_diff_forward
                / (
                    (1 - beta) * electrostatic_abs_squared
                    + beta * xp.max(electrostatic_abs_squared)
                )
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        electrostatic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * exit_wave_diff_forward
            / (
                (1 - alpha) * probe_forward_abs_squared
                + alpha * xp.max(probe_forward_abs_squared)
            )
        )

        return (
            (electrostatic_object, magnetic_object),
            (probe_forward, probe_reverse),
            position,
        )

    @staticmethod
    def _update_function(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        exit_waves: Sequence[np.ndarray],
        modified_exit_waves: Sequence[np.ndarray],
        diffraction_patterns: Sequence[np.ndarray],
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE objects and probes update static method.


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Current objects array estimate
        probes: Sequence[np.ndarray]
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: Sequence[np.ndarray]
            Exit waves array given by SimultaneousPtychographicOperator._overlap_projection method
        modified_exit_waves: Sequence[np.ndarray]
            Modified exit waves array given by SimultaneousPtychographicOperator._fourier_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: Sequence[np.ndarray]
            Updated electrostatic and magnetic object array estimates
        probes: Sequence[np.ndarray]
            Updated forward and reverse probe array estimates
        position: np.ndarray
            Updated probe position estimate
        """
        exit_wave_forward, exit_wave_reverse = exit_waves
        modified_exit_wave_forward, modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object = objects
        probe_forward, probe_reverse = probes

        exit_wave_diff_forward = modified_exit_wave_forward - exit_wave_forward
        exit_wave_diff_reverse = modified_exit_wave_reverse - exit_wave_reverse

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]
        magnetic_roi = magnetic_object[object_indices]

        probe_forward_conj = xp.conj(probe_forward)
        probe_reverse_conj = xp.conj(probe_reverse)
        electrostatic_conj = xp.conj(electrostatic_roi)
        magnetic_conj = xp.conj(magnetic_roi)

        probe_forward_magnetic_abs_squared = xp.abs(probe_forward * magnetic_roi) ** 2
        probe_reverse_magnetic_abs_squared = xp.abs(probe_reverse * magnetic_roi) ** 2
        probe_forward_electrostatic_abs_squared = (
            xp.abs(probe_forward * electrostatic_roi) ** 2
        )
        probe_reverse_electrostatic_abs_squared = (
            xp.abs(probe_reverse * electrostatic_roi) ** 2
        )
        electrostatic_magnetic_abs_squared = (
            xp.abs(electrostatic_roi * magnetic_roi) ** 2
        )

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            probe_forward += (
                probe_step_size
                * electrostatic_conj
                * magnetic_conj
                * exit_wave_diff_forward
                / (
                    (1 - beta) * electrostatic_magnetic_abs_squared
                    + beta * xp.max(electrostatic_magnetic_abs_squared)
                )
                / 2
            )
            probe_reverse += (
                probe_step_size
                * electrostatic_conj
                * magnetic_roi
                * exit_wave_diff_reverse
                / (
                    (1 - beta) * electrostatic_magnetic_abs_squared
                    + beta * xp.max(electrostatic_magnetic_abs_squared)
                )
                / 2
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        electrostatic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * magnetic_conj
            * exit_wave_diff_forward
            / (
                (1 - alpha) * probe_forward_magnetic_abs_squared
                + alpha * xp.max(probe_forward_magnetic_abs_squared)
            )
            / 2
        )
        electrostatic_object[object_indices] += (
            object_step_size
            * probe_reverse_conj
            * magnetic_roi
            * exit_wave_diff_reverse
            / (
                (1 - alpha) * probe_reverse_magnetic_abs_squared
                + alpha * xp.max(probe_reverse_magnetic_abs_squared)
            )
            / 2
        )

        magnetic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * electrostatic_conj
            * exit_wave_diff_forward
            / (
                (1 - alpha) * probe_forward_electrostatic_abs_squared
                + alpha * xp.max(probe_forward_electrostatic_abs_squared)
            )
            / 2
        )
        magnetic_object[object_indices] -= (
            object_step_size
            * probe_reverse_conj
            * electrostatic_conj
            * exit_wave_diff_reverse
            / (
                (1 - alpha) * probe_reverse_electrostatic_abs_squared
                + alpha * xp.max(probe_reverse_electrostatic_abs_squared)
            )
            / 2
        )

        return (
            (electrostatic_object, magnetic_object),
            (probe_forward, probe_reverse),
            position,
        )

    @staticmethod
    def _alternative_update_function(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        exit_waves: Sequence[np.ndarray],
        modified_exit_waves: Sequence[np.ndarray],
        diffraction_patterns: Sequence[np.ndarray],
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE objects and probes update static method using a common probe.


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Current objects array estimate
        probes: Sequence[np.ndarray]
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: Sequence[np.ndarray]
            Exit waves array given by SimultaneousPtychographicOperator._overlap_projection method
        modified_exit_waves: Sequence[np.ndarray]
            Modified exit waves array given by SimultaneousPtychographicOperator._fourier_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: Sequence[np.ndarray]
            Updated electrostatic and magnetic object array estimates
        probes: Sequence[np.ndarray]
            Updated forward probe array estimate, dummy reverse probe array estimate
        position: np.ndarray
            Updated probe position estimate
        """
        exit_wave_forward, exit_wave_reverse = exit_waves
        modified_exit_wave_forward, modified_exit_wave_reverse = modified_exit_waves
        electrostatic_object, magnetic_object = objects
        probe_forward, probe_reverse = probes

        exit_wave_diff_forward = modified_exit_wave_forward - exit_wave_forward
        exit_wave_diff_reverse = modified_exit_wave_reverse - exit_wave_reverse

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        electrostatic_roi = electrostatic_object[object_indices]
        magnetic_roi = magnetic_object[object_indices]

        probe_forward_conj = xp.conj(probe_forward)
        electrostatic_conj = xp.conj(electrostatic_roi)
        magnetic_conj = xp.conj(magnetic_roi)

        probe_forward_magnetic_abs_squared = xp.abs(probe_forward * magnetic_roi) ** 2
        probe_forward_electrostatic_abs_squared = (
            xp.abs(probe_forward * electrostatic_roi) ** 2
        )
        electrostatic_magnetic_abs_squared = (
            xp.abs(electrostatic_roi * magnetic_roi) ** 2
        )

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            probe_forward += (
                probe_step_size
                * electrostatic_conj
                * magnetic_conj
                * exit_wave_diff_forward
                / (
                    (1 - beta) * electrostatic_magnetic_abs_squared
                    + beta * xp.max(electrostatic_magnetic_abs_squared)
                )
                / 2
            )
            probe_forward += (
                probe_step_size
                * electrostatic_conj
                * magnetic_roi
                * exit_wave_diff_reverse
                / (
                    (1 - beta) * electrostatic_magnetic_abs_squared
                    + beta * xp.max(electrostatic_magnetic_abs_squared)
                )
                / 2
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        electrostatic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * magnetic_conj
            * exit_wave_diff_forward
            / (
                (1 - alpha) * probe_forward_magnetic_abs_squared
                + alpha * xp.max(probe_forward_magnetic_abs_squared)
            )
            / 2
        )
        electrostatic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * magnetic_roi
            * exit_wave_diff_reverse
            / (
                (1 - alpha) * probe_forward_magnetic_abs_squared
                + alpha * xp.max(probe_forward_magnetic_abs_squared)
            )
            / 2
        )

        magnetic_object[object_indices] += (
            object_step_size
            * probe_forward_conj
            * electrostatic_conj
            * exit_wave_diff_forward
            / (
                (1 - alpha) * probe_forward_electrostatic_abs_squared
                + alpha * xp.max(probe_forward_electrostatic_abs_squared)
            )
            / 2
        )
        magnetic_object[object_indices] -= (
            object_step_size
            * probe_forward_conj
            * electrostatic_conj
            * exit_wave_diff_reverse
            / (
                (1 - alpha) * probe_forward_electrostatic_abs_squared
                + alpha * xp.max(probe_forward_electrostatic_abs_squared)
            )
            / 2
        )

        return (
            (electrostatic_object, magnetic_object),
            (probe_forward, probe_reverse),
            position,
        )

    @staticmethod
    def _constraints_function(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        pure_phase_object: bool,
        xp=np,
        **kwargs,
    ):
        """
        Simultaneous-PIE constraints static method:

        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        pure_phase_object:bool
            If True, constraints object to being a pure phase object, i.e. with unit amplitude
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Constrained objects array
        probes: np.ndarray
            Constrained probes array
        """
        electrostatic_object, magnetic_object = objects

        phase_e = xp.exp(1.0j * xp.angle(electrostatic_object))
        phase_m = xp.exp(1.0j * xp.angle(magnetic_object))
        if pure_phase_object:
            amplitude_e = 1.0
            amplitude_m = 1.0
        else:
            amplitude_e = xp.minimum(xp.abs(electrostatic_object), 1.0)
            amplitude_m = xp.minimum(xp.abs(magnetic_object), 1.0)
        return (amplitude_e * phase_e, amplitude_m * phase_m), probes

    @staticmethod
    def _position_correction(
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        position: np.ndarray,
        exit_wave: Sequence[np.ndarray],
        modified_exit_wave: Sequence[np.ndarray],
        diffraction_pattern: Sequence[np.ndarray],
        sobel: Callable,
        position_step_size: float = 1.0,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE probe position correction method.


        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Current objects array estimate
        probes: Sequence[np.ndarray]
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_wave: Sequence[np.ndarray]
            Exit wave array given by SimultaneousPtychographicOperator._overlap_projection method
        modified_exit_wave: Sequence[np.ndarray]
            Modified exit wave array given by SimultaneousPtychographicOperator._fourier_projection method
        diffraction_patterns: Sequence[np.ndarray]
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        position_step_size: float, optional
            Gradient step size for position update step
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        position: np.ndarray
            Updated probe position estimate
        """

        electrostatic_object, magnetic_object = objects
        probe_forward, probe_reverse = probes
        exit_wave_forward, exit_wave_reverse = exit_waves
        modified_exit_wave_forward, modified_exit_wave_reverse = modified_exit_waves
        exit_wave_diff_forward = modified_exit_wave_forward - exit_wave_forward

        object_dx = sobel(electrostatic_object, axis=0, mode="wrap")
        object_dy = sobel(electrostatic_object, axis=1, mode="wrap")

        object_indices = _wrapped_indices_2D_window(
            position, probe_forward.shape, electrostatic_object.shape
        )
        exit_wave_dx = object_dx[object_indices] * probe_forward
        exit_wave_dy = object_dy[object_indices] * probe_forward

        exit_wave_diff = modified_exit_wave - exit_wave
        displacement_x = xp.sum(
            xp.real(xp.conj(exit_wave_dx) * exit_wave_diff_forward)
        ) / xp.sum(xp.abs(exit_wave_dx) ** 2)
        displacement_y = xp.sum(
            xp.real(xp.conj(exit_wave_dy) * exit_wave_diff_forward)
        ) / xp.sum(xp.abs(exit_wave_dy) ** 2)

        return position + position_step_size * xp.array(
            [displacement_x, displacement_y]
        )

    @staticmethod
    def _fix_probe_center_of_mass(
        probes: Sequence[np.ndarray], center_of_mass: Callable, xp=np, **kwargs
    ):
        """
        Simultaneous-PIE probe center correction method.


        Parameters
        ----------
        probes: Sequence[np.ndarray]
            Current probes arrays estimate
        center_of_mass: Callable
            The scipy.ndimage module used to compute the array center of mass
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: Sequence[np.ndarray]
            Center-of-mass corrected probes array
        """

        probe_center = xp.array(probes[0].shape) / 2

        _probes = []
        for k in range(len(probes)):
            com = center_of_mass(xp.abs(probes[k]) ** 2)
            _probes.append(fft_shift(probes[k], probe_center - xp.array(com)))

        return tuple(_probes)

    def _prepare_functions_queue(
        self,
        max_iterations: int,
        warmup_update_steps: int = 0,
        common_probe: bool = False,
        pre_position_correction_update_steps: int = None,
        pre_probe_correction_update_steps: int = None,
        pure_phase_object_update_steps: int = None,
        **kwargs,
    ):
        """
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting.

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        warmup_update_steps: int, optional
            Number of update steps (not iterations) to perform using _warmup_ functions
        common_probe: bool, optional
            If True, use a common probe using _alternative_ functions
        pre_position_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling position correction
        pre_probe_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling probe correction

        Returns
        -------
        functions_queue: (max_iterations,J) list
            List of function calls
        queue_summary: str
            Summary of function calls the reconstruction loop will perform
        """
        _overlap_projection = (
            self._alternative_overlap_projection
            if common_probe
            else self._overlap_projection
        )
        _update_function = (
            self._alternative_update_function if common_probe else self._update_function
        )

        total_update_steps = max_iterations * self._num_diffraction_patterns
        queue_summary = "Ptychographic reconstruction will perform the following steps:"

        functions_tuple = (
            self._warmup_overlap_projection,
            self._warmup_fourier_projection,
            self._warmup_update_function,
            self._constraints_function,
            None,
        )
        functions_queue = [functions_tuple]

        if pre_position_correction_update_steps is None:
            functions_queue *= warmup_update_steps
            queue_summary += f"\n--Regularized PIE for {warmup_update_steps} steps"

            functions_tuple = (
                _overlap_projection,
                self._fourier_projection,
                _update_function,
                self._constraints_function,
                None,
            )
            remaining_update_steps = total_update_steps - warmup_update_steps
            functions_queue += [functions_tuple] * remaining_update_steps
            queue_summary += f"\n--Simultaneous PIE for {remaining_update_steps} steps"
        else:
            if warmup_update_steps <= pre_position_correction_update_steps:
                functions_queue *= warmup_update_steps
                queue_summary += f"\n--Regularized PIE for {warmup_update_steps} steps"

                functions_tuple = (
                    _overlap_projection,
                    self._fourier_projection,
                    _update_function,
                    self._constraints_function,
                    None,
                )
                remaining_update_steps = (
                    pre_position_correction_update_steps - warmup_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += (
                    f"\n--Simultaneous PIE for {remaining_update_steps} steps"
                )

                functions_tuple = (
                    _overlap_projection,
                    self._fourier_projection,
                    _update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = (
                    total_update_steps - pre_position_correction_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Simultaneous PIE with position correction for {remaining_update_steps} steps"
            else:
                functions_queue *= pre_position_correction_update_steps
                queue_summary += f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"

                functions_tuple = (
                    self._warmup_overlap_projection,
                    self._warmup_fourier_projection,
                    self._warmup_update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = (
                    warmup_update_steps - pre_position_correction_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"

                functions_tuple = (
                    _overlap_projection,
                    self._fourier_projection,
                    _update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = total_update_steps - warmup_update_steps
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Simultaneous PIE with position correction for {remaining_update_steps} steps"

        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"

        if common_probe:
            queue_summary += (
                f"\n--Using the first probe as a common probe for both objects"
            )

        if pure_phase_object_update_steps is not None:
            queue_summary += f"\n--Reconstructed object will be constrained to a pure-phase object for the first {pure_phase_object_update_steps} steps"

        functions_queue = [
            functions_queue[x : x + self._num_diffraction_patterns]
            for x in range(0, total_update_steps, self._num_diffraction_patterns)
        ]

        return functions_queue, queue_summary

    def reconstruct(
        self,
        max_iterations: int = 5,
        return_iterations: bool = False,
        warmup_update_steps: int = 0,
        common_probe: bool = False,
        fix_com: bool = True,
        random_seed=None,
        verbose: bool = False,
        functions_queue: Iterable = None,
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):
        """
        Main reconstruction loop method to do the following:
        - Precompute the order of function calls using the SimultaneousPtychographicOperator._prepare_functions_queue method
        - Iterate through function calls in queue
        - Pass reconstruction outputs to the SimultaneousPtychographicOperator._prepare_measurement_outputs method

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        return_iterations: bool, optional
            If True, method will return a list of current objects, probes, and positions estimates for each iteration
        warmup_update_steps: int, optional
            Number of warmup update steps to perform before simultaneous reconstruction begins
        common_probe: bool, optional
            If True, use a common probe for both sets of measurements
        fix_com: bool, optional
            If True, the center of mass of the probes array will be corrected at the end of the iteration
        random_seed
            If not None, used to seed the numpy random number generator
        verbose: bool, optional
            If True, prints functions queue and current iteration error
        functions_queue: (max_iterations, J) Iterable, optional
            If not None, the reconstruction algorithm will use the input functions queue instead
        parameters: dict, optional
            Dictionary specifying any of the abtem.reconstruct.recontruction_symbols parameters
            Additionally, these can also be specified using kwargs

        Returns
        -------
        reconstructed_object_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the objects estimate at each iteration is returned
        reconstructed_probes_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the probes estimate at each iteration is returned
        reconstructed_position_measurement: np.ndarray or Sequence[np.ndarray]
            If return_iterations, a list of position estimates at each iteration is returned
        reconstruction_error: float or Sequence[float]
            If return_iterations, a list of the reconstruction error at each iteration is returned
        """
        for key in kwargs.keys():
            if key not in reconstruction_symbols.keys():
                raise ValueError("{} not a recognized parameter".format(key))

        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)

        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                max_iterations,
                warmup_update_steps=warmup_update_steps,
                common_probe=common_probe,
                pre_position_correction_update_steps=self._reconstruction_parameters[
                    "pre_position_correction_update_steps"
                ],
                pre_probe_correction_update_steps=self._reconstruction_parameters[
                    "pre_probe_correction_update_steps"
                ],
                pure_phase_object_update_steps=self._reconstruction_parameters[
                    "pure_phase_object_update_steps"
                ],
            )
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [
                        [function_tuples] * self._num_diffraction_patterns
                        for function_tuples in functions_queue
                    ]
            elif (
                len(functions_queue) == max_iterations * self._num_diffraction_patterns
            ):
                functions_queue = [
                    functions_queue[x : x + self._num_diffraction_patterns]
                    for x in range(
                        0, total_update_steps, self._num_diffraction_patterns
                    )
                ]
            else:
                raise ValueError()

        self._functions_queue = functions_queue

        ### Main Loop
        xp = get_array_module_from_device(self._device)
        outer_pbar = ProgressBar(total=max_iterations, leave=False)
        inner_pbar = ProgressBar(total=self._num_diffraction_patterns, leave=False)
        indices = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(
            self._experimental_parameters["object_px_padding"]
        )
        center_of_mass = get_scipy_module(xp).ndimage.center_of_mass
        sobel = get_scipy_module(xp).ndimage.sobel

        if return_iterations:
            objects_iterations = []
            probes_iterations = []
            positions_iterations = []
            sse_iterations = []

        if random_seed is not None:
            np.random.seed(random_seed)

        for iteration_index, iteration_step in enumerate(self._functions_queue):

            inner_pbar.reset()

            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse = 0.0

            for update_index, update_step in enumerate(iteration_step):

                index = indices[update_index]
                position = self._positions_px[index]

                # Skip empty diffraction patterns
                diffraction_pattern = tuple(
                    dp[index] for dp in self._diffraction_patterns
                )

                if any(tuple(xp.sum(dp) == 0.0 for dp in diffraction_pattern)):
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i = (
                    iteration_index * self._num_diffraction_patterns + update_index
                )

                if (
                    self._reconstruction_parameters["pre_probe_correction_update_steps"]
                    is None
                ):
                    fix_probe = False
                else:
                    fix_probe = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pre_probe_correction_update_steps"
                        ]
                    )

                if (
                    self._reconstruction_parameters["pure_phase_object_update_steps"]
                    is None
                ):
                    pure_phase_object = False
                else:
                    pure_phase_object = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pure_phase_object_update_steps"
                        ]
                    )

                if warmup_update_steps != 0 and global_iteration_i == (
                    warmup_update_steps + 1
                ):
                    self._probes = (self._probes[0], self._probes[0].copy())

                (
                    _overlap_projection,
                    _fourier_projection,
                    _update_function,
                    _constraints_function,
                    _position_correction,
                ) = update_step

                self._probes, exit_wave = _overlap_projection(
                    self._objects, self._probes, position, old_position, xp=xp
                )

                modified_exit_wave, self._sse = _fourier_projection(
                    exit_wave, diffraction_pattern, self._sse, xp=xp
                )

                (
                    self._objects,
                    self._probes,
                    self._positions_px[index],
                ) = _update_function(
                    self._objects,
                    self._probes,
                    position,
                    exit_wave,
                    modified_exit_wave,
                    diffraction_pattern,
                    fix_probe=fix_probe,
                    position_correction=_position_correction,
                    sobel=sobel,
                    reconstruction_parameters=self._reconstruction_parameters,
                    xp=xp,
                )

                self._objects, self._probes = _constraints_function(
                    self._objects, self._probes, pure_phase_object, xp=xp
                )

                old_position = position
                inner_pbar.update(1)

            # Shift probe back to origin
            self._probes = tuple(
                fft_shift(_probe, xp.round(position) - position)
                for _probe in self._probes
            )

            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(
                    self._probes, center_of_mass, xp=xp
                )

            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (
                    xp.mean(self._positions_px, axis=0) - self._positions_px_com
                )
                self._reconstruction_parameters[
                    "position_step_size"
                ] *= self._reconstruction_parameters["step_size_damping_rate"]

            # Update Parameters
            self._reconstruction_parameters[
                "object_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._reconstruction_parameters[
                "probe_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._sse /= self._num_diffraction_patterns

            if return_iterations:
                objects_iterations.append(copy(self._objects))
                probes_iterations.append(copy(self._probes))
                positions_iterations.append(
                    self._positions_px.copy() * xp.array(self.sampling)
                )
                sse_iterations.append(self._sse)

            if verbose:
                print(
                    f"----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}"
                )

            outer_pbar.update(1)

        inner_pbar.close()
        outer_pbar.close()

        #  Return Results
        if return_iterations:
            results = map(
                self._prepare_measurement_outputs,
                objects_iterations,
                probes_iterations,
                positions_iterations,
                sse_iterations,
            )

            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(
                self._objects,
                self._probes,
                self._positions_px * xp.array(self.sampling),
                self._sse,
            )
            return results

    def _prepare_measurement_outputs(
        self,
        objects: Sequence[np.ndarray],
        probes: Sequence[np.ndarray],
        positions: np.ndarray,
        sse: np.ndarray,
    ):
        """
        Method to format the reconstruction outputs as Measurement objects.

        Parameters
        ----------
        objects: Sequence[np.ndarray]
            Reconstructed objects array
        probes: Sequence[np.ndarray]
            Reconstructed probes array
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error

        Returns
        -------
        objects_measurement: Sequence[Measurement]
            Reconstructed objects Measurement
        probes_measurement: Sequence[Measurement]
            Reconstructed probes Measurement
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error
        """

        calibrations = tuple(
            Calibration(0, s, units="Å", name=n, endpoint=False)
            for s, n in zip(self.sampling, ("x", "y"))
        )

        measurement_objects = tuple(
            Measurement(asnumpy(_object), calibrations) for _object in objects
        )
        measurement_probes = tuple(
            Measurement(asnumpy(_probe), calibrations) for _probe in probes
        )

        return measurement_objects, measurement_probes, asnumpy(positions), sse


class MixedStatePtychographicOperator(AbstractPtychographicOperator):
    """
    Mixed-State Ptychographic Iterative Engine (mix-PIE).
    Used to reconstruct weak-phase objects with partial coherence of the illuminating probe using a set of measured far-field CBED patterns with the following array dimensions:

    CBED pattern dimensions     : (J,M,N)
    objects dimensions          : (P,Q)
    probes dimensions           : (K,R,S)

    Parameters
    ----------
    diffraction_patterns: np.ndarray or Measurement
        Input 3D or 4D CBED pattern intensities with dimensions (M,N)
    energy: float
        Electron energy [eV]
    num_probes: int
        Number of mixed-state probes
    region_of_interest_shape: (2,) Sequence[int], optional
        Pixel dimensions (R,S) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the CBED dimensions (M,N)
    objects: np.ndarray, optional
        Initial objects guess with dimensions (P,Q) - Useful for restarting reconstructions
        If None, an array with 1.0j is initialized
    probes: np.ndarray or Probe, optional
        Initial probes guess with dimensions/gpts (R,S) - Useful for restarting reconstructions
        If None, a Probe with CTF given by the polar_parameters dictionary is initialized
    positions: np.ndarray, optional
        Initial positions guess [Å]
        If None, a raster scan with step sizes given by the experimental_parameters dictionary is initialized
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial Probe guess
    preprocess: bool, optional
        If True, it runs the preprocess method after initialization
    device: str, optional
        Device to perform Fourier-based reconstructrions - Either 'cpu' or 'gpu'
    parameters: dict, optional
       Dictionary specifying any of the abtem.transfer.polar_symbols or abtem.reconstruct.experimental_symbols parameters
       Additionally, these can also be specified using kwargs
    """

    def __init__(
        self,
        diffraction_patterns: Union[np.ndarray, Measurement],
        energy: float,
        num_probes: int,
        region_of_interest_shape: Sequence[int] = None,
        objects: np.ndarray = None,
        probes: Union[np.ndarray, Probe] = None,
        positions: np.ndarray = None,
        semiangle_cutoff: float = None,
        preprocess: bool = False,
        device: str = "cpu",
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        for key in kwargs.keys():
            if (
                (key not in polar_symbols)
                and (key not in polar_aliases.keys())
                and (key not in experimental_symbols)
            ):
                raise ValueError("{} not a recognized parameter".format(key))

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))
        self._experimental_parameters = dict(
            zip(experimental_symbols, [None] * len(experimental_symbols))
        )

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(
            parameters, self._polar_parameters, self._experimental_parameters
        )

        self._region_of_interest_shape = region_of_interest_shape
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._positions = positions
        self._device = device
        self._objects = objects
        self._probes = probes
        self._num_probes = num_probes
        self._diffraction_patterns = diffraction_patterns

        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False

    def preprocess(self):
        """
        Preprocess method to do the following:
        - Pads CBED patterns to region of interest dimensions
        - Prepares initial guess for scanning positions
        - Prepares initial guesses for the objects and probes arrays


        Returns
        -------
        preprocessed_ptychographic_operator: MixedStatePtychographicOperator
        """

        self._preprocessed = True

        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            (
                self._diffraction_patterns,
                angular_sampling,
                step_sizes,
            ) = self._extract_calibrations_from_measurement_object(
                self._diffraction_patterns, self._energy
            )
            self._experimental_parameters["angular_sampling"] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters["scan_step_sizes"] = step_sizes

        # Preprocess Diffraction Patterns
        xp = get_array_module_from_device(self._device)
        self._diffraction_patterns = copy_to_device(
            self._diffraction_patterns, self._device
        )

        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters[
                "grid_scan_shape"
            ] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns = self._diffraction_patterns.reshape(
                (-1,) + self._diffraction_patterns.shape[-2:]
            )

        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]

        self._diffraction_patterns = self._pad_diffraction_patterns(
            self._diffraction_patterns, self._region_of_interest_shape
        )
        self._num_diffraction_patterns = self._diffraction_patterns.shape[0]

        if self._experimental_parameters["background_counts_cutoff"] is not None:
            self._diffraction_patterns[
                self._diffraction_patterns
                < self._experimental_parameters["background_counts_cutoff"]
            ] = 0.0

        if self._experimental_parameters["counts_scaling_factor"] is not None:
            self._diffraction_patterns /= self._experimental_parameters[
                "counts_scaling_factor"
            ]

        self._mean_diffraction_intensity = (
            xp.sum(self._diffraction_patterns) / self._num_diffraction_patterns
        )
        self._diffraction_patterns = xp.fft.ifftshift(
            xp.sqrt(self._diffraction_patterns), axes=(-2, -1)
        )

        # Scan Positions Initialization
        (
            positions_px,
            self._experimental_parameters,
        ) = self._calculate_scan_positions_in_pixels(
            self._positions,
            self.sampling,
            self._region_of_interest_shape,
            self._experimental_parameters,
        )

        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y = self._experimental_parameters["object_px_padding"]
            p, q = np.max(positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._objects = xp.ones((p, q), dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects, self._device)

        self._positions_px = copy_to_device(positions_px, self._device)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)

        # Probes Initialization
        if self._probes is None:
            ctf = CTF(
                energy=self._energy,
                semiangle_cutoff=self._semiangle_cutoff,
                parameters=self._polar_parameters,
            )
            self._probes = (
                Probe(
                    semiangle_cutoff=self._semiangle_cutoff,
                    energy=self._energy,
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    ctf=ctf,
                    device=self._device,
                )
                .build()
                .array
            )
        else:
            if isinstance(self._probes, Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                self._probes = copy_to_device(self._probes.build().array, self._device)
            else:
                self._probes = copy_to_device(self._probes, self._device)

        probe_intensity = xp.sum(xp.abs(xp.fft.fft2(self._probes)) ** 2)
        self._probes *= np.sqrt(self._mean_diffraction_intensity / probe_intensity)

        self._probes = xp.tile(self._probes, (self._num_probes, 1, 1))
        self._probes /= xp.arange(self._num_probes)[:, None, None] + 1

        return self

    @staticmethod
    def _warmup_overlap_projection(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE overlap projection static method using a single probe:
        .. math::
            \psi^0_{R_j}(r) = O_{R_j}(r) * P^0(r)


        Parameters
        ----------
        objects: np.ndarray
            Object array to be illuminated
        probes: np.ndarray
            Probe window array to illuminate object with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Fractionally shifted probe window array
        exit_wave: np.ndarray
            Overlap projection of illuminated probe
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probes[0] = fft_shift(probes[0], fractional_position - old_fractional_position)
        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape
        )
        object_roi = objects[object_indices]
        exit_wave = object_roi * probes[0]

        return probes, exit_wave

    @staticmethod
    def _overlap_projection(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        old_position: np.ndarray,
        xp=np,
        **kwargs,
    ):
        """
        Mixed-State-PIE overlap projection static method:
        .. math::
            \psi^k_{R_j}(r) = O_{R_j}(r) * P^k(r)


        Parameters
        ----------
        objects: np.ndarray
            Object array to be illuminated
        probes: np.ndarray
            Probe window array to illuminate object with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Fractionally shifted probes window array
        exit_waves: np.ndarray
            Overlap projection of illuminated probes
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probes = fft_shift(probes, fractional_position - old_fractional_position)
        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape
        )
        object_roi = objects[object_indices]

        exit_waves = xp.empty_like(probes)
        for k in range(probes.shape[0]):
            exit_waves[k] = object_roi * probes[k]

        return probes, exit_waves

    @staticmethod
    def _warmup_fourier_projection(
        exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE fourier projection static method using a single probe:
        .. math::
            \psi'^0_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^0_{R_j}(u)] / |F[\psi^0_{R_j}(u)]|]


        Parameters
        ----------
        exit_waves: np.ndarray
            Exit waves array given by MixedStatePtychographicOperator._warmup_overlap_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: np.ndarray
            Fourier projection of illuminated probe
        sse: float
            Updated sum of squares error estimate
        """
        exit_wave_fft = xp.fft.fft2(exit_waves)
        sse += xp.mean(
            xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns) ** 2
        ) / xp.sum(diffraction_patterns**2)
        modified_exit_wave = xp.fft.ifft2(
            diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft))
        )

        return modified_exit_wave, sse

    @staticmethod
    def _fourier_projection(
        exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Mixed-State-PIE fourier projection static method:
        .. math::
            \psi'^k_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^k_{R_j}(u)] / \sqrt{\sum_k|F[\psi^k_{R_j}(u)]|^2}]


        Parameters
        ----------
        exit_waves: np.ndarray
            Exit waves array given by MixedStatePtychographicOperator._overlap_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: np.ndarray
            Fourier projection of illuminated probe
        sse: float
            Updated sum of squares error estimate
        """
        exit_waves_fft = xp.fft.fft2(exit_waves, axes=(-2, -1))
        intensity_norm = xp.sqrt(xp.sum(xp.abs(exit_waves_fft) ** 2, axis=0))
        amplitude_modification = diffraction_patterns / intensity_norm
        sse += xp.mean(xp.abs(intensity_norm - diffraction_patterns) ** 2) / xp.sum(
            diffraction_patterns**2
        )

        modified_exit_wave = xp.fft.ifft2(
            amplitude_modification[None] * exit_waves_fft, axes=(-2, -1)
        )

        return modified_exit_wave, sse

    @staticmethod
    def _warmup_update_function(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_waves: np.ndarray,
        modified_exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE objects and probes update static method using a single probe:
        .. math::
            O'_{R_j}(r)    &= O_{R_j}(r) + \frac{P^{0*}(r)}{\left(1-\alpha\right)|P^0(r)|^2 + \alpha|P^0(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right) \\
            P^{0'}(r)          &= P^0(r) + \frac{O^*_{R_j}(r)}{\left(1-\beta\right)|O_{R_j}(r)|^2 + \beta|O_{R_j}(r)|_{\mathrm{max}}^2} \left(\psi'_{R_j}(r) - \psi_{R_j}(r)\right)


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: np.ndarray
            Exit waves array given by MixedStatePtychographicOperator._warmup_overlap_projection method
        modified_exit_waves: np.ndarray
            Modified exit waves array given by MixedStatePtychographicOperator._warmup_fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Updated objects array estimate
        probes: np.ndarray
            Updated probes array estimate
        position: np.ndarray
            Updated probe position estimate
        """

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape
        )
        object_roi = objects[object_indices]

        exit_wave_diff = modified_exit_waves - exit_waves

        probe_conj = xp.conj(probes[0])
        probe_abs_squared = xp.abs(probes[0]) ** 2
        obj_conj = xp.conj(object_roi)
        obj_abs_squared = xp.abs(object_roi) ** 2

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        objects[object_indices] += (
            object_step_size
            * probe_conj
            * exit_wave_diff
            / ((1 - alpha) * probe_abs_squared + alpha * xp.max(probe_abs_squared))
        )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            probes[0] += (
                probe_step_size
                * obj_conj
                * exit_wave_diff
                / ((1 - beta) * obj_abs_squared + beta * xp.max(obj_abs_squared))
            )

        return objects, probes, position

    @staticmethod
    def _update_function(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_waves: np.ndarray,
        modified_exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        fix_probe: bool = False,
        orthogonalize_probes: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        xp=np,
        **kwargs,
    ):
        """
        Mixed-State-PIE objects and probes update static method:
        .. math::
            O'_{R_j}(r)    &= O_{R_j}(r) + \frac{1}{\left(1-\alpha\right)\sum_k|P^k(r)|^2 + \alpha\sum_k|P^k(r)|_{\mathrm{max}}^2} \left(\sum_k P^{k*}\psi^{k'}_{R_j}(r) - \psi^k_{R_j}(r)\right) \\
            P^{k'}(r)          &= P^k(r) + \frac{O^*_{R_j}(r)}{\left(1-\beta\right)|O_{R_j}(r)|^2 + \beta|O_{R_j}(r)|_{\mathrm{max}}^2} \left(\psi^{k'}_{R_j}(r) - \psi^k_{R_j}(r)\right)


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: np.ndarray
            Exit waves array given by MixedStatePtychographicOperator._overlap_projection method
        modified_exit_waves: np.ndarray
            Modified exit waves array given by MixedStatePtychographicOperator._fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Updated objects array estimate
        probes: np.ndarray
            Updated probes array estimate
        position: np.ndarray
            Updated probe position estimate
        """

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape
        )
        object_roi = objects[object_indices]

        exit_wave_diff = modified_exit_waves - exit_waves

        probe_conj = xp.conj(probes)
        probe_abs_squared_norm = xp.sum(xp.abs(probes) ** 2, axis=0)
        obj_conj = xp.conj(object_roi)
        obj_abs_squared = xp.abs(object_roi) ** 2

        if position_correction is not None:
            position_step_size = reconstruction_parameters["position_step_size"]
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        alpha = reconstruction_parameters["alpha"]
        object_step_size = reconstruction_parameters["object_step_size"]
        objects[object_indices] += (
            object_step_size
            * xp.sum(probe_conj * exit_wave_diff, axis=0)
            / (
                (1 - alpha) * probe_abs_squared_norm
                + alpha * xp.max(probe_abs_squared_norm)
            )
        )

        if not fix_probe:
            beta = reconstruction_parameters["beta"]
            probe_step_size = reconstruction_parameters["probe_step_size"]
            update_numerator = probe_step_size * obj_conj[None] * exit_wave_diff
            update_denominator = (1 - beta) * obj_abs_squared + beta * xp.max(
                obj_abs_squared
            )
            probes += update_numerator / update_denominator[None]

            if orthogonalize_probes:
                probes = _orthogonalize(probes.reshape((probes.shape[0], -1))).reshape(
                    probes.shape
                )

        return objects, probes, position

    @staticmethod
    def _constraints_function(
        objects: np.ndarray,
        probes: np.ndarray,
        pure_phase_object: bool,
        xp=np,
        **kwargs,
    ):
        """
        Mixed-State-PIE constraints static method:

        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        pure_phase_object:bool
            If True, constraints object to being a pure phase object, i.e. with unit amplitude
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Constrained objects array
        probes: np.ndarray
            Constrained probes array
        """
        phase = xp.exp(1.0j * xp.angle(objects))
        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(objects), 1.0)
        return amplitude * phase, probes

    @staticmethod
    def _position_correction(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_wave: np.ndarray,
        modified_exit_wave: np.ndarray,
        diffraction_pattern: np.ndarray,
        sobel: Callable,
        position_step_size: float = 1.0,
        xp=np,
        **kwargs,
    ):
        """
        Regularized-PIE probe position correction method.


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_wave: np.ndarray
            Exit wave array given by MixedStatePtychographicOperator._warmup_overlap_projection method
        modified_exit_wave: np.ndarray
            Modified exit wave array given by MixedStatePtychographicOperator._warmup_fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        position_step_size: float, optional
            Gradient step size for position update step
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        position: np.ndarray
            Updated probe position estimate
        """

        object_dx = sobel(objects, axis=0, mode="wrap")
        object_dy = sobel(objects, axis=1, mode="wrap")

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape
        )
        exit_wave_dx = object_dx[object_indices] * probes[0]
        exit_wave_dy = object_dy[object_indices] * probes[0]

        exit_wave_diff = modified_exit_wave[0] - exit_wave[0]
        displacement_x = xp.sum(
            xp.real(xp.conj(exit_wave_dx) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dx) ** 2)
        displacement_y = xp.sum(
            xp.real(xp.conj(exit_wave_dy) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dy) ** 2)

        return position + position_step_size * xp.array(
            [displacement_x, displacement_y]
        )

    @staticmethod
    def _fix_probe_center_of_mass(
        probes: np.ndarray, center_of_mass: Callable, xp=np, **kwargs
    ):
        """
        Mixed-State-PIE probe center correction method.


        Parameters
        ----------
        probes: np.ndarray
            Current probes array estimate
        center_of_mass: Callable
            The scipy.ndimage module used to compute the array center of mass
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Center-of-mass corrected probes array
        """

        probe_center = xp.array(probes.shape[-2:]) / 2
        for k in range(probes.shape[0]):
            com = center_of_mass(xp.abs(probes[k]) ** 2)
            probes[k] = fft_shift(probes[k], probe_center - xp.array(com))

        return probes

    def _prepare_functions_queue(
        self,
        max_iterations: int,
        warmup_update_steps: int = 0,
        pre_position_correction_update_steps: int = None,
        pre_probe_correction_update_steps: int = None,
        pure_phase_object_update_steps: int = None,
        **kwargs,
    ):
        """
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting.

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        warmup_update_steps: int, optional
            Number of update steps (not iterations) to perform using _warmup_ functions
        pre_position_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling position correction
        pre_probe_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling probe correction

        Returns
        -------
        functions_queue: (max_iterations,J) list
            List of function calls
        queue_summary: str
            Summary of function calls the reconstruction loop will perform
        """
        total_update_steps = max_iterations * self._num_diffraction_patterns
        queue_summary = "Ptychographic reconstruction will perform the following steps:"

        functions_tuple = (
            self._warmup_overlap_projection,
            self._warmup_fourier_projection,
            self._warmup_update_function,
            self._constraints_function,
            None,
        )
        functions_queue = [functions_tuple]

        if pre_position_correction_update_steps is None:
            functions_queue *= warmup_update_steps
            queue_summary += f"\n--Regularized PIE for {warmup_update_steps} steps"

            functions_tuple = (
                self._overlap_projection,
                self._fourier_projection,
                self._update_function,
                self._constraints_function,
                None,
            )
            remaining_update_steps = total_update_steps - warmup_update_steps
            functions_queue += [functions_tuple] * remaining_update_steps
            queue_summary += f"\n--Mixed-State PIE for {remaining_update_steps} steps"
        else:
            if warmup_update_steps <= pre_position_correction_update_steps:
                functions_queue *= warmup_update_steps
                queue_summary += f"\n--Regularized PIE for {warmup_update_steps} steps"

                functions_tuple = (
                    self._overlap_projection,
                    self._fourier_projection,
                    self._update_function,
                    self._constraints_function,
                    None,
                )
                remaining_update_steps = (
                    pre_position_correction_update_steps - warmup_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += (
                    f"\n--Mixed-State PIE for {remaining_update_steps} steps"
                )

                functions_tuple = (
                    self._overlap_projection,
                    self._fourier_projection,
                    self._update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = (
                    total_update_steps - pre_position_correction_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Mixed-State PIE with position correction for {remaining_update_steps} steps"
            else:
                functions_queue *= pre_position_correction_update_steps
                queue_summary += f"\n--Regularized PIE for {pre_position_correction_update_steps} steps"

                functions_tuple = (
                    self._warmup_overlap_projection,
                    self._warmup_fourier_projection,
                    self._warmup_update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = (
                    warmup_update_steps - pre_position_correction_update_steps
                )
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Regularized PIE with position correction for {remaining_update_steps} steps"

                functions_tuple = (
                    self._overlap_projection,
                    self._fourier_projection,
                    self._update_function,
                    self._constraints_function,
                    self._position_correction,
                )
                remaining_update_steps = total_update_steps - warmup_update_steps
                functions_queue += [functions_tuple] * remaining_update_steps
                queue_summary += f"\n--Mixed-State PIE with position correction for {remaining_update_steps} steps"

        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"

        if pure_phase_object_update_steps is not None:
            queue_summary += f"\n--Reconstructed object will be constrained to a pure-phase object for the first {pure_phase_object_update_steps} steps"

        functions_queue = [
            functions_queue[x : x + self._num_diffraction_patterns]
            for x in range(0, total_update_steps, self._num_diffraction_patterns)
        ]

        return functions_queue, queue_summary

    def reconstruct(
        self,
        max_iterations: int = 5,
        return_iterations: bool = False,
        probe_orthogonalization_frequency: int = None,
        warmup_update_steps: int = 0,
        fix_com: bool = True,
        random_seed=None,
        verbose: bool = False,
        parameters: Mapping[str, float] = None,
        functions_queue: Iterable = None,
        **kwargs,
    ):
        """
        Main reconstruction loop method to do the following:
        - Precompute the order of function calls using the MixedStatePtychographicOperator._prepare_functions_queue method
        - Iterate through function calls in queue
        - Pass reconstruction outputs to the MixedStatePtychographicOperator._prepare_measurement_outputs method

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        return_iterations: bool, optional
            If True, method will return a list of current objects, probes, and positions estimates for each iteration
        probe_orthogonalization_frequency: int, optional
            If not None, the probes array will be orthogonalized after this many update steps
        warmup_update_steps: int, optional
            Number of warmup update steps to perform before simultaneous reconstruction begins
        common_probe: bool, optional
            If True, use a common probe for both sets of measurements
        fix_com: bool, optional
            If True, the center of mass of the probes array will be corrected at the end of the iteration
        random_seed
            If not None, used to seed the numpy random number generator
        verbose: bool, optional
            If True, prints functions queue and current iteration error
        functions_queue: (max_iterations, J) Iterable, optional
            If not None, the reconstruction algorithm will use the input functions queue instead
        parameters: dict, optional
            Dictionary specifying any of the abtem.reconstruct.recontruction_symbols parameters
            Additionally, these can also be specified using kwargs

        Returns
        -------
        reconstructed_object_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the objects estimate at each iteration is returned
        reconstructed_probes_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the probes estimate at each iteration is returned
        reconstructed_position_measurement: np.ndarray or Sequence[np.ndarray]
            If return_iterations, a list of position estimates at each iteration is returned
        reconstruction_error: float or Sequence[float]
            If return_iterations, a list of the reconstruction error at each iteration is returned
        """
        for key in kwargs.keys():
            if key not in reconstruction_symbols.keys():
                raise ValueError("{} not a recognized parameter".format(key))

        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)

        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                max_iterations,
                warmup_update_steps=warmup_update_steps,
                pre_position_correction_update_steps=self._reconstruction_parameters[
                    "pre_position_correction_update_steps"
                ],
                pre_probe_correction_update_steps=self._reconstruction_parameters[
                    "pre_probe_correction_update_steps"
                ],
                pure_phase_object_update_steps=self._reconstruction_parameters[
                    "pure_phase_object_update_steps"
                ],
            )
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [
                        [function_tuples] * self._num_diffraction_patterns
                        for function_tuples in functions_queue
                    ]
            elif (
                len(functions_queue) == max_iterations * self._num_diffraction_patterns
            ):
                functions_queue = [
                    functions_queue[x : x + self._num_diffraction_patterns]
                    for x in range(
                        0, total_update_steps, self._num_diffraction_patterns
                    )
                ]
            else:
                raise ValueError()

        self._functions_queue = functions_queue

        ### Main Loop
        xp = get_array_module_from_device(self._device)
        outer_pbar = ProgressBar(total=max_iterations, leave=False)
        inner_pbar = ProgressBar(total=self._num_diffraction_patterns, leave=False)
        indices = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(
            self._experimental_parameters["object_px_padding"]
        )
        center_of_mass = get_scipy_module(xp).ndimage.center_of_mass
        sobel = get_scipy_module(xp).ndimage.sobel

        if return_iterations:
            objects_iterations = []
            probes_iterations = []
            positions_iterations = []
            sse_iterations = []

        if random_seed is not None:
            np.random.seed(random_seed)

        for iteration_index, iteration_step in enumerate(self._functions_queue):

            inner_pbar.reset()

            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse = 0.0

            for update_index, update_step in enumerate(iteration_step):

                index = indices[update_index]
                position = self._positions_px[index]

                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i = (
                    iteration_index * self._num_diffraction_patterns + update_index
                )

                if (
                    self._reconstruction_parameters["pre_probe_correction_update_steps"]
                    is None
                ):
                    fix_probe = False
                else:
                    fix_probe = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pre_probe_correction_update_steps"
                        ]
                    )

                if (
                    self._reconstruction_parameters["pure_phase_object_update_steps"]
                    is None
                ):
                    pure_phase_object = False
                else:
                    pure_phase_object = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pure_phase_object_update_steps"
                        ]
                    )

                if probe_orthogonalization_frequency is None:
                    orthogonalize_probes = False
                else:
                    orthogonalize_probes = not (
                        global_iteration_i % probe_orthogonalization_frequency
                    )

                (
                    _overlap_projection,
                    _fourier_projection,
                    _update_function,
                    _constraints_function,
                    _position_correction,
                ) = update_step

                self._probes, exit_wave = _overlap_projection(
                    self._objects, self._probes, position, old_position, xp=xp
                )

                modified_exit_wave, self._sse = _fourier_projection(
                    exit_wave, diffraction_pattern, self._sse, xp=xp
                )

                (
                    self._objects,
                    self._probes,
                    self._positions_px[index],
                ) = _update_function(
                    self._objects,
                    self._probes,
                    position,
                    exit_wave,
                    modified_exit_wave,
                    diffraction_pattern,
                    fix_probe=fix_probe,
                    orthogonalize_probes=orthogonalize_probes,
                    position_correction=_position_correction,
                    sobel=sobel,
                    reconstruction_parameters=self._reconstruction_parameters,
                    xp=xp,
                )

                self._objects, self._probes = _constraints_function(
                    self._objects, self._probes, pure_phase_object, xp=xp
                )

                old_position = position
                inner_pbar.update(1)

            # Shift probe back to origin
            self._probes = fft_shift(self._probes, xp.round(position) - position)

            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(
                    self._probes, center_of_mass, xp=xp
                )

            # Probe Orthogonalization
            if probe_orthogonalization_frequency is not None:
                self._probes = _orthogonalize(
                    self._probes.reshape((self._num_probes, -1))
                ).reshape(self._probes.shape)

            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (
                    xp.mean(self._positions_px, axis=0) - self._positions_px_com
                )
                self._reconstruction_parameters[
                    "position_step_size"
                ] *= self._reconstruction_parameters["step_size_damping_rate"]

            # Update Parameters
            self._reconstruction_parameters[
                "object_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._reconstruction_parameters[
                "probe_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._sse /= self._num_diffraction_patterns

            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(
                    self._positions_px.copy() * xp.array(self.sampling)
                )
                sse_iterations.append(self._sse)

            if verbose:
                print(
                    f"----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}"
                )

            outer_pbar.update(1)

        inner_pbar.close()
        outer_pbar.close()

        #  Return Results
        if return_iterations:
            results = map(
                self._prepare_measurement_outputs,
                objects_iterations,
                probes_iterations,
                positions_iterations,
                sse_iterations,
            )

            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(
                self._objects,
                self._probes,
                self._positions_px * xp.array(self.sampling),
                self._sse,
            )
            return results

    def _prepare_measurement_outputs(
        self,
        objects: np.ndarray,
        probes: np.ndarray,
        positions: np.ndarray,
        sse: np.ndarray,
    ):
        """
        Method to format the reconstruction outputs as Measurement objects.

        Parameters
        ----------
        objects: np.ndarray
            Reconstructed objects array
        probes: np.ndarray
            Reconstructed probes array
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error

        Returns
        -------
        objects_measurement: Measurement
            Reconstructed objects Measurement
        probes_measurement: Measurement
            Reconstructed probes Measurement
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error
        """

        calibrations = tuple(
            Calibration(0, s, units="Å", name=n, endpoint=False)
            for s, n in zip(self.sampling, ("x", "y"))
        )

        measurement_objects = Measurement(asnumpy(objects), calibrations)
        measurement_probes = [
            Measurement(asnumpy(probe), calibrations) for probe in probes
        ]

        return measurement_objects, measurement_probes, asnumpy(positions), sse


class MultislicePtychographicOperator(AbstractPtychographicOperator):
    """
    Multislice Ptychographic Iterative Engine (MS-PIE).
    Used to reconstruct _thick_ weak-phase objects using a set of measured far-field CBED patterns with the following array dimensions:

    CBED pattern dimensions     : (J,M,N)
    objects dimensions          : (T,P,Q)
    probes dimensions           : (T,R,S)

    Parameters
    ----------

    diffraction_patterns: np.ndarray or Measurement
        Input 3D or 4D CBED pattern intensities with dimensions (M,N)
    energy: float
        Electron energy [eV]
    num_slices: int
        Number of slices to use
    slice_thicknesses: float or Sequence[float]
        Slice thicknesses. If a float, all slices are assigned the same thickness
    region_of_interest_shape: (2,) Sequence[int], optional
        Pixel dimensions (R,S) of the region of interest (ROI)
        If None, the ROI dimensions are taken as the CBED dimensions (M,N)
    objects: np.ndarray, optional
        Initial objects guess with dimensions (P,Q) - Useful for restarting reconstructions
        If None, an array with 1.0j is initialized
    probes: np.ndarray or Probe, optional
        Initial probes guess with dimensions/gpts (R,S) - Useful for restarting reconstructions
        If None, a Probe with CTF given by the polar_parameters dictionary is initialized
    positions: np.ndarray, optional
        Initial positions guess [Å]
        If None, a raster scan with step sizes given by the experimental_parameters dictionary is initialized
    semiangle_cutoff: float, optional
        Semiangle cutoff for the initial Probe guess
    preprocess: bool, optional
        If True, it runs the preprocess method after initialization
    device: str, optional
        Device to perform Fourier-based reconstructrions - Either 'cpu' or 'gpu'
    parameters: dict, optional
       Dictionary specifying any of the abtem.transfer.polar_symbols or abtem.reconstruct.experimental_symbols parameters
       Additionally, these can also be specified using kwargs
    """

    def __init__(
        self,
        diffraction_patterns: Union[np.ndarray, Measurement],
        energy: float,
        num_slices: int,
        slice_thicknesses: Union[float, Sequence[float]],
        region_of_interest_shape: Sequence[int] = None,
        objects: np.ndarray = None,
        probes: Union[np.ndarray, Probe] = None,
        positions: np.ndarray = None,
        semiangle_cutoff: float = None,
        preprocess: bool = False,
        device: str = "cpu",
        parameters: Mapping[str, float] = None,
        **kwargs,
    ):

        for key in kwargs.keys():
            if (
                (key not in polar_symbols)
                and (key not in polar_aliases.keys())
                and (key not in experimental_symbols)
            ):
                raise ValueError("{} not a recognized parameter".format(key))

        self._polar_parameters = dict(zip(polar_symbols, [0.0] * len(polar_symbols)))
        self._experimental_parameters = dict(
            zip(experimental_symbols, [None] * len(experimental_symbols))
        )

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)
        self._polar_parameters, self._experimental_parameters = self._update_parameters(
            parameters, self._polar_parameters, self._experimental_parameters
        )

        slice_thicknesses = np.array(slice_thicknesses)
        if slice_thicknesses.shape == ():
            slice_thicknesses = np.tile(slice_thicknesses, num_slices)

        self._region_of_interest_shape = region_of_interest_shape
        self._energy = energy
        self._semiangle_cutoff = semiangle_cutoff
        self._positions = positions
        self._device = device
        self._objects = objects
        self._probes = probes
        self._num_slices = num_slices
        self._slice_thicknesses = slice_thicknesses
        self._diffraction_patterns = diffraction_patterns

        if preprocess:
            self.preprocess()
        else:
            self._preprocessed = False

    def preprocess(self):
        """
        Preprocess method to do the following:
        - Pads CBED patterns to region of interest dimensions
        - Prepares initial guess for scanning positions
        - Prepares initial guesses for the objects and probes arrays


        Returns
        -------
        preprocessed_ptychographic_operator: MultislicePtychographicOperator
        """

        self._preprocessed = True

        # Convert Measurement Objects
        if isinstance(self._diffraction_patterns, Measurement):
            (
                self._diffraction_patterns,
                angular_sampling,
                step_sizes,
            ) = self._extract_calibrations_from_measurement_object(
                self._diffraction_patterns, self._energy
            )
            self._experimental_parameters["angular_sampling"] = angular_sampling
            if step_sizes is not None:
                self._experimental_parameters["scan_step_sizes"] = step_sizes

        # Preprocess Diffraction Patterns
        xp = get_array_module_from_device(self._device)
        self._diffraction_patterns = copy_to_device(
            self._diffraction_patterns, self._device
        )

        if len(self._diffraction_patterns.shape) == 4:
            self._experimental_parameters[
                "grid_scan_shape"
            ] = self._diffraction_patterns.shape[:2]
            self._diffraction_patterns = self._diffraction_patterns.reshape(
                (-1,) + self._diffraction_patterns.shape[-2:]
            )

        if self._region_of_interest_shape is None:
            self._region_of_interest_shape = self._diffraction_patterns.shape[-2:]

        self._diffraction_patterns = self._pad_diffraction_patterns(
            self._diffraction_patterns, self._region_of_interest_shape
        )
        self._num_diffraction_patterns = self._diffraction_patterns.shape[0]

        if self._experimental_parameters["background_counts_cutoff"] is not None:
            self._diffraction_patterns[
                self._diffraction_patterns
                < self._experimental_parameters["background_counts_cutoff"]
            ] = 0.0

        if self._experimental_parameters["counts_scaling_factor"] is not None:
            self._diffraction_patterns /= self._experimental_parameters[
                "counts_scaling_factor"
            ]

        self._mean_diffraction_intensity = (
            xp.sum(self._diffraction_patterns) / self._num_diffraction_patterns
        )
        self._diffraction_patterns = xp.fft.ifftshift(
            xp.sqrt(self._diffraction_patterns), axes=(-2, -1)
        )

        # Scan Positions Initialization
        (
            positions_px,
            self._experimental_parameters,
        ) = self._calculate_scan_positions_in_pixels(
            self._positions,
            self.sampling,
            self._region_of_interest_shape,
            self._experimental_parameters,
        )

        # Objects Initialization
        if self._objects is None:
            pad_x, pad_y = self._experimental_parameters["object_px_padding"]
            p, q = np.max(positions_px, axis=0)
            p = np.max([np.round(p + pad_x), self._region_of_interest_shape[0]]).astype(
                int
            )
            q = np.max([np.round(q + pad_y), self._region_of_interest_shape[1]]).astype(
                int
            )
            self._objects = xp.ones((self._num_slices, p, q), dtype=xp.complex64)
        else:
            self._objects = copy_to_device(self._objects, self._device)

        self._positions_px = copy_to_device(positions_px, self._device)
        self._positions_px_com = xp.mean(self._positions_px, axis=0)

        # Probes Initialization
        if self._probes is None:
            ctf = CTF(
                energy=self._energy,
                semiangle_cutoff=self._semiangle_cutoff,
                parameters=self._polar_parameters,
            )
            _probes = (
                Probe(
                    semiangle_cutoff=self._semiangle_cutoff,
                    energy=self._energy,
                    gpts=self._region_of_interest_shape,
                    sampling=self.sampling,
                    ctf=ctf,
                    device=self._device,
                )
                .build()
                .array
            )

        else:
            if isinstance(self._probes, Probe):
                if self._probes.gpts != self._region_of_interest_shape:
                    raise ValueError()
                _probes = copy_to_device(self._probes.build().array, self._device)
            else:
                _probes = copy_to_device(self._probes, self._device)

        probe_intensity = xp.sum(xp.abs(xp.fft.fft2(_probes)) ** 2)
        _probes *= np.sqrt(self._mean_diffraction_intensity / probe_intensity)

        self._probes = xp.zeros((self._num_slices,) + _probes.shape, dtype=xp.complex64)
        self._probes[0] = _probes

        return self

    @staticmethod
    def _overlap_projection(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        old_position: np.ndarray,
        propagator: FresnelPropagator = None,
        slice_thicknesses: Sequence[float] = None,
        sampling: Sequence[float] = None,
        wavelength: float = None,
        fft2_convolve: Callable = None,
        xp=np,
        **kwargs,
    ):
        """
        Multislice-PIE overlap projection static method:
        .. math::
            \psi^n_{R_j}(r) = O^n_{R_j}(r) * P^n(r)


        Parameters
        ----------
        objects: np.ndarray
            Object array to be illuminated
        probes: np.ndarray
            Probe window array to illuminate object with
        position: np.ndarray
            Center position of probe window
        old_position: np.ndarray
            Old center position of probe window
            Used for fractionally shifting probe sequentially
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Fractionally shifted probe window array
        exit_wave: np.ndarray
            Overlap projection of illuminated probe
        """

        fractional_position = position - xp.round(position)
        old_fractional_position = old_position - xp.round(old_position)

        probes[0] = fft_shift(probes[0], fractional_position - old_fractional_position)
        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape[-2:]
        )
        exit_waves = xp.empty_like(probes)

        # Removed antialiasing - didn't seem to add much, and more consistent w/o modifying self._objects here
        # objects                  = antialias_filter._bandlimit(objects)

        num_slices = slice_thicknesses.shape[0]
        for s in range(num_slices):
            exit_waves[s] = objects[s][object_indices] * probes[s]
            if s + 1 < num_slices:
                probes[s + 1] = _propagate_array(
                    propagator,
                    exit_waves[s],
                    sampling,
                    wavelength,
                    slice_thicknesses[s],
                    fft2_convolve=fft2_convolve,
                    overwrite=False,
                    xp=xp,
                )

        return probes, exit_waves

    @staticmethod
    def _fourier_projection(
        exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        sse: float,
        xp=np,
        **kwargs,
    ):
        """
        Multislice-PIE fourier projection static method:
        .. math::
            \psi^{N'}_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^N_{R_j}(u)] / |F[\psi^N_{R_j}(u)]|]


        Parameters
        ----------
        exit_waves: np.ndarray
            Exit waves array given by MultislicePtychographicOperator._overlap_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sse: float
            Current sum of squares error estimate
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        modified_exit_wave: np.ndarray
            Fourier projection of illuminated probe
        sse: float
            Updated sum of squares error estimate
        """
        """
        MS-PIE Fourier-amplitude modification projection:
        \psi'^N_{R_j}(r) = F^{-1}[\sqrt{I_j(u)} F[\psi^N_{R_j}(u)] / |F[\psi^N_{R_j}(u)]|]
        """

        modified_exit_waves = xp.empty_like(exit_waves)
        exit_wave_fft = xp.fft.fft2(exit_waves[-1])
        sse += xp.mean(
            xp.abs(xp.abs(exit_wave_fft) - diffraction_patterns) ** 2
        ) / xp.sum(diffraction_patterns**2)
        modified_exit_waves[-1] = xp.fft.ifft2(
            diffraction_patterns * xp.exp(1j * xp.angle(exit_wave_fft))
        )

        return modified_exit_waves, sse

    @staticmethod
    def _update_function(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_waves: np.ndarray,
        modified_exit_waves: np.ndarray,
        diffraction_patterns: np.ndarray,
        fix_probe: bool = False,
        position_correction: Callable = None,
        sobel: Callable = None,
        reconstruction_parameters: Mapping[str, float] = None,
        propagator: FresnelPropagator = None,
        slice_thicknesses: Sequence[float] = None,
        sampling: Sequence[float] = None,
        wavelength: float = None,
        fft2_convolve: Callable = None,
        xp=np,
        **kwargs,
    ):
        """
        Multislice-PIE objects and probes update static method:

        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_waves: np.ndarray
            Exit waves array given by MultislicePtychographicOperator._overlap_projection method
        modified_exit_waves: np.ndarray
            Modified exit waves array given by MultislicePtychographicOperator._fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        fix_probe: bool, optional
            If True, the probe will not be updated by the algorithm. Default is False
        position_correction: Callable, optional
            If not None, the function used to update the current probe position
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        reconstruction_parameters: dict, optional
            Dictionary with common reconstruction parameters
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Updated objects array estimate
        probes: np.ndarray
            Updated probes array estimate
        position: np.ndarray
            Updated probe position estimate
        """

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape[-2:]
        )
        num_slices = slice_thicknesses.shape[0]

        if position_correction is not None:
            position = position_correction(
                objects,
                probes,
                position,
                exit_waves,
                modified_exit_waves,
                diffraction_patterns,
                sobel=sobel,
                position_step_size=position_step_size,
                xp=xp,
            )

        for s in reversed(range(num_slices)):
            exit_wave = exit_waves[s]
            modified_exit_wave = modified_exit_waves[s]
            exit_wave_diff = modified_exit_wave - exit_wave

            probe_conj = xp.conj(probes[s])
            probe_abs_squared = xp.abs(probes[s]) ** 2
            obj_conj = xp.conj(objects[s][object_indices])
            obj_abs_squared = xp.abs(objects[s][object_indices]) ** 2

            alpha = reconstruction_parameters["alpha"]
            object_step_size = reconstruction_parameters["object_step_size"]
            objects[s][object_indices] += (
                object_step_size
                * probe_conj
                * exit_wave_diff
                / ((1 - alpha) * probe_abs_squared + alpha * xp.max(probe_abs_squared))
            )

            if not fix_probe or s > 0:
                beta = reconstruction_parameters["beta"]
                probe_step_size = reconstruction_parameters["probe_step_size"]
                probes[s] += (
                    probe_step_size
                    * obj_conj
                    * exit_wave_diff
                    / ((1 - beta) * obj_abs_squared + beta * xp.max(obj_abs_squared))
                )

            if s > 0:
                modified_exit_waves[s - 1] = _propagate_array(
                    propagator,
                    probes[s],
                    sampling,
                    wavelength,
                    -slice_thicknesses[s - 1],
                    fft2_convolve=fft2_convolve,
                    overwrite=False,
                    xp=xp,
                )

        return objects, probes, position

    @staticmethod
    def _constraints_function(
        objects: np.ndarray,
        probes: np.ndarray,
        pure_phase_object: bool,
        xp=np,
        **kwargs,
    ):
        """
        Multislice-PIE constraints static method:

        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        pure_phase_object:bool
            If True, constraints object to being a pure phase object, i.e. with unit amplitude
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        objects: np.ndarray
            Constrained objects array
        probes: np.ndarray
            Constrained probes array
        """
        phase = xp.exp(1.0j * xp.angle(objects))
        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(objects), 1.0)
        return amplitude * phase, probes

    @staticmethod
    def _position_correction(
        objects: np.ndarray,
        probes: np.ndarray,
        position: np.ndarray,
        exit_wave: np.ndarray,
        modified_exit_wave: np.ndarray,
        diffraction_pattern: np.ndarray,
        sobel: Callable,
        position_step_size: float = 1.0,
        xp=np,
        **kwargs,
    ):
        """
        Multislice-PIE probe position correction method using the last slice.


        Parameters
        ----------
        objects: np.ndarray
            Current objects array estimate
        probes: np.ndarray
            Current probes array estimate
        position: np.ndarray
            Current probe position estimate
        exit_wave: np.ndarray
            Exit wave array given by MultislicePtychographicOperator._overlap_projection method
        modified_exit_wave: np.ndarray
            Modified exit wave array given by MultislicePtychographicOperator._fourier_projection method
        diffraction_patterns: np.ndarray
            Square-root of CBED intensities array used to modify exit_waves amplitude
        sobel: Callable, optional
            The scipy.ndimage module used to compute the object gradients. Passed to the position correction function
        position_step_size: float, optional
            Gradient step size for position update step
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        position: np.ndarray
            Updated probe position estimate
        """

        object_dx = sobel(objects[-1], axis=0, mode="wrap")
        object_dy = sobel(objects[-1], axis=1, mode="wrap")

        object_indices = _wrapped_indices_2D_window(
            position, probes.shape[-2:], objects.shape[-2:]
        )
        exit_wave_dx = object_dx[object_indices] * probes[-1]
        exit_wave_dy = object_dy[object_indices] * probes[-1]

        exit_wave_diff = modified_exit_wave[-1] - exit_wave[-1]
        displacement_x = xp.sum(
            xp.real(xp.conj(exit_wave_dx) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dx) ** 2)
        displacement_y = xp.sum(
            xp.real(xp.conj(exit_wave_dy) * exit_wave_diff)
        ) / xp.sum(xp.abs(exit_wave_dy) ** 2)

        return position + position_step_size * xp.array(
            [displacement_x, displacement_y]
        )

    @staticmethod
    def _fix_probe_center_of_mass(
        probes: np.ndarray, center_of_mass: Callable, xp=np, **kwargs
    ):
        """
        Multislice-PIE probe center correction method.


        Parameters
        ----------
        probes: np.ndarray
            Current probes array estimate
        center_of_mass: Callable
            The scipy.ndimage module used to compute the array center of mass
        xp
            Numerical programming module to use - either np or cp

        Returns
        -------
        probes: np.ndarray
            Center-of-mass corrected probes array
        """

        probe_center = xp.array(probes.shape[-2:]) / 2
        com = center_of_mass(xp.abs(probes[0]) ** 2)
        probes[0] = fft_shift(probes[0], probe_center - xp.array(com))

        return probes

    def _prepare_functions_queue(
        self,
        max_iterations: int,
        pre_position_correction_update_steps: int = None,
        pre_probe_correction_update_steps: int = None,
        pure_phase_object_update_steps: int = None,
        **kwargs,
    ):
        """
        Precomputes the order in which functions will be called in the reconstruction loop.
        Additionally, prepares a summary of steps to be printed for reporting.

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        pre_position_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling position correction
        pre_probe_correction_update_steps: int, optional
            Number of update steps (not iterations) to perform before enabling probe correction

        Returns
        -------
        functions_queue: (max_iterations,J) list
            List of function calls
        queue_summary: str
            Summary of function calls the reconstruction loop will perform
        """
        total_update_steps = max_iterations * self._num_diffraction_patterns
        queue_summary = "Ptychographic reconstruction will perform the following steps:"

        functions_tuple = (
            self._overlap_projection,
            self._fourier_projection,
            self._update_function,
            self._constraints_function,
            None,
        )
        functions_queue = [functions_tuple]
        if pre_position_correction_update_steps is None:
            functions_queue *= total_update_steps
            queue_summary += f"\n--Multislice PIE for {total_update_steps} steps"
        else:
            functions_queue *= pre_position_correction_update_steps
            queue_summary += (
                f"\n--Multislice PIE for {pre_position_correction_update_steps} steps"
            )

            functions_tuple = (
                self._overlap_projection,
                self._fourier_projection,
                self._update_function,
                self._constraints_function,
                self._position_correction,
            )

            remaining_update_steps = (
                total_update_steps - pre_position_correction_update_steps
            )
            functions_queue += [functions_tuple] * remaining_update_steps
            queue_summary += f"\n--Multislice PIE with position correction for {remaining_update_steps} steps"

        if pre_probe_correction_update_steps is None:
            queue_summary += f"\n--Probe correction is enabled"
        elif pre_probe_correction_update_steps > total_update_steps:
            queue_summary += f"\n--Probe correction is disabled"
        else:
            queue_summary += f"\n--Probe correction will be enabled after the first {pre_probe_correction_update_steps} steps"

        if pure_phase_object_update_steps is not None:
            queue_summary += f"\n--Reconstructed object will be constrained to a pure-phase object for the first {pure_phase_object_update_steps} steps"

        functions_queue = [
            functions_queue[x : x + self._num_diffraction_patterns]
            for x in range(0, total_update_steps, self._num_diffraction_patterns)
        ]

        return functions_queue, queue_summary

    def reconstruct(
        self,
        max_iterations: int = 5,
        return_iterations: bool = False,
        fix_com: bool = True,
        random_seed=None,
        verbose: bool = False,
        parameters: Mapping[str, float] = None,
        measurement_output_view: str = "padded",
        functions_queue: Iterable = None,
        **kwargs,
    ):
        """
        Main reconstruction loop method to do the following:
        - Precompute the order of function calls using the MultislicePtychographicOperator._prepare_functions_queue method
        - Iterate through function calls in queue
        - Pass reconstruction outputs to the MultislicePtychographicOperator._prepare_measurement_outputs method

        Parameters
        ----------
        max_iterations: int
            Maximum number of iterations to run reconstruction algorithm
        return_iterations: bool, optional
            If True, method will return a list of current objects, probes, and positions estimates for each iteration
        fix_com: bool, optional
            If True, the center of mass of the probes array will be corrected at the end of the iteration
        random_seed
            If not None, used to seed the numpy random number generator
        verbose: bool, optional
            If True, prints functions queue and current iteration error
        functions_queue: (max_iterations, J) Iterable, optional
            If not None, the reconstruction algorithm will use the input functions queue instead
        parameters: dict, optional
            Dictionary specifying any of the abtem.reconstruct.recontruction_symbols parameters
            Additionally, these can also be specified using kwargs

        Returns
        -------
        reconstructed_object_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the objects estimate at each iteration is returned
        reconstructed_probes_measurement: Measurement or Sequence[Measurement]
            If return_iterations, a list of Measurements for the probes estimate at each iteration is returned
        reconstructed_position_measurement: np.ndarray or Sequence[np.ndarray]
            If return_iterations, a list of position estimates at each iteration is returned
        reconstruction_error: float or Sequence[float]
            If return_iterations, a list of the reconstruction error at each iteration is returned
        """
        for key in kwargs.keys():
            if key not in reconstruction_symbols.keys():
                raise ValueError("{} not a recognized parameter".format(key))

        if parameters is None:
            parameters = {}
        self._reconstruction_parameters = reconstruction_symbols.copy()
        self._reconstruction_parameters.update(parameters)
        self._reconstruction_parameters.update(kwargs)

        if functions_queue is None:
            functions_queue, summary = self._prepare_functions_queue(
                max_iterations,
                pre_position_correction_update_steps=self._reconstruction_parameters[
                    "pre_position_correction_update_steps"
                ],
                pre_probe_correction_update_steps=self._reconstruction_parameters[
                    "pre_probe_correction_update_steps"
                ],
                pure_phase_object_update_steps=self._reconstruction_parameters[
                    "pure_phase_object_update_steps"
                ],
            )
            if verbose:
                print(summary)
        else:
            if len(functions_queue) == max_iterations:
                if callable(functions_queue[0]):
                    functions_queue = [
                        [function_tuples] * self._num_diffraction_patterns
                        for function_tuples in functions_queue
                    ]
            elif (
                len(functions_queue) == max_iterations * self._num_diffraction_patterns
            ):
                functions_queue = [
                    functions_queue[x : x + self._num_diffraction_patterns]
                    for x in range(
                        0, total_update_steps, self._num_diffraction_patterns
                    )
                ]
            else:
                raise ValueError()

        self._functions_queue = functions_queue

        ### Main Loop
        xp = get_array_module_from_device(self._device)
        outer_pbar = ProgressBar(total=max_iterations, leave=False)
        inner_pbar = ProgressBar(total=self._num_diffraction_patterns, leave=False)
        indices = np.arange(self._num_diffraction_patterns)
        position_px_padding = xp.array(
            self._experimental_parameters["object_px_padding"]
        )
        center_of_mass = get_scipy_module(xp).ndimage.center_of_mass
        sobel = get_scipy_module(xp).ndimage.sobel
        fft2_convolve = get_device_function(xp, "fft2_convolve")
        propagator = FresnelPropagator()
        wavelength = energy2wavelength(self._energy)

        if return_iterations:
            objects_iterations = []
            probes_iterations = []
            positions_iterations = []
            sse_iterations = []

        if random_seed is not None:
            np.random.seed(random_seed)

        for iteration_index, iteration_step in enumerate(self._functions_queue):

            inner_pbar.reset()

            # Set iteration-specific parameters
            np.random.shuffle(indices)
            old_position = position_px_padding
            self._sse = 0.0

            for update_index, update_step in enumerate(iteration_step):

                index = indices[update_index]
                position = self._positions_px[index]

                # Skip empty diffraction patterns
                diffraction_pattern = self._diffraction_patterns[index]
                if xp.sum(diffraction_pattern) == 0.0:
                    inner_pbar.update(1)
                    continue

                # Set update-specific parameters
                global_iteration_i = (
                    iteration_index * self._num_diffraction_patterns + update_index
                )

                if (
                    self._reconstruction_parameters["pre_probe_correction_update_steps"]
                    is None
                ):
                    fix_probe = False
                else:
                    fix_probe = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pre_probe_correction_update_steps"
                        ]
                    )

                if (
                    self._reconstruction_parameters["pure_phase_object_update_steps"]
                    is None
                ):
                    pure_phase_object = False
                else:
                    pure_phase_object = (
                        global_iteration_i
                        < self._reconstruction_parameters[
                            "pure_phase_object_update_steps"
                        ]
                    )

                (
                    _overlap_projection,
                    _fourier_projection,
                    _update_function,
                    _constraints_function,
                    _position_correction,
                ) = update_step

                self._probes, exit_wave = _overlap_projection(
                    self._objects,
                    self._probes,
                    position,
                    old_position,
                    propagator=propagator,
                    slice_thicknesses=self._slice_thicknesses,
                    sampling=self.sampling,
                    wavelength=wavelength,
                    fft2_convolve=fft2_convolve,
                    xp=xp,
                )

                modified_exit_wave, self._sse = _fourier_projection(
                    exit_wave, diffraction_pattern, self._sse, xp=xp
                )

                (
                    self._objects,
                    self._probes,
                    self._positions_px[index],
                ) = _update_function(
                    self._objects,
                    self._probes,
                    position,
                    exit_wave,
                    modified_exit_wave,
                    diffraction_pattern,
                    fix_probe=fix_probe,
                    position_correction=_position_correction,
                    sobel=sobel,
                    reconstruction_parameters=self._reconstruction_parameters,
                    propagator=propagator,
                    slice_thicknesses=self._slice_thicknesses,
                    sampling=self.sampling,
                    wavelength=wavelength,
                    fft2_convolve=fft2_convolve,
                    xp=xp,
                )

                self._objects, self._probes = _constraints_function(
                    self._objects, self._probes, pure_phase_object, xp=xp
                )

                old_position = position
                inner_pbar.update(1)

            # Shift probe back to origin
            self._probes = fft_shift(self._probes, xp.round(position) - position)

            # Probe CoM
            if fix_com:
                self._probes = self._fix_probe_center_of_mass(
                    self._probes, center_of_mass, xp=xp
                )

            # Positions CoM
            if _position_correction is not None:
                self._positions_px -= (
                    xp.mean(self._positions_px, axis=0) - self._positions_px_com
                )
                self._reconstruction_parameters[
                    "position_step_size"
                ] *= self._reconstruction_parameters["step_size_damping_rate"]

            # Update Parameters
            self._reconstruction_parameters[
                "object_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._reconstruction_parameters[
                "probe_step_size"
            ] *= self._reconstruction_parameters["step_size_damping_rate"]
            self._sse /= self._num_diffraction_patterns

            if return_iterations:
                objects_iterations.append(self._objects.copy())
                probes_iterations.append(self._probes.copy())
                positions_iterations.append(
                    self._positions_px.copy() * xp.array(self.sampling)
                )
                sse_iterations.append(self._sse)

            if verbose:
                print(
                    f"----Iteration {iteration_index:<{len(str(max_iterations))}}, SSE = {float(self._sse):.3e}"
                )

            outer_pbar.update(1)

        inner_pbar.close()
        outer_pbar.close()

        #  Return Results
        if return_iterations:
            mapfunc = partial(
                self._prepare_measurement_outputs,
                slice_thicknesses=self._slice_thicknesses,
            )
            results = map(
                mapfunc,
                objects_iterations,
                probes_iterations,
                positions_iterations,
                sse_iterations,
            )

            return tuple(map(list, zip(*results)))
        else:
            results = self._prepare_measurement_outputs(
                self._objects,
                self._probes,
                self._positions_px * xp.array(self.sampling),
                self._sse,
                slice_thicknesses=self._slice_thicknesses,
            )
            return results

    def _prepare_measurement_outputs(
        self,
        objects: np.ndarray,
        probes: np.ndarray,
        positions: np.ndarray,
        sse: np.ndarray,
        slice_thicknesses: Sequence[float] = None,
    ):
        """
        Method to format the reconstruction outputs as Measurement objects.

        Parameters
        ----------
        objects: np.ndarray
            Reconstructed objects array
        probes: np.ndarray
            Reconstructed probes array
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error
        slice_thicknesses: Sequence[float]
            Slice thicknesses

        Returns
        -------
        objects_measurement: Measurement
            Reconstructed objects Measurement
        probes_measurement: Measurement
            Reconstructed probes Measurement
        positions: np.ndarray
            Reconstructed positions array
        sse: float
            Reconstruction error
        """

        calibrations = tuple(
            Calibration(0, s, units="Å", name=n, endpoint=False)
            for s, n in zip(self.sampling, ("x", "y"))
        )
        calibrations = (
            Calibration(0, slice_thicknesses[0], units="Å", name="z", endpoint=False),
        ) + calibrations

        measurement_objects = Measurement(asnumpy(objects), calibrations)
        measurement_probes = Measurement(asnumpy(probes), calibrations)

        return measurement_objects, measurement_probes, asnumpy(positions), sse
