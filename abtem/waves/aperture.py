from abtem.core.backend import get_array_module
from abtem.waves.transfer import AbstractAperture
from typing import Union
import numpy as np

class Spokes(AbstractAperture):
    '''
	Amplitude plate with wedges of the beam blocked
	
	Parameters
    ----------
		spoke_num (float)		: number of spokes (int)
		spoke_width (float)		: width of spokes (radians)
		semiangle_cutoff (float): probe semiangle (mrad)
	
	Returns
	----------
		aperture: AbstractAperture

	'''
    def __init__(
    	self, 
    	spoke_num: float, 
    	spoke_width: float, 
    	semiangle_cutoff: float = None, 
    	energy: float = None
    ):
        self._spoke_num = spoke_num
        self._spoke_width = spoke_width

        super().__init__(energy=energy, semiangle_cutoff=semiangle_cutoff)

    @property
    def spoke_num(self):
        return self._spoke_num

    @property
    def spoke_width(self):
        return self._spoke_width

    @property
    def metadata(self):
        metadata = {}
        return metadata

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)
        
        semiangle_cutoff = self.semiangle_cutoff / 1e3
        
        array = alpha < semiangle_cutoff
        array = array * (((phi + self.spoke_width / 2) * self.spoke_num) % (2 * np.pi) > (
                self.spoke_width * self.spoke_num))
        
        return array

class Bullseye(AbstractAperture):
    '''
    vortex beam
    Parameters
    ----------
        spoke_num (float)       : number of spokes (int)
        spoke_width (float)     : width of spokes (radians)
        ring_num (float)        : number of rings (int)
        ring_width (float)      : width of rings (mrad)
        semiangle_cutoff (float): probe semiangle (mrad)
    
    Returns
    ----------
        aperture: AbstractAperture

    '''
    def __init__(
        self, 
        spoke_num: float, 
        spoke_width: float, 
        ring_num: float, 
        ring_width: float, 
        semiangle_cutoff: float, 
        energy: float = None
    ):
        self._spoke_num = spoke_num
        self._spoke_width = spoke_width
        self._ring_num = ring_num
        self._ring_width = ring_width

        super().__init__(energy=energy, semiangle_cutoff=semiangle_cutoff)

    @property
    def spoke_num(self):
        return self._spoke_num

    @property
    def spoke_width(self):
        return self._spoke_width

    @property
    def ring_num(self):
        return self._ring_num

    @property
    def ring_width(self):
        return self._ring_width

    @property
    def metadata(self):
        metadata = {}
        return metadata

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)
        
        semiangle_cutoff = self.semiangle_cutoff / 1e3

        array = alpha < semiangle_cutoff

        #add cross bars
        array = array * (((phi + self.spoke_width / 2) * self.spoke_num) % (2 * np.pi) > (
                self.spoke_width * self.spoke_num))

        #add ring bars
        end_edges = np.linspace(semiangle_cutoff/self.ring_num, semiangle_cutoff, self.ring_num)
        start_edges = end_edges - self.ring_width/1e3
        
        for start_edge, end_edge in zip(start_edges, end_edges):
                array[(alpha > start_edge) * (alpha < end_edge)] = 0.

        return array

class Vortex(AbstractAperture):
    '''
	vortex beam
	Parameters
    ----------
		m (float)				: quantum number of vortex beam
		semiangle_cutoff (float): max angle of probe (mrad)
	
	Returns
	----------
		aperture: AbstractAperture

	'''
    def __init__(
    	self, 
    	m: float, 
    	semiangle_cutoff: float, 
    	energy: float = None
    ):
        self._m = m
        super().__init__(energy=energy, semiangle_cutoff=semiangle_cutoff)

    @property
    def m(self):
        return self._m

    @property
    def metadata(self):
        metadata = {}
        return metadata

    def evaluate_with_alpha_and_phi(self, alpha: Union[float, np.ndarray], phi) -> Union[float, np.ndarray]:
        xp = get_array_module(alpha)
        alpha = xp.array(alpha)
        
        semiangle_cutoff = self.semiangle_cutoff / 1e3

        array = alpha < semiangle_cutoff
        array = array * np.exp(1j*phi*self.m)
        
        return array