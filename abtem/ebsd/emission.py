"""Where backscattered electrons are generated: at the atoms.

Backscattering is scattering through more than ninety degrees, a momentum
transfer near ``q = 2 / λ`` -- far inside the screening radius, so the nucleus
sets it and the cross-section is the electron scattering factor there,
squared. Taken from abTEM's parametrization, that is Rutherford's ``Z²`` to
within 2% from carbon to gold. EMsoft builds its master patterns from the same
source (``CalcSgh``: ``Z²`` times the site occupation, times a Debye-Waller
factor).

The atoms are laid on the multislice grid exactly as abTEM lays them for an
infinite projection -- each in the one slice that holds it
(:class:`~abtem.slicing.SliceIndexedAtoms`), as a sub-pixel delta
(:func:`~abtem.integrals.superpose_deltas`) corrected by the same sinc -- here
weighted atom by atom, since each may be lit differently by the incident beam.
The thermal cloud of the emitting atom is abTEM's too: build the specimen from
:class:`~abtem.FrozenPhonons`, and the average over its configurations is the
Debye-Waller smearing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from abtem.antialias import antialias_aperture
from abtem.core.backend import get_array_module
from abtem.core.energy import energy2wavelength
from abtem.core.fft import fft2, ifft2
from abtem.core.utils import get_dtype
from abtem.integrals import sinc, superpose_deltas
from abtem.parametrizations import validate_parametrization
from abtem.slicing import SliceIndexedAtoms

__all__ = ["backscatter_cross_section", "EmissionSlices"]


def backscatter_cross_section(
    numbers: np.ndarray, energy: float, parametrization: str = "lobato"
) -> np.ndarray:
    """Relative cross-section of each atom for scattering straight back.

    The square of the electron scattering factor at ``q = 2 / λ``, where the
    nucleus alone matters -- Rutherford's ``Z²``, to within 2% from carbon to
    gold.

    Parameters
    ----------
    numbers : np.ndarray
        Atomic numbers.
    energy : float
        Electron energy [eV].
    parametrization : str, optional
        abTEM parametrization of the scattering factors (default 'lobato').

    Returns
    -------
    cross_section : np.ndarray
        One value per atom, relative to hydrogen -- so about ``Z²``.
    """
    from ase.data import chemical_symbols

    numbers = np.asarray(numbers)
    parametrization = validate_parametrization(parametrization)
    q2 = np.array((2.0 / energy2wavelength(energy)) ** 2)

    def scattering_factor(number):
        return float(parametrization.scattering_factor(chemical_symbols[number])(q2))

    # Relative to hydrogen, so the values are of order Z^2 rather than 1e-6.
    hydrogen = scattering_factor(1)
    values = {
        number: (scattering_factor(number) / hydrogen) ** 2
        for number in np.unique(numbers)
    }
    return np.array([values[number] for number in numbers])


def configuration_atoms(potential) -> Atoms:
    """The atoms a single-configuration potential is built from.

    Its frozen-phonon configuration, displaced exactly as the potential
    displaces it; :class:`~abtem.slicing.SliceIndexedAtoms` wraps it into the
    cell the same way.
    """
    return potential.frozen_phonons.randomize(potential.get_transformed_atoms())


class EmissionSlices:
    """The emitting atoms of one configuration, on the grid, slice by slice.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms of the configuration.
    slice_thickness : sequence of float
        Thickness of each slice [Å], as the potential slices it.
    gpts : two int
        Grid points of the potential.
    extent : two float
        Lateral size of the potential [Å].
    energy : float
        Electron energy [eV], which sets the cross-sections.
    illumination : np.ndarray, optional
        How brightly the incident beam lights each atom, of shape ``(atoms,)``
        or ``(sources, atoms)`` for several illuminations at once. Defaults to
        one for every atom.
    device : str, optional
        Device to build the slices on.
    """

    def __init__(
        self,
        atoms: Atoms,
        slice_thickness,
        gpts: tuple[int, int],
        extent: tuple[float, float],
        energy: float,
        illumination: Optional[np.ndarray] = None,
        device: str = "cpu",
    ):
        emission = backscatter_cross_section(atoms.numbers, energy)

        if illumination is None:
            emission = emission[None]
        else:
            illumination = np.atleast_2d(np.asarray(illumination, dtype=float))
            if illumination.shape[-1] != len(atoms):
                raise ValueError(
                    f"illumination has {illumination.shape[-1]} values per source "
                    f"but there are {len(atoms)} atoms"
                )
            emission = emission[None] * illumination

        atoms = atoms.copy()
        atoms.set_array("emission", emission.T.copy())

        self._sliced = SliceIndexedAtoms(atoms, slice_thickness)
        self._n_sources = len(emission)
        self._gpts = (int(gpts[0]), int(gpts[1]))
        self._sampling = (extent[0] / self._gpts[0], extent[1] / self._gpts[1])
        self._device = device

        # The Fourier-space kernel of abTEM's infinite projection, band-limited
        # like the waves: |psi|^2 reaches 4/3 of Nyquist and aliases on the
        # grid, and an emitter limited to 2/3 of it cannot see what folded in.
        xp = get_array_module(device)
        self._kernel = antialias_aperture(self._gpts, self._sampling, xp=xp) / sinc(
            self._gpts, self._sampling, device
        )

    @property
    def n_sources(self) -> int:
        """Number of illuminations the slices are built for."""
        return self._n_sources

    def __len__(self) -> int:
        return len(self._sliced)

    def totals(self) -> np.ndarray:
        """Each slice's emission summed over its atoms, ``(slices, sources)``.

        Proportional to what the maps of :meth:`__getitem__` sum to, which the
        band-limited deltas preserve, without laying anything on the grid.
        """
        totals = np.zeros((len(self), self._n_sources))
        for index in range(len(self)):
            atoms = self._sliced.get_atoms_in_slices(index)
            if len(atoms):
                emission = atoms.arrays["emission"].reshape(len(atoms), -1)
                totals[index] = emission.sum(axis=0)
        return totals

    def __getitem__(self, index: int):
        """The emission of one slice, of shape ``(sources, x, y)``, or None.

        Per unit area [1 / Å²], as abTEM's infinite projection lays out the
        potential: integrated over the slice, it is the cross-sections of the
        slice's atoms, times their illumination.
        """
        atoms = self._sliced.get_atoms_in_slices(index)
        if len(atoms) == 0:
            return None

        xp = get_array_module(self._device)
        positions = atoms.positions[:, :2] / np.array(self._sampling)
        weights = atoms.arrays["emission"].reshape(len(atoms), -1)

        real, complex_ = get_dtype(complex=False), get_dtype(complex=True)
        maps = []
        for source in range(self._n_sources):
            array = xp.zeros(self._gpts, dtype=real)
            array = superpose_deltas(
                positions, array, weights=xp.asarray(weights[:, source], real)
            )
            maps.append(ifft2(fft2(array.astype(complex_)) * self._kernel).real)

        return xp.stack(maps).astype(real)
