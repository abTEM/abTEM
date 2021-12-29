from ase import Atoms

from abtem.core.antialias import AntialiasAperture
from abtem.potentials.potentials import validate_potential
from abtem.structures.slicing import SliceIndexedAtoms
from abtem.waves.fresnel import FresnelPropagator


def validate_sites(sites, potential):
    if sites is None:
        if hasattr(potential, 'atoms'):
            sites = potential.atoms
        else:
            raise RuntimeError(f'transition sites cannot be inferred from potential of type {type(potential)}')

    if isinstance(sites, SliceIndexedAtoms):
        if len(potential) != len(sites):
            raise RuntimeError(f'transition sites slices({len(sites)}) != potential slices({len(potential)})')
    elif isinstance(sites, Atoms):
        sites = SliceIndexedAtoms(sites, slice_thickness=potential.slice_thickness)

    else:
        raise RuntimeError(f'transition sites must be Atoms or SliceIndexedAtoms, received {type(sites)}')

    return sites


def _transition_potential_multislice(waves, potential, transition_potentials, sites, detectors, ctf=None):
    potential = validate_potential(potential)
    sites = validate_sites(sites, potential)

    antialias_aperture = AntialiasAperture().match_grid(waves)
    propagator = FresnelPropagator().match_waves(waves)

    potential = potential.build()
    transmission_function = potential.transmission_function(energy=waves.energy)
    transmission_function = antialias_aperture.bandlimit(transmission_function)

    measurements = []
    for i, (transmission_function_slice, sites_slice) in enumerate(zip(transmission_function, sites)):
        scattered_waves = transition_potentials.scatter(waves, sites_slice)
        scattered_waves = scattered_waves.multislice(potential[i:])

        if ctf is not None:
            scattered_waves = scattered_waves.apply_ctf(ctf)

        propagator.thickness = transmission_function_slice.thickness
        waves = transmission_function_slice.transmit(waves)
        waves = propagator.propagate(waves)

        for i, detector in enumerate(detectors):
            try:
                measurements[i].add(detector.detect(scattered_waves).sum(0))
            except IndexError:
                measurements.append(detector.detect(scattered_waves).sum(0))

    return measurements


def transition_potential_multislice(waves, potential, transition_potentials, sites, detectors, ctf):
    if not waves.is_lazy:
        return _transition_potential_multislice(waves, potential, transition_potentials, sites, detectors, ctf)

    waves.array.map_blocks()
