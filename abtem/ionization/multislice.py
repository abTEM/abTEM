import numpy as np
from ase import Atoms

from abtem.core.antialias import AntialiasAperture
from abtem.potentials.potentials import validate_potential
from abtem.structures.slicing import SliceIndexedAtoms
from abtem.waves.multislice import FresnelPropagator, multislice


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


def transition_potential_multislice(waves, potential, detectors, transition_potentials, sites=None, ctf=None):
    potential = validate_potential(potential)

    sites = potential.atoms
    sites = validate_sites(sites, potential)

    transition_potentials.grid.match(waves)
    transition_potentials.accelerator.match(waves)

    antialias_aperture = AntialiasAperture().match_grid(waves)
    propagator = FresnelPropagator().match_waves(waves)

    potential = potential.build()
    transmission_function = potential.transmission_function(energy=waves.energy)
    transmission_function = antialias_aperture.bandlimit(transmission_function)

    measurements = []
    for i, (transmission_function_slice, sites_slice) in enumerate(zip(transmission_function, sites)):
        sites_slice = transition_potentials.validate_sites(sites_slice)

        if len(sites_slice) == 0:
            continue

        scattered_waves = transition_potentials.scatter(waves, sites_slice)
        scattered_waves = multislice(scattered_waves,
                                     potential[i:],
                                     propagator=propagator,
                                     antialias_aperture=antialias_aperture)

        if ctf is not None:
            scattered_waves = scattered_waves.apply_ctf(ctf)

        for i, detector in enumerate(detectors):
            try:
                measurements[i].add(detector.detect(scattered_waves).sum(0))
            except IndexError:
                measurements.append(detector.detect(scattered_waves).sum(0))

        propagator.thickness = transmission_function_slice.thickness
        waves = transmission_function_slice.transmit(waves)
        waves = propagator.propagate(waves)

    return measurements