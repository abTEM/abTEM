import numpy as np
from ase import Atoms
from typing import TYPE_CHECKING

from abtem.core.antialias import AntialiasAperture
from abtem.core.complex import complex_exponential
from abtem.potentials.potentials import validate_potential
from abtem.structures.slicing import SliceIndexedAtoms
from abtem.waves.multislice import FresnelPropagator, multislice
from abtem.measure.measure import Images
from abtem.waves.prism_utils import wrapped_crop_2d

if TYPE_CHECKING:
    from abtem.waves.prism import SMatrix


def validate_sites(potential=None, sites=None):
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


def transition_potential_multislice(waves,
                                    potential,
                                    detectors,
                                    transition_potentials,
                                    sites=None,
                                    ctf=None,
                                    scan=None):
    if hasattr(waves, 'reduce') and scan is None:
        raise RuntimeError()

    if not hasattr(waves, 'reduce') and scan is not None:
        raise RuntimeError()

    potential = validate_potential(potential)
    sites = validate_sites(potential, sites)

    transition_potentials.grid.match(waves)
    transition_potentials.accelerator.match(waves)

    antialias_aperture = AntialiasAperture().match_grid(waves)
    propagator = FresnelPropagator().match_waves(waves)

    transmission_function = potential.build().transmission_function(energy=waves.energy)
    transmission_function = antialias_aperture.bandlimit(transmission_function)

    measurements = [detector.allocate_measurement(waves, scan=scan) for detector in detectors]
    for i, (transmission_function_slice, sites_slice) in enumerate(zip(transmission_function, sites)):
        sites_slice = transition_potentials.validate_sites(sites_slice)

        for scattered_waves in transition_potentials.generate_scattered_waves(waves, sites_slice):
            scattered_waves = multislice(scattered_waves,
                                         transmission_function,
                                         start=i,
                                         propagator=propagator,
                                         antialias_aperture=antialias_aperture)

            if ctf is not None:
                scattered_waves = scattered_waves.apply_ctf(ctf)

            for j, detector in enumerate(detectors):
                if hasattr(scattered_waves, 'reduce'):
                    measurement = scattered_waves.reduce(positions=scan, detectors=detector).sum(0)
                else:
                    measurement = detector.detect(scattered_waves).sum(0)

                measurements[j].add(measurement)

        propagator.thickness = transmission_function_slice.thickness
        waves = transmission_function_slice.transmit(waves)
        waves = propagator.propagate(waves)

    return measurements


def linear_scaling_transition_multislice(S1: 'SMatrix', S2: 'SMatrix', scan, transition_potentials):
    positions = scan.get_positions(lazy=False).reshape((-1, 2))

    prism_region = (S1.extent[0] / S1.interpolation[0] / 2, S1.extent[1] / S1.interpolation[1] / 2)

    coefficients = complex_exponential(-2. * np.pi * positions[:, 0, None] * S1.wave_vectors[None, :, 0])
    coefficients *= complex_exponential(-2. * np.pi * positions[:, 1, None] * S1.wave_vectors[None, :, 1])
    coefficients = coefficients / np.sqrt(coefficients.shape[1])

    potential = S1.potential

    sites = validate_sites(potential, sites=None)

    S1 = S1.build(lazy=False, stop=0)

    images = np.zeros(len(positions), dtype=np.float32)
    for i in range(len(potential)):
        S1 = S1.multislice(potential, start=max(i - 1, 0), stop=i)
        S2_multislice = S2.build(lazy=False, start=len(potential), stop=i)

        sites_slice = transition_potentials.validate_sites(sites[i])

        for site, scattered_S1 in transition_potentials.generate_scattered_waves(S1, sites_slice, chunks=1):

            if S1.interpolation == (1, 1):
                cropped_coefficients = coefficients
                mask = None
            else:
                mask = np.ones(len(coefficients), dtype=bool)
                if S1.interpolation[0] > 1:
                    mask *= (np.abs(positions[:, 0] - site[0, 0]) % (S1.extent[0] - prism_region[0])) <= prism_region[0]
                if S1.interpolation[1] > 1:
                    mask *= (np.abs(positions[:, 1] - site[0, 1]) % (S1.extent[1] - prism_region[1])) <= prism_region[1]

                cropped_coefficients = coefficients[mask]

            S2_multislice = S2_multislice.crop_to_positions(site)
            scattered_S1 = scattered_S1.crop_to_positions(site)

            for j in range(scattered_S1.shape[0]):
                SHn0 = np.tensordot(S2_multislice.array.reshape((len(S2), -1)),
                                    scattered_S1.array[j].reshape((len(S1), -1)).T, axes=1)

                if mask is not None:
                    images[mask] += (np.abs(np.dot(SHn0, cropped_coefficients.T)) ** 2).sum(0)  # .reshape(scan.gpts)
                else:
                    images += (np.abs(np.dot(SHn0, cropped_coefficients.T)) ** 2).sum(0)

    images = Images(images.reshape(scan.gpts), sampling=scan.sampling)
    return images
