import numpy as np

from abtem import GridScan, AnnularDetector, PixelatedDetector, Probe, Potential, PlaneWave
import pytest
from abtem.noise import poisson_noise


def test_calibration_coordinates():
    for endpoint in (True, False):
        gridscan = GridScan(sampling=.7, start=(0, 0), end=(4, 4), endpoint=endpoint)
        calibration_coordinates = gridscan.calibrations[0].coordinates(gridscan.gpts[0])
        assert np.all(calibration_coordinates == gridscan.get_positions()[::gridscan.gpts[0], 0])


@pytest.fixture()
def stem_images(data_path, graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.05)
    probe = Probe(energy=200e3, semiangle_cutoff=30)
    probe.grid.match(potential)

    scan = GridScan(start=[0, 0], end=[0, potential.extent[1]], gpts=(10, 9))

    detector = PixelatedDetector()
    annular_detector = AnnularDetector(50, 150)

    return probe.scan(scan, [detector, annular_detector], potential, pbar=False)


@pytest.fixture()
def hrtem_image(data_path, graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.05)
    planewave = PlaneWave(energy=200e3)
    planewave.grid.match(potential)

    exit_wave = planewave.multislice(potential)

    return exit_wave.intensity()


def test_poisson_noise(stem_images, hrtem_image):
    poisson_noise(stem_images[0], 1e3)
    poisson_noise(stem_images[1], 1e3)
    poisson_noise(hrtem_image, 1e3)
