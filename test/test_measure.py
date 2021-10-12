import numpy as np
import pytest

from abtem import GridScan, AnnularDetector, PixelatedDetector, Probe, Potential, PlaneWave, FlexibleAnnularDetector, \
    Measurement
from abtem.measure import center_of_mass
from abtem.noise import poisson_noise


def test_calibration_coordinates():
    for endpoint in (True, False):
        gridscan = GridScan(sampling=.7, start=(0, 0), end=(4, 4), endpoint=endpoint)
        calibration_coordinates = gridscan.calibrations[0].coordinates(gridscan.gpts[0])
        assert np.all(calibration_coordinates == gridscan.get_positions()[::gridscan.gpts[0], 0])


@pytest.fixture()
def stem_data(data_path, graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.05)
    probe = Probe(energy=200e3, semiangle_cutoff=30)
    probe.grid.match(potential)

    scan = GridScan(start=[0, 0], end=[potential.extent[0], potential.extent[1]], gpts=(10, 9))

    pixelated_detector = PixelatedDetector()
    flexible_annular_detector = FlexibleAnnularDetector()
    annular_detector = AnnularDetector(50, 150)
    measurements = probe.scan(scan, [annular_detector, flexible_annular_detector, pixelated_detector], potential,
                              pbar=False)
    return {'annular': measurements[0], 'flexible': measurements[1], 'pixelated': measurements[2]}


@pytest.fixture()
def hrtem_image(data_path, graphene_atoms):
    potential = Potential(atoms=graphene_atoms, sampling=.05)
    planewave = PlaneWave(energy=200e3)
    planewave.grid.match(potential)

    exit_wave = planewave.multislice(potential, pbar=False)

    return exit_wave.intensity()


def test_poisson_noise(stem_data, hrtem_image):
    poisson_noise(stem_data['annular'], 1e3)
    poisson_noise(stem_data['pixelated'], 1e3)
    poisson_noise(hrtem_image, 1e3)


def test_integrate_flexible_annular_measurement(stem_data):
    integrated = stem_data['flexible'].integrate(50, 150)
    assert np.allclose(integrated.array, stem_data['annular'].array)


def test_add_measurements(stem_data):
    stem_data = stem_data['annular'].copy()
    stem_data = stem_data + stem_data
    stem_data += stem_data


def test_subtract_measurements(stem_data):
    stem_data = stem_data['annular'].copy()
    stem_data = stem_data - stem_data
    stem_data -= stem_data

@pytest.mark.hyperspy
@pytest.mark.parametrize("signal_type", [None, "diffraction"])
def test_to_hyperspy(stem_data, signal_type):
    for key in stem_data:
        sig = stem_data[key].to_hyperspy(signal_type=signal_type)
        assert isinstance(sig, hs.signals.BaseSignal)


@pytest.mark.hyperspy
def test_to_hyperspy_hrtem(hrtem_image):
    hrtem_image.to_hyperspy()


def test_read_write_measurement(tmp_path, stem_data, hrtem_image):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'measurement.hdf5'

    stem_data['annular'].write(path)
    Measurement.read(path)

    stem_data['flexible'].write(path)
    Measurement.read(path)

    stem_data['pixelated'].write(path)
    Measurement.read(path)

    #hrtem_image.write(path)
    #Measurement.read(path)


@pytest.mark.hyperspy
def test_write_hspy(tmp_path, stem_data, hrtem_image):
    d = tmp_path / 'sub'
    d.mkdir()
    path = d / 'measurement.hspy'

    stem_data['annular'].write(path, format="hspy")

    stem_data['flexible'].write(path, format="hspy")

    stem_data['pixelated'].write(path, format="hspy")

    #hrtem_image.write(path, format="hspy")


def test_indexing(stem_data):
    diffraction_pattern = stem_data['pixelated'][0, 0]
    assert len(diffraction_pattern.calibrations) == 2
    assert diffraction_pattern.calibrations[0].units == 'mrad'


def test_center_of_mass(stem_data):
    center_of_mass(stem_data['pixelated'])
    center_of_mass(stem_data['pixelated'], return_icom=True)
