import dask
import numpy as np
from ase.build import bulk

from abtem.measure.detect import PixelatedDetector, FlexibleAnnularDetector, AnnularDetector
from abtem.measure.measure import Images, from_zarr
from abtem.potentials import Potential, PotentialArray
from abtem.waves.scan import GridScan
from abtem.waves.waves import Probe, Waves


def test_export_import_potential(tmp_path, graphene_atoms):
    d = tmp_path / 'sub'
    d.mkdir()
    path = str(d / 'potential.zarr')

    potential = Potential(graphene_atoms, sampling=.05)
    precalculated_potential = potential.build()
    precalculated_potential.to_zarr(path)
    imported_potential = PotentialArray.from_zarr(path)
    assert np.allclose(imported_potential.array, precalculated_potential.array)
    assert np.allclose(imported_potential.extent, precalculated_potential.extent)
    assert np.allclose(imported_potential.slice_thickness, precalculated_potential.slice_thickness)


def test_export_import_waves(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    path = str(d / 'waves.zarr')

    waves = Probe(semiangle_cutoff=30, sampling=.05, extent=10, energy=80e3, tilt=(5, 5)).build()
    waves.to_zarr(path)
    imported_waves = Waves.from_zarr(path)

    assert np.allclose(waves.array, imported_waves.array)
    assert np.allclose(waves.extent, imported_waves.extent)
    assert np.allclose(waves.energy, imported_waves.energy)
    assert np.allclose(waves.tilt, imported_waves.tilt)
    assert waves.axes_metadata == imported_waves.axes_metadata


def test_export_import_images(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    path = str(d / 'images.zarr')

    images = Images(np.random.rand(1, 512, 256), sampling=.1, axes_metadata=[{'a': 'b', 'c': 0.1}])
    images.to_zarr(path)
    imported_images = Images.from_zarr(path)
    assert np.allclose(images.array, imported_images.array)
    assert np.allclose(images.sampling, imported_images.sampling)
    assert images.axes_metadata == imported_images.axes_metadata


def test_scan_to_file(tmp_path):
    flexible_detector = FlexibleAnnularDetector()
    annular_detector = AnnularDetector(inner=50, outer=90)
    pixelated_detector = PixelatedDetector()
    detectors = [annular_detector, flexible_detector, pixelated_detector]

    atoms = bulk('Si', 'diamond', a=5.43, cubic=True)

    potential = Potential(atoms,
                          gpts=128,
                          device='cpu',
                          projection='infinite',
                          slice_thickness=6)

    probe = Probe(energy=100e3, semiangle_cutoff=20, device='cpu')
    scan = GridScan(sampling=.5)

    d = tmp_path / 'sub'
    d.mkdir()

    for detector, name in zip(detectors, ['a', 'b', 'c']):
        path = str(d / f'{name}.zarr')
        measurement = probe.scan(scan, detector, potential)
        dask.compute([measurement.as_delayed(), measurement.to_zarr(path, compute=False)])
        imported_measurement = from_zarr(path)
        assert np.allclose(measurement.array, imported_measurement.array)
