"""Every concrete ArrayObject subclass survives the operations that rebuild it
from its constructor: a copy, partitioning into ensemble blocks (the rebuild
multislice and apply_transform use) and a to_zarr/from_zarr round trip.

from_zarr resolves the stored class name as an attribute of ``abtem``, so every
concrete subclass is exported there.
"""

import importlib
import inspect
import pkgutil

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from utils import assert_array_objects_equal

import abtem
from abtem.array import ArrayObject
from abtem.core.axes import EnergyLossAxis, OrdinalAxis

ENERGY = 100e3


def _atoms():
    return Atoms(
        "CO",
        positions=[(1.0, 1.0, 0.5), (2.5, 1.7, 2.5)],
        cell=(4.0, 3.0, 4.0),
        pbc=True,
    )


def _potential_array():
    return abtem.Potential(_atoms(), gpts=(40, 30), slice_thickness=1.0).build(
        lazy=False
    )


def _waves():
    return abtem.PlaneWave(energy=ENERGY, extent=(4.0, 3.0), gpts=(40, 30)).build(
        lazy=False
    )


def _diffraction_patterns():
    probe = abtem.Probe(
        energy=ENERGY, semiangle_cutoff=20, extent=(4.0, 3.0), gpts=(40, 30)
    )
    return probe.build(lazy=False).diffraction_patterns(max_angle=None)


def _magnetic_atoms():
    atoms = Atoms("Fe", positions=[(1.0, 1.0, 1.0)], cell=(4.0, 3.0, 2.0), pbc=True)
    atoms.set_array("magnetic_moments", np.array([[0.0, 0.0, 2.0]]))
    return atoms


def _transition_potential_array():
    from abtem.inelastic.core_loss import TransitionPotentialArray

    rng = np.random.default_rng(0)
    array = rng.standard_normal((3, 32, 24)) + 1j * rng.standard_normal((3, 32, 24))
    return TransitionPotentialArray(
        Z=5,
        energy=ENERGY,
        extent=(8.0, 6.0),
        array=array.astype(np.complex64),
        ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))],
    )


def _momentum_resolved_spectrum():
    from abtem.detectors import SpectralSlitDetector

    array = np.random.default_rng(1).random((3, 32, 24))
    dp = abtem.DiffractionPatterns(
        array,
        sampling=0.5,
        metadata={"energy": 300e3},
        ensemble_axes_metadata=[
            EnergyLossAxis(label="E", values=(0.02, 0.05, 0.1), units="eV")
        ],
    )
    return abtem.momentum_resolved_spectrum(
        dp, SpectralSlitDetector(width=3.0, q_max=8.0)
    )


CASES = {
    "Waves": _waves,
    "PotentialArray": _potential_array,
    "TransmissionFunction": lambda: _potential_array().transmission_function(ENERGY),
    "SMatrixArray": lambda: abtem.SMatrix(
        potential=_potential_array(), energy=ENERGY, semiangle_cutoff=10
    ).build(lazy=False),
    "Images": lambda: _waves().intensity(),
    "DiffractionPatterns": _diffraction_patterns,
    "PolarMeasurements": lambda: _diffraction_patterns().polar_binning(
        nbins_radial=4, nbins_azimuthal=3
    ),
    "IndexedDiffractionPatterns": lambda: _diffraction_patterns().index_diffraction_spots(
        cell=_atoms().cell
    ),
    "RealSpaceLineProfiles": lambda: _waves()
    .intensity()
    .interpolate_line(start=(0, 0), end=(3, 2)),
    "ReciprocalSpaceLineProfiles": lambda: _diffraction_patterns().interpolate_line(
        start=(0, 0), end=(10, 10)
    ),
    "MomentumResolvedSpectrum": _momentum_resolved_spectrum,
    "MeasurementsEnsemble": lambda: abtem.MeasurementsEnsemble(
        np.arange(3.0), ensemble_axes_metadata=[OrdinalAxis(values=(0, 1, 2))]
    ),
    "TransitionPotentialArray": _transition_potential_array,
    "MagneticFieldArray": lambda: __import__(
        "abtem.magnetism.iam", fromlist=["MagneticField"]
    )
    .MagneticField(_magnetic_atoms(), gpts=(20, 16), slice_thickness=1.0)
    .build(lazy=False),
    "VectorPotentialArray": lambda: __import__(
        "abtem.magnetism.iam", fromlist=["VectorPotential"]
    )
    .VectorPotential(_magnetic_atoms(), gpts=(20, 16), slice_thickness=1.0)
    .build(lazy=False),
    "StructureFactorArray": lambda: abtem.StructureFactor(
        bulk("Si", cubic=True), g_max=2.0
    ).build(lazy=False),
}


def _concrete_array_object_classes():
    for module in pkgutil.walk_packages(abtem.__path__, "abtem."):
        try:
            importlib.import_module(module.name)
        except (ImportError, AttributeError):
            pass  # optional dependency missing; test_imports reports it

    def subclasses(cls):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclasses(subclass)

    return {cls for cls in subclasses(ArrayObject) if not inspect.isabstract(cls)}


def test_every_array_object_subclass_has_a_case_and_is_exported():
    classes = _concrete_array_object_classes()
    assert {cls.__name__ for cls in classes} == set(CASES)
    for cls in classes:
        assert getattr(abtem, cls.__name__, None) is cls, cls.__name__


@pytest.mark.parametrize("name", sorted(CASES))
def test_array_object_round_trip(name, tmp_path):
    array_object = CASES[name]()
    assert type(array_object).__name__ == name

    assert_array_objects_equal(array_object.copy(), array_object)

    for _, slics, wrapped in array_object.generate_blocks():
        block = wrapped.item()
        assert type(block) is type(array_object)
        np.testing.assert_array_equal(
            np.asarray(block.array), np.asarray(array_object.array[slics])
        )

    url = str(tmp_path / "array_object.zarr")
    array_object.to_zarr(url)
    assert_array_objects_equal(abtem.from_zarr(url), array_object)
