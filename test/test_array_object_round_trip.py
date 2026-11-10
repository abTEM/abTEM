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
from dask import is_dask_collection
from test_imports import NEEDS_CUPY
from utils import assert_array_objects_equal

import abtem
from abtem.array import ArrayObject
from abtem.core.axes import EnergyLossAxis, OrdinalAxis
from abtem.core.backend import cp
from abtem.magnetism.iam import MagneticField, VectorPotential

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


def _spherical_pattern():
    from abtem.ebsd import SphericalPattern, SquareLambertProjection

    directions = SquareLambertProjection().grid(21)
    return SphericalPattern(1.0 + 0.3 * directions[:, 0], directions)


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
    "IndexedDiffractionPatterns": lambda: (
        _diffraction_patterns().index_diffraction_spots(cell=_atoms().cell)
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
    "MagneticFieldArray": lambda: MagneticField(
        _magnetic_atoms(), gpts=(20, 16), slice_thickness=1.0
    ).build(lazy=False),
    "VectorPotentialArray": lambda: VectorPotential(
        _magnetic_atoms(), gpts=(20, 16), slice_thickness=1.0
    ).build(lazy=False),
    "EBSDPatternImages": lambda: _spherical_pattern().project(
        abtem.ebsd.EBSDGeometry(shape=(8, 6)), euler=(30.0, 54.7, 45.0)
    ),
    "ReferencePatternImages": lambda: _spherical_pattern().interpolate(9, "lambert"),
    "StructureFactorArray": lambda: abtem.StructureFactor(
        bulk("Si", cubic=True), g_max=2.0
    ).build(lazy=False),
}


def _concrete_array_object_classes():
    for module in pkgutil.walk_packages(abtem.__path__, "abtem."):
        if module.name in NEEDS_CUPY and cp is None:
            continue
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as error:
            if error.name and error.name.split(".")[0] != "abtem":
                continue  # an optional package is not installed
            raise

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


@pytest.mark.parametrize("name", sorted(CASES))
def test_compute_leaves_no_lazy_attribute(name, tmp_path):
    """compute() turns every array the object holds eager, not only its main one,
    so the computed object compares equal to the eagerly built one.

    CPU only: which attributes compute() replaces is bookkeeping shared by every
    device.
    """
    array_object = CASES[name]()
    url = str(tmp_path / "array_object.zarr")
    array_object.to_zarr(url)

    for computed in (
        array_object.ensure_lazy().compute(),
        abtem.from_zarr(url).compute(),
    ):
        lazy = [k for k, v in vars(computed).items() if is_dask_collection(v)]
        assert not computed.is_lazy
        assert lazy == []
        assert computed == array_object
        assert_array_objects_equal(computed, array_object)


def test_indexed_diffraction_patterns_unpickled_from_an_older_version_compare_equal():
    """abTEM 1.0.11 and earlier pickled `_intensities`, an alias of the array, with
    the object; one restored from such a pickle compares equal both ways.

    CPU only: unpickling is bookkeeping shared by every device.
    """
    import pickle

    old = CASES["IndexedDiffractionPatterns"]()
    old.__dict__["_intensities"] = old.array
    restored = pickle.loads(pickle.dumps(old))
    new = CASES["IndexedDiffractionPatterns"]()

    assert "_intensities" not in vars(restored)
    assert restored == new
    assert new == restored


@pytest.mark.parametrize("name", sorted(CASES))
def test_zarr_round_trip_keeps_array_valued_attributes(name, tmp_path):
    """Every array-valued attribute (Miller indices, cell, wave vectors, ...) reads
    back from zarr with its type, dtype, shape and values; the zarr attributes hold
    it as nested JSON lists, which `==` (np.allclose) cannot tell from the array.

    CPU only: zarr is written and read on the host.
    """
    array_object = CASES[name]()
    url = str(tmp_path / "array_object.zarr")
    array_object.to_zarr(url)
    read = abtem.from_zarr(url).compute()

    for key, value in vars(array_object).items():
        if key == "_array" or not hasattr(value, "dtype") or np.ndim(value) == 0:
            continue
        other = vars(read)[key]
        assert type(other) is type(value), (key, type(other), type(value))
        assert other.dtype == value.dtype, (key, other.dtype, value.dtype)
        np.testing.assert_array_equal(np.asarray(other), np.asarray(value), err_msg=key)


def test_structure_factor_read_from_zarr_gives_the_same_bloch_wave_patterns(tmp_path):
    """The Miller indices and the cell of a structure factor read from zarr are
    arrays, so `BlochWaves` accepts it. `lazy=False`: a lazy `BlochWaves` does not
    take an eager structure factor.

    CPU only: zarr is written and read on the host.
    """
    structure_factor = CASES["StructureFactorArray"]()
    url = str(tmp_path / "structure_factor.zarr")
    structure_factor.to_zarr(url)
    read = abtem.from_zarr(url).compute()

    def patterns(structure_factor):
        bloch_waves = abtem.BlochWaves(structure_factor, energy=ENERGY, sg_max=0.05)
        return bloch_waves.calculate_diffraction_patterns(10.0, lazy=False)

    assert_array_objects_equal(patterns(read), patterns(structure_factor))


def test_s_matrix_read_from_zarr_reduces_to_the_same_waves(tmp_path):
    """The wave vectors of an S-matrix read from zarr keep the float64 they were
    written in, so reducing it gives the waves of the S-matrix that was written.
    The 5 x 4 scan differs in size from the 9 plane waves and from each grid axis.

    CPU only: zarr is written and read on the host.
    """
    s_matrix = CASES["SMatrixArray"]()
    url = str(tmp_path / "s_matrix.zarr")
    s_matrix.to_zarr(url)
    read = abtem.from_zarr(url).compute()

    scan = abtem.GridScan((0.0, 0.0), (4.0, 3.0), gpts=(5, 4))
    assert_array_objects_equal(
        read.reduce(scan=scan).compute(), s_matrix.reduce(scan=scan).compute()
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_transmission_function_with_ensemble_axes_and_exit_planes_round_trip(
    lazy, tmp_path
):
    frozen_phonons = abtem.FrozenPhonons(_atoms(), num_configs=3, sigmas=0.1, seed=1)
    potential = abtem.Potential(
        frozen_phonons, gpts=(40, 30), slice_thickness=1.0, exit_planes=1
    ).build(lazy=lazy)
    potential.metadata["label"] = "kept"
    original = potential.transmission_function(ENERGY)

    assert original.exit_planes == potential.exit_planes
    assert original.ensemble_axes_metadata == potential.ensemble_axes_metadata
    assert original.metadata["label"] == "kept"

    url = str(tmp_path / "transmission_function.zarr")
    original.to_zarr(url)
    rebuilt = abtem.from_zarr(url)

    assert_array_objects_equal(rebuilt, original)
    assert rebuilt.energy == ENERGY
    assert rebuilt.exit_planes == original.exit_planes
    assert rebuilt.metadata["label"] == "kept"
    assert rebuilt.array.shape == (3, 4, 40, 30)
