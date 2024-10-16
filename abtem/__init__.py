"""Main abTEM module."""

import importlib
import sys

from abtem import distributions, transfer
from abtem._version import __version__
from abtem.array import concatenate, from_zarr, stack
from abtem.atoms import orthogonalize_cell, standardize_cell
from abtem.bloch import BlochWaves, StructureFactor
from abtem.core import axes, config
from abtem.detectors import (
    AnnularDetector,
    FlexibleAnnularDetector,
    PixelatedDetector,
    SegmentedDetector,
    WavesDetector,
)
from abtem.inelastic.phonons import AtomsEnsemble, FrozenPhonons
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    IndexedDiffractionPatterns,
    PolarMeasurements,
    RealSpaceLineProfiles,
    ReciprocalSpaceLineProfiles,
)
from abtem.potentials.charge_density import ChargeDensityPotential
from abtem.potentials.iam import CrystalPotential, Potential, PotentialArray
from abtem.prism.s_matrix import SMatrix, SMatrixArray
from abtem.scan import CustomScan, GridScan, LineScan
from abtem.transfer import CTF, Aperture, SpatialEnvelope, TemporalEnvelope
from abtem.visualize.visualizations import show_atoms
from abtem.waves import PlaneWave, Probe, Waves


def _dynamic_import_gpaw():
    module_name = "abtem.potentials.gpaw"
    class_name = "GPAWPotential"
    module = importlib.import_module(module_name)
    setattr(sys.modules[__name__], class_name, getattr(module, class_name))

_dynamic_import_gpaw()

__all__ = [
    "__version__",
    "distributions",
    "orthogonalize_cell",
    "standardize_cell",
    "ChargeDensityPotential",
    "axes",
    "config",
    "concatenate",
    "stack",
    "from_zarr",
    "AnnularDetector",
    "SegmentedDetector",
    "FlexibleAnnularDetector",
    "PixelatedDetector",
    "WavesDetector",
    "GPAWPotential",
    "Images",
    "DiffractionPatterns",
    "RealSpaceLineProfiles",
    "ReciprocalSpaceLineProfiles",
    "PolarMeasurements",
    "IndexedDiffractionPatterns",
    "SMatrix",
    "SMatrixArray",
    "FrozenPhonons",
    "AtomsEnsemble",
    "Potential",
    "CrystalPotential",
    "PotentialArray",
    "CustomScan",
    "LineScan",
    "GridScan",
    "CTF",
    "Aperture",
    "TemporalEnvelope",
    "SpatialEnvelope",
    "show_atoms",
    "Waves",
    "Probe",
    "PlaneWave",
    "transfer",
    "BlochWaves",
    "StructureFactor",
]