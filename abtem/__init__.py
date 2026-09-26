"""Main abTEM module."""

# Runs its seeding as an import-time side effect (see the module docstring):
# must be imported before abtem.distributions below, which is what first
# reaches abtem.core.backend, which is what first imports numba. Kept as
# its own statement rather than merged with the later `from abtem.core
# import axes, config` below -- an isort-style merge would silently lose
# this ordering requirement.
from abtem.core import _numba_threads  # noqa: F401, I001

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
    SpectralAnnularDetector,
    SpectralSlitDetector,
    WavesDetector,
    WindowedPixelatedDetector,
)
from abtem.inelastic.phonons import (
    AtomsEnsemble,
    EnergyResolvedAtomsEnsemble,
    FrozenPhonons,
)
from abtem.measurements import (
    DiffractionPatterns,
    Images,
    IndexedDiffractionPatterns,
    MeasurementsEnsemble,
    MomentumResolvedSpectrum,
    PolarMeasurements,
    RealSpaceLineProfiles,
    ReciprocalSpaceLineProfiles,
    momentum_resolved_spectrum,
    phonon_loss_diffraction_patterns,
)
from abtem.potentials.iam import CrystalPotential, Potential, PotentialArray
from abtem.prism.s_matrix import CompressedSMatrixArray, SMatrix, SMatrixArray
from abtem.scan import CustomScan, GridScan, LineScan
from abtem.transfer import CTF, Aperture, SpatialEnvelope, TemporalEnvelope
from abtem.visualize.visualizations import show_atoms
from abtem.waves import PlaneWave, Probe, Waves

# Registers dask.sizeof.sizeof for abTEM's large payload carriers (and
# ase.Atoms). Import last: by this point every class it registers is
# already loaded, and nothing above this line instantiates one of them,
# so there is no risk of a sizeof() call reaching an unregistered type
# before this module's registrations take effect (dask memoizes its
# dispatch per type -- see abtem.core.dask_sizeof's own docstring).
from abtem.core import dask_sizeof  # noqa: F401

__all__ = [
    "__version__",
    "distributions",
    "orthogonalize_cell",
    "standardize_cell",
    "axes",
    "config",
    "concatenate",
    "stack",
    "from_zarr",
    "AnnularDetector",
    "SpectralAnnularDetector",
    "SpectralSlitDetector",
    "SegmentedDetector",
    "FlexibleAnnularDetector",
    "PixelatedDetector",
    "WindowedPixelatedDetector",
    "WavesDetector",
    "Images",
    "DiffractionPatterns",
    "RealSpaceLineProfiles",
    "ReciprocalSpaceLineProfiles",
    "MeasurementsEnsemble",
    "MomentumResolvedSpectrum",
    "momentum_resolved_spectrum",
    "phonon_loss_diffraction_patterns",
    "PolarMeasurements",
    "IndexedDiffractionPatterns",
    "SMatrix",
    "SMatrixArray",
    "CompressedSMatrixArray",
    "FrozenPhonons",
    "AtomsEnsemble",
    "EnergyResolvedAtomsEnsemble",
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
