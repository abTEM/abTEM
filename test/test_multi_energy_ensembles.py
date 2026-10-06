"""Multi-energy ensembles: S-matrix scans (#502) and diffraction patterns (#503).

Detectors are matched against the whole energy ensemble before it is split into
energies, and the wavelength-dependent methods of a multi-energy
`DiffractionPatterns` are evaluated one energy at a time.
"""

import ase.build
import numpy as np
import pytest
from utils import devices

import abtem

ENERGIES = (50e3, 60e3, 70e3)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The interpolation factor does not exactly divide:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The scan step is not a whole number of pixels:UserWarning"
    ),
]


@pytest.fixture(autouse=True)
def _float64():
    with abtem.config.set({"precision": "float64"}):
        yield


def _potential(device="cpu"):
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    return abtem.Potential(atoms, sampling=0.15, slice_thickness=2, device=device)


def _scan(potential, gpts=(3, 4)):
    return abtem.GridScan(
        (0, 0), (1, 1), gpts=gpts, fractional=True, potential=potential
    )


def _exit_waves(potential, energy, lazy=False):
    probe = abtem.Probe(energy=energy, semiangle_cutoff=20, device=potential.device)
    return probe.multislice(potential, scan=_scan(potential), lazy=lazy)


def _close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol * np.abs(b).max())


# --- multi-energy S-matrix scans (#502) -----------------------------------------


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(interpolation=2), dict(interpolation=2, upsample=True)],
    ids=["interpolation_1", "interpolation_2", "interpolation_2_upsampled"],
)
def test_multi_energy_prism_pixelated_crop_stacks_like_multislice(kwargs, lazy):
    potential = _potential()
    s_matrix = abtem.SMatrix(
        potential=potential, energy=list(ENERGIES), semiangle_cutoff=20, **kwargs
    )
    prism = s_matrix.scan(
        scan=_scan(potential),
        detectors=abtem.PixelatedDetector(max_angle=60),
        lazy=lazy,
    )
    prism = prism.compute(progress_bar=False) if lazy else prism

    probe = abtem.Probe(energy=list(ENERGIES), semiangle_cutoff=20)
    probe.grid.match(potential)
    multislice = probe.scan(
        potential,
        scan=_scan(potential),
        detectors=abtem.PixelatedDetector(max_angle=60),
        lazy=False,
    )

    assert prism.shape[0] == len(ENERGIES)
    # the pixel count of the whole ensemble, and its axis labels, as multislice
    if not kwargs:
        assert prism.shape[-2:] == multislice.shape[-2:]
        np.testing.assert_allclose(prism.sampling, multislice.sampling, rtol=1e-12)


@pytest.mark.parametrize("make", [lambda: abtem.FlexibleAnnularDetector()])
def test_multi_energy_prism_refuses_an_auto_sized_radial_detector(make):
    potential = _potential()
    s_matrix = abtem.SMatrix(
        potential=potential, energy=list(ENERGIES), semiangle_cutoff=20
    )
    with pytest.raises(RuntimeError, match="cannot auto-size its outer angle"):
        s_matrix.scan(scan=_scan(potential), detectors=make(), lazy=False)


# --- per-energy diffraction patterns (#503) -----------------------------------


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        lambda p: p.polar_binning(4, 4, 30, 100),
        lambda p: p.radial_binning(step_size=5, inner=30, outer=100),
        lambda p: p.integrate_radial(30, 100),
        lambda p: p.center_of_mass(units="mrad"),
        lambda p: p.block_direct(),
        lambda p: p.bandlimit(30, 100),
    ],
    ids=[
        "polar_binning",
        "radial_binning",
        "integrate_radial",
        "center_of_mass_mrad",
        "block_direct",
        "bandlimit",
    ],
)
def test_diffraction_pattern_methods_equal_single_energy_runs(
    operation, lazy, device
):
    potential = _potential(device)
    multi = _exit_waves(potential, list(ENERGIES)).diffraction_patterns(
        max_angle="full"
    )
    if lazy:
        multi = multi.lazy()
    result = operation(multi)
    result = result.compute(progress_bar=False) if lazy else result

    axis = [type(a).__name__ for a in result.axes_metadata].index("EnergyAxis")
    for i, energy in enumerate(ENERGIES):
        single = operation(
            _exit_waves(potential, energy).diffraction_patterns(max_angle="full")
        )
        _close(np.take(result.array, i, axis=axis), single.array, atol=1e-10)


@devices
def test_pixelated_integration_equals_the_annular_detector_for_each_energy(device):
    potential = _potential(device)
    waves = _exit_waves(potential, list(ENERGIES))
    pixelated = (
        abtem.PixelatedDetector(max_angle="full")
        .detect(waves)
        .integrate_radial(30, 100)
    )
    annular = abtem.AnnularDetector(30, 100).detect(waves)

    # the annular detector moves the energy axis behind the scan axes
    axis = [type(a).__name__ for a in pixelated.axes_metadata].index("EnergyAxis")
    annular_axis = [type(a).__name__ for a in annular.axes_metadata].index(
        "EnergyAxis"
    )
    for i in range(len(ENERGIES)):
        _close(
            np.take(pixelated.array, i, axis=axis),
            np.take(annular.array, i, axis=annular_axis),
            atol=1e-6,
        )
