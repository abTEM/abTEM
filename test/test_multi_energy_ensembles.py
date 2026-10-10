"""Multi-energy ensembles: S-matrix scans and reductions (#502) and diffraction
patterns (#503).

Detectors are matched against the whole energy ensemble before it is split into
energies, and the wavelength-dependent methods of a multi-energy
`DiffractionPatterns` are evaluated one energy at a time. Reductions of
frozen-phonon scattering matrices follow the frozen phonons' `ensemble_mean`,
eager and lazy.
"""

import ase.build
import numpy as np
import pytest
from utils import devices

import abtem
from abtem.core.backend import asnumpy

# not sorted, and a different count from the scan axes (3 and 5)
ENERGIES = (70e3, 50e3, 80e3, 60e3)

pytestmark = [
    pytest.mark.float64,
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


def _scan(potential, gpts=(3, 5)):
    return abtem.GridScan(
        (0, 0), (1, 1), gpts=gpts, fractional=True, potential=potential
    )


def _exit_waves(potential, energy, lazy=False):
    probe = abtem.Probe(energy=energy, semiangle_cutoff=20, device=potential.device)
    return probe.multislice(potential, scan=_scan(potential), lazy=lazy)


def _close(a, b, atol=1e-12):
    # the arrays of a GPU run are CuPy arrays, which NumPy does not convert
    a, b = asnumpy(a), asnumpy(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=atol * np.abs(b).max())


# --- multi-energy S-matrix scans (#502) -----------------------------------------

_PRISM_CONFIGURATIONS = pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(interpolation=2), dict(interpolation=2, upsample=True)],
    ids=["interpolation_1", "interpolation_2", "interpolation_2_upsampled"],
)


def _prism(potential, energy, detector, lazy=False, **kwargs):
    s_matrix = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=20, **kwargs
    )
    out = s_matrix.scan(scan=_scan(potential), detectors=detector, lazy=lazy)
    return out.compute(progress_bar=False) if lazy else out


def _same_pixels(a, b):
    """The patterns `a` and `b` cropped to the pixels they share."""
    shape = tuple(min(m, n) for m, n in zip(a.shape[-2:], b.shape[-2:]))
    return a.crop(gpts=shape).array, b.crop(gpts=shape).array


@pytest.mark.parametrize("lazy", [False, True])
@_PRISM_CONFIGURATIONS
@pytest.mark.parametrize("max_angle", ["valid", "cutoff", "full", 60])
def test_multi_energy_prism_pixelated_equals_single_energy_runs(
    max_angle, kwargs, lazy
):
    potential = _potential()
    multi = _prism(
        potential,
        list(ENERGIES),
        abtem.PixelatedDetector(max_angle=max_angle),
        lazy,
        **kwargs,
    )
    axis = [type(a).__name__ for a in multi.axes_metadata].index("EnergyAxis")
    assert multi.shape[axis] == len(ENERGIES)

    shapes = []
    for i, energy in enumerate(ENERGIES):
        single = _prism(
            potential,
            energy,
            abtem.PixelatedDetector(max_angle=max_angle),
            lazy,
            **kwargs,
        )
        member = multi[(slice(None),) * axis + (i,)]
        shapes.append(single.shape[-2:])
        # a string max_angle crops every energy alike; a number of mrad crops the
        # ensemble to the pixel count of its highest energy, and what is there
        # equals the energy's own run
        a, b = _same_pixels(member, single)
        # lazy blocks are computed in the default single precision
        _close(a, b, atol=1e-5 if lazy else 1e-10)
        if isinstance(max_angle, str):
            assert member.shape[-2:] == single.shape[-2:]

    if isinstance(max_angle, str):
        assert len(set(shapes)) == 1
    else:
        highest = ENERGIES.index(max(ENERGIES))
        assert multi.shape[-2:] == shapes[highest]


@pytest.mark.parametrize(
    "make",
    [
        lambda: abtem.FlexibleAnnularDetector(),
        lambda: abtem.SegmentedDetector(
            nbins_radial=2, nbins_azimuthal=4, inner=30, outer=None
        ),
    ],
    ids=["flexible_annular", "segmented"],
)
def test_multi_energy_prism_refuses_an_auto_sized_radial_detector(make):
    potential = _potential()
    s_matrix = abtem.SMatrix(
        potential=potential, energy=list(ENERGIES), semiangle_cutoff=20
    )
    with pytest.raises(RuntimeError, match="cannot auto-size its outer angle"):
        s_matrix.scan(scan=_scan(potential), detectors=make(), lazy=False)


def _build_and_reduce(potential, energy, detectors, lazy=False, **kwargs):
    s_matrix = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=20, **kwargs
    )
    out = s_matrix.build(lazy=lazy).reduce(scan=_scan(potential), detectors=detectors)
    return out.compute(progress_bar=False) if lazy else out


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(interpolation=2)],
    ids=["interpolation_1", "interpolation_2"],
)
def test_multi_energy_build_reduce_equals_single_energy_runs(kwargs, lazy):
    """A multi-energy `SMatrix.build()` could not be reduced at all: the
    `SMatrixArray` has no energy of its own and raised `EnergyUndefinedError`.
    Each energy is now reduced from its own rows of the zero-padded expansion,
    at its own wavelength, and the measurements are stacked."""

    def detectors():
        return [
            abtem.AnnularDetector(30, 60),
            abtem.AnnularDetector(30),
            abtem.PixelatedDetector(max_angle=60),
            abtem.WavesDetector(),
        ]

    potential = _potential()
    multi = _build_and_reduce(potential, list(ENERGIES), detectors(), lazy, **kwargs)
    singles = [
        _build_and_reduce(potential, energy, detectors(), lazy, **kwargs)
        for energy in ENERGIES
    ]

    assert len(multi) == len(detectors())
    for j, measurement in enumerate(multi):
        names = [type(a).__name__ for a in measurement.axes_metadata]
        axis = names.index("EnergyAxis")
        assert measurement.axes_metadata[axis].values == ENERGIES

        for i, single in enumerate(singles):
            member = measurement[(slice(None),) * axis + (i,)]
            if member.shape == single[j].shape:
                a, b = member.array, single[j].array
            else:
                # a max_angle in mrad crops to the highest energy's pixel count
                a, b = _same_pixels(member, single[j])
            _close(a, b, atol=1e-5 if lazy else 1e-10)


@pytest.mark.parametrize("lazy", [False, True])
def test_multi_energy_scan_with_a_ctf_uses_each_energys_wavelength(lazy):
    """Every energy of a multi-energy scan matched the same CTF object to its own
    energy in place, so a lazy scan computed every energy with the last energy's
    wavelength (the lowest energy was 15 % off its own run)."""

    def scan(energy):
        s_matrix = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=20
        )
        ctf = abtem.CTF(defocus=50.0, semiangle_cutoff=20)
        out = s_matrix.scan(
            scan=_scan(potential),
            detectors=abtem.AnnularDetector(30, 60),
            ctf=ctf,
            lazy=lazy,
        )
        return out.compute(progress_bar=False) if lazy else out

    potential = _potential()
    multi = scan(list(ENERGIES))
    for i, energy in enumerate(ENERGIES):
        _close(multi[i].array, scan(energy).array, atol=1e-5 if lazy else 1e-10)


def _frozen_phonon_potential(ensemble_mean=True):
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    frozen_phonons = abtem.FrozenPhonons(
        atoms, num_configs=2, sigmas=0.1, seed=1, ensemble_mean=ensemble_mean
    )
    return abtem.Potential(frozen_phonons, sampling=0.15, slice_thickness=2)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(interpolation=2)],
    ids=["interpolation_1", "interpolation_2"],
)
def test_multi_energy_frozen_phonon_build_reduce_equals_single_energy_runs(
    kwargs, lazy
):
    """A multi-energy `SMatrix.build()` over a frozen-phonon potential crashed:
    it took the wave-vector axis to be the first, where the frozen-phonon axis
    is. Eagerly the zero-padding to the union of the wave vectors failed with a
    shape mismatch, lazily the union was taken from the wrong energy and the
    wave-vector lookup raised a KeyError."""

    def detectors():
        return [abtem.AnnularDetector(30, 60), abtem.PixelatedDetector(max_angle=60)]

    potential = _frozen_phonon_potential()
    multi = _build_and_reduce(potential, list(ENERGIES), detectors(), lazy, **kwargs)

    for j, measurement in enumerate(multi):
        names = [type(a).__name__ for a in measurement.axes_metadata]
        assert names[0] == "EnergyAxis"
        assert measurement.axes_metadata[0].values == ENERGIES

    for i, energy in enumerate(ENERGIES):
        # the same seed gives the same configurations
        single = _build_and_reduce(
            _frozen_phonon_potential(), energy, detectors(), lazy, **kwargs
        )
        for j, measurement in enumerate(multi):
            member = measurement[i]
            if member.shape == single[j].shape:
                a, b = member.array, single[j].array
            else:
                # a max_angle in mrad crops to the highest energy's pixel count
                a, b = _same_pixels(member, single[j])
            _close(a, b, atol=1e-5 if lazy else 1e-10)


def test_frozen_phonon_build_reduce_follows_ensemble_mean():
    """An eager `SMatrixArray.reduce` returned the frozen-phonon axis unreduced,
    while a lazy one averaged it, as `SMatrix.scan` and multislice do. Both
    now average the configurations when the frozen phonons ask for the
    ensemble mean (the default), and keep them otherwise."""
    energy = ENERGIES[0]

    def axis_names(measurement):
        return [type(a).__name__ for a in measurement.axes_metadata]

    for ensemble_mean in (True, False):
        potential = _frozen_phonon_potential(ensemble_mean)
        detector = abtem.AnnularDetector(30, 60)
        eager = _build_and_reduce(potential, energy, detector)
        lazy = _build_and_reduce(potential, energy, detector, lazy=True)
        prism = _prism(potential, energy, detector)
        probe = abtem.Probe(energy=energy, semiangle_cutoff=20)
        multislice = probe.scan(
            potential, scan=_scan(potential), detectors=detector, lazy=False
        )

        expected = ["RealSpaceAxis", "RealSpaceAxis"]
        if not ensemble_mean:
            expected = ["FrozenPhononsAxis"] + expected
        for measurement in (eager, lazy, prism, multislice):
            assert axis_names(measurement) == expected

        # lazy blocks are computed in the default single precision
        _close(lazy.array, eager.array, atol=1e-5)
        _close(eager.array, prism.array, atol=1e-10)
        # at interpolation 1 the PRISM reduction equals multislice
        _close(eager.array, multislice.array, atol=1e-5)


def test_multi_energy_frozen_phonon_build_reduce_averages_the_configurations():
    """Each energy of an eager multi-energy reduction averages the
    configurations, as a separate single-energy `SMatrix.scan` of the same
    frozen phonons does. (A lazy reduction already averaged them.)"""
    potential = _frozen_phonon_potential()
    detector = abtem.AnnularDetector(30, 60)
    multi = _build_and_reduce(potential, list(ENERGIES), detector)
    assert [type(a).__name__ for a in multi.axes_metadata] == [
        "EnergyAxis",
        "RealSpaceAxis",
        "RealSpaceAxis",
    ]
    for i, energy in enumerate(ENERGIES):
        single = _prism(_frozen_phonon_potential(), energy, detector)
        _close(multi[i].array, single.array, atol=1e-10)


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
        _close(np.take(asnumpy(result.array), i, axis=axis), single.array, atol=1e-10)


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
            np.take(asnumpy(pixelated.array), i, axis=axis),
            np.take(asnumpy(annular.array), i, axis=annular_axis),
            atol=1e-6,
        )


@devices
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "name, args", [("polar_binning", (4, 4)), ("radial_binning", ())]
)
def test_default_outer_is_the_same_for_every_energy(name, args, lazy, device):
    # each energy's own maximum angle would give radial axes that cannot stack, or
    # that are labelled with the first energy's; the ensemble's, the highest
    # energy's, is reached by every energy
    potential = _potential(device)
    multi = _exit_waves(potential, list(ENERGIES)).diffraction_patterns(
        max_angle="full"
    )
    shared = min(multi.max_angles)
    if lazy:
        multi = multi.lazy()
    result = getattr(multi, name)(*args)
    result = result.compute(progress_bar=False) if lazy else result

    axis = [type(a).__name__ for a in result.axes_metadata].index("EnergyAxis")
    for i, energy in enumerate(ENERGIES):
        single = getattr(
            _exit_waves(potential, energy).diffraction_patterns(max_angle="full"),
            name,
        )(*args, outer=shared)
        _close(np.take(asnumpy(result.array), i, axis=axis), single.array, atol=1e-10)
        assert result.radial_sampling == pytest.approx(single.radial_sampling)
