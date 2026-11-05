"""A probe with several energies, through multislice and a detector.

Each energy of a multi-energy result must equal a separate run at that energy alone.
The sizes all differ (3 energies, 2 frozen-phonon configurations, a 4 x 7 scan, 5 exit
planes), so a result whose axes are mislabelled or stacked in the wrong place shows up
as a wrong shape or wrong values rather than passing by coincidence. One test
deliberately uses 3 configurations or 3 exit planes against the 3 energies: with equal
sizes a mislabelled axis keeps the shape and passes silently unless the values are
compared.
"""

import ase.build
import dask
import numpy as np
import pytest
from utils import devices

import abtem
from abtem.core.axes import EnergyAxis
from abtem.core.backend import asnumpy

ENERGIES = [50e3, 60e3, 70e3]

DETECTORS = {
    "annular": lambda: abtem.AnnularDetector(30, 100),
    "annular-auto": lambda: abtem.AnnularDetector(30),
    "annular-offset": lambda: abtem.AnnularDetector(30, 100, offset=(5.0, -3.0)),
    "flexible": lambda: abtem.FlexibleAnnularDetector(step_size=10, inner=10, outer=80),
    "segmented": lambda: abtem.SegmentedDetector(
        nbins_radial=2, nbins_azimuthal=3, inner=20, outer=80
    ),
    "pixelated": lambda: abtem.PixelatedDetector(max_angle=60),
    "pixelated-full": lambda: abtem.PixelatedDetector(max_angle=None),
    "pixelated-resampled": lambda: abtem.PixelatedDetector(
        max_angle=60, resample="uniform"
    ),
    "spectral-slit": lambda: abtem.SpectralSlitDetector(width=20, q_max=60),
    "spectral-annular": lambda: abtem.SpectralAnnularDetector(
        outer=15, q_min=5, q_max=60, angle=30.0
    ),
    "waves": lambda: abtem.WavesDetector(),
    # one detector reorders the scan axes, the other does not
    "annular+waves": lambda: [abtem.AnnularDetector(30, 100), abtem.WavesDetector()],
}

_references = {}


def _potential(frozen_phonons, device, exit_planes=None, num_configs=2):
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    if frozen_phonons:
        atoms = abtem.FrozenPhonons(atoms, num_configs=num_configs, sigmas=0.08, seed=1)
    return abtem.Potential(
        atoms, sampling=0.1, slice_thickness=2, exit_planes=exit_planes, device=device
    )


def _scan(kind, potential):
    if kind == "gridscan":
        return abtem.GridScan(
            (0, 0), (1, 1), gpts=(4, 7), fractional=True, potential=potential
        )
    if kind == "linescan":
        return abtem.LineScan(start=(0, 0), end=(3, 5), gpts=4)
    if kind == "customscan":
        return abtem.CustomScan(
            np.array([[0.5, 0.5], [1.0, 2.0], [2.0, 1.5], [3.0, 3.0]])
        )
    return None


def _run(energy, detector, frozen_phonons, scan, lazy, device, exit_planes=None):
    potential = _potential(frozen_phonons, device, exit_planes)
    probe = abtem.Probe(energy=energy, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    scan = _scan(scan, potential)
    detectors = DETECTORS[detector]()
    with dask.config.set(scheduler="synchronous"):
        if scan is None:
            result = probe.multislice(potential, detectors=detectors, lazy=lazy)
        else:
            result = probe.scan(potential, scan=scan, detectors=detectors, lazy=lazy)
        if lazy:
            result = result.compute(progress_bar=False)
    return result


def _reference(energy, detector, frozen_phonons, scan, device, exit_planes=None):
    """The arrays of a single-energy run, one per detector."""
    key = (energy, detector, frozen_phonons, scan, device, exit_planes)
    if key not in _references:
        results = _run(
            energy, detector, frozen_phonons, scan, False, device, exit_planes
        )
        if not isinstance(results, list):
            results = [results]
        _references[key] = [asnumpy(result.array) for result in results]
    return _references[key]


_exit_waves_cache = {}


def _exit_waves(energy, device):
    """The eager exit waves of a GridScan at `energy` (one or several), as `_run`
    computes them. They are the input of most detector tests here, so each is
    computed once per worker; every call returns a copy, so a test that changes
    the waves it gets does not change those of the next."""
    key = (tuple(np.atleast_1d(energy)), np.ndim(energy), device)
    if key not in _exit_waves_cache:
        _exit_waves_cache[key] = _run(energy, "waves", False, "gridscan", False, device)
    return _exit_waves_cache[key].copy()


def _centre_crop(array, gpts):
    """The central `gpts` of fftshifted diffraction patterns."""
    starts = [n // 2 - m // 2 for n, m in zip(array.shape[-2:], gpts)]
    return array[..., starts[0] : starts[0] + gpts[0], starts[1] : starts[1] + gpts[1]]


def _assert_members_match_single_energy_runs(
    results, detector, frozen_phonons, scan, device, exit_planes=None, energies=ENERGIES
):
    detectors = DETECTORS[detector]()
    if not isinstance(results, list):
        results = [results]
    assert len(results) == (len(detectors) if isinstance(detectors, list) else 1)

    for k, result in enumerate(results):
        kinds = [type(axis) for axis in result.axes_metadata]
        assert kinds.count(EnergyAxis) == 1
        energy_axis = kinds.index(EnergyAxis)
        assert list(result.axes_metadata[energy_axis].values) == energies

        for i, energy in enumerate(energies):
            member = np.take(asnumpy(result.array), i, axis=energy_axis)
            args = (frozen_phonons, scan, device, exit_planes)
            if detector == "pixelated":
                # Every energy is cropped to the pixel count of the highest energy,
                # as Waves.diffraction_patterns crops a whole ensemble.
                gpts = _reference(max(energies), detector, *args)[k].shape[-2:]
                assert member.shape[-2:] == gpts
                full = _reference(energy, "pixelated-full", *args)[k]
                reference = _centre_crop(full, gpts)
            else:
                reference = _reference(energy, detector, *args)[k]
            assert member.shape == reference.shape
            np.testing.assert_allclose(
                member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
            )


def _assert_each_energy_matches_a_single_energy_run(
    detector, frozen_phonons, scan, lazy, device, exit_planes=None, energies=ENERGIES
):
    results = _run(energies, detector, frozen_phonons, scan, lazy, device, exit_planes)
    _assert_members_match_single_energy_runs(
        results, detector, frozen_phonons, scan, device, exit_planes, energies
    )


def _lazy(waves, chunks):
    """`waves` as a dask array, with `chunks` blocks along the energy axis."""
    energy_axis = [type(axis) for axis in waves.axes_metadata].index(EnergyAxis)
    return waves.ensure_lazy(
        chunks=tuple(
            chunks if i == energy_axis else -1 for i in range(len(waves.shape))
        )
    )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("scan", ["gridscan", None], ids=["gridscan", "no-scan"])
@pytest.mark.parametrize(
    "frozen_phonons", [False, True], ids=["static", "frozen-phonons"]
)
@pytest.mark.parametrize("detector", ["annular", "waves"])
@devices
def test_each_energy_matches_a_single_energy_run(
    detector, frozen_phonons, scan, lazy, device
):
    """Covers the energy axis being stacked in front of the potential's ensemble axes
    (eager, frozen phonons), a detector's reordering of the scan axes not being shifted
    past them (frozen phonons with a GridScan), and a lazy block returning its declared
    rather than its natural axis order (lazy GridScan)."""
    _assert_each_energy_matches_a_single_energy_run(
        detector, frozen_phonons, scan, lazy, device
    )


@pytest.mark.parametrize(
    "frozen_phonons", [False, True], ids=["static", "frozen-phonons"]
)
@pytest.mark.parametrize("detector", ["flexible", "segmented"])
@devices
def test_binned_detectors_with_a_multi_energy_scan(detector, frozen_phonons, device):
    _assert_each_energy_matches_a_single_energy_run(
        detector, frozen_phonons, "gridscan", False, device
    )


@pytest.mark.parametrize(
    "scan, frozen_phonons, lazy",
    [
        ("linescan", False, True),
        ("linescan", True, False),
        ("customscan", False, False),
        ("customscan", True, False),
    ],
)
@devices
def test_other_scan_types_with_a_multi_energy_probe(scan, frozen_phonons, lazy, device):
    _assert_each_energy_matches_a_single_energy_run(
        "annular", frozen_phonons, scan, lazy, device
    )


@pytest.mark.parametrize("detector, scan", [("annular", "gridscan"), ("waves", None)])
@devices
def test_exit_planes_with_a_multi_energy_probe(detector, scan, device):
    """The exit-plane axis is one of the leading axes the energy axis must be stacked
    behind."""
    _assert_each_energy_matches_a_single_energy_run(
        detector, True, scan, False, device, exit_planes=1
    )


@pytest.mark.parametrize(
    "frozen_phonons, exit_planes",
    [(True, None), (False, 2)],
    ids=["3-frozen-phonons", "3-exit-planes"],
)
@devices
def test_a_leading_axis_as_long_as_the_energy_axis_keeps_its_values(
    frozen_phonons, exit_planes, device
):
    """With as many configurations or exit planes as energies, an energy axis stacked
    in the wrong place keeps the shape, so only the values tell."""
    potential = _potential(frozen_phonons, device, exit_planes, num_configs=3)
    num_leading = np.prod(potential.ensemble_shape) * potential.num_exit_planes
    assert num_leading == len(ENERGIES)
    probe = abtem.Probe(energy=ENERGIES, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    result = probe.multislice(potential, lazy=False)

    energy_axis = [type(axis) for axis in result.axes_metadata].index(EnergyAxis)
    for i, energy in enumerate(ENERGIES):
        single = abtem.Probe(energy=energy, semiangle_cutoff=20, device=device)
        single.grid.match(potential)
        reference = asnumpy(single.multislice(potential, lazy=False).array)
        member = np.take(asnumpy(result.array), i, axis=energy_axis)
        assert member.shape == reference.shape
        np.testing.assert_allclose(
            member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
        )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize(
    "frozen_phonons", [False, True], ids=["static", "frozen-phonons"]
)
@devices
def test_several_detectors_with_a_multi_energy_scan(frozen_phonons, lazy, device):
    """Each output must be paired with its own detector's axis order."""
    _assert_each_energy_matches_a_single_energy_run(
        "annular+waves", frozen_phonons, "gridscan", lazy, device
    )


UNORDERED = [60e3, 70e3, 50e3]


# The order of the energies matters only for the crop to the highest energy's pixel
# count, so it is varied with the crop; the frozen phonons change only the axis
# order, which `test_each_energy_matches_a_single_energy_run` covers in full.
@pytest.mark.parametrize(
    "frozen_phonons, lazy, energies",
    [
        (False, True, UNORDERED),
        (True, False, ENERGIES),
        (True, True, UNORDERED),
    ],
    ids=[
        "static-lazy-unordered",
        "frozen-phonons-eager",
        "frozen-phonons-lazy-unordered",
    ],
)
@devices
def test_pixelated_detector_with_a_multi_energy_scan(
    frozen_phonons, lazy, energies, device
):
    _assert_each_energy_matches_a_single_energy_run(
        "pixelated", frozen_phonons, "gridscan", lazy, device, energies=energies
    )


# Every detector is applied eagerly. The lazy blocks of 3 and of 2 and 1 energies
# are taken by one detector of each code path: one that splits the ensemble into
# energies and reorders the scan axes (annular), the auto-sized annular detector
# (sized per energy inside the blocks), a radial-binned detector that keeps the axis
# order (flexible), the spectral slit (not a radial detector), and the two that
# detect the whole ensemble at once (pixelated, cropped to the highest energy, and
# waves). The detectors that split the ensemble share one
# `_calculate_new_array_per_energy`, so the offset, segmented and spectral-annular
# ones add nothing lazily. Unordered energies
# go to the crop (pixelated) and the per-energy sizing (annular-auto).
_EXIT_WAVE_DETECTION_CASES = (
    [
        (detector, None, ENERGIES)
        for detector in (
            "annular",
            "annular-auto",
            "annular-offset",
            "flexible",
            "segmented",
            "pixelated",
            "spectral-slit",
            "spectral-annular",
            "waves",
        )
    ]
    + [
        (detector, chunks, ENERGIES)
        for detector in ("annular", "pixelated", "waves")
        for chunks in (3, (2, 1))
    ]
    + [
        (detector, (2, 1), ENERGIES)
        for detector in ("annular-auto", "flexible", "spectral-slit")
    ]
    + [("pixelated", chunks, UNORDERED) for chunks in (None, 3, (2, 1))]
    + [("annular-auto", (2, 1), UNORDERED)]
)


@pytest.mark.parametrize(
    "detector, chunks, energies",
    _EXIT_WAVE_DETECTION_CASES,
    ids=[
        f"{detector}-{chunks}-{'ascending' if energies == ENERGIES else 'unordered'}"
        for detector, chunks, energies in _EXIT_WAVE_DETECTION_CASES
    ],
)
@devices
def test_detect_on_multi_energy_exit_waves(detector, chunks, energies, device):
    """A detector applied to precomputed exit waves detects each energy with its own
    angular sampling, eagerly (`chunks` None) and lazily, with every energy in one
    block or split into blocks of 2 and 1."""
    waves = _exit_waves(energies, device)
    if chunks is not None:
        waves = _lazy(waves, chunks)

    with dask.config.set(scheduler="synchronous"):
        detected = DETECTORS[detector]().detect(waves)
        if chunks is not None:
            detected = detected.compute(progress_bar=False)

    _assert_members_match_single_energy_runs(
        detected, detector, False, "gridscan", device, energies=energies
    )


# The crop is fixed by `_match_ensemble` on the whole ensemble before a lazy scan or
# detection is split into blocks, so one lazy and one eager entry cover both.
@pytest.mark.parametrize(
    "entry, lazy", [("scan", True), ("detect", False)], ids=["scan-lazy", "detect"]
)
@devices
def test_resampled_pixelated_detector_with_multi_energy_waves(entry, lazy, device):
    """With `resample`, every energy is cropped to the pixel count of the highest
    energy before it is resampled."""
    if entry == "scan":
        result = _run(ENERGIES, "pixelated-resampled", False, "gridscan", lazy, device)
    else:
        waves = _exit_waves(ENERGIES, device)
        if lazy:
            waves = _lazy(waves, 3)
        with dask.config.set(scheduler="synchronous"):
            result = DETECTORS["pixelated-resampled"]().detect(waves)
            if lazy:
                result = result.compute(progress_bar=False)

    energy_axis = [type(axis) for axis in result.axes_metadata].index(EnergyAxis)
    gpts = _reference(max(ENERGIES), "pixelated", False, "gridscan", device)[0]
    gpts = gpts.shape[-2:]
    for i, energy in enumerate(ENERGIES):
        single = _exit_waves(energy, device)
        patterns = single.diffraction_patterns(max_angle="full", parity="same")
        cropped = abtem.DiffractionPatterns(
            _centre_crop(patterns.array, gpts),
            sampling=patterns.sampling,
            fftshift=True,
            ensemble_axes_metadata=patterns.ensemble_axes_metadata,
            metadata=patterns.metadata,
        )
        reference = asnumpy(cropped.interpolate("uniform").array)
        member = np.take(asnumpy(result.array), i, axis=energy_axis)
        assert member.shape == reference.shape
        np.testing.assert_allclose(
            member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
        )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("scan", ["gridscan", None], ids=["gridscan", "no-scan"])
@devices
def test_auto_sized_annular_detector_with_a_multi_energy_probe(scan, lazy, device):
    """Without an outer angle, each energy is integrated up to its own cutoff, as a
    separate run of that energy is."""
    _assert_each_energy_matches_a_single_energy_run(
        "annular-auto", False, scan, lazy, device
    )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
def test_flexible_annular_detector_refuses_an_auto_sized_outer_angle(lazy):
    """Its radial bins must be shared by every energy, so they cannot follow each
    energy's cutoff."""
    potential = _potential(False, "cpu")
    probe = abtem.Probe(energy=ENERGIES, semiangle_cutoff=20)
    probe.grid.match(potential)
    with pytest.raises(RuntimeError, match="cannot auto-size its outer angle"):
        result = probe.scan(
            potential,
            scan=_scan("gridscan", potential),
            detectors=abtem.FlexibleAnnularDetector(inner=10),
            lazy=lazy,
        )
        if lazy:
            result.compute(progress_bar=False)


def _stacked_exit_waves(energies, device):
    """Single-energy exit waves stacked along an energy axis behind the scan axes."""
    singles = [_exit_waves(energy, device) for energy in energies]
    return abtem.stack(singles, EnergyAxis(values=tuple(energies)), axis=2)


# Stacked waves differ from a multi-energy run only in where their energies come
# from, which matters to a detector that splits the ensemble into energies
# (annular) and to the crop to the highest energy (pixelated); the other detectors
# of either kind read the energies the same way.
@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@pytest.mark.parametrize("detector", ["annular", "pixelated"])
@devices
def test_detect_on_stacked_single_energy_waves(detector, lazy, device):
    """Waves stacked along an energy axis carry each member's energy on the axis,
    not the first member's as a scalar energy."""
    waves = _stacked_exit_waves(ENERGIES, device)
    assert waves.accelerator.energy is None
    assert waves.metadata.get("energy") is None
    if lazy:
        waves = _lazy(waves, 3)

    with dask.config.set(scheduler="synchronous"):
        detected = DETECTORS[detector]().detect(waves)
        if lazy:
            detected = detected.compute(progress_bar=False)

    _assert_members_match_single_energy_runs(
        detected, detector, False, "gridscan", device
    )


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy", "eager"])
@devices
def test_multislice_with_a_waves_detector_on_stacked_exit_waves(lazy, device):
    """Waves stacked along an energy axis are propagated, each energy with its own
    wavelength, and returned as waves."""
    energies = [70e3, 50e3, 60e3]
    potential = _potential(False, device)
    waves = _stacked_exit_waves(energies, device)
    singles = [_exit_waves(energy, device) for energy in energies]
    assert waves.accelerator.energy is None
    if lazy:
        waves = _lazy(waves, 3)

    with dask.config.set(scheduler="synchronous"):
        result = waves.multislice(potential, detectors=abtem.WavesDetector())
        if lazy:
            result = result.compute(progress_bar=False)
        references = [
            single.multislice(potential, detectors=abtem.WavesDetector())
            for single in singles
        ]

    assert list(result.ensemble_axes_metadata[2].values) == energies
    for i, reference in enumerate(references):
        member = np.take(asnumpy(result.array), i, axis=2)
        reference = asnumpy(reference.array)
        assert member.shape == reference.shape
        np.testing.assert_allclose(
            member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
        )


@devices
def test_detect_on_multi_energy_prism_exit_waves(device):
    """An S-matrix with several energies stacks its exit waves along an energy
    axis."""
    potential = _potential(False, device)

    def exit_waves(energy):
        s_matrix = abtem.SMatrix(
            potential=potential, energy=energy, semiangle_cutoff=20, interpolation=1
        )
        return s_matrix.scan(
            scan=abtem.GridScan(
                (0, 0), (1, 1), gpts=(3, 4), fractional=True, potential=potential
            ),
            detectors=abtem.WavesDetector(),
            lazy=False,
        )

    multi_energy = exit_waves(ENERGIES)
    for detector in (abtem.AnnularDetector(30, 100), abtem.WavesDetector()):
        detected = detector.detect(multi_energy)
        energy_axis = [type(axis) for axis in detected.axes_metadata].index(EnergyAxis)
        for i, energy in enumerate(ENERGIES):
            reference = asnumpy(detector.detect(exit_waves(energy)).array)
            member = np.take(asnumpy(detected.array), i, axis=energy_axis)
            np.testing.assert_allclose(
                member, reference, rtol=0, atol=1e-5 * np.abs(reference).max()
            )


def test_the_users_detectors_are_not_changed():
    """The ensemble's crop is fixed on a copy of the detector."""
    detector = abtem.PixelatedDetector(max_angle=60)
    waves = _exit_waves(ENERGIES, "cpu")
    detector.detect(waves)
    potential = _potential(False, "cpu")
    probe = abtem.Probe(energy=ENERGIES, semiangle_cutoff=20)
    probe.grid.match(potential)
    detectors = [detector, abtem.AnnularDetector(30, 100)]
    probe.scan(
        potential, scan=_scan("gridscan", potential), detectors=detectors
    ).compute(progress_bar=False)

    assert detector._ensemble_gpts is None
    assert detectors[0] is detector and len(detectors) == 2
