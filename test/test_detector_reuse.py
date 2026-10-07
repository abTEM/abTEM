"""A detector object passed to a PRISM scan must not be modified by it.

``AnnularDetector(inner)`` without an outer angle integrates up to the antialias
cutoff of the waves it detects. PRISM detects on downsampled waves, whose cutoff is
well below the one of a multislice probe on the same potential. A detector reused
after a PRISM run must therefore give the same result as a new one.
"""

import ase.build
import numpy as np
import pytest

import abtem

ENERGIES = (50e3, 60e3, 70e3)

# the 43 x 74 grid is not divisible by an interpolation factor of 2, and the 3 x 5
# scan is not a whole number of pixels on the upsampled reduction
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The interpolation factor does not exactly divide:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The scan step is not a whole number of pixels:UserWarning"
    ),
]


@pytest.fixture(autouse=True)
def _cpu_float64():
    with abtem.config.set({"device": "cpu", "precision": "float64"}):
        yield


def _potential():
    atoms = ase.build.mx2("WSe2", vacuum=2) * (2, 1, 1)
    return abtem.Potential(atoms, sampling=0.15, slice_thickness=2)


def _scan(potential):
    # 3 x 5 positions: the scan axes differ in size from each other and from
    # the three energies
    return abtem.GridScan(
        (0, 0), (1, 1), gpts=(3, 5), fractional=True, potential=potential
    )


def _prism(potential, energy, detector, lazy=False, **kwargs):
    s_matrix = abtem.SMatrix(
        potential=potential, energy=energy, semiangle_cutoff=20, **kwargs
    )
    out = s_matrix.scan(scan=_scan(potential), detectors=detector, lazy=lazy)
    return out.compute(progress_bar=False) if lazy else out


def _probe(potential, energy, detector, lazy):
    out = abtem.Probe(energy=energy, semiangle_cutoff=20).scan(
        potential, scan=_scan(potential), detectors=detector, lazy=lazy
    )
    return out.compute(progress_bar=False) if lazy else out


_MULTI_ENERGY_AUTO_OUTER_ERRORS = (
    "cannot auto-size its outer angle for a multi-energy ensemble",
    "number of values for ordinal axis",
)


def _assert_equal_to_fresh(reused, fresh):
    np.testing.assert_allclose(
        reused, fresh, rtol=0, atol=1e-12 * np.abs(fresh).max()
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("interpolation", [1, 2])
@pytest.mark.parametrize(
    "make",
    [
        lambda: abtem.AnnularDetector(20),
        lambda: abtem.FlexibleAnnularDetector(step_size=5),
    ],
    ids=["annular", "flexible_annular"],
)
def test_prism_scan_leaves_the_users_detector_unmatched(make, interpolation, lazy):
    detector = make()
    _prism(_potential(), 60e3, detector, lazy=lazy, interpolation=interpolation)

    assert detector.outer is None
    assert not detector._outer_is_explicit


@pytest.mark.parametrize("upsample", [False, True])
def test_s_matrix_array_reduce_leaves_the_users_detector_unmatched(upsample):
    potential = _potential()
    kwargs = dict(upsample=True, blend_angle="auto") if upsample else {}
    s_matrix = abtem.SMatrix(
        potential=potential,
        energy=60e3,
        semiangle_cutoff=20,
        interpolation=2,
        **kwargs,
    ).build(lazy=False)
    detector = abtem.AnnularDetector(20)
    reduced = s_matrix.reduce(scan=_scan(potential), detectors=detector)

    assert detector.outer is None
    assert not detector._outer_is_explicit

    # the detector's own result is that of a new detector
    fresh = s_matrix.reduce(scan=_scan(potential), detectors=abtem.AnnularDetector(20))
    _assert_equal_to_fresh(reduced.array, fresh.array)


def test_an_explicit_outer_is_not_changed_by_a_prism_scan():
    detector = abtem.AnnularDetector(20, 30)
    _prism(_potential(), 60e3, detector)

    assert detector.outer == 30
    assert detector._outer_is_explicit


@pytest.mark.parametrize("lazy", [False, True])
def test_annular_detector_reused_after_prism_matches_a_fresh_one(lazy):
    potential = _potential()
    detector = abtem.AnnularDetector(20)
    _prism(potential, 60e3, detector)

    reused = _probe(potential, 50e3, detector, lazy).array
    fresh = _probe(potential, 50e3, abtem.AnnularDetector(20), lazy).array
    _assert_equal_to_fresh(reused, fresh)


def test_annular_detector_reused_after_prism_detects_like_a_fresh_one():
    potential = _potential()
    detector = abtem.AnnularDetector(20)
    _prism(potential, 60e3, detector)

    waves = abtem.Probe(energy=50e3, semiangle_cutoff=20).multislice(
        potential, scan=_scan(potential), lazy=False
    )
    reused = detector.detect(waves).array
    fresh = abtem.AnnularDetector(20).detect(waves).array
    _assert_equal_to_fresh(reused, fresh)


def test_prism_multi_energy_lazy_equals_eager_for_an_auto_outer():
    potential = _potential()
    eager = _prism(potential, list(ENERGIES), abtem.AnnularDetector(20)).array
    lazy = _prism(
        potential, list(ENERGIES), abtem.AnnularDetector(20), lazy=True
    ).array

    np.testing.assert_allclose(lazy, eager, rtol=0, atol=1e-6 * np.abs(eager).max())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(interpolation=1),
        dict(interpolation=2),
        dict(interpolation=2, upsample=True, blend_angle="auto"),
    ],
    ids=["interpolation_1", "interpolation_2", "interpolation_2_upsampled"],
)
def test_prism_auto_outer_is_the_dummy_probes_cutoff(kwargs):
    potential = _potential()
    s_matrix = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=20, **kwargs
    )
    outer = min(s_matrix.dummy_probes().cutoff_angles)

    auto = _prism(potential, 60e3, abtem.AnnularDetector(20), **kwargs).array
    pinned = _prism(potential, 60e3, abtem.AnnularDetector(20, outer), **kwargs).array

    np.testing.assert_array_equal(auto, pinned)


@pytest.mark.parametrize("lazy", [False, True])
def test_annular_detector_reused_after_prism_matches_a_fresh_one_for_each_energy(
    lazy,
):
    # An auto-sized outer angle for a multi-energy probe scan exists only where the
    # energy-ensemble detector convention does. Elsewhere a new detector is refused
    # (eager) or fails on the energy axis (lazy), and there is no reference to
    # compare a reused one against. Any other error is a failure.
    potential = _potential()
    try:
        _probe(potential, list(ENERGIES), abtem.AnnularDetector(20), lazy)
    except RuntimeError as error:
        if not any(text in str(error) for text in _MULTI_ENERGY_AUTO_OUTER_ERRORS):
            raise
        pytest.skip(
            "multi-energy probe scans with an auto outer angle are refused (eager) "
            "or fail on the energy axis (lazy)"
        )

    detector = abtem.AnnularDetector(20)
    _prism(potential, 60e3, detector)
    reused = _probe(potential, list(ENERGIES), detector, lazy)

    axis = [type(a).__name__ for a in reused.axes_metadata].index("EnergyAxis")
    for i, energy in enumerate(ENERGIES):
        fresh = _probe(potential, energy, abtem.AnnularDetector(20), False).array
        _assert_equal_to_fresh(np.take(reused.array, i, axis=axis), fresh)
