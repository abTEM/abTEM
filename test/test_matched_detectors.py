"""Detectors sized for waves (`_matched`).

An automatic outer angle is sized from the waves a detector detects, on a copy:
the detector the caller holds is never written to.
"""

import ase.build
import numpy as np
import pytest
from utils import devices

import abtem
from abtem.detectors import _AbstractRadialDetector
from abtem.waves import Waves

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


def _plane_waves(device="cpu"):
    xp = np
    if device != "cpu":
        from abtem.core.backend import get_array_module

        xp = get_array_module(device)
    return Waves(xp.ones((64, 64), "complex128"), energy=100e3, sampling=0.1)


def _close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol * np.abs(b).max())


# --- _matched -----------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: abtem.AnnularDetector(inner=50),
        lambda: abtem.FlexibleAnnularDetector(),
    ],
    ids=["annular", "flexible_annular"],
)
@devices
def test_matched_is_a_copy_and_leaves_the_detector_unsized(make, device):
    waves = _plane_waves(device)
    detector = make()
    matched = detector._matched(waves)

    assert matched is not detector
    assert detector.outer is None and not detector._outer_is_explicit
    assert matched.outer == min(waves.cutoff_angles)
    # sized, but not pinned: the next waves size it again
    assert not matched._outer_is_explicit


def test_matched_returns_an_explicit_detector_itself():
    detector = abtem.FlexibleAnnularDetector(outer=100)
    assert detector._matched(_plane_waves()) is detector


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(inner=50),
        abtem.FlexibleAnnularDetector(),
        abtem.SegmentedDetector(30, 4, 30, None),
    ],
    ids=["annular", "flexible_annular", "segmented"],
)
def test_show_regions_and_detect_do_not_write_the_outer_angle(detector):
    waves = _plane_waves()
    detector.show(waves)
    detector.get_detector_regions(waves)
    detector.detect(waves)
    assert detector.outer is None


def test_matched_is_sized_again_by_other_waves():
    detector = abtem.AnnularDetector(20)
    first = detector._matched(_plane_waves())
    other = Waves(np.ones((64, 64), "complex128"), energy=300e3, sampling=0.1)
    second = first._matched(other)
    assert second.outer == min(other.cutoff_angles) != first.outer


def test_annular_detector_integrates_to_the_outer_angle_that_show_draws():
    # Plane waves diffract into a delta at k=0, which no annulus contains; the
    # exit waves of a potential scatter into the annulus.
    waves = abtem.PlaneWave(energy=60e3, sampling=0.15).multislice(
        _potential(), lazy=False
    )
    detector = abtem.AnnularDetector(20)
    outer = detector.angular_limits(waves)[1]
    explicit = abtem.AnnularDetector(20, outer).detect(waves).array

    assert explicit > 0
    _close(detector.detect(waves).array, explicit, atol=1e-10)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
@pytest.mark.parametrize(
    "make",
    [
        lambda: abtem.FlexibleAnnularDetector(),
        lambda: abtem.SegmentedDetector(2, 4, 30, None),
    ],
    ids=["flexible_annular", "segmented"],
)
def test_probe_scan_does_not_write_the_outer_angle(make, lazy):
    potential = _potential()
    detector = make()
    measurement = abtem.Probe(energy=60e3, semiangle_cutoff=20).scan(
        scan=_scan(potential), detectors=detector, potential=potential, lazy=lazy
    )
    assert detector.outer is None
    if lazy:
        measurement.compute()
        assert detector.outer is None


def test_prism_match_does_not_pin_the_first_outer_angle(monkeypatch):
    # SMatrix.reduce and SMatrixArray.reduce both match: the reduction must
    # integrate to the second, the cutoff of its own waves.
    seen = []
    original = _AbstractRadialDetector._matched

    def traced(self, waves):
        matched = original(self, waves)
        seen.append(matched)
        return matched

    monkeypatch.setattr(_AbstractRadialDetector, "_matched", traced)

    potential = _potential()
    detector = abtem.AnnularDetector(20)
    auto = abtem.SMatrix(potential=potential, energy=60e3, semiangle_cutoff=20).scan(
        scan=_scan(potential), detectors=detector, lazy=False
    )
    assert detector.outer is None
    assert all(not m._outer_is_explicit for m in seen)

    outer = min(
        abtem.SMatrix(potential=potential, energy=60e3, semiangle_cutoff=20)
        .dummy_probes()
        .cutoff_angles
    )
    pinned = abtem.SMatrix(
        potential=potential, energy=60e3, semiangle_cutoff=20
    ).scan(scan=_scan(potential), detectors=abtem.AnnularDetector(20, outer), lazy=False)
    np.testing.assert_array_equal(auto.array, pinned.array)
