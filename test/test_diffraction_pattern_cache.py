"""Tests for the shared diffraction-pattern FFT (abtem.core.fft:
``share_diffraction_pattern_fft`` / ``get_shared_diffraction_pattern_fft``,
and ``Waves._share_diffraction_pattern_fft``).

Motivation: every radial detector (AnnularDetector, FlexibleAnnularDetector,
SegmentedDetector, PixelatedDetector) recomputed its own 2D FFT of the wave
function array via ``Waves.diffraction_patterns``, even when several such
detectors are evaluated against the exact same array in one detection pass
(as happens once per detector, per site batch, per slice, in e.g.
``transition_potential_multislice_and_detect``). These tests check that:

1. The FFT is actually shared/computed only once per detection pass when the
   optimization applies (the performance goal).
2. Detector results are unaffected -- bit-identical (to within tiny floating-
   point accumulation-order noise) to the pre-sharing behaviour -- for
   AnnularDetector, FlexibleAnnularDetector, SegmentedDetector and
   PixelatedDetector.
3. Sharing cannot leak between different Waves objects (there is no global
   state to leak through -- the precomputed value is a plain attribute on
   one specific Waves instance), is a no-op outside an explicit
   ``share_diffraction_pattern_fft``/``_share_diffraction_pattern_fft``
   block, and -- the subtle case this module exists to guard against --
   cannot return a stale result when an in-place multislice step reuses the
   very same array object across depths while overwriting its contents.

All device-parametrised tests use ``["cpu", gpu]`` via ``test/utils.py:gpu``.
"""
import ase
import numpy as np
import pytest

import abtem
from abtem import (
    AnnularDetector,
    FlexibleAnnularDetector,
    PixelatedDetector,
    Potential,
    Probe,
    SegmentedDetector,
)
from abtem.antialias import AntialiasAperture
from abtem.core.axes import OrdinalAxis
from abtem.core.fft import (
    get_shared_diffraction_pattern_fft,
    share_diffraction_pattern_fft,
)
from abtem.inelastic.core_loss import TransitionPotentialArray
from abtem.multislice import (
    FresnelPropagator,
    conventional_multislice_step,
    transition_potential_multislice_and_detect,
)

try:
    import cupy as cp
except ImportError:
    cp = None

from utils import gpu  # noqa: E402  -- pytest.param('gpu', skipif no cupy)


class _FakeWaves:
    """Minimal stand-in for a Waves object in the low-level primitive tests
    below -- just something ``share_diffraction_pattern_fft`` can attach a
    plain attribute to (a raw numpy array can't: it has no ``__dict__``)."""


# ---------------------------------------------------------------------------
# Low-level primitive tests (no real Waves objects involved).
# ---------------------------------------------------------------------------


def test_shared_fft_is_a_noop_by_default():
    """Outside any ``share_diffraction_pattern_fft`` block,
    ``get_shared_diffraction_pattern_fft`` must call ``compute`` every time
    -- sharing must never happen implicitly."""
    waves = _FakeWaves()
    calls = []

    def compute():
        calls.append(1)
        return object()

    get_shared_diffraction_pattern_fft(waves, False, compute)
    get_shared_diffraction_pattern_fft(waves, False, compute)
    get_shared_diffraction_pattern_fft(waves, False, compute)

    assert len(calls) == 3


def test_share_block_computes_once_and_is_reused():
    """Inside a ``share_diffraction_pattern_fft`` block, repeated
    ``get_shared_diffraction_pattern_fft`` calls with the same ``normalize``
    must reuse the precomputed value (compute called once, for the
    ``share_diffraction_pattern_fft`` call itself)."""
    waves = _FakeWaves()
    calls = []

    def compute():
        calls.append(1)
        return "the-fft"

    with share_diffraction_pattern_fft(waves, False, compute):
        r1 = get_shared_diffraction_pattern_fft(waves, False, lambda: calls.append(1))
        r2 = get_shared_diffraction_pattern_fft(waves, False, lambda: calls.append(1))

    assert len(calls) == 1
    assert r1 == "the-fft" and r2 == "the-fft"


def test_share_block_does_not_leak_to_a_different_object():
    """Sharing is attached to one specific object -- a *different* object
    (even created inside the same block) must never see it."""
    waves_a = _FakeWaves()
    waves_b = _FakeWaves()
    calls = []

    with share_diffraction_pattern_fft(waves_a, False, lambda: "a-value"):
        result_a = get_shared_diffraction_pattern_fft(
            waves_a, False, lambda: calls.append("a") or "a-value"
        )
        result_b = get_shared_diffraction_pattern_fft(
            waves_b, False, lambda: calls.append("b") or "b-value"
        )

    assert result_a == "a-value"
    assert result_b == "b-value"
    assert calls == ["b"]  # waves_b always recomputes; waves_a never does


def test_share_block_does_not_share_across_different_normalize():
    """The same object with a different ``normalize`` flag must miss the
    shared value and recompute."""
    waves = _FakeWaves()
    calls = []

    with share_diffraction_pattern_fft(waves, False, lambda: "normalize=False"):
        get_shared_diffraction_pattern_fft(
            waves, False, lambda: calls.append(1) or "x"
        )
        get_shared_diffraction_pattern_fft(
            waves, True, lambda: calls.append(1) or "x"
        )

    assert len(calls) == 1  # only the normalize=True lookup recomputed


def test_share_block_clears_on_exit():
    """The precomputed value must not survive past the block: a lookup after
    the block closes must always recompute, never returning a value shared
    by an earlier, now-closed block."""
    waves = _FakeWaves()
    calls = []

    def compute():
        calls.append(1)
        return "value"

    with share_diffraction_pattern_fft(waves, False, compute):
        get_shared_diffraction_pattern_fft(waves, False, compute)

    # Outside the block: must recompute.
    get_shared_diffraction_pattern_fft(waves, False, compute)

    assert len(calls) == 2


def test_share_block_reflects_the_object_it_was_set_on_not_a_stale_copy():
    """The critical safety property: two *separate* blocks on the same
    object (with a real mutation of the underlying data source in between)
    must each produce their own value -- there is no shared/global slot for
    a later block to accidentally see an earlier block's stale entry."""
    waves = _FakeWaves()
    source = {"value": 1.0}

    with share_diffraction_pattern_fft(waves, False, lambda: source["value"]):
        first = get_shared_diffraction_pattern_fft(waves, False, lambda: None)

    # "Mutate" between blocks.
    source["value"] = 2.0

    with share_diffraction_pattern_fft(waves, False, lambda: source["value"]):
        second = get_shared_diffraction_pattern_fft(waves, False, lambda: None)

    assert first == 1.0
    assert second == 2.0


def test_nested_share_blocks_restore_outer_value():
    waves = _FakeWaves()

    with share_diffraction_pattern_fft(waves, False, lambda: "outer"):
        assert get_shared_diffraction_pattern_fft(waves, False, lambda: None) == "outer"
        with share_diffraction_pattern_fft(waves, False, lambda: "inner"):
            assert (
                get_shared_diffraction_pattern_fft(waves, False, lambda: None)
                == "inner"
            )
        # Back in the outer block: must see the outer value again, not
        # leftover state from the (now-exited) inner block.
        assert get_shared_diffraction_pattern_fft(waves, False, lambda: None) == "outer"

    # And after both blocks: back to always recomputing.
    assert get_shared_diffraction_pattern_fft(waves, False, lambda: "fresh") == "fresh"


# ---------------------------------------------------------------------------
# Detector-level correctness: sharing must not change results.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def si_atoms():
    return ase.build.bulk("Si", cubic=True)


def _make_potential(atoms, device, gpts=48, num_slices=3):
    slice_thickness = atoms.cell[2, 2] / num_slices
    return Potential(atoms, gpts=gpts, slice_thickness=slice_thickness, device=device)


def _make_transition_potential(potential, device, n_transitions=4, seed=0):
    rng = np.random.default_rng(seed)
    array = (
        rng.standard_normal((n_transitions, *potential.gpts))
        + 1j * rng.standard_normal((n_transitions, *potential.gpts))
    ).astype(np.complex64)
    if device == "gpu":
        array = cp.asarray(array)
    return TransitionPotentialArray(
        Z=14,
        array=array,
        energy=100e3,
        extent=potential.extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": 14, "n": 1, "l": 0},
    )


def _make_probe_waves(potential, device, scan_gpts=(2, 2)):
    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    scan = abtem.GridScan(
        start=(0, 0), end=(1, 1), fractional=True, potential=potential,
        endpoint=False, gpts=scan_gpts,
    )
    return probe.build(scan=scan, lazy=False)


ALL_RADIAL_DETECTORS = [
    lambda: AnnularDetector(inner=0, outer=15),
    lambda: FlexibleAnnularDetector(step_size=10),
    lambda: SegmentedDetector(inner=0, outer=30, nbins_radial=3, nbins_azimuthal=4),
    lambda: PixelatedDetector(),
]
DETECTOR_IDS = ["annular", "flexible_annular", "segmented", "pixelated"]


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("double_channel", [True, False])
def test_transition_potential_multiple_radial_detectors_match_uncached(
    si_atoms, device, double_channel
):
    """Running all four radial detector types together in one
    ``transition_potential_multislice_and_detect`` pass (which shares the FFT
    across them) must give the same result as running each detector alone
    (which never shares anything), to within tiny floating-point
    accumulation-order noise.
    """
    potential = _make_potential(si_atoms, device)
    tp = _make_transition_potential(potential, device)
    waves = _make_probe_waves(potential, device)

    detectors = [factory() for factory in ALL_RADIAL_DETECTORS]

    combined = transition_potential_multislice_and_detect(
        waves=waves.copy(),
        potential=potential,
        transition_potential=tp,
        detectors=detectors,
        sites=None,
        double_channel=double_channel,
    )

    for name, factory, combined_result in zip(DETECTOR_IDS, ALL_RADIAL_DETECTORS, combined):
        solo = transition_potential_multislice_and_detect(
            waves=waves.copy(),
            potential=potential,
            transition_potential=tp,
            detectors=[factory()],
            sites=None,
            double_channel=double_channel,
        )[0]

        combined_array = np.asarray(combined_result.array)
        solo_array = np.asarray(solo.array)
        if device == "gpu":
            combined_array = cp.asnumpy(combined_array)
            solo_array = cp.asnumpy(solo_array)

        np.testing.assert_allclose(
            combined_array,
            solo_array,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"{name} detector result changed when sharing the FFT",
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_standalone_detect_calls_never_share(si_atoms, device):
    """A plain, standalone ``detector.detect(waves)`` (not wrapped in
    ``waves._share_diffraction_pattern_fft()``) must never see sharing --
    this is the default, always-safe path used throughout the rest of the
    codebase and by any user calling ``detect`` directly."""
    potential = _make_potential(si_atoms, device)
    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    calls = []
    from abtem.waves import Waves

    original = Waves._diffraction_pattern_fft

    def counting(array, normalize):
        calls.append(1)
        return original(array, normalize)

    Waves._diffraction_pattern_fft = staticmethod(counting)
    try:
        detector = AnnularDetector(inner=0, outer=20)
        detector.detect(waves)
        detector.detect(waves)
        detector.detect(waves)
    finally:
        Waves._diffraction_pattern_fft = staticmethod(original)

    assert len(calls) == 3, (
        "standalone detect() calls on the same waves object must each "
        "recompute the FFT -- sharing must require an explicit block"
    )


# ---------------------------------------------------------------------------
# The regression this module exists to prevent: in-place mutation across
# multislice depths must never be masked by a stale shared value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_in_place_multislice_reuses_array_identity(si_atoms, device):
    """Sanity check for the hazard this design guards against:
    ``FresnelPropagator.propagate(..., in_place=True)`` (used by
    ``conventional_multislice_step``) legitimately reuses the very same
    ``waves.array`` Python object across multislice depths while
    overwriting its contents. If this ever stops being true, the other
    tests in this module would no longer be exercising the hazard they
    claim to."""
    potential = _make_potential(si_atoms, device)
    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    propagator = FresnelPropagator()
    aperture = AntialiasAperture()

    w = waves.copy()
    array_ids = set()
    for potential_slice in potential.generate_slices():
        w = conventional_multislice_step(w, potential_slice, propagator, aperture)
        array_ids.add(id(w.array))

    assert len(array_ids) == 1


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_unshared_loop_survives_in_place_mutation_across_depths(si_atoms, device):
    """A bare loop calling ``detector.detect(w)`` once per multislice depth,
    with no ``_share_diffraction_pattern_fft`` block, must give the same
    per-depth results as an uncached reference -- even though ``w.array``
    keeps the same identity across depths (see
    ``test_in_place_multislice_reuses_array_identity``). This is the
    scenario that would silently break with a cache keyed on array identity
    alone.
    """
    potential = _make_potential(si_atoms, device, gpts=48, num_slices=3)
    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    propagator = FresnelPropagator()
    aperture = AntialiasAperture()
    detector = AnnularDetector(inner=0, outer=20)

    def run_pass():
        w = waves.copy()
        results = []
        for potential_slice in potential.generate_slices():
            w = conventional_multislice_step(w, potential_slice, propagator, aperture)
            result = detector.detect(w).array
            if device == "gpu":
                result = cp.asnumpy(result)
            results.append(np.array(result, copy=True))
        return results

    reference = run_pass()
    repeated = run_pass()

    for depth, (ref, rep) in enumerate(zip(reference, repeated)):
        np.testing.assert_allclose(
            ref, rep, rtol=1e-4, atol=1e-6,
            err_msg=f"depth {depth} result changed between repeated passes",
        )

    # And the per-depth values must actually differ from each other (the
    # potential does scatter a little at each slice) -- a stale value from
    # an earlier depth would collapse these to be identical.
    assert len(set(round(float(v), 5) for v in reference)) > 1


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_share_per_depth_shares_within_but_not_across_depths(si_atoms, device):
    """Using ``waves._share_diffraction_pattern_fft()`` per depth (the
    pattern used by the real multislice call sites) must (a) give the same
    per-depth, per-detector results as no sharing at all, and (b) actually
    compute the FFT only once per depth even though two detectors are
    evaluated there.
    """
    potential = _make_potential(si_atoms, device, gpts=48, num_slices=3)
    probe = Probe(energy=100e3, semiangle_cutoff=20, device=device)
    probe.grid.match(potential)
    waves = probe.build(lazy=False)

    propagator = FresnelPropagator()
    aperture = AntialiasAperture()
    d1 = AnnularDetector(inner=0, outer=20)
    d2 = AnnularDetector(inner=20, outer=40)

    from abtem.waves import Waves

    def run_pass(share):
        original = Waves._diffraction_pattern_fft
        calls = []

        def counting(array, normalize):
            calls.append(1)
            return original(array, normalize)

        Waves._diffraction_pattern_fft = staticmethod(counting)
        try:
            w = waves.copy()
            results = []
            for potential_slice in potential.generate_slices():
                w = conventional_multislice_step(w, potential_slice, propagator, aperture)
                if share:
                    with w._share_diffraction_pattern_fft():
                        r1 = d1.detect(w).array
                        r2 = d2.detect(w).array
                else:
                    r1 = d1.detect(w).array
                    r2 = d2.detect(w).array
                if device == "gpu":
                    r1 = cp.asnumpy(r1)
                    r2 = cp.asnumpy(r2)
                results.append((np.array(r1, copy=True), np.array(r2, copy=True)))
        finally:
            Waves._diffraction_pattern_fft = staticmethod(original)
        return results, len(calls)

    unshared_results, unshared_calls = run_pass(share=False)
    shared_results, shared_calls = run_pass(share=True)

    n_slices = potential.num_slices
    assert unshared_calls == 2 * n_slices  # one FFT per detector per depth
    assert shared_calls == n_slices  # shared: one FFT per depth

    for depth, ((u1, u2), (s1, s2)) in enumerate(
        zip(unshared_results, shared_results)
    ):
        np.testing.assert_allclose(u1, s1, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(u2, s2, rtol=1e-4, atol=1e-6)
