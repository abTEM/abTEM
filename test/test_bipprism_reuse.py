"""Regressions for bounded, per-slice BiP-PRISM reconstruction reuse."""

import numpy as np
import pytest
from test_bipprism import _prism_eels_setup, _run
from utils import gpu

import abtem
from abtem.core.backend import asnumpy, get_array_module
from abtem.inelastic.core_loss import (
    prism_transition_potential_scan_beam_basis as driver,
)
from abtem.prism import _bipartite as bip


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "legs,double_channel",
    [((2, None), False), ((2, None), True), ((None, 2), True), ((2, 2), True)],
)
@pytest.mark.parametrize("mag_preserve", [False, True])
def test_reuse_images_and_call_count(
    monkeypatch, device, precision, legs, double_channel, mag_preserve
):
    calls = []
    original = bip.windowed_reconstruct

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result.shape)
        assert result.dtype == (
            np.complex64 if precision == "float32" else np.complex128
        )
        return result

    monkeypatch.setattr(bip, "windowed_reconstruct", record)
    with abtem.config.set({"precision": precision}):
        sm, tp, scan, det, atoms = _prism_eels_setup(
            gpts=(24, 32), reps=(1, 1, 2), device=device
        )
        options = dict(
            partitions_s1=legs[0],
            partitions_s2=legs[1],
            double_channel=double_channel,
            collection_angle=25,
            mag_preserve=mag_preserve,
        )
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 0, raising=False)
        reference = _run(driver, sm, tp, scan, det, atoms, **options)
        uncached_calls = len(calls)
        calls.clear()
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 1 << 31, raising=False)
        actual = _run(driver, sm, tp, scan, det, atoms, **options)
    assert len(calls) < uncached_calls
    assert len(calls) == 2 * sum(v is not None for v in legs)
    assert np.isfinite(actual).all()
    error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    assert error < (1e-5 if precision == "float32" else 1e-11)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("reason", ["budget", "sparse", "one_site", "focal"])
def test_reconstruction_guard_does_no_work(monkeypatch, device, precision, reason):
    assert hasattr(bip, "_reconstruct_slice"), (
        "bounded reconstruction helper is missing"
    )
    xp = get_array_module(device)
    with abtem.config.set({"precision": precision}):
        dtype = np.complex64 if precision == "float32" else np.complex128
        parents = xp.ones((2, 8, 12), dtype=dtype)
        opts = dict(n_active=4, window_gpts=(8, 12), budget=1 << 20, focal=False)
        if reason == "budget":
            opts["budget"] = 0
        if reason == "sparse":
            opts["window_gpts"] = (2, 2)
        if reason == "one_site":
            opts["n_active"] = 1
        if reason == "focal":
            opts["focal"] = True

        def unexpected(*args, **kwargs):
            pytest.fail("ineligible reconstruction allocated/interpolated")

        monkeypatch.setattr(bip, "windowed_reconstruct", unexpected)
        assert (
            bip._reconstruct_slice(
                parents,
                np.eye(2),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                (8.0, 12.0),
                (8, 12),
                **opts,
            )
            is None
        )


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_prepared_geometry_does_not_copy_to_host(monkeypatch, device):
    assert hasattr(bip, "_prepare_reconstruction_geometry"), (
        "prepared geometry is missing"
    )
    xp = get_array_module(device)
    with abtem.config.set({"precision": "float32"}):
        weights = xp.asarray(np.eye(2), dtype=np.complex64)
        k = np.array([[0.0, 0.0], [1 / 8, -1 / 12]])
        geometry = bip._prepare_reconstruction_geometry(weights, k, k, xp, np.complex64)

        def unexpected(*args):
            pytest.fail("device geometry copied back to host")

        monkeypatch.setattr(bip, "_to_numpy", unexpected)
        parents = xp.ones((2, 8, 12), dtype=np.complex64)
        r = bip.windowed_reconstruct(
            parents,
            weights,
            k,
            k,
            np.arange(8),
            np.arange(12),
            (8.0, 12.0),
            (8, 12),
            geometry=geometry,
        )
        np.testing.assert_allclose(asnumpy(r), 1, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("remaining", [0.0, 10.0, 100.0])
def test_cached_s2_vacuum_phase_and_support(monkeypatch, device, remaining):
    import ase
    from test_bipprism import _vacuum_eels_setup

    with abtem.config.set({"precision": "float64"}):
        sm, tp, scan, det, _ = _vacuum_eels_setup((1.0, remaining), device=device)
        atoms = ase.Atoms(
            "Si2",
            positions=[(8, 8, 0.5), (0.3, 15.7, 0.5)],
            cell=(16, 16, 1 + remaining),
            pbc=True,
        )
        options = dict(double_channel=True, partitions_s1=2, partitions_s2=4)
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 0)
        ref = _run(driver, sm, tp, scan, det, atoms, **options)
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 1 << 31)
        got = _run(driver, sm, tp, scan, det, atoms, **options)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize(
    "partitions,focal,active",
    [(2, "centroid", True), (100, "centroid", False), (2, 0.0, False)],
)
def test_driver_focal_cache_guard(monkeypatch, device, partitions, focal, active):
    original = bip._reconstruct_slice
    decisions = []

    def record(*args, **kwargs):
        r = original(*args, **kwargs)
        decisions.append((kwargs.get("focal", False), r is not None))
        return r

    monkeypatch.setattr(bip, "_reconstruct_slice", record)
    with abtem.config.set({"precision": "float64"}):
        setup = _prism_eels_setup(gpts=(24, 32), device=device)
        opts = dict(
            partitions_s1=partitions, focal_backprop=focal, double_channel=False
        )
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 0)
        ref = _run(driver, *setup, **opts)
        decisions.clear()
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 1 << 31)
        got = _run(driver, *setup, **opts)
    assert decisions == [(active, not active)] * 2
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-18)


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("position,expected_active", [((8.0, 8.0), 1), ((4.0, 4.0), 0)])
def test_driver_counts_only_contributing_sites(
    monkeypatch, device, position, expected_active
):
    import ase
    from test_bipprism import _vacuum_eels_setup

    original = bip._reconstruct_slice
    counts = []

    def record(*args, **kwargs):
        counts.append(kwargs["n_active"])
        r = original(*args, **kwargs)
        assert r is None
        return r

    monkeypatch.setattr(bip, "_reconstruct_slice", record)
    with abtem.config.set({"precision": "float64"}):
        sm, tp, _, det, _ = _vacuum_eels_setup(interpolation=4, device=device)
        atoms = ase.Atoms(
            "Si2", positions=[(8, 8, 0.5), (1, 1, 0.5)], cell=(16, 16, 1), pbc=True
        )
        with pytest.warns(UserWarning, match="no scan position"):
            got = _run(
                driver,
                sm,
                tp,
                abtem.CustomScan([position]),
                det,
                atoms,
                partitions_s1=2,
                partitions_s2=2,
                double_channel=True,
            )
    assert counts == ([1, 1] if expected_active else [])
    if not expected_active:
        assert not np.any(got)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_cache_lifetime_and_joint_budget(monkeypatch, device):
    import weakref

    import abtem.multislice

    original = bip._reconstruct_slice
    step = abtem.multislice.conventional_multislice_step
    refs, allocations = [], []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        refs.append(weakref.ref(result))
        allocations.append((kwargs["budget"], result.nbytes))
        return result

    def step_check(*args, **kwargs):
        if not kwargs.get("conjugate", False):
            assert all(ref() is None for ref in refs)
        return step(*args, **kwargs)

    monkeypatch.setattr(bip, "_reconstruct_slice", record)
    monkeypatch.setattr(abtem.multislice, "conventional_multislice_step", step_check)
    with abtem.config.set({"precision": "float32"}):
        _run(
            driver,
            *_prism_eels_setup(gpts=(24, 32), reps=(1, 1, 3), device=device),
            partitions_s1=2,
            partitions_s2=2,
            double_channel=True,
            collection_angle=25,
        )
    assert all(ref() is None for ref in refs)
    assert len(allocations) == 6
    for a, b in zip(allocations[::2], allocations[1::2]):
        assert b[0] == a[0] - a[1]


def test_host_reconstruction_accepts_transferable_weights():
    class DeviceWeights:
        def get(self):
            return np.eye(2)

        def __array__(self, *args, **kwargs):
            raise TypeError("device array requires an explicit host transfer")

    parents = np.ones((2, 8, 12), dtype=np.complex64)
    k = np.zeros((2, 2))
    result = bip.windowed_reconstruct(
        parents,
        DeviceWeights(),
        k,
        k,
        np.arange(8),
        np.arange(12),
        (8.0, 12.0),
        (8, 12),
    )
    np.testing.assert_array_equal(result, parents)


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_joint_budget_falls_back_for_second_leg(monkeypatch, device):
    original = bip._reconstruct_slice
    decisions = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        decisions.append(result is not None)
        return result

    monkeypatch.setattr(bip, "_reconstruct_slice", record)
    with abtem.config.set({"precision": "float32"}):
        setup = _prism_eels_setup(gpts=(24, 32), device=device)
        # Full S2 disk is larger than S1; select a budget from actual geometry
        # so S1 fits while S2 fails after subtracting retained S1 output.
        weights = bip.natural_neighbor_weights
        counts = []

        def geometry(parents, targets, *args, **kwargs):
            counts.append((len(parents), len(targets)))
            return weights(parents, targets, *args, **kwargs)

        monkeypatch.setattr(bip, "natural_neighbor_weights", geometry)
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", 0)
        options = dict(
            partitions_s1=2, partitions_s2=2, double_channel=True, collection_angle=25
        )
        ref = _run(driver, *setup, **options)
        budget = 8 * 16 * sum(counts[0]) * 24 * 32
        assert 8 * 16 * sum(counts[1]) * 24 * 32 > budget
        decisions.clear()
        monkeypatch.setattr(bip, "_EELS_RECONSTRUCTION_BYTES", budget)
        got = _run(driver, *setup, **options)
    assert decisions == [True, False] * 2
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 1e-5


@pytest.mark.parametrize("device", ["cpu", gpu])
def test_equal_coverage_requires_reuse_savings_on_cpu(device):
    xp = get_array_module(device)
    parents = xp.ones((2, 8, 12), dtype=np.complex64)
    k = np.zeros((2, 2))
    result = bip._reconstruct_slice(
        parents,
        np.eye(2),
        k,
        k,
        (8.0, 12.0),
        (8, 12),
        n_active=4,
        window_gpts=(4, 6),
        budget=1 << 20,
    )
    if device == "cpu":
        assert result is None
    else:
        np.testing.assert_array_equal(asnumpy(result), asnumpy(parents))
