"""Tests for the BiP-PRISM (bi-partitioned PRISM core-loss EELS) port.

BiP-PRISM (paper Alg. 5) partitions BOTH the probe-forming matrix S1 and the
detector/exit matrix S2 onto sparse hex-ring *parent* beams and reconstructs the
full beam set locally at each ionized atom by windowed natural-neighbor
interpolation (+ magnitude preservation). This module first tests the geometry /
reconstruction helpers in ``abtem.prism._bipartite`` (M0), then the partitioned
beam-basis driver (M1+).
"""

import numpy as np
import pytest

from abtem.prism.utils import plane_waves


def _hex_beam_grid(radius=3, scale=0.1):
    """A small circular-aperture-like set of 2D wave vectors (1/Å)."""
    ks = np.arange(-radius, radius + 1)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    mask = KX**2 + KY**2 <= radius**2
    return np.stack([KX[mask], KY[mask]], axis=1).astype(np.float64) * scale


# --------------------------------------------------------------------------- M0
# parent selection


def test_select_parent_beams_returns_unique_subset():
    from abtem.prism._bipartite import select_parent_beams

    wv = _hex_beam_grid()
    idx = select_parent_beams(wv, n_radial=2, n_angular=6)

    assert idx.ndim == 1
    assert np.array_equal(idx, np.unique(idx))          # sorted + de-duplicated
    assert idx.min() >= 0 and idx.max() < len(wv)       # valid indices into wv
    assert len(idx) < len(wv)                            # genuinely a subsample
    dc = int(np.argmin(np.linalg.norm(wv, axis=1)))      # DC beam is a parent
    assert dc in idx


# natural-neighbor weights


def test_natural_neighbor_weights_full_parents_is_identity():
    from abtem.prism._bipartite import natural_neighbor_weights

    wv = _hex_beam_grid()
    w = natural_neighbor_weights(wv, wv, method="linear")  # parents == queries
    assert w.shape == (len(wv), len(wv))
    np.testing.assert_allclose(w, np.eye(len(wv)), atol=1e-9)


def test_natural_neighbor_weights_partition_of_unity():
    from abtem.prism._bipartite import (
        natural_neighbor_weights,
        select_parent_beams,
    )

    wv = _hex_beam_grid()
    idx = select_parent_beams(wv, n_radial=2, n_angular=6)
    w = natural_neighbor_weights(wv[idx], wv, method="linear")
    assert w.shape == (len(wv), len(idx))
    np.testing.assert_allclose(w.sum(axis=1), np.ones(len(wv)), atol=1e-9)
    assert (w >= 0).all()


# windowed reconstruction


def test_windowed_reconstruct_full_parents_recovers_exact_columns():
    from abtem.prism._bipartite import windowed_reconstruct

    extent = (10.0, 10.0)
    gpts = (16, 16)
    wv = _hex_beam_grid()

    cols = np.asarray(plane_waves(wv, extent, gpts))     # (B, gy, gx) exact columns
    iy = np.arange(3, 11)
    ix = np.arange(5, 13)
    cols_win = cols[:, iy][:, :, ix]                     # (B, wy, wx)

    recon = windowed_reconstruct(
        parent_cols=cols_win,
        weights=np.eye(len(wv)),
        k_parents=wv,
        k_targets=wv,
        iy=iy,
        ix=ix,
        extent=extent,
        gpts=gpts,
        mag_preserve=True,
    )
    np.testing.assert_allclose(recon, cols_win, atol=1e-5)


def test_windowed_reconstruct_detilt_removes_carrier():
    """De-tilting a plane-wave column by its own wave vector yields a constant
    (DC) field on the window, proving the phase convention matches plane_waves."""
    from abtem.prism._bipartite import windowed_reconstruct

    extent = (10.0, 10.0)
    gpts = (16, 16)
    wv = _hex_beam_grid()
    b = int(np.argmax(np.linalg.norm(wv, axis=1)))       # a high-angle beam
    k = wv[b : b + 1]                                     # single "parent" == "target"

    cols = np.asarray(plane_waves(k, extent, gpts))       # (1, gy, gx)
    iy = np.arange(2, 12)
    ix = np.arange(4, 14)
    cols_win = cols[:, iy][:, :, ix]

    # Reconstruct with a single parent/target: de-tilt(k) then re-tilt(k) == identity.
    recon = windowed_reconstruct(
        parent_cols=cols_win,
        weights=np.ones((1, 1)),
        k_parents=k,
        k_targets=k,
        iy=iy,
        ix=ix,
        extent=extent,
        gpts=gpts,
        mag_preserve=False,
    )
    np.testing.assert_allclose(recon, cols_win, atol=1e-5)


# --------------------------------------------------------------------------- M1
# Partitioned beam-basis driver (BiP-PRISM). Its no-partition limit is the
# existing bit-exact beam-basis path (guarded by test_ionization.py); here we
# check that the partitioned path stays highly correlated with the
# unpartitioned reference (the paper's Pearson metric).

import ase  # noqa: E402
import abtem  # noqa: E402
from abtem.core.axes import OrdinalAxis  # noqa: E402
from abtem.inelastic.core_loss import TransitionPotentialArray  # noqa: E402


def _synthetic_tp(Z, gpts, extent, energy=100e3, n_transitions=1, seed=0):
    rng = np.random.default_rng(seed)
    array = (
        rng.standard_normal((n_transitions, *gpts))
        + 1j * rng.standard_normal((n_transitions, *gpts))
    ).astype(np.complex64)
    return TransitionPotentialArray(
        Z=Z, array=array, energy=energy, extent=extent,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_transitions)))],
        metadata={"Z": Z, "n": 1, "l": 0},
    )


def _prism_eels_setup(gpts=(24, 24), reps=(1, 1, 2)):
    unit = ase.build.bulk("Si", cubic=True)
    atoms = unit * reps
    st = float(unit.cell[2, 2])
    pot = abtem.Potential(atoms, gpts=gpts, slice_thickness=st, device="cpu")
    tp = _synthetic_tp(14, gpts, pot.extent, energy=100e3)
    scan = abtem.GridScan(
        start=(0, 0), end=(unit.cell[0, 0], unit.cell[1, 1]),
        sampling=0.5, endpoint=False,
    )
    sm = abtem.SMatrix(
        potential=pot, energy=100e3, semiangle_cutoff=20.0,
        interpolation=1, downsample=False, device="cpu",
    )
    det = abtem.FlexibleAnnularDetector(to_cpu=True)
    return sm, tp, scan, det, atoms


def _pearson(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def _run(driver, sm, tp, scan, det, atoms, **kw):
    return np.asarray(driver(sm, tp, scan, det, sites=atoms, **kw).array)


@pytest.mark.parametrize("double_channel", [False, True])
def test_bipprism_s1_partition_high_correlation(double_channel):
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, det, atoms = _prism_eels_setup()
    ref = _run(drv, sm, tp, scan, det, atoms, double_channel=double_channel)
    par = _run(drv, sm, tp, scan, det, atoms, double_channel=double_channel,
               partitions_s1=4)
    assert par.shape == ref.shape
    assert np.all(np.isfinite(par))
    assert _pearson(par, ref) > 0.99


def test_bipprism_s2_partition_converges_to_exact():
    """S2 partitioned over the *full* reciprocal grid converges monotonically to
    the exact (unpartitioned) map as the parent count grows — the paper's
    eq-part-exact full-parent limit. (BiP-PRISM proper partitions S2 over the
    smaller detector-collection disk, which is far more accurate at low parent
    counts; that is the detector-disk milestone.)"""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, det, atoms = _prism_eels_setup()
    ref = _run(drv, sm, tp, scan, det, atoms, double_channel=True)

    corr = []
    for n_radial in (3, 10):
        par = _run(drv, sm, tp, scan, det, atoms, double_channel=True,
                   partitions_s2=n_radial)
        assert par.shape == ref.shape
        assert np.all(np.isfinite(par))
        corr.append(_pearson(par, ref))

    assert corr[0] < corr[1]                       # more parents -> closer to exact
    assert corr[-1] > 0.98                          # approaches exact at full parents


def test_bipprism_both_partitions_converges_to_exact():
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, det, atoms = _prism_eels_setup()
    ref = _run(drv, sm, tp, scan, det, atoms, double_channel=True)

    par_coarse = _run(drv, sm, tp, scan, det, atoms, double_channel=True,
                      partitions_s1=3, partitions_s2=3)
    par_fine = _run(drv, sm, tp, scan, det, atoms, double_channel=True,
                    partitions_s1=6, partitions_s2=10)
    assert par_coarse.shape == ref.shape
    assert np.all(np.isfinite(par_coarse)) and np.all(np.isfinite(par_fine))
    # both S1 and S2 partitioned; finer parents -> closer to exact
    assert _pearson(par_fine, ref) > _pearson(par_coarse, ref)
    assert _pearson(par_fine, ref) > 0.98


def test_bipprism_mag_preserve_flag_runs_both_ways():
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, det, atoms = _prism_eels_setup()
    with_mp = _run(drv, sm, tp, scan, det, atoms, double_channel=True,
                   partitions_s1=3, partitions_s2=3, mag_preserve=True)
    no_mp = _run(drv, sm, tp, scan, det, atoms, double_channel=True,
                 partitions_s1=3, partitions_s2=3, mag_preserve=False)
    assert with_mp.shape == no_mp.shape
    assert np.all(np.isfinite(with_mp)) and np.all(np.isfinite(no_mp))
    # both should still correlate with the pattern
    assert _pearson(with_mp, no_mp) > 0.9


# --------------------------------------------------------------------------- M2
# Detector-collection-disk S2 (the paper's BiP-PRISM map regime).


def test_bipprism_collection_angle_matches_full_grid():
    """Restricting S2 to a collection disk that covers the detector's outer angle
    leaves the detector-integrated map unchanged (the beam-basis reduction already
    sums |a_q|^2 over the detector); it only skips building S2 for beams the
    detector discards."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    ref = _run(drv, sm, tp, scan, ann, atoms, double_channel=True)
    disk = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
                collection_angle=35)
    assert disk.shape == ref.shape
    np.testing.assert_allclose(disk, ref, rtol=1e-4, atol=1e-8)


def test_bipprism_collection_angle_s2_partition_accurate():
    """Over the small detector-collection disk, S2 partitioning at a modest parent
    count is highly accurate — the paper's BiP-PRISM operating regime."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    ref = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=35)
    par = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=35, partitions_s2=4)
    assert par.shape == ref.shape
    assert np.all(np.isfinite(par))
    assert _pearson(par, ref) > 0.99


def test_bipprism_full_partitioned_map_collection_disk():
    """Full BiP-PRISM: partition both S1 and S2 over the collection disk; the map
    stays highly correlated with the exact dual-matrix map."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    ref = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=35)
    par = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=35, partitions_s1=4, partitions_s2=4)
    assert par.shape == ref.shape
    assert np.all(np.isfinite(par))
    assert _pearson(par, ref) > 0.99


# --------------------------------------------------------------- public API (M2b)


def test_bipprism_public_api_dispatch_matches_driver():
    """SMatrix.transition_potential_scan auto-selects the beam-basis backend when
    partition kwargs are given and reproduces the direct driver call."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    direct = np.asarray(
        drv(sm, tp, scan, ann, sites=atoms, double_channel=True,
            collection_angle=35, partitions_s1=4, partitions_s2=4).array
    )
    pub = sm.transition_potential_scan(
        tp, scan=scan, detectors=ann, sites=atoms, double_channel=True,
        collection_angle=35, partitions_s1=4, partitions_s2=4, lazy=False,
    )
    pub_arr = np.asarray(pub.array)
    assert pub_arr.shape == direct.shape
    np.testing.assert_allclose(pub_arr, direct, rtol=1e-5, atol=1e-8)


def test_bipprism_public_api_lazy_matches_eager():
    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    eager = np.asarray(
        sm.transition_potential_scan(
            tp, scan=scan, detectors=ann, sites=atoms, double_channel=True,
            reduction="bipartite", collection_angle=35,
            partitions_s1=4, partitions_s2=4, lazy=False,
        ).array
    )
    lazy = np.asarray(
        sm.transition_potential_scan(
            tp, scan=scan, detectors=ann, sites=atoms, double_channel=True,
            reduction="bipartite", collection_angle=35,
            partitions_s1=4, partitions_s2=4, lazy=True,
        ).compute().array
    )
    np.testing.assert_allclose(lazy, eager, rtol=1e-5, atol=1e-8)


# ------------------------------------------------------------- validation (M3)


def test_bipprism_mag_preserve_recovers_scale():
    """Quantitative check of the magnitude-preserving reconstruction (paper
    eq-magpreserve / Sec. 3.3.6): the plain complex NNW average decoheres and
    *loses* absolute map intensity, growing with thickness; mag_preserve pairs the
    interpolated magnitude with the complex-sum phase and recovers the scale."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    # a thicker slab makes the decoherence (which grows with propagation) visible
    sm, tp, scan, _, atoms = _prism_eels_setup(reps=(1, 1, 4))
    ann = abtem.AnnularDetector(inner=0, outer=30)

    ref = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=35)
    mp = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
              collection_angle=35, partitions_s1=3, partitions_s2=3,
              mag_preserve=True)
    no = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
              collection_angle=35, partitions_s1=3, partitions_s2=3,
              mag_preserve=False)

    tot = lambda a: float(np.asarray(a).sum())
    r_mp = tot(mp) / tot(ref)
    r_no = tot(no) / tot(ref)

    assert r_no < r_mp                      # decoherence lowers intensity
    assert abs(r_mp - 1.0) < abs(r_no - 1.0)  # mag_preserve is closer to exact scale
    assert abs(r_mp - 1.0) < 0.02             # ... and recovers scale to ~1%


# ------------------------------------------------------ focal back-prop helpers


def test_focal_backprop_distance_modes():
    from abtem.prism._bipartite import focal_backprop_distance

    assert focal_backprop_distance(10.0, None) == 0.0
    assert focal_backprop_distance(10.0, 0) == 0.0
    assert focal_backprop_distance(10.0, "centroid") == 5.0
    assert focal_backprop_distance(10.0, 0.3) == 3.0
    with pytest.raises(ValueError):
        focal_backprop_distance(10.0, "middle")


def test_pad_window_recovers_inner():
    from abtem.prism._bipartite import pad_window

    padded, inner = pad_window(corner=5, length=8, margin=3, n=100)
    assert len(padded) == 8 + 2 * 3
    np.testing.assert_array_equal(padded[inner], 5 + np.arange(8))


def test_pad_window_caps_to_full_axis():
    from abtem.prism._bipartite import pad_window

    padded, inner = pad_window(corner=0, length=24, margin=10, n=24)
    assert len(padded) == 24                       # capped, not 44
    np.testing.assert_array_equal(padded[inner], np.arange(24))


def test_fresnel_margin_bounds():
    from abtem.prism._bipartite import fresnel_margin

    assert fresnel_margin(0.0, 0.037, 0.5, (0.2, 0.2), (24, 24)) == 0
    m = fresnel_margin(20.0, 0.037, 0.5, (0.2, 0.2), (64, 64))
    assert 4 <= m <= 32                             # positive, capped at min(gpts)//2


# --------------------------------------------------- focal back-prop (driver)


def _relL2(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_bipprism_focal_backprop_reduces_s1_error():
    """S1 focal back-propagation lowers the on-atom probe reconstruction error
    (paper Sec. 3.3.6, 2-3x for thick specimens). Isolate S1 by keeping S2 exact
    over the collection disk (partitions_s2=None) and partitioning only S1."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    # thicker slab -> more channelling -> larger S1 reconstruction error to reduce
    sm, tp, scan, _, atoms = _prism_eels_setup(reps=(1, 1, 6))
    ann = abtem.AnnularDetector(inner=0, outer=25)

    ref = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
               collection_angle=30)  # exact S1 & S2 over the disk
    no_fb = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
                 collection_angle=30, partitions_s1=2)
    fb = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
              collection_angle=30, partitions_s1=2, focal_backprop="centroid")

    err_no = _relL2(no_fb, ref)
    err_fb = _relL2(fb, ref)
    assert np.all(np.isfinite(fb))
    assert err_fb < err_no          # focal back-prop improves S1 reconstruction
    assert err_fb < 0.5 * err_no + 1e-6  # ... substantially (paper: 2-3x)


def test_bipprism_focal_backprop_noop_without_s1_partition():
    """focal_backprop only touches the S1 partition; with partitions_s1=None it is
    a no-op (S2's parallel detector beams have no crossover plane)."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup()
    ann = abtem.AnnularDetector(inner=0, outer=30)
    base = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
                collection_angle=35, partitions_s2=4)
    with_fb = _run(drv, sm, tp, scan, ann, atoms, double_channel=True,
                   collection_angle=35, partitions_s2=4, focal_backprop="centroid")
    np.testing.assert_allclose(with_fb, base, rtol=1e-6, atol=1e-10)


def test_bipprism_focal_backprop_public_api():
    """focal_backprop flows through SMatrix.transition_potential_scan and (with
    partition kwargs) auto-selects the beam-basis backend."""
    from abtem.inelastic.core_loss import (
        prism_transition_potential_scan_beam_basis as drv,
    )

    sm, tp, scan, _, atoms = _prism_eels_setup(reps=(1, 1, 4))
    ann = abtem.AnnularDetector(inner=0, outer=25)
    direct = np.asarray(
        drv(sm, tp, scan, ann, sites=atoms, double_channel=True,
            collection_angle=30, partitions_s1=3, focal_backprop="centroid").array
    )
    pub = np.asarray(
        sm.transition_potential_scan(
            tp, scan=scan, detectors=ann, sites=atoms, double_channel=True,
            collection_angle=30, partitions_s1=3, focal_backprop="centroid",
            lazy=False,
        ).array
    )
    np.testing.assert_allclose(pub, direct, rtol=1e-5, atol=1e-8)
