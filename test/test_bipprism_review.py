"""Regressions for the review of the BiP-PRISM entry point (PR #338).

Each test reproduces one reported defect: lazy double-channel sampling,
frozen-phonon potentials, options that silently do nothing, a collection disk
smaller than the detector, the full S1 stack kept next to its parents, dense
interpolation weights, and per-atom focal propagators.
"""

import ase
import numpy as np
import pytest
from test_bipprism import _prism_eels_setup, _synthetic_tp
from utils import gpu

import abtem
from abtem.core.backend import asnumpy
from abtem.inelastic.core_loss import (
    prism_transition_potential_scan_beam_basis as driver,
)
from abtem.prism import _bipartite as bip


def _si_smatrix(atoms, gpts=(32, 32), interpolation=2, device="cpu"):
    unit = ase.build.bulk("Si", cubic=True)
    potential = abtem.Potential(
        atoms, gpts=gpts, slice_thickness=float(unit.cell[2, 2]), device=device
    )
    tp = _synthetic_tp(14, gpts, potential.extent)
    sm = abtem.SMatrix(
        potential=potential, energy=100e3, semiangle_cutoff=20,
        interpolation=interpolation, downsample=False, device=device,
    )
    scan = abtem.GridScan(
        start=(0, 0), end=(unit.cell[0, 0], unit.cell[1, 1]),
        sampling=0.7, endpoint=False,
    )
    return sm, tp, scan


@pytest.mark.parametrize("device", ["cpu", gpu])
@pytest.mark.parametrize("double_channel", [False, True])
def test_lazy_beam_basis_matches_eager_sampling(device, double_channel):
    # The double-channel beam-basis driver detects on the native grid. The lazy
    # path declared the PRISM-cell grid, so at interpolation=2 every angle of
    # the lazy diffraction patterns was off by 2x.
    atoms = ase.build.bulk("Si", cubic=True) * (2, 2, 1)
    sm, tp, scan = _si_smatrix(atoms, device=device)
    options = dict(
        scan=scan, sites=atoms, detectors=abtem.PixelatedDetector(max_angle=30),
        reduction="beam_basis", double_channel=double_channel,
    )
    eager = sm.transition_potential_scan(tp, lazy=False, **options)
    lazy = sm.transition_potential_scan(tp, lazy=True, **options).compute()
    assert lazy.shape == eager.shape
    np.testing.assert_allclose(lazy.sampling, eager.sampling, rtol=1e-12)
    np.testing.assert_allclose(asnumpy(lazy.array), asnumpy(eager.array),
                               rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("lazy", [False, True])
def test_beam_basis_frozen_phonons_match_real_space(lazy):
    # SMatrix.transition_potential_scan hands the driver one configuration at a
    # time, with a length-1 ensemble axis; the driver refused all of them. At
    # interpolation=1 both drivers are exact, so the averages must agree.
    atoms = ase.build.bulk("Si", cubic=True) * (1, 1, 2)
    frozen = abtem.FrozenPhonons(atoms, num_configs=2, sigmas=0.1, seed=1)
    sm, tp, scan = _si_smatrix(frozen, gpts=(24, 24), interpolation=1)
    detector = abtem.AnnularDetector(inner=0, outer=20)

    def run(**options):
        result = sm.transition_potential_scan(
            tp, scan=scan, detectors=detector, sites=atoms, lazy=lazy, **options
        )
        return asnumpy((result.compute() if lazy else result).array)

    real_space = run(reduction="real_space", double_channel=True)
    beam_basis = run(reduction="beam_basis", double_channel=True)
    assert beam_basis.shape == real_space.shape
    np.testing.assert_allclose(beam_basis, real_space, rtol=1e-4,
                               atol=1e-6 * np.abs(real_space).max())
    bipartite = run(partitions_s1=3, partitions_s2=3, double_channel=True,
                    collection_angle=20)
    assert bipartite.shape == real_space.shape
    assert np.isfinite(bipartite).all()


@pytest.mark.parametrize(
    "options,match",
    [
        (dict(partitions_s2=4), "partitions_s2 applies only"),
        (dict(collection_angle=25), "collection_angle applies only"),
        (dict(double_channel=True, focal_backprop="centroid"),
         "focal_backprop applies only"),
        (dict(reduction="bipartite"), "needs partitions_s1"),
        (dict(partitions_s1=2, n_angular=0), "n_angular"),
        (dict(partitions_s1=0), "partitions_s1"),
        (dict(partitions_s1=2, focal_backprop="middle"), "centroid"),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_ineffective_options_raise(options, match, lazy):
    # Each of these used to run without the option taking effect (or crashed
    # with ZeroDivisionError), so a user could believe BiP-PRISM had been used.
    # A lazy scan must fail at once, not when it is computed.
    sm, tp, scan, _, atoms = _prism_eels_setup()
    with pytest.raises(ValueError, match=match):
        sm.transition_potential_scan(tp, scan=scan, sites=atoms, lazy=lazy,
                                     **options)


def test_collection_angle_must_cover_detector():
    # A 15 mrad disk under a 20 mrad detector gave 0.555 of the correct signal
    # with no warning. A disk that covers the detector leaves the map exact.
    sm, tp, scan, _, atoms = _prism_eels_setup()
    detector = abtem.AnnularDetector(inner=0, outer=20)
    options = dict(scan=scan, detectors=detector, sites=atoms,
                   double_channel=True, lazy=False)
    with pytest.raises(ValueError, match="collection_angle=15"):
        sm.transition_potential_scan(tp, collection_angle=15, **options)
    with pytest.raises(ValueError, match="collection_angle=15"):
        _run_driver(sm, tp, scan, detector, atoms, collection_angle=15)

    exact = sm.transition_potential_scan(tp, reduction="beam_basis", **options)
    disk = sm.transition_potential_scan(tp, collection_angle=20, **options)
    np.testing.assert_allclose(asnumpy(disk.array), asnumpy(exact.array),
                               rtol=1e-5, atol=1e-12)


def test_collection_angle_covers_auto_sized_radial_detector():
    # This detector stops at its automatically chosen radial cutoff. A disk
    # larger than that cutoff covers its bins; it does not collect the full grid.
    sm, tp, scan, _, atoms = _prism_eels_setup()
    cutoff = max(sm.dummy_probes().cutoff_angles)
    result = _run_driver(sm, tp, scan, abtem.FlexibleAnnularDetector(), atoms,
                         collection_angle=1.5 * cutoff)
    assert np.isfinite(result).all()


def _run_driver(sm, tp, scan, detector, atoms, **options):
    return asnumpy(driver(sm, tp, scan, detector, sites=atoms,
                          double_channel=True, **options).array)


def test_partitioned_s1_builds_only_parent_columns(monkeypatch):
    # The setup built the full (n_k, *gpts) S1 stack and kept it alive next to
    # the parents for the whole scan, cancelling the S1 memory saving.
    import abtem.prism.utils

    sm, tp, scan, det, atoms = _prism_eels_setup()
    n_k = len(sm.wave_vectors)
    plane_waves = abtem.prism.utils.plane_waves
    built = []

    def record(wave_vectors, *args, **kwargs):
        built.append(len(wave_vectors))
        return plane_waves(wave_vectors, *args, **kwargs)

    monkeypatch.setattr(abtem.prism.utils, "plane_waves", record)
    driver(sm, tp, scan, det, sites=atoms, double_channel=False, partitions_s1=2)
    n_parents = len(bip.select_parent_beams(sm.wave_vectors, 2))
    assert n_parents < n_k
    assert built == [n_parents]


def test_sparse_weights_match_dense_and_nearest_matches_argmin():
    # The reconstruction stores at most 3 weights per target instead of a dense
    # (B, Bp) matrix, and snaps with a KD-tree instead of an (n, N) distance
    # array. Both must give exactly what the dense forms gave.
    fy = np.fft.fftfreq(40, 0.25)
    ky, kx = np.meshgrid(fy, fy, indexing="ij")
    k = np.stack([ky.ravel(), kx.ravel()], axis=1)
    for n_radial in (1, 3, 8):
        parents = bip.select_parent_beams(k, n_radial)
        dense = bip.natural_neighbor_weights(k[parents], k)
        indices, values = bip.natural_neighbor_weights(k[parents], k, sparse=True)
        assert indices.shape == values.shape == (len(k), 3)
        np.testing.assert_array_equal(
            bip._dense_weights(indices, values, len(parents)), dense
        )

    # Queries at grid midpoints are exact ties; argmin takes the lowest index.
    rng = np.random.default_rng(0)
    queries = np.concatenate([
        rng.uniform(-2, 2, (200, 2)),
        (k[:50] + k[1:51]) / 2,
        np.zeros((1, 2)),
    ])
    expected = np.argmin(
        np.linalg.norm(queries[:, None, :] - k[None, :, :], axis=2), axis=1
    )
    np.testing.assert_array_equal(bip._nearest_indices(k, queries), expected)


def test_focal_propagators_are_built_once_per_slice(monkeypatch):
    # The padded-window margin and forward propagator depend only on the slice,
    # but were rebuilt for every atom in it.
    import abtem.multislice

    propagator = abtem.multislice._fresnel_propagator_array
    distances = []

    def record(*args, **kwargs):
        # The driver passes the distance by position; the multislice steps,
        # which are not counted here, pass it as ``thickness=``.
        if args:
            distances.append(args[0])
        return propagator(*args, **kwargs)

    monkeypatch.setattr(abtem.multislice, "_fresnel_propagator_array", record)
    sm, tp, scan, det, atoms = _prism_eels_setup(reps=(1, 1, 2))
    driver(sm, tp, scan, det, sites=atoms, double_channel=False,
           partitions_s1=2, focal_backprop="centroid")
    backward = [d for d in distances if d < 0]
    forward = [d for d in distances if d > 0]
    assert backward
    assert forward == [-d for d in backward]
