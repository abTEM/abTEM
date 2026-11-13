"""Independent review regressions for PR #338, pinned initially to 77d488e1."""

import ase.build
import numpy as np
import pytest
from utils import (  # noqa: F401
    angle_by_hand,
    each_angle_convention,
    synthetic_transition_potential,
)

import abtem


def setup_case(interpolation=1, phonons=False, ensemble_mean=True):
    atoms = ase.build.bulk("Si", cubic=True)
    source = (
        abtem.FrozenPhonons(
            atoms, num_configs=2, sigmas=0.05, seed=42, ensemble_mean=ensemble_mean
        )
        if phonons
        else atoms
    )
    potential = abtem.Potential(source, gpts=(24, 24), slice_thickness=3.0)
    sm = abtem.SMatrix(
        potential=potential,
        energy=100e3,
        semiangle_cutoff=20.0,
        interpolation=interpolation,
        downsample=False,
        device="cpu",
    )
    tp = synthetic_transition_potential(
        14, (24, 24), potential.extent, energy=100e3, n_transitions=2
    )
    scan = abtem.GridScan(
        start=(0, 0), end=potential.extent, gpts=(2, 2), endpoint=False
    )
    return sm, tp, scan, atoms


def run(sm, tp, scan, atoms, **options):
    out = sm.transition_potential_scan(tp, scan=scan, sites=atoms, **options)
    return out.compute() if out.is_lazy else out


@pytest.mark.parametrize("double_channel", [False, True])
def test_review_lazy_metadata_matches_eager(double_channel):
    sm, tp, scan, atoms = setup_case(interpolation=2)
    options = dict(
        reduction="beam_basis",
        double_channel=double_channel,
        detectors=abtem.PixelatedDetector(max_angle=30),
    )
    eager = run(sm, tp, scan, atoms, lazy=False, **options)
    lazy = run(sm, tp, scan, atoms, lazy=True, **options)
    np.testing.assert_allclose(lazy.array, eager.array, rtol=1e-6, atol=0)
    assert lazy.shape == eager.shape
    assert lazy.sampling == eager.sampling
    assert lazy.axes_metadata == eager.axes_metadata


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("ensemble_mean", [False, True])
def test_review_beam_basis_frozen_phonons(lazy, ensemble_mean):
    sm, tp, scan, atoms = setup_case(phonons=True, ensemble_mean=ensemble_mean)
    options = dict(double_channel=False, detectors=abtem.AnnularDetector(0, 20))
    reference = run(sm, tp, scan, atoms, reduction="real_space", lazy=lazy, **options)
    actual = run(sm, tp, scan, atoms, reduction="beam_basis", lazy=lazy, **options)
    assert actual.shape == reference.shape
    assert actual.axes_metadata == reference.axes_metadata
    np.testing.assert_allclose(actual.array, reference.array, rtol=1e-4, atol=1e-12)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"partitions_s2": 2}, "double_channel"),
        ({"collection_angle": 25.0}, "double_channel"),
        ({"focal_backprop": "centroid"}, "partitions_s1"),
        ({"reduction": "bipartite"}, "partitions"),
        ({"partitions_s1": 2, "n_angular": 0}, "n_angular"),
        ({"partitions_s1": 2, "n_angular": -1}, "n_angular"),
    ],
)
def test_review_inactive_or_invalid_options_are_refused(options, match):
    sm, tp, scan, atoms = setup_case()
    with pytest.raises(ValueError, match=match):
        run(
            sm,
            tp,
            scan,
            atoms,
            lazy=False,
            detectors=abtem.AnnularDetector(0, 20),
            **options,
        )


@pytest.mark.parametrize(
    "detector",
    [
        abtem.AnnularDetector(0, 20),
        abtem.AnnularDetector(0, 20, offset=(10.0, 0.0)),
    ],
)
def test_review_collection_covers_detector(detector):
    sm, tp, scan, atoms = setup_case()
    with pytest.raises(ValueError, match="collection_angle"):
        run(
            sm,
            tp,
            scan,
            atoms,
            lazy=False,
            reduction="beam_basis",
            double_channel=True,
            collection_angle=15.0,
            detectors=detector,
        )


def test_review_partitioned_s1_never_builds_full_stack(monkeypatch):
    from abtem.prism import utils as prism_utils
    from abtem.prism._bipartite import select_parent_beams

    sm, tp, scan, atoms = setup_case()
    n_parents = len(select_parent_beams(sm.wave_vectors, 1))
    assert n_parents < len(sm.wave_vectors)
    counts = []
    original = prism_utils.plane_waves

    def record(wave_vectors, *args, **kwargs):
        counts.append(len(wave_vectors))
        return original(wave_vectors, *args, **kwargs)

    monkeypatch.setattr(prism_utils, "plane_waves", record)
    run(
        sm,
        tp,
        scan,
        atoms,
        reduction="beam_basis",
        partitions_s1=1,
        detectors=abtem.AnnularDetector(0, 20),
        lazy=False,
    )
    assert counts and max(counts) <= n_parents


def test_review_focal_kernel_is_built_once_per_slice(monkeypatch):
    import abtem.multislice as multislice

    sm, tp, scan, atoms = setup_case(interpolation=2)
    kernels = []
    original = multislice._fresnel_propagator_array

    def record(thickness, gpts, *args, **kwargs):
        if thickness > 0 and tuple(gpts) != tuple(sm.gpts):
            kernels.append((float(thickness), tuple(gpts)))
        return original(thickness, gpts, *args, **kwargs)

    monkeypatch.setattr(multislice, "_fresnel_propagator_array", record)
    run(
        sm,
        tp,
        scan,
        atoms,
        reduction="beam_basis",
        partitions_s1=1,
        focal_backprop="centroid",
        detectors=abtem.AnnularDetector(0, 20),
        lazy=False,
    )
    assert kernels
    assert len(kernels) == len(set(kernels))


@pytest.mark.parametrize(
    "detector",
    [
        abtem.PixelatedDetector(max_angle="full"),
        abtem.PixelatedDetector(max_angle="cutoff"),
        abtem.PixelatedDetector(max_angle="valid"),
        abtem.PixelatedDetector(max_angle=4.8),
        abtem.PixelatedDetector(max_angle=4.8, reciprocal_space=False),
        abtem.AnnularDetector(0, 8, offset=(3.6, 0)),
        abtem.AnnularDetector(0, None, offset=(3.6, 0)),
        abtem.FlexibleAnnularDetector(step_size=3, outer=8),
        abtem.WavesDetector(),
        abtem.WavesDetector(gpts=(4, 6)),
    ],
)
def test_review_collection_guard_matches_detector_response(
    detector, each_angle_convention
):
    """Measure support using isolated Fourier beams through the actual detector."""
    from test_bipprism import _vacuum_eels_setup

    from abtem.core.axes import OrdinalAxis
    from abtem.core.energy import energy2wavelength
    from abtem.inelastic.core_loss import _check_collection_angle

    sm, _, _, _, _ = _vacuum_eels_setup()
    gpts = sm.gpts
    n_pixels = int(np.prod(gpts))
    waves = abtem.Waves(
        np.eye(n_pixels, dtype=np.complex64).reshape(n_pixels, *gpts),
        energy=sm.energy,
        extent=sm.extent,
        reciprocal_space=True,
        ensemble_axes_metadata=[OrdinalAxis(values=tuple(range(n_pixels)))],
    )
    response = detector.detect(waves).array
    # WavesDetector transforms back through real space. Ignore only FFT
    # roundoff relative to the response of a transmitted unit Fourier beam.
    noise = 8 * np.finfo(response.real.dtype).eps * np.abs(response).max()
    collected = np.max(np.abs(response).reshape(n_pixels, -1), axis=1) > noise
    frequencies = [np.fft.fftfreq(n, e / n) for n, e in zip(gpts, sm.extent)]
    alpha = np.hypot(frequencies[0][:, None], frequencies[1][None, :])
    alpha *= energy2wavelength(sm.energy) * 1e3
    # the angle of the outermost collected beam, in the configured convention
    edge = float(angle_by_hand(alpha.ravel()[collected].max()))
    with pytest.raises(ValueError, match="collection_angle"):
        _check_collection_angle(sm, [detector], edge - 1e-3)
    _check_collection_angle(sm, [detector], edge + 1e-3)


def test_review_last_slice_ionization_exceeds_antialias_support():
    """The last transition is detected without another antialiasing propagation."""
    from test_bipprism import _vacuum_eels_setup

    from abtem.antialias import antialias_aperture
    from abtem.core.energy import energy2wavelength

    sm, tp, scan, detector, atoms = _vacuum_eels_setup()
    sampling = tuple(e / n for e, n in zip(sm.extent, sm.gpts))
    fy, fx = [np.fft.fftfreq(n, d) for n, d in zip(sm.gpts, sampling)]
    alpha = np.hypot(fy[:, None], fx[None, :])
    alpha *= energy2wavelength(sm.energy) * 1e3
    support = antialias_aperture(sm.gpts, sampling, np) > 0
    angle = float(alpha[support].max()) + 0.1
    options = dict(
        reduction="beam_basis", double_channel=True, detectors=detector, lazy=False
    )
    full = run(sm, tp, scan, atoms, **options)
    omitted = np.fft.fftshift(alpha > angle)
    assert full.array[..., omitted].sum() > 0.5 * full.array.sum()
    with pytest.raises(ValueError, match="collection_angle"):
        run(sm, tp, scan, atoms, collection_angle=angle, **options)
