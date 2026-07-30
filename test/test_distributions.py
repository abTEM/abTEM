import numpy as np

from abtem import distributions
from abtem.transfer import CTF, Aperture, Vortex
from abtem.waves import PlaneWave, Probe


def test_gaussian_distribution_normalized():
    defocus = distributions.gaussian(1.0, num_samples=11, center=3)
    wave = Probe(energy=100e3, semiangle_cutoff=30, defocus=0.0, extent=10, gpts=64)
    assert np.allclose(
        wave.build().diffraction_patterns().reduce_ensemble().array.sum().compute(), 1.0
    )


def test_focal_series_with_incoherent_spread():
    # Answers https://github.com/abTEM/abTEM/issues/168: a focal series (kept as
    # its own axis) where each defocus step is itself an incoherent
    # (temporal-coherence) average. Composing two apply_ctf() calls works because
    # the defocus phase term is linear in defocus, so applying CTF twice with
    # independent defocus distributions equals a single application with summed
    # defocus.
    #
    # Both distributions use ensemble_mean=False and the incoherent spread axis is
    # reduced with a plain .sum(): abTEM pre-multiplies the wave amplitude by the
    # L2-normalized distribution weight (so I_i = w_i**2 * I(x_i) with sum w_i**2 =
    # 1), which makes .sum() the correct incoherent average. This is the same
    # reduction the partial-coherence tutorial performs by hand.
    import ase

    atoms = ase.build.mx2(vacuum=2)
    exit_wave = PlaneWave(energy=80e3, sampling=0.1).multislice(atoms).compute()

    focal_series_values = np.array([-100.0, 0.0, 100.0])
    focal_series = distributions.from_values(
        focal_series_values, ensemble_mean=False
    )
    spread = distributions.gaussian(
        20.0, num_samples=7, sampling_limit=2, ensemble_mean=False
    )

    images = (
        exit_wave.apply_ctf(CTF(energy=80e3, defocus=focal_series))
        .apply_ctf(CTF(energy=80e3, defocus=spread))
        .intensity()
        .compute()
    )

    # locate the spread (7 values) and series (3 values) axes and reduce the spread
    spread_axis = next(
        i
        for i, ax in enumerate(images.axes_metadata)
        if getattr(ax, "values", None) is not None and len(ax.values) == 7
    )
    result = images.sum(spread_axis)
    series_axis = next(
        i
        for i, ax in enumerate(result.axes_metadata)
        if getattr(ax, "values", None) is not None and len(ax.values) == 3
    )
    reduced = np.moveaxis(result.array, series_axis, 0)

    # brute-force reference: per focus step, weighted incoherent average over the
    # spread (weights pre-multiplied as w_i**2, matching abTEM's convention)
    raw = distributions.gaussian(20.0, num_samples=7, sampling_limit=2)
    deltas, weights = np.array(raw.values), np.array(raw.weights)
    ref = np.zeros((len(focal_series_values),) + exit_wave.array.shape)
    for i, series_val in enumerate(focal_series_values):
        acc = np.zeros(exit_wave.array.shape)
        for delta, w in zip(deltas, weights):
            acc += (w**2) * (
                exit_wave.apply_ctf(CTF(energy=80e3, defocus=series_val + delta))
                .intensity()
                .compute()
                .array
            )
        ref[i] = acc

    assert reduced.shape[0] == 3  # focal series axis survives
    assert np.allclose(reduced, ref, atol=1e-4)


def test_vortex_quantum_number_ensemble():
    # Vortex(quantum_number=[...]) should build a l-ensemble in one Waves object,
    # matching per-l scalar Vortex runs bit-for-bit through multislice -- same
    # ensemble mechanism Aperture(semiangle_cutoff=[...]) already uses.
    import ase

    from abtem import Potential

    atoms = ase.build.bulk("Fe", "bcc", a=2.87, cubic=True)
    potential = Potential(atoms, sampling=0.1, slice_thickness=1.0)

    ls = (-1, 0, 1)
    probe = Probe(energy=100e3, semiangle_cutoff=20.0)
    probe.aperture = Vortex(quantum_number=ls, semiangle_cutoff=20.0, soft=True)
    probe.grid.match(potential)

    exit_waves = probe.build().multislice(potential).compute()
    assert exit_waves.array.shape[0] == len(ls)

    for i, l in enumerate(ls):
        probe_scalar = Probe(energy=100e3, semiangle_cutoff=20.0)
        probe_scalar.aperture = Vortex(quantum_number=l, semiangle_cutoff=20.0, soft=True)
        probe_scalar.grid.match(potential)
        exit_scalar = probe_scalar.build().multislice(potential).compute()
        assert np.allclose(exit_waves.array[i], exit_scalar.array, atol=1e-8)


def _probe_array(aperture, energy=100e3, gpts=64, extent=10.0):
    probe = Probe(energy=energy, semiangle_cutoff=20.0, gpts=gpts, extent=extent)
    probe.aperture = aperture
    return probe.build().compute().array


def test_vortex_l0_soft_matches_plain_soft_aperture():
    # The ensemble rewrite inlines soft_aperture()'s formula (it has to: that
    # helper sets its centre point by in-place index assignment, which does not
    # generalize to broadcast ensemble axes). For l=0 the vortex phase is unity,
    # so the result must be identical to the ordinary soft aperture.
    vortex = _probe_array(Vortex(quantum_number=0, semiangle_cutoff=20.0, soft=True))
    plain = _probe_array(Aperture(semiangle_cutoff=20.0, soft=True))
    assert np.allclose(vortex, plain, atol=1e-12)


def test_vortex_hard_aperture_ensemble_matches_scalars():
    ls = (-2, 0, 3)
    ens = _probe_array(Vortex(quantum_number=ls, semiangle_cutoff=20.0, soft=False))
    assert ens.shape[0] == len(ls)
    for i, l in enumerate(ls):
        scalar = _probe_array(Vortex(quantum_number=l, semiangle_cutoff=20.0, soft=False))
        assert np.allclose(ens[i], scalar, atol=1e-12)


def test_vortex_quantum_number_and_semiangle_ensembles_combine():
    ls, cutoffs = (-1, 1), (15.0, 20.0)
    ens = _probe_array(Vortex(quantum_number=ls, semiangle_cutoff=cutoffs, soft=True))
    assert ens.shape[:2] == (len(ls), len(cutoffs))
    for i, l in enumerate(ls):
        for j, cutoff in enumerate(cutoffs):
            scalar = _probe_array(
                Vortex(quantum_number=l, semiangle_cutoff=cutoff, soft=True)
            )
            assert np.allclose(ens[i, j], scalar, atol=1e-12)


def test_vortex_ensemble_preserves_mirror_symmetry():
    # +l and -l must remain exact mirror images: magnetic signals are extracted
    # as the difference between them, so any asymmetry becomes a spurious signal.
    # Compared on the angular grid, where the relation is plain conjugation --
    # in real space it would additionally involve the r -> -r inversion that
    # conjugating a Fourier transform induces.
    aperture = Vortex(quantum_number=(1, -1), semiangle_cutoff=20.0, soft=True,
                      energy=100e3, gpts=64, extent=10.0)
    kernel = aperture._evaluate_kernel()
    assert kernel.shape[0] == 2
    assert np.allclose(kernel[0], np.conj(kernel[1]), atol=1e-12)


def test_vortex_ensemble_axes_metadata():
    aperture = Vortex(quantum_number=(-1, 0, 1), semiangle_cutoff=(15.0, 20.0), soft=True)
    axes = aperture.ensemble_axes_metadata
    assert [ax.label for ax in axes] == ["quantum_number", "semiangle_cutoff"]
    assert [len(ax.values) for ax in axes] == [3, 2]
