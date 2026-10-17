"""Tests for energy ensemble support in PlaneWave, Probe, Waves, BlochWaves, and SMatrix."""

import numpy as np
import os
import tempfile

import ase
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import abtem
from abtem.bloch.dynamical import BlochWaves
from abtem.core.axes import EnergyAxis, ThicknessAxis
from abtem.measurements import IndexedDiffractionPatterns
from abtem.prism.s_matrix import SMatrix, SMatrixArray
from abtem.multislice import RealSpaceMultislice
from abtem.waves import PlaneWave, Probe, Waves
from utils import devices, gpu, lazy_params, si_cubic_atoms

ENERGIES = [80e3, 200e3, 300e3]

# Shared by the BlochWaves and SMatrix (PRISM) energy-ensemble test classes
# below -- previously duplicated as BLOCH_ENERGIES / PRISM_ENERGIES.
TEST_ENERGIES = [100e3, 200e3, 300e3]


class TestPlaneWaveEnergyEnsemble:
    def test_build(self):
        """A list or an array of energies gives an EnergyAxis, built eagerly or
        lazily alike. (The members of a plane wave are all ones at any energy;
        their energy dependence is tested through multislice below.)"""
        eager = None
        for energy in (ENERGIES, np.array(ENERGIES)):
            pw = PlaneWave(energy=energy, gpts=32, sampling=0.1)
            assert pw.ensemble_shape == (3,)
            for lazy in (False, True):
                waves = pw.build(lazy=lazy)
                assert waves.shape == (3, 32, 32)
                assert isinstance(waves.ensemble_axes_metadata[0], EnergyAxis)
                assert waves.ensemble_axes_metadata[0].values == tuple(ENERGIES)
                waves = waves.compute()
                if eager is None:
                    eager = waves.array
                np.testing.assert_array_equal(waves.array, eager)

    def test_to_zarr_from_zarr(self):
        waves = PlaneWave(energy=ENERGIES, gpts=32, sampling=0.1).build(lazy=False)
        # a directory store and a zip store
        for filename in ("waves.zarr", "waves.zarr.zip"):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, filename)
                waves.to_zarr(path)
                loaded = Waves.from_zarr(path).compute()
            assert isinstance(loaded.ensemble_axes_metadata[0], EnergyAxis), filename
            assert loaded.ensemble_axes_metadata[0].values == tuple(ENERGIES), filename
            np.testing.assert_array_equal(loaded.array, waves.array, err_msg=filename)
            assert loaded[1].wavelength == waves[1].wavelength, filename

    # The energies are split in the multislice transform, ahead of the
    # algorithm, eagerly over the whole array and lazily per block; the lazy
    # path does not depend on the algorithm, so RealSpaceMultislice runs eagerly.
    @pytest.mark.parametrize(
        "algorithm, lazy",
        [(None, False), (None, True), (RealSpaceMultislice(order=1), False)],
        ids=["default-eager", "default-lazy", "realspace-eager"],
    )
    def test_multislice_members_match_single_energy(self, algorithm, lazy):
        """Oracle: each member of an energy ensemble is an independent
        simulation at that energy; the ensemble must reproduce each
        single-energy run, not just have the right shape. The energies give
        exit waves that differ by O(1), so a member computed with another
        member's wavelength, interaction constant or propagator is caught."""
        potential = abtem.Potential(
            _srtio3_atoms(), sampling=0.1, projection="finite"
        )
        kwargs = {} if algorithm is None else {"algorithm": algorithm}
        energies = [100e3, 200e3, 300e3]

        def run(energy):
            plane_wave = abtem.PlaneWave(energy=energy).match_grid(potential)
            return plane_wave.multislice(potential, lazy=lazy, **kwargs).compute()

        ensemble = run(energies)
        assert ensemble.shape[0] == len(energies)
        members = [np.asarray(run(energy).array) for energy in energies]
        # the members really are distinguishable
        assert np.abs(members[0] - members[-1]).max() > 0.1
        for i, member in enumerate(members):
            # the same single-precision operations in the same order; the
            # bound only allows for a different batching of the FFTs
            np.testing.assert_allclose(
                np.asarray(ensemble.array[i]), member, rtol=0, atol=1e-5,
                err_msg=f"energy member {i} ({energies[i]:g} eV)",
            )


class TestProbeEnergyEnsemble:
    @lazy_params
    def test_multislice_members_match_single_energy(self, lazy):
        """Oracle: each energy member equals an independent single-energy
        probe simulation, for a scanned (off-centre, non-square) set of probe
        positions, so the aperture (in mrad, hence energy dependent in
        1/A), the propagator and the interaction constant must all use the
        member's own energy. The potential is a CrystalPotential, which
        generates its slices with its own code."""
        potential = abtem.CrystalPotential(
            abtem.Potential(_srtio3_atoms(), sampling=0.1, projection="finite"),
            repetitions=(1, 1, 2),
        )
        energies = [60e3, 100e3, 200e3]
        positions = [(0.7, 1.3), (2.2, 0.4)]

        def run(energy):
            probe = Probe(energy=energy, semiangle_cutoff=20).match_grid(potential)
            return probe.multislice(potential, scan=positions, lazy=lazy).compute()

        ensemble = run(energies)
        energy_axis = [
            i for i, axis in enumerate(ensemble.ensemble_axes_metadata)
            if isinstance(axis, EnergyAxis)
        ]
        assert len(energy_axis) == 1
        ensemble = np.moveaxis(np.asarray(ensemble.array), energy_axis[0], 0)
        assert ensemble.shape[:2] == (len(energies), len(positions))
        members = [np.asarray(run(energy).array) for energy in energies]
        scale = max(np.abs(member).max() for member in members)
        assert np.abs(members[0] - members[-1]).max() > 0.1 * scale
        for i, member in enumerate(members):
            np.testing.assert_allclose(
                ensemble[i], member, rtol=0, atol=1e-5 * scale,
                err_msg=f"energy member {i} ({energies[i]:g} eV)",
            )


class TestWavesBuilderEnergyProperty:
    """Regression tests: PlaneWave.energy / Probe.energy must return the
    unwrapped value (float or BaseDistribution), not the internal
    EnergyEnsemble wrapper -- and detector.show() must accept a WavesBuilder
    directly (not just a built Waves object)."""

    def test_energy_is_a_plain_float_or_a_distribution(self):
        """A scalar energy stays a float with no ensemble axes; several energies
        are a distribution with one EnergyAxis and leave the accelerator without
        an energy."""
        from abtem.distributions import BaseDistribution

        builders = {
            PlaneWave: {},
            Probe: dict(semiangle_cutoff=30),
            SMatrix: dict(semiangle_cutoff=PRISM_SEMIANGLE),
        }
        for cls, kwargs in builders.items():
            name = cls.__name__
            scalar = cls(energy=100e3, gpts=32, sampling=0.1, **kwargs)
            assert scalar.energy == 100e3, name
            assert isinstance(scalar.energy, float), name
            assert scalar.accelerator.energy == 100e3, name
            assert scalar.ensemble_shape == (), name
            assert scalar.ensemble_axes_metadata == [], name

            ensemble = cls(energy=TEST_ENERGIES, gpts=32, sampling=0.1, **kwargs)
            assert isinstance(ensemble.energy, BaseDistribution), name
            np.testing.assert_array_equal(ensemble.energy.values, TEST_ENERGIES)
            assert ensemble.accelerator.energy is None, name
            assert ensemble.metadata["energy"] is None, name
            assert ensemble.ensemble_shape == (3,), name
            (axis,) = ensemble.ensemble_axes_metadata
            assert isinstance(axis, EnergyAxis), name
            assert axis.values == tuple(TEST_ENERGIES), name

        waves = PlaneWave(energy=100e3, gpts=32, sampling=0.1).build(lazy=False)
        assert waves.shape == (32, 32)
        assert waves.energy == 100e3

    def test_annular_detector_show_unbuilt_probe(self):
        """AnnularDetector.show() must accept a Probe directly, without build()."""
        probe = Probe(
            energy=80e3, semiangle_cutoff=20, sampling=0.1, extent=10
        )
        abtem.AnnularDetector(inner=40, outer=80).show(probe)
        plt.close("all")

    def test_segmented_detector_show_unbuilt_plane_wave(self):
        """SegmentedDetector.show() must accept a PlaneWave directly, without build()."""
        from abtem.detectors import SegmentedDetector

        pw = PlaneWave(energy=80e3, sampling=0.1, extent=10)
        SegmentedDetector(
            nbins_radial=2, nbins_azimuthal=4, inner=10, outer=80
        ).show(pw)
        plt.close("all")


class TestWavesEnergyEnsemble:
    def test_energy_of_the_ensemble_and_its_members(self):
        """A list of energies gives an EnergyAxis and no scalar energy. The full
        ensemble's angular sampling uses the highest energy (the finest grid
        in mrad, conservative for every member); an indexed member, or an
        ensemble of one energy given only by its axis, uses its own energy."""
        from abtem.core.energy import energy2wavelength

        scalar = Waves(np.ones((32, 32), dtype=complex), energy=100e3, sampling=0.1)
        assert scalar.energy == 100e3
        assert scalar.ensemble_axes_metadata == []

        energies = [80e3, 100e3, 120e3]
        w = Waves(np.ones((3, 32, 32), dtype=complex), energy=energies, sampling=0.1)
        assert isinstance(w.ensemble_axes_metadata[0], EnergyAxis)
        assert w.ensemble_axes_metadata[0].values == tuple(energies)
        assert w.metadata.get("energy") is None

        def angular_sampling(energy):
            wavelength = energy2wavelength(energy)
            return tuple(d * wavelength * 1e3 for d in w.reciprocal_space_sampling)

        assert np.allclose(w.angular_sampling, angular_sampling(max(energies)))
        for i, energy in enumerate(energies):
            assert w[i].metadata["energy"] == energy
            assert w[i].wavelength == energy2wavelength(energy)
            assert np.allclose(w[i].angular_sampling, angular_sampling(energy))

        single = Waves(
            np.ones((1, 32, 32), dtype=complex),
            sampling=0.1,
            ensemble_axes_metadata=[EnergyAxis(values=(100e3,))],
        )
        assert single.wavelength == energy2wavelength(100e3)

    def test_a_scalar_energy_contradicting_the_energy_axis_is_refused(self):
        """The EnergyAxis is the one record of the energies of an ensemble: a
        scalar energy, given as the energy or in the metadata, is refused unless
        it is the energy of every member."""
        array = np.ones((2, 8, 8), dtype=complex)
        axis = EnergyAxis(values=(80e3, 200e3))
        with pytest.raises(ValueError, match="contradicts the EnergyAxis"):
            Waves(array, energy=80e3, sampling=0.1, ensemble_axes_metadata=[axis])
        with pytest.raises(ValueError, match="contradicts the EnergyAxis"):
            abtem.DiffractionPatterns(
                np.ones((2, 8, 8)),
                sampling=0.1,
                ensemble_axes_metadata=[axis],
                metadata={"energy": 80e3},
            )
        one = [EnergyAxis(values=(80e3,))]
        with pytest.raises(ValueError, match="contradicts the EnergyAxis"):
            Waves(array[:1], energy=200e3, sampling=0.1, ensemble_axes_metadata=one)
        member = Waves(array[:1], energy=80e3, sampling=0.1, ensemble_axes_metadata=one)
        assert member.energy == 80e3

    def test_stack_along_an_energy_axis_leaves_no_scalar_energy(self):
        """The axis carries each member's energy; the scalar energy of the first
        member would contradict it."""
        members = [
            Waves(np.ones((32, 32), dtype=complex), energy=energy, sampling=0.1)
            for energy in ENERGIES
        ]
        stacked = abtem.stack(members, EnergyAxis(values=tuple(ENERGIES)))
        assert stacked.accelerator.energy is None
        # the key stays, as for a probe built with several energies
        assert "energy" in stacked.metadata
        assert stacked.metadata["energy"] is None

        built = Waves(np.ones((3, 32, 32), dtype=complex), energy=ENERGIES, sampling=0.1)
        assert stacked.angular_sampling == built.angular_sampling
        for i, member in enumerate(members):
            assert stacked[i].wavelength == member.wavelength

    @pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
    def test_waves_without_an_energy_metadata_key_can_be_detected(self, lazy):
        """A producer that leaves out metadata["energy"] does not make a transform
        that returns Waves raise."""
        waves = Waves(
            np.ones((3, 32, 32), dtype=complex),
            sampling=0.1,
            ensemble_axes_metadata=[EnergyAxis(values=tuple(ENERGIES))],
        )
        assert waves.accelerator.energy is None
        assert "energy" not in waves.metadata
        if lazy:
            waves = waves.ensure_lazy()

        detected = abtem.WavesDetector().detect(waves)
        if lazy:
            detected = detected.compute(progress_bar=False)

        assert detected.array.shape == (3, 32, 32)
        assert detected.metadata.get("energy") is None
        assert [type(axis) for axis in detected.ensemble_axes_metadata] == [EnergyAxis]


class TestWavesEnergyEnsembleDiffractionPatterns:
    """Regression tests for three failures reported by a user working with an
    energy-ensemble exit-wave stack produced by RealSpaceMultislice.

    All three failures share the same root cause: when energy is stored as an
    EnergyAxis in ensemble_axes_metadata, metadata["energy"] is None, causing
    energy-dependent geometry calculations to raise TypeError / EnergyUndefinedError.
    """

    def _exit_waves(self):
        """Minimal energy-ensemble Waves (3 energies, 32×32 grid)."""
        arr = np.ones((3, 32, 32), dtype=complex)
        return Waves(arr, energy=[80e3, 100e3, 120e3], sampling=0.1)

    def test_energy_of_the_patterns_and_their_members(self):
        """diffraction_patterns(), with and without max_angle, keeps the
        EnergyAxis. The full ensemble's geometry is that of its highest energy
        (block_direct and the other methods are compared with single-energy
        runs in test_multi_energy_ensembles.py); an indexed member carries its
        own energy and angular sampling."""
        energies = [80e3, 100e3, 120e3]

        def single(energy, **kwargs):
            waves = Waves(np.ones((32, 32), dtype=complex), energy=energy, sampling=0.1)
            return waves.diffraction_patterns(**kwargs)

        for kwargs in ({}, dict(max_angle=30)):
            dp = self._exit_waves().diffraction_patterns(**kwargs)
            assert isinstance(dp.ensemble_axes_metadata[0], EnergyAxis), kwargs
            assert dp.metadata.get("energy") is None, kwargs

            highest = single(max(energies), **kwargs)
            assert dp.shape[1:] == highest.shape, kwargs
            assert np.allclose(dp.angular_sampling, highest.angular_sampling), kwargs
            assert np.allclose(dp.max_angles, highest.max_angles), kwargs
            for i, energy in enumerate(energies):
                assert dp[i].metadata["energy"] == energy, kwargs
                assert np.allclose(
                    dp[i].angular_sampling, single(energy, **kwargs).angular_sampling
                ), (kwargs, energy)


class TestEnergyEnsembleSpotIndexing:
    """index_diffraction_spots on an un-indexed energy ensemble must index each
    member with its own energy -- filtering and overlapping-spot assignment
    both depend on it -- not with one energy for the whole ensemble (it used
    to take _get_energy()'s fallback, the maximum, for every member)."""

    ENERGIES = (80e3, 300e3)
    ROTATIONS = (0.0, 10.0, 20.0)
    # 49 x 49 pixels at 0.1 1/A hold 150 and 63 spots at 80 and 300 keV, so
    # indexing a member at the other energy changes its spots
    GPTS = (49, 49)

    @staticmethod
    def _cell():
        from ase.build import bulk

        return bulk("Si", cubic=True).cell

    def _orientation_matrices(self):
        from scipy.spatial.transform import Rotation

        # (N, 1) angles for a single-axis sequence: newer SciPy rejects (N,)
        angles = np.asarray(self.ROTATIONS)[:, None]
        return Rotation.from_euler("x", angles, degrees=True).as_matrix()

    def _patterns(self, chunks=None):
        from abtem.core.axes import NonLinearAxis
        from abtem.measurements import DiffractionPatterns

        # indexing only reads the pattern array and the geometry, so random
        # patterns exercise it fully
        array = np.random.default_rng(0).random(
            (len(self.ROTATIONS), len(self.ENERGIES)) + self.GPTS
        )
        if chunks is not None:
            import dask.array as da

            array = da.from_array(array, chunks=chunks + self.GPTS)
        return DiffractionPatterns(
            array,
            sampling=(0.1, 0.1),
            fftshift=True,
            ensemble_axes_metadata=[
                NonLinearAxis(label="x_rotation", units="deg", values=self.ROTATIONS),
                EnergyAxis(values=self.ENERGIES),
            ],
        )

    @staticmethod
    def _assert_member_matches(ensemble, values, single, rtol=0.0):
        """`values`, one energy member of the array of `ensemble`, equals the
        single-energy `single` on its spots and is zero on the other spots."""
        from abtem.core.backend import asnumpy

        index = {tuple(h): i for i, h in enumerate(ensemble.miller_indices)}
        idx = [index[tuple(h)] for h in single.miller_indices]
        others = np.setdiff1d(np.arange(len(index)), idx)
        values = asnumpy(values)
        np.testing.assert_allclose(
            values[..., idx], np.asarray(single.array), rtol=rtol, atol=0
        )
        assert np.all(values[..., others] == 0)

    # a rotation series is only indexable lazily, one orientation per block
    @pytest.mark.parametrize("energy_chunk", [1, 2])
    def test_lazy_rotation_series_matches_single_energy(self, energy_chunk):
        dp = self._patterns(chunks=(1, energy_chunk))
        om = self._orientation_matrices()
        ensemble = dp.index_diffraction_spots(
            cell=self._cell(), orientation_matrices=om[:, None], centering="F"
        ).compute()
        for j in range(len(self.ENERGIES)):
            single = (
                dp[:, j]
                .index_diffraction_spots(
                    cell=self._cell(), orientation_matrices=om, centering="F"
                )
                .compute()
            )
            self._assert_member_matches(
                ensemble, np.asarray(ensemble.array)[:, j], single
            )

    # an integration radius has its own device handling (the disk weights)
    @pytest.mark.parametrize("device", ["cpu", gpu])
    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("radius", [None, 0.25])
    def test_members_match_single_energy_on_device(self, device, lazy, radius):
        """Each member of a single orientation equals the member indexed on its
        own on the CPU. Every array the indexing combines with the patterns
        lives on their device (CuPy refuses implicit NumPy conversion), and the
        result stays there."""
        from abtem.core.backend import copy_to_device, get_array_module
        from abtem.measurements import DiffractionPatterns

        reference = self._patterns()[1]
        om = self._orientation_matrices()[1]
        array = copy_to_device(np.asarray(reference.array), device)
        if lazy:
            import dask.array as da

            array = da.from_array(array, chunks=(1,) + self.GPTS)
        dp = DiffractionPatterns(
            array,
            sampling=reference.sampling,
            fftshift=True,
            ensemble_axes_metadata=reference.ensemble_axes_metadata,
        )
        kwargs = dict(cell=self._cell(), orientation_matrices=om, centering="F",
                      radius=radius)
        result = dp.index_diffraction_spots(**kwargs)
        if lazy:
            result = result.compute()
        assert get_array_module(result.array) is get_array_module(
            copy_to_device(np.zeros(1), device)
        )
        singles = [
            reference[j].index_diffraction_spots(**kwargs)
            for j in range(len(self.ENERGIES))
        ]
        # indexing every member at one energy would fail the comparison
        assert len(singles[0].miller_indices) != len(singles[1].miller_indices)
        for j, single in enumerate(singles):
            self._assert_member_matches(result, result.array[j], single, rtol=1e-6)


class TestCTFEnergyEnsemble:
    """Regression tests for CTF / Aperture with energy-ensemble Probe."""

    def _probe(self):
        return Probe(
            sampling=0.05, extent=20, energy=[40e3, 60e3, 80e3], semiangle_cutoff=20
        )

    def test_probe_ctf_is_energy_ensemble(self):
        """probe.ctf carries the probe's energies as an EnergyAxis, and its
        diffraction patterns are on the grid of the highest energy, whose
        member is that energy's own (see test_to_diffraction_patterns_grid)."""
        ctf = self._probe().ctf
        assert len(ctf.ensemble_axes_metadata) == 1
        assert isinstance(ctf.ensemble_axes_metadata[0], EnergyAxis)
        assert tuple(ctf.ensemble_axes_metadata[0].values) == (40e3, 60e3, 80e3)

        patterns = ctf.to_diffraction_patterns(max_angle=20)
        assert isinstance(patterns.ensemble_axes_metadata[0], EnergyAxis)
        single = Probe(
            sampling=0.05, extent=20, energy=80e3, semiangle_cutoff=20
        ).ctf.to_diffraction_patterns(max_angle=20)
        np.testing.assert_allclose(patterns.array[-1], single.array)

    def test_apply_ctf_on_multi_energy_waves_can_be_detected(self):
        """The stacked result of a per-energy CTF stays a multi-energy ensemble that
        a `WavesDetector` accepts."""
        from abtem.transfer import CTF

        waves = Probe(
            sampling=0.1, extent=10, energy=[40e3, 60e3, 80e3], semiangle_cutoff=20
        ).build(lazy=False)
        applied = waves.apply_ctf(CTF(defocus=50))
        assert applied.accelerator.energy is None
        assert applied.metadata.get("energy") is None

        detected = abtem.WavesDetector().detect(applied)
        assert detected.array.shape == applied.array.shape
        np.testing.assert_array_equal(detected.array, applied.array)


class TestTransferFunctionEnergyMatching:
    """A transfer function whose energy is a distribution is matched to the
    EnergyAxis of a multi-energy ensemble and evaluated per member, rather than
    adding a second energy axis. Oracle: the same simulation run separately at
    each single energy."""

    CTF_KWARGS = dict(defocus=50.0, Cs=-20e4, semiangle_cutoff=20.0, focal_spread=20.0)

    @staticmethod
    def _potential(device="cpu"):
        atoms = ase.Atoms(
            "CSi", positions=[(1.5, 2.0, 1.0), (3.5, 3.0, 2.0)], cell=(5, 5, 3)
        )
        return abtem.Potential(atoms, gpts=48, slice_thickness=1.0, device=device)

    def _exit_waves(self, energy, lazy, device="cpu"):
        return PlaneWave(energy=energy, device=device).multislice(
            self._potential(device), lazy=lazy
        )

    @staticmethod
    def _intensity(waves):
        return waves.intensity().compute().to_cpu().array

    def _oracle(self, ctf_kwargs=None):
        ctf_kwargs = self.CTF_KWARGS if ctf_kwargs is None else ctf_kwargs
        return np.stack(
            [
                self._exit_waves(energy, lazy=False)
                .apply_ctf(abtem.CTF(energy=energy, **ctf_kwargs))
                .intensity()
                .array
                for energy in TEST_ENERGIES
            ]
        )

    # Waves.apply_ctf and CTF.apply are one-line wrappers of
    # Waves.apply_transform: they share one code path, so they are looped over
    _APPLY = {
        "apply_ctf": lambda waves, ctf: waves.apply_ctf(ctf),
        "ctf.apply": lambda waves, ctf: ctf.apply(waves),
        "apply_transform": lambda waves, ctf: waves.apply_transform(ctf),
    }

    @devices
    @lazy_params
    def test_matches_single_energy_simulations(self, lazy, device):
        waves = self._exit_waves(TEST_ENERGIES, lazy, device)
        oracle = self._oracle()
        for name, apply in self._APPLY.items():
            ctf = abtem.CTF(energy=TEST_ENERGIES, **self.CTF_KWARGS)
            result = apply(waves, ctf)

            assert len(result.ensemble_axes_metadata) == 1, name
            assert isinstance(result.ensemble_axes_metadata[0], EnergyAxis), name
            assert result.ensemble_axes_metadata[0].values == tuple(TEST_ENERGIES)
            assert result.accelerator.energy is None, name
            np.testing.assert_allclose(
                self._intensity(result), oracle, rtol=1e-4, atol=1e-6, err_msg=name
            )

    def test_matches_ctf_without_energy(self):
        """Matching the energies is equivalent to leaving the CTF energy unset, and
        neither CTF is left with the energy of a member."""
        waves = self._exit_waves(TEST_ENERGIES, lazy=False)
        matched_ctf = abtem.CTF(energy=TEST_ENERGIES, **self.CTF_KWARGS)
        unset_ctf = abtem.CTF(**self.CTF_KWARGS)
        matched = waves.apply_ctf(matched_ctf)
        unset = waves.apply_ctf(unset_ctf)
        assert np.array_equal(matched.array, unset.array)
        assert tuple(matched_ctf.energy.values) == tuple(TEST_ENERGIES)
        assert unset_ctf.energy is None
        # ... so both can be applied again
        assert np.array_equal(waves.apply_ctf(unset_ctf).array, unset.array)

    @devices
    def test_lazy_block_holding_several_energies(self, device):
        """A lazy block holding every energy is split into its energies too."""
        waves = self._exit_waves(TEST_ENERGIES, lazy=True, device=device)
        waves = waves.rechunk((len(TEST_ENERGIES), -1, -1))
        assert waves.array.chunks[0] == (len(TEST_ENERGIES),)
        ctf = abtem.CTF(energy=TEST_ENERGIES, **self.CTF_KWARGS)
        np.testing.assert_allclose(
            self._intensity(waves.apply_ctf(ctf)),
            self._oracle(),
            rtol=1e-4,
            atol=1e-6,
        )

    @devices
    def test_ensemble_mean(self, device):
        """The reported example: an ensemble-mean energy distribution on both."""
        distribution = abtem.distributions.from_values(
            TEST_ENERGIES, ensemble_mean=True
        )
        waves = self._exit_waves(distribution, lazy=True, device=device)
        ctf = abtem.CTF(energy=distribution, **self.CTF_KWARGS)
        intensity = waves.apply_ctf(ctf).intensity().reduce_ensemble().compute()
        assert intensity.ensemble_shape == ()
        np.testing.assert_allclose(
            intensity.to_cpu().array, self._oracle().mean(0), rtol=1e-4, atol=1e-6
        )

    @devices
    def test_weighted_ensemble_mean(self, device):
        """The weights of the wave functions decide the ensemble mean; those of the
        CTF's distribution are not used."""
        weights = [0.2, 0.7, 0.1]
        waves = self._exit_waves(
            abtem.distributions.from_values(
                TEST_ENERGIES, weights=weights, ensemble_mean=True
            ),
            lazy=True,
            device=device,
        )
        ctf = abtem.CTF(
            energy=abtem.distributions.from_values(
                TEST_ENERGIES, weights=[0.6, 0.2, 0.2], ensemble_mean=True
            ),
            **self.CTF_KWARGS,
        )
        intensity = waves.apply_ctf(ctf).intensity().reduce_ensemble().compute()
        assert intensity.ensemble_shape == ()
        np.testing.assert_allclose(
            intensity.to_cpu().array,
            np.tensordot(weights, self._oracle(), axes=1),
            rtol=1e-4,
            atol=1e-6,
        )

    @devices
    @lazy_params
    def test_with_defocus_series(self, lazy, device):
        """The CTF's other ensemble axes precede the matched energy axis, as for
        a single energy, and each (defocus, energy) member is the oracle's, also
        when each lazy block holds one energy."""
        defocus = abtem.distributions.from_values([0.0, 40.0, 80.0])
        waves = self._exit_waves(TEST_ENERGIES, lazy, device)
        if lazy:
            assert waves.array.chunks[0] == (1,) * len(TEST_ENERGIES)
        ctf = abtem.CTF(energy=TEST_ENERGIES, defocus=defocus, semiangle_cutoff=20.0)
        result = waves.apply_ctf(ctf)

        assert result.ensemble_shape == (3, len(TEST_ENERGIES))
        assert isinstance(result.ensemble_axes_metadata[1], EnergyAxis)
        if lazy:
            assert result.array.chunks[1] == (1,) * len(TEST_ENERGIES)

        intensity = self._intensity(result)
        for i, value in enumerate(defocus.values):
            oracle = self._oracle(dict(defocus=float(value), semiangle_cutoff=20.0))
            np.testing.assert_allclose(intensity[i], oracle, rtol=1e-4, atol=1e-6)

    def test_mismatched_energies_raise(self):
        waves = self._exit_waves(TEST_ENERGIES, lazy=False)
        with pytest.raises(ValueError, match="do not match the energies"):
            waves.apply_ctf(abtem.CTF(energy=[100e3, 200e3, 250e3], defocus=50))
        with pytest.raises(ValueError, match="do not match the energies"):
            waves.apply_ctf(abtem.CTF(energy=TEST_ENERGIES[::-1], defocus=50))
        with pytest.raises(ValueError, match="do not match the energies"):
            waves.apply_ctf(abtem.CTF(energy=TEST_ENERGIES[:2], defocus=50))

    def test_fixed_energy_on_ensemble_raises(self):
        waves = self._exit_waves(TEST_ENERGIES, lazy=False)
        with pytest.raises(ValueError, match="fixed energy"):
            abtem.Aperture(semiangle_cutoff=20, energy=100e3).apply(waves)

    @pytest.mark.parametrize("index", [None, 1])
    def test_single_energy_waves_raise(self, index):
        if index is None:
            waves = self._exit_waves(200e3, lazy=False)
        else:
            waves = self._exit_waves(TEST_ENERGIES, lazy=False)[index]
        with pytest.raises(ValueError, match="not an energy ensemble"):
            waves.apply_ctf(abtem.CTF(energy=TEST_ENERGIES, defocus=50))

    @devices
    def test_reused_after_single_energy_waves(self, device):
        """Applying a CTF to single-energy wave functions leaves it without their
        energy, so it can then be applied to a multi-energy ensemble."""
        single = self._exit_waves(200e3, lazy=False, device=device)
        ensemble = self._exit_waves(TEST_ENERGIES, lazy=False, device=device)
        oracle = self._oracle()
        for name in ("apply_ctf", "ctf.apply"):
            ctf = abtem.CTF(**self.CTF_KWARGS)
            for waves in (single, ensemble):
                result = self._APPLY[name](waves, ctf)
                assert ctf.energy is None, name
            np.testing.assert_allclose(
                self._intensity(result), oracle, rtol=1e-4, atol=1e-6, err_msg=name
            )

    def test_fixed_energy_on_mismatched_member_raises(self):
        """A fixed energy is checked against an indexed member's own energy.
        The check reads the metadata when the transform is applied, before any
        lazy graph is built, so eager wave functions stand for lazy ones."""
        member = PlaneWave(energy=TEST_ENERGIES, extent=5, gpts=32).build(lazy=False)[1]
        for name in ("apply_ctf", "ctf.apply"):
            apply = self._APPLY[name]
            ctf = abtem.CTF(energy=TEST_ENERGIES[0], **self.CTF_KWARGS)
            with pytest.raises(RuntimeError, match="Inconsistent energies"):
                apply(member, ctf)
            assert ctf.energy == TEST_ENERGIES[0], name

            matching = abtem.CTF(energy=TEST_ENERGIES[1], **self.CTF_KWARGS)
            assert apply(member, matching).metadata["energy"] == TEST_ENERGIES[1]

    def test_single_valued_energy_distribution_is_scalar(self):
        for energy in (200e3, [200e3]):
            ctf = abtem.CTF(energy=energy, defocus=50)
            assert ctf.energy == 200e3, energy
            assert ctf.ensemble_axes_metadata == [], energy

    def test_profiles(self):
        """Each energy member of the profiles is the single-energy profile on the
        common spatial-frequency grid, which reaches max_angle at the highest
        energy."""
        from abtem.core.energy import energy2wavelength

        max_angle = 30.0
        profiles = abtem.CTF(energy=TEST_ENERGIES, **self.CTF_KWARGS).profiles(
            max_angle=max_angle
        )
        assert isinstance(profiles.ensemble_axes_metadata[0], EnergyAxis)
        assert profiles.ensemble_shape[0] == len(TEST_ENERGIES)
        assert "energy" not in profiles.metadata

        shortest = energy2wavelength(max(TEST_ENERGIES))
        for i, energy in enumerate(TEST_ENERGIES):
            member_max_angle = max_angle * energy2wavelength(energy) / shortest
            oracle = abtem.CTF(energy=energy, **self.CTF_KWARGS).profiles(
                max_angle=member_max_angle
            )
            assert np.isclose(profiles.sampling, oracle.sampling)
            np.testing.assert_allclose(
                profiles.array[i], oracle.array, rtol=1e-4, atol=1e-6
            )
            assert profiles[i].metadata["energy"] == energy

    def test_profiles_require_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            abtem.CTF(energy=200e3, semiangle_cutoff=20).profiles(gpts=1)

    def test_to_diffraction_patterns_grid(self):
        """The default grid is sized from the highest energy, so that it reaches
        the semiangle cutoff for every member; each member is the single-energy
        aperture on that grid."""
        aperture = abtem.Aperture(semiangle_cutoff=20, energy=TEST_ENERGIES)
        patterns = aperture.to_diffraction_patterns()
        assert patterns.ensemble_shape == (len(TEST_ENERGIES),)
        assert isinstance(patterns.ensemble_axes_metadata[0], EnergyAxis)
        assert "energy" not in patterns.metadata

        highest = abtem.Aperture(
            semiangle_cutoff=20, energy=max(TEST_ENERGIES)
        ).to_diffraction_patterns()
        assert np.allclose(patterns.sampling, highest.sampling)
        np.testing.assert_allclose(
            patterns.array[TEST_ENERGIES.index(max(TEST_ENERGIES))], highest.array
        )


# ---------------------------------------------------------------------------
# SrTiO3 fixture used by BlochWaves tests
# ---------------------------------------------------------------------------

def _srtio3_atoms():
    return ase.Atoms(
        symbols="SrTiO3",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ],
        cell=[3.9127, 3.9127, 3.9127],
        pbc=True,
    )


# g_max/sg_max are as small as they can be while the two energies below still
# give clearly distinct active-beam sets: a 373-beam union, of which 40 % / 42 %
# are inactive at 100 / 200 keV (test_inactive_beams_are_zero needs > 10 %).
# g_max = 3 is too small (only ~7 % inactive at 200 keV). The previous g_max = 8,
# sg_max = 0.1 gave a ~4200-beam union, i.e. ~2400 x 2400 eigenproblems that
# made these classes the critical path of the whole --runslow suite (~9 min).
BLOCH_SG_MAX = 0.03
BLOCH_G_MAX = 4.5
BLOCH_THICKNESS = [20.0, 40.0]

# 2 energies is the minimum that exercises "a list of energies" (vs. the
# scalar-energy code path covered by test_scalar_energy_unchanged) and gives
# distinct per-energy active-beam sets for test_inactive_beams_are_zero.
BLOCH_ENERGIES = [100e3, 200e3]


class TestBlochWavesEnergyEnsemble:
    """Verify that BlochWaves accepts a list of energies and returns correctly
    shaped, stacked results without requiring manual loops."""

    @pytest.fixture(scope="class")
    @classmethod
    def bw_multi(cls):
        """Shared BlochWaves object with multiple energies (expensive to create)."""
        atoms = _srtio3_atoms()
        return BlochWaves(
            atoms, energy=BLOCH_ENERGIES, sg_max=BLOCH_SG_MAX, g_max=BLOCH_G_MAX
        )

    @pytest.fixture(scope="class")
    @classmethod
    def dp_single(cls, bw_multi):
        """Cached single-thickness diffraction patterns (used by multiple tests)."""
        return bw_multi.calculate_diffraction_patterns(BLOCH_THICKNESS[0]).compute()

    @pytest.fixture(scope="class")
    @classmethod
    def dp_multi(cls, bw_multi):
        """Cached multi-thickness diffraction patterns."""
        return bw_multi.calculate_diffraction_patterns(BLOCH_THICKNESS).compute()

    @pytest.fixture(scope="class")
    @classmethod
    def ew_single(cls, bw_multi):
        """Cached single-thickness exit waves."""
        return bw_multi.calculate_exit_waves(BLOCH_THICKNESS[0]).compute()

    @pytest.fixture(scope="class")
    @classmethod
    def ew_multi(cls, bw_multi):
        """Cached multi-thickness exit waves."""
        return bw_multi.calculate_exit_waves(BLOCH_THICKNESS).compute()

    def test_members_match_single_energy(
        self, dp_single, dp_multi, ew_single, ew_multi
    ):
        """Oracle: each energy member is an independent Bloch-wave calculation
        at that energy, for a scalar thickness and a list of thicknesses. Its
        beams (on the union beam set) must equal those of a single-energy run
        on that energy's own beam set, every other beam must be zero, and its
        exit wave must be the single-energy exit wave. The single-energy runs
        have no energy axis."""
        assert isinstance(dp_single, IndexedDiffractionPatterns)
        assert isinstance(ew_single, Waves)
        for result, axes in (
            (dp_single, [EnergyAxis]),
            (dp_multi, [EnergyAxis, ThicknessAxis]),
            (ew_single, [EnergyAxis]),
            (ew_multi, [EnergyAxis, ThicknessAxis]),
        ):
            assert [type(a) for a in result.ensemble_axes_metadata] == axes
            assert result.ensemble_axes_metadata[0].values == tuple(BLOCH_ENERGIES)
            assert result.shape[0] == len(BLOCH_ENERGIES)

        union = {tuple(hkl): j for j, hkl in enumerate(dp_single.miller_indices)}
        for i, energy in enumerate(BLOCH_ENERGIES):
            single = BlochWaves(
                _srtio3_atoms(), energy=energy, sg_max=BLOCH_SG_MAX,
                g_max=BLOCH_G_MAX,
            )
            for thicknesses, patterns, exit_waves in (
                (BLOCH_THICKNESS[0], dp_single, ew_single),
                (BLOCH_THICKNESS, dp_multi, ew_multi),
            ):
                message = f"member {i} ({energy:g} eV), thicknesses {thicknesses}"
                dp = single.calculate_diffraction_patterns(thicknesses).compute()
                ew = single.calculate_exit_waves(thicknesses).compute()
                for result in (dp, ew):
                    assert not any(
                        isinstance(a, EnergyAxis) for a in result.axes_metadata
                    ), message

                indices = [union[tuple(hkl)] for hkl in dp.miller_indices]
                member = np.asarray(patterns.array[i])
                reference = np.asarray(dp.array)
                np.testing.assert_allclose(
                    member[..., indices], reference, rtol=0,
                    atol=1e-6 * np.abs(reference).max(), err_msg=message,
                )
                # every beam outside the single-energy set carries no intensity
                outside = np.ones(member.shape[-1], dtype=bool)
                outside[indices] = False
                np.testing.assert_array_equal(member[..., outside], 0.0, message)

                reference = np.asarray(ew.array)
                np.testing.assert_allclose(
                    np.asarray(exit_waves.array[i]), reference, rtol=0,
                    atol=1e-5 * np.abs(reference).max(), err_msg=message,
                )

    def test_inactive_beams_are_zero(self, bw_multi, dp_single):
        """Beams inactive at a given energy must have zero intensity in the output.

        The `if inactive.any()` guard below would let this test pass
        vacuously if BLOCH_ENERGIES ever stopped producing distinct
        per-energy active-beam sets (e.g. too few/too-close energies for
        this cell). The assertion after the loop turns that silent
        degenerate case into a hard failure, rather than relying on a
        one-off empirical check quoted only in a comment (see
        BLOCH_ENERGIES above) staying true forever.
        """
        n_union = int(bw_multi._hkl_mask.sum())
        for i, sub in enumerate(bw_multi._energy_hkl_masks):
            inactive = ~sub  # positions active in union but NOT at energy i
            # A large fraction inactive at every energy is what makes this
            # test meaningful; BLOCH_ENERGIES was last verified to give
            # 40-42 % inactive beams per energy (out of a 373-beam union).
            assert inactive.sum() > 0.1 * n_union, (
                f"Energy index {i} has only {inactive.sum()}/{n_union} inactive "
                "beams -- BLOCH_ENERGIES may no longer give distinct enough "
                "per-energy active-beam sets for this test to be meaningful"
            )
            np.testing.assert_array_equal(
                dp_single.array[i, inactive], 0.0,
                err_msg=f"Inactive beams non-zero at energy index {i}",
            )


# ---------------------------------------------------------------------------
# SMatrix (PRISM) energy ensemble tests
# ---------------------------------------------------------------------------

PRISM_SEMIANGLE = 20.0
PRISM_GPTS = 64
PRISM_SAMPLING = 0.1


class TestSMatrixEnergyEnsemble:
    def test_build(self):
        """build() gives an SMatrixArray with the EnergyAxis, eagerly or lazily
        (its members are compared with single-energy runs after reduction in
        test_multi_energy_ensembles.py::
        test_multi_energy_build_reduce_equals_single_energy_runs)."""
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=TEST_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        eager = s.build(lazy=False)
        lazy = s.build(lazy=True)
        assert lazy.is_lazy
        for sma in (eager, lazy.compute()):
            assert isinstance(sma, SMatrixArray)
            assert sma.array.shape[0] == len(TEST_ENERGIES)
            assert isinstance(sma.ensemble_axes_metadata[0], EnergyAxis)
            assert sma.ensemble_axes_metadata[0].values == tuple(TEST_ENERGIES)
        np.testing.assert_allclose(lazy.compute().array, eager.array, atol=1e-6)

        # the expansion of the ensemble is that of the built array, the union of
        # the energies' expansions, which the highest energy's holds
        highest = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=max(TEST_ENERGIES),
                          gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)
        assert len(s) == len(eager.wave_vectors) == len(highest)
        assert len(SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=min(TEST_ENERGIES),
                           gpts=PRISM_GPTS, sampling=PRISM_SAMPLING)) < len(s)
        assert s.shape == eager.shape[:2] + s.gpts
        # the constructor arguments, from which every block is rebuilt
        # (_from_partitioned_args), keep every energy
        copied = SMatrix(**s._copy_kwargs())
        np.testing.assert_array_equal(copied.energy.values, TEST_ENERGIES)


# ---------------------------------------------------------------------------
# SMatrix reduce/scan energy ensemble tests
#
# The reduction contracts away the expansion axis, so each energy is reduced
# separately and only the measurements are stacked. This needs no shared
# expansion basis, hence it also covers the compressed (C-PRISM) reduction,
# whose per-energy SVD bases have different ranks and cannot be stacked.
# ---------------------------------------------------------------------------


# Each energy is reduced on its own and the measurements are stacked, eagerly
# or lazily alike for every mode; the lazy stacking is run on C-PRISM.
REDUCE_MODES = [
    pytest.param(False, 1, False, id="prism_interp1"),
    pytest.param(False, 2, False, id="prism_interp2"),
    pytest.param(True, 2, False, id="cprism_interp2"),
    pytest.param(True, 2, True, id="cprism_interp2-lazy"),
]


def _reduce_potential():
    # gpts must be divisible by the interpolation factor, otherwise SMatrix
    # warns about normalization and the test config turns that into an error
    atoms = abtem.orthogonalize_cell(si_cubic_atoms())
    return abtem.Potential(atoms, gpts=68)


class TestSMatrixReduceEnergyEnsemble:
    @pytest.mark.parametrize("upsample, interpolation, lazy", REDUCE_MODES)
    def test_scan_matches_single_energy(self, upsample, interpolation, lazy):
        """Each energy slice of every detector's measurement must equal an
        independent single-energy scan, which has no EnergyAxis."""
        potential = _reduce_potential()
        scan = abtem.GridScan(start=(0, 0), end=potential.extent, sampling=0.6)

        def detectors():
            return [
                abtem.AnnularDetector(inner=30, outer=70),
                abtem.PixelatedDetector(max_angle=None),
            ]

        kwargs = dict(semiangle_cutoff=PRISM_SEMIANGLE, potential=potential,
                      interpolation=interpolation, upsample=upsample)

        ensemble = SMatrix(energy=TEST_ENERGIES, **kwargs).scan(
            scan=scan, detectors=detectors(), lazy=lazy
        )
        if lazy:
            ensemble = ensemble.compute()
        assert len(ensemble) == len(detectors())
        for measurement in ensemble:
            assert measurement.shape[0] == len(TEST_ENERGIES)
            assert isinstance(measurement.ensemble_axes_metadata[0], EnergyAxis)
            assert measurement.ensemble_axes_metadata[0].values == tuple(
                TEST_ENERGIES
            )

        for i, energy in enumerate(TEST_ENERGIES):
            references = SMatrix(energy=energy, **kwargs).scan(
                scan=scan, detectors=detectors(), lazy=False
            )
            for measurement, reference in zip(ensemble, references):
                name = f"{type(reference).__name__} at {energy:g} eV"
                assert not any(
                    isinstance(axis, EnergyAxis)
                    for axis in reference.ensemble_axes_metadata
                ), name
                # max_angle=None is a pixel count, the same for every energy
                member = np.asarray(measurement.array[i])
                reference = np.asarray(reference.array)
                assert member.shape == reference.shape, name
                # lazy blocks are computed in the default single precision
                np.testing.assert_allclose(
                    member, reference, rtol=1e-5 if lazy else 1e-7,
                    atol=1e-5 * np.abs(reference).max(), err_msg=name,
                )

    def test_cprism_build_rejects_multiple_energies(self):
        """C-PRISM build() must reject energy ensembles with a clear error.

        CompressedSMatrixArray has no ensemble machinery to carry an EnergyAxis,
        so rather than fail obscurely downstream it refuses up front and points
        at reduce()/scan(), which support energy ensembles.
        """
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=TEST_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING,
                    interpolation=2, upsample=True)
        with pytest.raises(NotImplementedError, match="multiple energies"):
            s.build(lazy=False)

    def test_upsample_without_compression_allows_multiple_energies(self):
        """The guard is scoped to an active compression.

        At interpolation (1, 1) the expansion is already complete, so build()
        returns a plain SMatrixArray and the energy ensemble must still work.
        """
        s = SMatrix(semiangle_cutoff=PRISM_SEMIANGLE, energy=TEST_ENERGIES,
                    gpts=PRISM_GPTS, sampling=PRISM_SAMPLING,
                    interpolation=1, upsample=True)
        sma = s.build(lazy=False)
        assert isinstance(sma, SMatrixArray)
        assert sma.array.shape[0] == len(TEST_ENERGIES)
        assert isinstance(sma.ensemble_axes_metadata[0], EnergyAxis)
