import ase
import ase.build
import numpy as np
import pytest
from utils import devices, gpu, to_host_array

import abtem
from abtem.multislice import FourierMultislice, RealSpaceMultislice

# Suppress Numba performance warnings during tests
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning",
    "ignore::UserWarning"
)


def create_sto_atoms():
    """Create a SrTiO3 unit cell and supercell."""
    unit_cell = ase.Atoms(
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
    atoms = unit_cell * (2, 2, 4)
    return atoms, unit_cell


@pytest.fixture
def test_system(request):
    """Create complete test system with atoms, potential, probe, and scans."""
    # Get device from parametrization if available, otherwise default to "cpu"
    device = getattr(request, "param", "cpu")

    atoms, unit_cell = create_sto_atoms()

    # Standard potential
    potential = abtem.Potential(
        atoms,
        gpts=(80, 80),
        slice_thickness=0.75,
        projection="finite",
        device=device,
    )

    # Potential with exit planes
    potential_exit_planes = abtem.Potential(
        atoms,
        gpts=(80, 80),
        slice_thickness=0.75,
        exit_planes=1,
        projection="finite",
        device=device,
    )

    # Probe
    probe = abtem.Probe(
        semiangle_cutoff=20,
        energy=30e3,
        device=device,
    ).match_grid(potential)

    # Scans
    single_point_scan = [[0, 0]]
    grid_scan = abtem.GridScan(
        start=(0, 0),
        end=(unit_cell.cell[0, 0], unit_cell.cell[1, 1]),
        gpts=2,
    )

    return {
        "atoms": atoms,
        "unit_cell": unit_cell,
        "potential": potential,
        "potential_exit_planes": potential_exit_planes,
        "probe": probe,
        "single_point_scan": single_point_scan,
        "grid_scan": grid_scan,
        "device": device,
    }


class TestLazyVsEager:
    """Test that lazy and eager computations produce similar results."""

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.parametrize(
        "algorithm",
        [
            FourierMultislice(order=1),
            FourierMultislice(order=2),
            FourierMultislice(order='exact'),
            pytest.param(RealSpaceMultislice(order=1), marks=pytest.mark.slow),
            pytest.param(RealSpaceMultislice(order=2), marks=pytest.mark.slow),
            pytest.param(RealSpaceMultislice(order=3), marks=pytest.mark.slow),
        ],
    )
    def test_lazy_vs_eager_single_point(self, test_system, algorithm):
        """Test that lazy and eager give similar results for single point."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        # Lazy computation
        lazy_result = probe.multislice(
            potential=potential,
            scan=scan,
            algorithm=algorithm,
        ).compute()

        # Eager computation
        eager_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=algorithm,
        )

        # Check shapes match
        assert lazy_result.array.shape == eager_result.array.shape

        # Check values are close (allowing for numerical differences)
        np.testing.assert_allclose(
            to_host_array(lazy_result.array),
            to_host_array(eager_result.array),
            rtol=1e-5,
            atol=1e-8,
        )


class TestFourierMultislice:
    """Test FourierMultislice algorithm with various configurations."""

    @pytest.mark.parametrize("order", [1, 2,"exact"])
    def test_fourier_orders(self, test_system, order):
        """Test that FourierMultislice accepts valid orders."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=order),
        )
        assert result is not None
        assert hasattr(result, "array")

    def test_fourier_invalid_order(self, test_system):
        """Test that FourierMultislice rejects invalid orders."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        with pytest.raises(ValueError, match="Only order 1, 2, and 'exact' are supported"):
            probe.multislice(
                potential=potential,
                scan=scan,
                lazy=False,
                algorithm=FourierMultislice(order=3),  # type: ignore
            )

    @pytest.mark.parametrize("conjugate", [True, False])
    @pytest.mark.parametrize("transpose", [True, False])
    def test_fourier_conjugate_transpose(self, test_system, conjugate, transpose):
        """Test conjugate and transpose parameters."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(conjugate=conjugate, transpose=transpose),
        )
        assert result is not None


class TestRealSpaceMultislice:
    """Test RealSpaceMultislice algorithm with various configurations."""

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.parametrize("expansion_scope", ["propagator", "full"])
    @pytest.mark.slow
    def test_realspace_expansion_scope(self, test_system, expansion_scope):
        """Test different expansion scopes."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope=expansion_scope),
        )
        assert result is not None

    @pytest.mark.parametrize("derivative_accuracy", [4, 6, 8])
    def test_realspace_derivative_accuracy(self, test_system, derivative_accuracy):
        """Test different derivative accuracies."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(derivative_accuracy=derivative_accuracy),
        )
        assert result is not None

    def test_realspace_max_terms(self, test_system):
        """Test max_terms parameter."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(max_terms=100),
        )
        assert result is not None


class TestOutputShapes:
    """Test that output shapes match expected dimensions."""

    def test_single_point_shape(self, test_system):
        """Test that single point scan returns array with potential.gpts shape."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
        )

        # Shape should be potential.gpts for single scan point
        expected_shape = potential.gpts
        assert result.array.shape == expected_shape

    def test_grid_scan_shape(self, test_system):
        """Test that grid scan returns array with grid_scan.shape + potential.gpts."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        grid_scan = test_system["grid_scan"]

        result = probe.multislice(
            potential=potential,
            scan=grid_scan,
            lazy=False,
        )

        # Shape should be grid_scan.shape + potential.gpts
        expected_shape = grid_scan.shape + potential.gpts
        assert result.array.shape == expected_shape

    def test_exit_planes_single_point_shape(self, test_system):
        """Test that exit planes potential returns correct shape."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            lazy=False,
        )

        # Shape should be (num_exit_planes,) + potential.gpts
        expected_shape = (
            potential_exit_planes.num_exit_planes,
        ) + potential_exit_planes.gpts
        assert result.array.shape == expected_shape

    def test_exit_planes_grid_scan_shape(self, test_system):
        """Test exit planes with grid scan."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        grid_scan = test_system["grid_scan"]

        result = probe.multislice(
            potential=potential_exit_planes,
            scan=grid_scan,
            lazy=False,
        )

        # Shape should be grid_scan.shape + (num_exit_planes,) + potential.gpts
        expected_shape = (
            (potential_exit_planes.num_exit_planes,)
            + grid_scan.shape
            + potential_exit_planes.gpts
        )
        assert result.array.shape == expected_shape


class TestDetectors:
    """Test multislice with various detector configurations."""

    def test_single_detector(self, test_system):
        """Test with a single detector."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        detector = abtem.PixelatedDetector()
        result = probe.multislice(
            potential=potential,
            scan=scan,
            detectors=detector,
            lazy=False,
        )
        assert result is not None

    def test_multiple_detectors(self, test_system):
        """Test with multiple detectors."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
        ]
        results = probe.multislice(
            potential=potential,
            scan=scan,
            detectors=detectors,
            lazy=False,
        )

        # Should return a list of measurements
        assert isinstance(results, list)
        assert len(results) == 2

    def test_no_detector_returns_waves(self, test_system):
        """Test that no detector returns Waves object."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
        )

        # Should return Waves object
        assert hasattr(result, "array")


class TestBackscattering:
    """Test backscattering calculations."""

    def test_backscattering_requires_full_expansion(self, test_system):
        """Test that backscattering requires expansion_scope='full'."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        with pytest.raises(
            ValueError,
            match="Backscattering contributions require expansion_scope='full'",
        ):
            probe.multislice(
                potential=potential_exit_planes,
                scan=scan,
                lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="propagator"),
                return_backscattered=True,
            )

    def test_backscattering_requires_exit_planes(self, test_system):
        """Test that backscattering requires exit_planes."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        with pytest.raises(
            ValueError,
            match="Backscattering contributions require potential.exit_planes",
        ):
            probe.multislice(
                potential=potential,
                scan=scan,
                lazy=False,
                algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
                return_backscattered=True,
            )

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.slow
    def test_backscattering_returns_extra_waves(self, test_system):
        """Test that backscattering adds an extra detector (WavesDetector)."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        result = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            return_backscattered=True,
        )

        # Should return a tuple: (forward_waves, backward_waves)
        assert isinstance(result, (list, tuple))
        assert len(result) == 2

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.slow
    def test_backscattering_with_detectors(self, test_system):
        """Test backscattering with additional detectors."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
        ]
        results = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            detectors=detectors,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            return_backscattered=True,
        )

        # Should return N+1 results (N detectors + 1 backscattered waves)
        assert isinstance(results, (list, tuple))
        assert len(results) == len(detectors) + 1

    @pytest.mark.parametrize("test_system", ["cpu", gpu], indirect=True)
    @pytest.mark.slow
    def test_backscattering_shape_consistency(self, test_system):
        """Test that forward and backward waves have consistent shapes."""
        probe = test_system["probe"]
        potential_exit_planes = test_system["potential_exit_planes"]
        scan = test_system["single_point_scan"]

        forward, backward = probe.multislice(
            potential=potential_exit_planes,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3, expansion_scope="full"),
            return_backscattered=True,
        )

        # Forward and backward should have same spatial dimensions
        assert forward.array.shape[-2:] == backward.array.shape[-2:]


def _multislice_arrays(
    potential, lazy, scan=None, backscattered=True, potential_chunk_size="auto"
):
    """The arrays of a full-expansion real-space multislice: the transmitted
    waves, and with `backscattered` also the backscattered waves."""
    algorithm = RealSpaceMultislice(expansion_scope="full")
    kwargs = dict(
        lazy=lazy,
        return_backscattered=backscattered,
        potential_chunk_size=potential_chunk_size,
        algorithm=algorithm,
    )
    if scan is None:
        result = abtem.PlaneWave(energy=100e3).multislice(potential, **kwargs)
    else:
        result = abtem.Probe(energy=100e3, semiangle_cutoff=20).multislice(
            potential, scan=scan, **kwargs
        )
    result = result if backscattered else [result]
    if lazy:
        result = [
            r.compute(scheduler="synchronous", progress_bar=False) for r in result
        ]
    return [r.array for r in result]


class TestBackscatteringEnsemble:
    """The backscattered waves of an ensemble potential are those of each
    configuration run on its own."""

    @pytest.mark.parametrize(
        "num_configs, slices_per_cell, scan, lazy",
        [
            pytest.param(
                *case,
                lazy,
                marks=[pytest.mark.slow] if lazy and case[0] == 3 else [],
            )
            for case in [
                (2, 2, None),  # 3 exit planes
                (3, 2, None),  # as many configurations as exit planes
                (2, 3, None),  # 4 exit planes
                (3, 2, [(1.0, 1.0), (2.0, 3.0)]),  # two probe positions
            ]
            for lazy in (False, True)
        ],
    )
    def test_frozen_phonons(self, num_configs, slices_per_cell, scan, lazy):
        atoms = ase.build.bulk("Si", cubic=True)
        frozen_phonons = abtem.FrozenPhonons(atoms, num_configs, sigmas=0.1, seed=1)
        kwargs = dict(
            gpts=(24, 20),
            slice_thickness=atoms.cell[2, 2] / slices_per_cell,
            exit_planes=1,
        )

        members = [
            _multislice_arrays(abtem.Potential(config, **kwargs), False, scan)
            for config in frozen_phonons
        ]
        result = _multislice_arrays(
            abtem.Potential(frozen_phonons, **kwargs), lazy, scan
        )

        for i, output in enumerate(result):
            expected = np.stack([member[i] for member in members])
            assert output.shape == expected.shape
            np.testing.assert_allclose(
                output, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
            )

    @pytest.mark.parametrize(
        "num_frozen_phonons, lazy",
        [
            pytest.param(
                num_frozen_phonons,  # 5 exit planes
                lazy,
                marks=[pytest.mark.slow] if lazy and num_frozen_phonons == 5 else [],
            )
            for num_frozen_phonons in (2, 5)
            for lazy in (False, True)
        ],
    )
    def test_crystal_potential(self, num_frozen_phonons, lazy):
        atoms = ase.build.bulk("Si", cubic=True)

        def crystal():
            unit = abtem.Potential(
                abtem.FrozenPhonons(atoms, 4, sigmas=0.1, seed=1),
                gpts=(24, 20),
                slice_thickness=atoms.cell[2, 2] / 2,
            )
            return abtem.CrystalPotential(
                unit,
                (2, 1, 2),
                num_frozen_phonons=num_frozen_phonons,
                seeds=tuple(range(1, num_frozen_phonons + 1)),
                exit_planes=1,
            )

        members = []
        for _, _, member in crystal().generate_blocks():
            member = member.item()
            chunks = list(member.generate_chunked_slices())
            slices = abtem.PotentialArray(
                np.concatenate([chunk.array for chunk in chunks]),
                slice_thickness=[t for chunk in chunks for t in chunk.slice_thickness],
                sampling=member.sampling,
                exit_planes=1,
            )
            members.append(_multislice_arrays(slices, False))

        result = _multislice_arrays(crystal(), lazy)

        for i, output in enumerate(result):
            expected = np.stack([member[i] for member in members])
            assert output.shape == expected.shape
            np.testing.assert_allclose(
                output, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
            )


def _full_expansion_potentials():
    atoms = ase.build.bulk("Si", cubic=True)
    grid = dict(gpts=(24, 20), slice_thickness=atoms.cell[2, 2] / 3)  # 6 slices

    def displaced(atoms):
        return list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]

    return {
        "potential": lambda: abtem.Potential(
            displaced(atoms * (1, 1, 2)), exit_planes=1, **grid
        ),
        "frozen_phonons": lambda: abtem.Potential(
            abtem.FrozenPhonons(atoms * (1, 1, 2), 4, sigmas=0.1, seed=1),
            exit_planes=2,
            **grid,
        ),
        "crystal_potential": lambda: abtem.CrystalPotential(
            abtem.Potential(abtem.FrozenPhonons(atoms, 4, sigmas=0.1, seed=1), **grid),
            (1, 1, 2),
            num_frozen_phonons=3,
            seeds=(1, 2, 3),
            exit_planes=1,
        ),
    }


class TestFullExpansionChunking:
    @pytest.mark.parametrize("backscattered", [False, True])
    @pytest.mark.parametrize(
        "name, lazy",
        [
            pytest.param(
                name,
                lazy,
                marks=[pytest.mark.slow] if lazy and name != "potential" else [],
            )
            for name in _full_expansion_potentials()
            for lazy in (False, True)
        ],
    )
    def test_independent_of_the_potential_chunk_size(self, name, backscattered, lazy):
        make = _full_expansion_potentials()[name]

        def arrays(chunk_size):
            return _multislice_arrays(
                make(),
                lazy,
                backscattered=backscattered,
                potential_chunk_size=chunk_size,
            )

        expected = arrays(6)
        # 4 does not divide the 6 slices; the lazy runs are slow, so skip chunk size 2
        for chunk_size in (1, 4) if lazy else (1, 2, 4):
            result = arrays(chunk_size)
            for array, reference in zip(result, expected):
                assert array.shape == reference.shape
                np.testing.assert_allclose(
                    array,
                    reference,
                    rtol=0,
                    atol=1e-6 * np.abs(reference).max(),
                    err_msg=f"chunk size {chunk_size}",
                )

    @pytest.mark.parametrize("lazy", [False, True])
    def test_backscattering_between_identical_slices_is_zero(self, lazy):
        # One slice per unit cell: every slice equals the next, so the correction
        # term (the difference between consecutive slices) vanishes exactly.
        atoms = ase.build.bulk("Si", cubic=True)
        displaced = list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]
        potential = abtem.Potential(
            displaced * (1, 1, 3),
            gpts=(24, 20),
            slice_thickness=atoms.cell[2, 2],
            exit_planes=1,
        )
        transmitted, backscattered = _multislice_arrays(potential, lazy)
        (alone,) = _multislice_arrays(potential, lazy, backscattered=False)

        np.testing.assert_array_equal(transmitted, alone)
        assert not np.any(backscattered)

    def test_backscattering_of_a_slab_followed_by_vacuum(self):
        atoms = ase.build.bulk("Si", cubic=True)
        slab = list(abtem.FrozenPhonons(atoms, 1, sigmas=0.1, seed=1))[0]
        slab.positions[:, 2] += 2.0
        slab.cell[2, 2] = 3 * atoms.cell[2, 2]
        potential = abtem.Potential(
            slab, gpts=(24, 20), slice_thickness=atoms.cell[2, 2] / 3, exit_planes=1
        )
        transmitted, backscattered = _multislice_arrays(potential, False)
        (alone,) = _multislice_arrays(potential, False, backscattered=False)

        np.testing.assert_array_equal(transmitted, alone)
        per_plane = np.abs(backscattered).max(axis=(-2, -1))
        # nothing comes back from the vacuum behind the slab
        assert per_plane[:5].all() and not per_plane[5:].any()


class TestAlgorithmComparison:
    """Test that different algorithms produce reasonable results."""

    def test_fourier_vs_realspace_shapes_match(self, test_system):
        """Test that Fourier and RealSpace produce same output shapes."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        fourier_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=1),
        )

        realspace_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=1),
        )

        # Shapes should match
        assert fourier_result.array.shape == realspace_result.array.shape

        # Both should produce non-zero results (sanity check)
        assert np.abs(to_host_array(fourier_result.array)).sum() > 0
        assert np.abs(to_host_array(realspace_result.array)).sum() > 0

    @pytest.mark.slow
    def test_higher_orders_differ(self, test_system):
        """Test that higher orders produce different results."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        order1_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=1),
        )

        order3_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3),
        )

        # Shapes should match
        assert order1_result.array.shape == order3_result.array.shape

        # Results should be different (if identical, something's wrong)
        assert not np.allclose(
            to_host_array(order1_result.array), to_host_array(order3_result.array), rtol=1e-10
        )

    def test_fourier_order2_differs_from_order1(self, test_system):
        """Test that Fourier order 2 differs from order 1."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        scan = test_system["single_point_scan"]

        order1_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=1),
        )

        order2_result = probe.multislice(
            potential=potential,
            scan=scan,
            lazy=False,
            algorithm=FourierMultislice(order=2),
        )

        # Shapes should match
        assert order1_result.array.shape == order2_result.array.shape

        # Results should be different
        assert not np.allclose(
            to_host_array(order1_result.array), to_host_array(order2_result.array), rtol=1e-10
        )


class TestComplexWorkflows:
    """Test complex multislice workflows."""

    @pytest.mark.slow
    def test_realspace_with_scan_and_detectors(self, test_system):
        """Test RealSpace multislice with scan and detectors."""
        probe = test_system["probe"]
        potential = test_system["potential"]
        grid_scan = test_system["grid_scan"]

        detectors = [
            abtem.PixelatedDetector(),
            abtem.AnnularDetector(inner=30, outer=100),
        ]

        results = probe.multislice(
            potential=potential,
            scan=grid_scan,
            detectors=detectors,
            lazy=False,
            algorithm=RealSpaceMultislice(order=3),
        )

        assert isinstance(results, list)
        assert len(results) == len(detectors)

    def test_works_with_plane_waves(self, test_system):
        """Test that multislice works with plane waves."""
        potential = test_system["potential"]
        plane_wave = abtem.PlaneWave(energy=30e3).match_grid(potential)

        result = plane_wave.multislice(
            potential=potential,
            lazy=False,
        )

        assert result is not None


class TestStencilNumericalAccuracy:
    """Verify the fast stencils match the scipy reference implementation."""

    @devices
    @pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
    def test_laplace_stencil_matches_scipy_reference(self, accuracy, device):
        """Compare the fast Laplacian stencils against scipy.ndimage.convolve."""
        from abtem.core.backend import get_array_module
        from abtem.finite_difference import (
            _laplace_operator_func_slow,
            _laplace_operator_stencil,
        )

        prefactor = np.complex64(1.0 + 0.5j)
        rng = np.random.default_rng(42)
        a = (
            rng.standard_normal((2, 24, 24))
            + 1j * rng.standard_normal((2, 24, 24))
        ).astype(np.complex64)

        ref = np.stack(
            [
                _laplace_operator_func_slow(accuracy, prefactor)(a[m])
                for m in range(a.shape[0])
            ]
        )
        xp = get_array_module(device)
        result = _laplace_operator_stencil(
            accuracy, prefactor, mode="wrap", dtype=np.complex64, device=device
        )(xp.asarray(a))

        np.testing.assert_allclose(
            to_host_array(result),
            ref,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Stencil mismatch at accuracy={accuracy} on {device}",
        )

    @pytest.mark.parametrize("device", [gpu])
    def test_gpu_stencil_rejects_non_complex_dtype(self, device):
        """The raw GPU kernel only ships complex specializations; a real array
        must raise instead of silently reinterpreting the buffer."""
        import cupy as cp

        from abtem.finite_difference import _laplace_operator_stencil

        stencil = _laplace_operator_stencil(
            4, 1.0, mode="wrap", dtype=np.complex64, device="gpu"
        )
        with pytest.raises(TypeError, match="complex64 or complex128"):
            stencil(cp.ones((8, 8), dtype=cp.float32))
