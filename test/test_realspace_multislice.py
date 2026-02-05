import ase
import numpy as np
import pytest
from utils import gpu

import abtem
from abtem.multislice import FourierMultislice, RealSpaceMultislice

# Suppress Numba performance warnings during tests
pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaPerformanceWarning"
)


def to_numpy(array):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if hasattr(array, "get"):  # CuPy array
        return np.asarray(array.get())
    return np.asarray(array)


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
            FourierMultislice(),
            FourierMultislice(order=2),
            RealSpaceMultislice(),
            RealSpaceMultislice(order=3),
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
            to_numpy(lazy_result.array),
            to_numpy(eager_result.array),
            rtol=1e-5,
            atol=1e-8,
        )


class TestFourierMultislice:
    """Test FourierMultislice algorithm with various configurations."""

    @pytest.mark.parametrize("order", [1, 2])
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

        with pytest.raises(ValueError, match="Only orders 1 and 2 are supported"):
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
        assert np.abs(to_numpy(fourier_result.array)).sum() > 0
        assert np.abs(to_numpy(realspace_result.array)).sum() > 0

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
            to_numpy(order1_result.array), to_numpy(order3_result.array), rtol=1e-10
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
            to_numpy(order1_result.array), to_numpy(order2_result.array), rtol=1e-10
        )


class TestComplexWorkflows:
    """Test complex multislice workflows."""

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
