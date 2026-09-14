import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import abtem
from abtem.integrals import (
    _MAX_CACHE_ENTRIES,
    _MAX_SCATTERING_FACTOR_ENTRIES,
    GaussianProjectionIntegrals,
    ScatteringFactorProjectionIntegrals,
)
from utils import assert_array_matches_device, gpu

# from abtem.integrals import GaussianProjectionIntegrals
from abtem.parametrizations import (
    KirklandParametrization,
    LobatoParametrization,
    PengParametrization,
)


@given(atomic_number=st.integers(min_value=1, max_value=98))
@pytest.mark.parametrize(
    "parametrization_a",
    [
        LobatoParametrization(),
        KirklandParametrization(),
        PengParametrization(),
    ],
    ids=["lobato", "kirkland", "peng"],
)
@pytest.mark.parametrize(
    "parametrization_b",
    [LobatoParametrization(), KirklandParametrization(), PengParametrization()],
    ids=["lobato", "kirkland", "peng"],
)
def test_parametrizations(atomic_number, parametrization_a, parametrization_b):
    k = np.linspace(0, 5, 100)
    assert np.allclose(
        parametrization_a.projected_scattering_factor(atomic_number)(k),
        parametrization_b.projected_scattering_factor(atomic_number)(k),
        atol=10,
        rtol=0.1,
    )
    r = np.linspace(0.2, 5, 100)
    assert np.allclose(
        parametrization_a.projected_potential(atomic_number)(r),
        parametrization_b.projected_potential(atomic_number)(r),
        atol=20,
        rtol=0.1,
    )


# @pytest.mark.parametrize('parameters',
#                          [{'gaussian_projection_integrals':
#                                GaussianProjectionIntegrals(correction_parametrization=None),
#                            'parametrization':
#                                PengParametrization()
#                            },
#                           {'gaussian_projection_integrals':
#                                GaussianProjectionIntegrals(correction_parametrization='lobato'),
#                            'parametrization': LobatoParametrization()
#                            }
#                           ], ids=['uncorrected', 'corrected'])
# def test_gaussian_projection_integrals(parameters):
#     parametrization = parameters['parametrization']
#     gaussian_projection_integrals = parameters['gaussian_projection_integrals']
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.1, 0.1)
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#     a = -np.inf
#     b = np.inf
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build('C', gpts, sampling)
#
#     projections = gaussian_scattering_factors.integrate_on_grid(
#         positions,
#         a,
#         b,
#         gpts,
#         sampling,
#         fourier_space=True)
#
#     k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
#     analytical = parametrization.projected_scattering_factor(symbol)(k ** 2)
#     assert np.allclose(analytical, projections[0].real, rtol=1e-6)
#
#
# @pytest.mark.parametrize('fourier_space', [True, False])
# def test_finite_gaussian_projection_integrals(fourier_space):
#     gaussian_projection_integrals = GaussianProjectionIntegrals(correction_parametrization=None)
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.02, 0.02)
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#     a = -.2
#     b = .2
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build('C', gpts, sampling)
#
#     projections = gaussian_scattering_factors.integrate_on_grid(
#         positions,
#         a,
#         b,
#         gpts,
#         sampling,
#         fourier_space=fourier_space)
#
#     if fourier_space:
#         k = np.abs(np.fft.fftfreq(gpts[0], sampling[0])).astype(np.float32)
#         analytical = PengParametrization().finite_projected_scattering_factor(symbol)(k, a, b)
#         assert np.allclose(analytical, projections[0].real, rtol=1e-6)
#
#     else:
#         r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False)
#         analytical = PengParametrization().finite_projected_potential(symbol)(r, a, b)
#         assert np.allclose(analytical[:len(r) // 2], projections[0][:len(r) // 2], rtol=1e-6, atol=5)


# @pytest.mark.parametrize('parametrization',
#                          [LobatoParametrization(),
#                           KirklandParametrization()])
# def test_quadrature(parametrization):
#     quadrature = QuadratureProjectionIntegrals(parametrization, quad_order=20, cutoff_tolerance=1e-9)
#
#     symbol = 'Au'
#     gpts = (256, 256)
#     sampling = (0.05, 0.05)
#     xp = np
#     a = -20
#     b = 20
#
#     positions = xp.array([[0, 0, 0]], dtype=xp.float32)
#
#     table = quadrature.build_integral_table(symbol, min(sampling) / 2)
#
#     integrated = table.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     r = np.linspace(0, gpts[0] * sampling[0], gpts[0], endpoint=False).astype(np.float32)
#     analytical = parametrization.projected_potential(symbol)(r[1:])
#
#     assert np.allclose(analytical, integrated[0, 1:], atol=2)
#

# def test_finite_projections():
#     quadrature = ProjectionQuadratureRule('lobato', quad_order=8, cutoff_tolerance=1e-4)
#     gaussian_projection_integrals = GaussianProjectionIntegrals()
#
#     symbol = 'C'
#     gpts = (256, 256)
#     sampling = (0.05, 0.05)
#     a = .1
#     b = 1
#
#     positions = np.array([[0, 0, 0]], dtype=np.float32)
#
#     table = quadrature.build_integral_table(symbol, min(sampling) / 2)
#
#     quadrature_potential = table.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     gaussian_scattering_factors = gaussian_projection_integrals.build(symbol, gpts, sampling)
#
#     gaussian_potential = gaussian_scattering_factors.integrate_on_grid(positions, a, b, gpts, sampling)
#
#     assert np.allclose(quadrature_potential[0, :gpts[1] // 2], gaussian_potential[0, :gpts[1] // 2], atol=2)


class TestScatteringFactorCacheKey:
    """``get_scattering_factor`` cached on the chemical symbol alone.

    The cached array depends on the grid and on the device it was allocated
    on as well, so reusing one integrator across two potentials served the
    first potential's array to the second: a broadcast error for a new grid,
    and a numpy array handed to a cupy kernel for a new device. ``integrator``
    is a documented ``Potential`` parameter, so sharing one is ordinary use.
    """

    @staticmethod
    def _atoms():
        import ase.build

        return ase.build.bulk("Si", cubic=True)

    def test_second_grid_is_not_served_the_first_grids_array(self):
        integrator = ScatteringFactorProjectionIntegrals()
        for gpts, sampling in (((64, 64), (0.125, 0.125)), ((128, 128), (0.0625,) * 2)):
            array = integrator.get_scattering_factor("Si", gpts, sampling, "cpu")
            assert array.shape == gpts

    def test_potentials_on_two_grids_may_share_an_integrator(self):
        atoms = self._atoms()
        shared = ScatteringFactorProjectionIntegrals()
        for gpts in ((64, 64), (128, 128)):
            got = abtem.Potential(
                atoms, gpts=gpts, slice_thickness=1.0, integrator=shared
            ).build(lazy=False)
            reference = abtem.Potential(
                atoms,
                gpts=gpts,
                slice_thickness=1.0,
                integrator=ScatteringFactorProjectionIntegrals(),
            ).build(lazy=False)
            assert np.array_equal(got.array, reference.array)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_array_lands_on_the_requested_device(self, device):
        integrator = ScatteringFactorProjectionIntegrals()
        # Warm the cache on the cpu first: the array served for ``device``
        # must still be the one belonging to ``device``.
        integrator.get_scattering_factor("Si", (64, 64), (0.125, 0.125), "cpu")
        array = integrator.get_scattering_factor("Si", (64, 64), (0.125, 0.125), device)
        assert_array_matches_device(array, device)

    @pytest.mark.parametrize("device", ["cpu", gpu])
    def test_potential_may_share_an_integrator_across_devices(self, device):
        atoms = self._atoms()
        shared = ScatteringFactorProjectionIntegrals()
        abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=1.0, integrator=shared, device="cpu"
        ).build(lazy=False)
        got = abtem.Potential(
            atoms, gpts=(64, 64), slice_thickness=1.0, integrator=shared, device=device
        ).build(lazy=False)
        reference = abtem.Potential(
            atoms,
            gpts=(64, 64),
            slice_thickness=1.0,
            integrator=ScatteringFactorProjectionIntegrals(),
            device=device,
        ).build(lazy=False)
        assert np.allclose(
            np.asarray(got.to_cpu().array), np.asarray(reference.to_cpu().array)
        )

    def test_cache_is_bounded(self):
        """A full key admits an entry per grid, so it needs a bound."""
        integrator = ScatteringFactorProjectionIntegrals()
        for n in range(_MAX_SCATTERING_FACTOR_ENTRIES + 8):
            integrator.get_scattering_factor("Si", (8 + n,) * 2, (0.1, 0.1), "cpu")
        assert len(integrator.scattering_factors) <= _MAX_SCATTERING_FACTOR_ENTRIES

    def test_cache_is_least_recently_used_not_first_in_first_out(self):
        """A hit must refresh recency, or a sweep gets no benefit from it.

        This needs *more* distinct keys than the bound: with exactly maxsize
        keys nothing is ever evicted and FIFO and LRU are indistinguishable.
        """
        integrator = ScatteringFactorProjectionIntegrals()
        kept = (8, 8)
        integrator.get_scattering_factor("Si", kept, (0.1, 0.1), "cpu")
        # Insert well past the bound, touching `kept` between each insertion so
        # it stays the most recently used entry. Under FIFO it would be evicted
        # first regardless; under LRU it survives.
        for n in range(_MAX_SCATTERING_FACTOR_ENTRIES + 8):
            integrator.get_scattering_factor("Si", (16 + n,) * 2, (0.1, 0.1), "cpu")
            integrator.get_scattering_factor("Si", kept, (0.1, 0.1), "cpu")
        assert len(integrator.scattering_factors) <= _MAX_SCATTERING_FACTOR_ENTRIES
        keys = list(integrator.scattering_factors)
        assert any(k[1] == kept for k in keys), "most recently used entry was evicted"

    def test_concurrent_access_does_not_race(self):
        """Hand-rolled dict eviction raced: two threads evicting the same key
        gave KeyError, and iteration could see the dict resized underneath."""
        import sys
        import threading

        errors = []

        def hammer(integrator, seed):
            try:
                for n in range(4000):
                    g = 8 + ((seed * 7919 + n * 13) % 200)
                    integrator.get_scattering_factor(
                        "Si", (g, g), (0.1, 0.1), "cpu"
                    )
            except Exception as exc:  # noqa: BLE001 -- report, don't mask
                errors.append(exc)

        integrator = ScatteringFactorProjectionIntegrals()
        # A short switch interval makes the window reachable in seconds
        # instead of relying on luck.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [
                threading.Thread(target=hammer, args=(integrator, i))
                for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)

        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"


class TestScatteringFactorIntegratorIsAPlainObject:
    """The cache must not make the integrator unpicklable, unequal or uncopyable.

    A per-instance ``functools.lru_cache`` around a bound method was tried and
    broke all three: the wrapper resolves by ``__qualname__`` so neither pickle
    nor cloudpickle can serialise it (which disables dask's processes scheduler
    and distributed, and therefore the multi-GPU path the cache exists to
    serve); ``EqualityMixin`` compares ``__dict__`` and a wrapper has identity
    equality; and ``_lru_cache_wrapper.__deepcopy__`` returns ``self``, so a
    "copy" shared the original's cache and stayed bound to the original.
    """

    @staticmethod
    def _warmed():
        integrator = ScatteringFactorProjectionIntegrals()
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), "cpu")
        return integrator

    def _potential(self, **kwargs):
        import ase.build

        return abtem.Potential(
            ase.build.bulk("Si", cubic=True),
            gpts=(64, 64),
            slice_thickness=1.0,
            **kwargs,
        )

    def test_pickles(self):
        import pickle

        assert isinstance(
            pickle.loads(pickle.dumps(self._warmed())),
            ScatteringFactorProjectionIntegrals,
        )

    def test_potential_cloudpickles(self):
        cloudpickle = pytest.importorskip("cloudpickle")
        assert isinstance(
            cloudpickle.loads(cloudpickle.dumps(self._potential())), abtem.Potential
        )

    def test_builds_under_the_processes_scheduler(self):
        """The processes scheduler round-trips the graph through pickle."""
        import dask

        with dask.config.set(scheduler="processes"):
            built = self._potential().build(lazy=True).compute(progress_bar=False)
        assert built.array.shape == (6, 64, 64)

    def test_tokenizes_deterministically(self):
        """dask falls back to a random token when pickling fails."""
        import dask

        assert len({dask.base.tokenize(self._potential()) for _ in range(3)}) == 1

    def test_two_fresh_integrators_compare_equal(self):
        assert ScatteringFactorProjectionIntegrals() == (
            ScatteringFactorProjectionIntegrals()
        )
        assert self._potential() == self._potential()

    def test_deepcopy_is_independent(self):
        import copy

        original = self._warmed()
        clone = copy.deepcopy(original)
        assert clone.scattering_factors is not original.scattering_factors
        clone.get_scattering_factor("Si", (32, 32), (0.1, 0.1), "cpu")
        assert len(clone.scattering_factors) == len(original.scattering_factors) + 1

    def test_cached_arrays_are_freed_without_a_cyclic_collection(self):
        """The wrapper made instance -> cache -> bound method -> instance."""
        import gc
        import weakref

        gc.disable()
        try:
            integrator = self._warmed()
            reference = weakref.ref(integrator)
            del integrator
            assert reference() is None
        finally:
            gc.enable()

    @pytest.mark.parametrize(
        "device", ["cpu", np, pytest.param(np.zeros(3), id="array")]
    )
    def test_device_may_be_any_form_get_array_module_accepts(self, device):
        """``device`` is not part of the key, so it need not be hashable."""
        integrator = ScatteringFactorProjectionIntegrals()
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), device)
        integrator.get_scattering_factor("Si", (64, 64), (0.1, 0.1), "cpu")
        # All spellings of the same physical device share one canonical entry.
        assert len(integrator.scattering_factors) == 1


class TestIntegratorCaches:
    """Four caches in integrals.py were broken in two different ways.

    ``GaussianProjectionIntegrals.get_gaussians`` and ``get_corrections`` each
    computed a key, checked the dict, missed, recomputed and never wrote back,
    so both stayed empty for the life of the object. Its ``_sinc_cache`` and
    ``QuadratureProjectionIntegrals._device_arrays`` keyed device-resident
    arrays on the literal ``"cpu"``/``"gpu"`` string, which does not
    distinguish one GPU from another.

    All four now share one bounded, LRU, race-free container.
    """

    @staticmethod
    def _atoms():
        import ase.build

        return ase.build.bulk("Si", cubic=True)

    def _build(self, integrator, gpts=(64, 64)):
        return abtem.Potential(
            self._atoms(), gpts=gpts, slice_thickness=1.0, integrator=integrator
        ).build(lazy=False)

    def test_gaussian_caches_actually_store(self):
        integrator = GaussianProjectionIntegrals()
        self._build(integrator)
        assert len(integrator._gaussians) > 0
        assert len(integrator._corrections) > 0
        assert len(integrator._sinc_cache) > 0

    def test_a_cache_hit_returns_the_same_object(self):
        integrator = GaussianProjectionIntegrals()
        first = integrator.get_gaussians("Si", (64, 64), (0.1, 0.1))
        assert integrator.get_gaussians("Si", (64, 64), (0.1, 0.1)) is first

    def test_a_different_grid_is_not_served_the_first_grids_array(self):
        integrator = GaussianProjectionIntegrals()
        small = integrator.get_gaussians("Si", (64, 64), (0.1, 0.1))
        large = integrator.get_gaussians("Si", (128, 128), (0.05, 0.05))
        assert small.shape[-2:] == (64, 64)
        assert large.shape[-2:] == (128, 128)

    def test_results_do_not_depend_on_cache_state(self):
        """A shared integrator must match a fresh one, on every grid."""
        shared = GaussianProjectionIntegrals()
        for gpts in ((64, 64), (96, 96), (64, 64)):
            got = self._build(shared, gpts)
            reference = self._build(GaussianProjectionIntegrals(), gpts)
            assert np.array_equal(got.array, reference.array)

    def test_sinc_is_cached_per_grid_and_device(self):
        """The key must carry the grid, which the old one did too, and the
        device, which it recorded only as the literal "cpu"/"gpu" string.

        Note this class is host-only today -- it returns numpy arrays from
        get_gaussians and fails on GPU with ``TypeError: Unsupported type
        <class 'numpy.ndarray'>`` on this branch and on dev alike -- so the
        device half of the key is defensive, not currently exercisable. It is
        included because the cache is migrating to the shared container that
        the GPU-capable integrator also uses.
        """
        integrator = GaussianProjectionIntegrals()
        self._build(integrator, gpts=(64, 64))
        self._build(integrator, gpts=(96, 96))
        keys = list(integrator._sinc_cache)
        assert len(keys) == 2, "the grid must be part of the key"
        assert all(k[-1] == "cpu" for k in keys)

    def test_caches_are_bounded(self):
        integrator = GaussianProjectionIntegrals()
        for n in range(_MAX_CACHE_ENTRIES + 8):
            integrator.get_gaussians("Si", (8 + n,) * 2, (0.1, 0.1))
        assert len(integrator._gaussians) <= _MAX_CACHE_ENTRIES

    def test_cache_is_least_recently_used(self):
        integrator = GaussianProjectionIntegrals()
        kept = (8, 8)
        integrator.get_gaussians("Si", kept, (0.1, 0.1))
        for n in range(_MAX_CACHE_ENTRIES + 8):
            integrator.get_gaussians("Si", (16 + n,) * 2, (0.1, 0.1))
            integrator.get_gaussians("Si", kept, (0.1, 0.1))
        assert any(k[1] == kept for k in integrator._gaussians)

    def test_the_integrator_stays_picklable_comparable_and_copyable(self):
        """The three things a per-instance lru_cache broke in #388."""
        import copy
        import pickle

        used, fresh = GaussianProjectionIntegrals(), GaussianProjectionIntegrals()
        self._build(used)
        assert isinstance(pickle.loads(pickle.dumps(used)), GaussianProjectionIntegrals)
        # A cache is incidental state, never identity.
        assert used == fresh
        clone = copy.deepcopy(used)
        assert clone._gaussians is not used._gaussians

    def test_concurrent_access_does_not_race(self):
        import sys
        import threading

        errors = []

        def hammer(integrator, seed):
            try:
                for n in range(4000):
                    g = 8 + ((seed * 7919 + n * 13) % 200)
                    integrator.get_gaussians("Si", (g, g), (0.1, 0.1))
            except Exception as exc:  # noqa: BLE001 -- report, don't mask
                errors.append(exc)

        integrator = GaussianProjectionIntegrals()
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [
                threading.Thread(target=hammer, args=(integrator, i)) for i in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            sys.setswitchinterval(previous)
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
