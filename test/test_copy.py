import hypothesis.strategies as st
import pytest
import strategies as abtem_st
from hypothesis import given


@given(data=st.data())
@pytest.mark.parametrize(
    "copyable",
    [
        abtem_st.atoms,
        abtem_st.frozen_phonons,
        abtem_st.grid_scan,
        abtem_st.line_scan,
        abtem_st.custom_scan,
        abtem_st.potential,
        abtem_st.potential_array,
        abtem_st.aberrations,
        abtem_st.aperture,
        abtem_st.temporal_envelope,
        abtem_st.spatial_envelope,
        # abtem_st.composite_wave_transform,
        abtem_st.ctf,
        abtem_st.probe,
        abtem_st.plane_wave,
        abtem_st.waves,
        abtem_st.s_matrix,
        # # # prism_st.random_s_matrix,
        abtem_st.images,
        abtem_st.diffraction_patterns,
        abtem_st.line_profiles,
        abtem_st.polar_measurements,
    ],
)
def test_copy_equals(data, copyable):
    original = data.draw(copyable())
    assert original.copy() == original


class TestEqualityDiscriminates:
    """`test_copy_equals` above compares an object with itself.

    A self-comparison cannot fail for an over-permissive ``__eq__`` -- and
    ``safe_equality`` was over-permissive in two independent ways, so nothing
    in the suite noticed. These tests compare objects that genuinely DIFFER.
    """

    @staticmethod
    def _atoms(dx=0.0):
        import ase.build

        atoms = ase.build.bulk("Si", cubic=True)
        if dx:
            atoms.positions[0] += [dx, 0.0, 0.0]
        return atoms

    def _potential(self, dx=0.0, **kwargs):
        import abtem

        kwargs.setdefault("gpts", (64, 64))
        kwargs.setdefault("slice_thickness", 1.0)
        return abtem.Potential(self._atoms(dx), **kwargs)

    def test_potentials_with_different_atomic_positions_are_not_equal(self):
        """`safe_equality` tested `equal is False`, which only catches the
        `False` singleton. `ase.Atoms.__eq__` returns `numpy.bool_`, and
        `np.False_ is False` is False -- so the difference was dropped and two
        physically different potentials compared equal."""
        assert self._potential() != self._potential(dx=1.234)

    def test_potentials_with_identical_atoms_are_still_equal(self):
        """The fix must not make everything unequal."""
        assert self._potential() == self._potential()

    @pytest.mark.parametrize(
        "kwargs", [{"gpts": (32, 32)}, {"slice_thickness": 2.0}], ids=["gpts", "dz"]
    )
    def test_other_differences_are_still_caught(self, kwargs):
        assert self._potential() != self._potential(**kwargs)

    def test_a_built_potential_equals_an_identical_unbuilt_one(self):
        """`_sliced_atoms` is populated lazily by get_sliced_atoms(), so a
        built potential stopped comparing equal to an identical unbuilt one --
        equality depended on whether a result had been computed from it."""
        import abtem

        built, unbuilt = self._potential(), self._potential()
        with abtem.config.set({"fft": "numpy"}):
            built.build(lazy=False)
        assert built == unbuilt

    def test_an_empty_ensemble_attribute_does_not_short_circuit_the_rest(self):
        """The EmptyEnsemble branch did `return True` for the whole object
        rather than `continue` for that attribute, so anything declared after
        one was never compared.

        Latent in abtem today -- nothing instantiates EmptyEnsemble or its
        subclass EmptyTransform -- so this is demonstrated on a local class,
        which is also the only way to pin it against reintroduction.
        """
        from abtem.core.ensemble import EmptyEnsemble
        from abtem.core.utils import EqualityMixin

        class Holder(EqualityMixin):
            def __init__(self, tail):
                self._ensemble = EmptyEnsemble()  # declared FIRST
                self._tail = tail

        assert Holder("one") == Holder("one")
        assert Holder("one") != Holder("two")

    def test_a_built_potential_equals_an_unbuilt_one_inside_a_container(self):
        """`_eq_exclude` is applied by `EqualityMixin.__eq__`, but
        `safe_equality` recurses into a nested `EqualityMixin` directly and so
        bypasses it. Without forwarding the exclusion, the invariant above held
        only for a top-level operand: the same potential held by an SMatrix
        went back to comparing unequal once it had been built."""
        import abtem

        built, unbuilt = self._potential(), self._potential()
        s_built = abtem.SMatrix(potential=built, energy=100e3, semiangle_cutoff=20)
        s_unbuilt = abtem.SMatrix(potential=unbuilt, energy=100e3, semiangle_cutoff=20)
        with abtem.config.set({"fft": "numpy"}):
            built.build(lazy=False)
        assert s_built == s_unbuilt

    def test_a_built_crystal_potential_equals_an_identical_unbuilt_one(self):
        """CrystalPotential creates its own `_sliced_atoms` and descends from
        `_PotentialBuilder`, not from `_FieldBuilderFromAtoms`, so it needs the
        exclusion declared in its own right -- inheriting it through Potential
        does not reach it."""
        import abtem

        built = abtem.CrystalPotential(self._potential(), repetitions=(1, 1, 2))
        unbuilt = abtem.CrystalPotential(self._potential(), repetitions=(1, 1, 2))

        # get_sliced_atoms(), not build(): build() populates the *unit*
        # potential's cache, while this populates the crystal's own. It is the
        # public entry point, and the core-loss driver calls it on the caller's
        # potential to extract EELS sites.
        built.get_sliced_atoms()
        assert built.__dict__["_sliced_atoms"] is not None

        assert built == unbuilt

    def test_equality_does_not_execute_a_dask_graph(self):
        """`np.all` on a dask value returns another dask value, and the `not`
        in front of it calls `__bool__`, which computes it. Comparing two lazy
        objects would then run both simulations and discard the results.

        This pins the cost, not the verdict: what `==` should mean for a lazy
        object is a separate question, and today a deferred attribute is simply
        skipped -- the behaviour that has always been in place.
        """
        import dask
        import dask.array as da

        from abtem.measurements import Images

        calls = []
        real_compute = dask.base.compute

        def counting_compute(*args, **kwargs):
            calls.append(args)
            return real_compute(*args, **kwargs)

        dask.base.compute = counting_compute
        try:
            a = Images(da.zeros((16, 16), dtype="float32"), sampling=0.1)
            b = Images(da.ones((16, 16), dtype="float32"), sampling=0.1)
            a == b
        finally:
            dask.base.compute = real_compute

        assert not calls, f"`==` executed {len(calls)} dask graph(s)"

    def test_slice_indexed_atoms_equals_an_identical_twin(self):
        """`_slice_index` is a list of integer arrays, one per slice.
        `list.__eq__` reduces each pairwise `==` (itself an array) to a bool,
        which numpy refuses for anything but a length-1 array -- the generic
        safe_equality catches that ValueError and reports "unequal", so this
        class never compared equal to anything, including an identical twin.
        """
        from abtem.slicing import SliceIndexedAtoms

        a, b = SliceIndexedAtoms(self._atoms(), 1.0), SliceIndexedAtoms(self._atoms(), 1.0)
        assert a == b

    def test_slice_indexed_atoms_with_different_binning_are_not_equal(self):
        """The fix must not make every SliceIndexedAtoms equal regardless of
        content -- different slice thicknesses bin the same atoms differently
        and must still compare unequal."""
        from abtem.slicing import SliceIndexedAtoms

        a = SliceIndexedAtoms(self._atoms(), 1.0)
        b = SliceIndexedAtoms(self._atoms(), 0.5)
        assert a != b

    def test_magnetic_field_equals_its_own_copy(self):
        """`QuasiDipoleProjections` (the integrator `MagneticField`/
        `VectorPotential` hold as `_integrator`) was a plain class with no
        `__eq__`, so it fell back to object identity and every `MagneticField`
        compared unequal to its own copy, including an untouched one."""
        from abtem.magnetism.iam import MagneticField

        atoms = self._atoms()
        m = MagneticField(atoms, gpts=(32, 32), slice_thickness=1.0)
        assert m == m.copy()

    def test_magnetic_field_with_different_atoms_is_not_equal(self):
        from abtem.magnetism.iam import MagneticField

        a = MagneticField(self._atoms(), gpts=(32, 32), slice_thickness=1.0)
        b = MagneticField(self._atoms(dx=1.234), gpts=(32, 32), slice_thickness=1.0)
        assert a != b

    def test_quasi_dipole_integrator_ignores_its_table_cache(self):
        """`_tables` is populated lazily by `get_integral_table`, one entry
        per element on first use -- the same shape of defect `_sliced_atoms`
        had for `Potential`, and doubly so: an unpopulated cache made a used
        integrator stop comparing equal to a fresh one, and a *populated* one
        made two integrators that cached the identical element compare
        unequal anyway, since dict.__eq__ on numpy-array values hits the same
        ValueError the list case above does.
        """
        from abtem.magnetism.iam import MagneticField

        atoms = self._atoms()
        atoms.numbers[:] = 26  # Fe, present in the Lyon parametrization
        m1 = MagneticField(atoms, gpts=(32, 32), slice_thickness=1.0)
        m2 = m1.copy()
        assert m1 == m2

        m1._integrator.get_integral_table("Fe")
        assert m1.__dict__["_integrator"].__dict__["_tables"]
        assert not m2.__dict__["_integrator"].__dict__["_tables"]
        assert m1 == m2

        m2._integrator.get_integral_table("Fe")
        assert m1 == m2

    def test_transition_potential_array_ignores_its_device_cache(self):
        """`_local_potential_device_cache` is populated lazily by
        `_local_potential_on_device()`, a normal side effect of core-loss
        multislice -- the same shape of defect `_sliced_atoms` had for
        `Potential`. Two transition potentials with identical arrays stopped
        comparing equal once one of them had been used on a device."""
        import numpy as np

        from abtem.core.axes import OrdinalAxis
        from abtem.inelastic.core_loss import TransitionPotentialArray

        def make():
            rng = np.random.default_rng(0)
            array = (
                rng.standard_normal((2, 16, 16))
                + 1j * rng.standard_normal((2, 16, 16))
            ).astype(np.complex64)
            return TransitionPotentialArray(
                Z=14,
                array=array,
                energy=100e3,
                extent=4.0,
                ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
                metadata={"Z": 14, "n": 1, "l": 0},
            )

        tp1, tp2 = make(), make()
        assert tp1 == tp2

        tp1._local_potential_on_device(np.zeros((16, 16), dtype=np.complex64))
        assert tp1.__dict__["_local_potential_device_cache"] is not None
        assert tp2.__dict__["_local_potential_device_cache"] is None

        assert tp1 == tp2
