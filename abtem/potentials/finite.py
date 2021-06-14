class PotentialIntegrator:
    """
    Perform finite integrals of a radial function along a straight line.

    Parameters
    ----------
    function: callable
        Radial function to integrate.
    r: array of float
        The evaluation points of the integrals.
    cutoff: float, optional
        The radial function is assumed to be zero outside this threshold.
    cache_size: int, optional
        The maximum number of integrals that will be cached.
    cache_key_decimals: int, optional
        The number of decimals used in the cache keys.
    tolerance: float, optional
        The absolute error tolerance of the integrals.
    """

    def __init__(self,
                 function: Callable,
                 r: np.ndarray,
                 max_interval,
                 cutoff: float = None,
                 cache_size: int = 4096,
                 cache_key_decimals: int = 2,
                 tolerance: float = 1e-6):

        self._function = function
        self._r = r

        if cutoff is None:
            self._cutoff = r[-1]
        else:
            self._cutoff = cutoff

        self._cache = Cache(cache_size)
        self._cache_key_decimals = cache_key_decimals
        self._tolerance = tolerance

        def f(z):
            return self._function(np.sqrt(self.r[0] ** 2 + (z * max_interval / 2 + max_interval / 2) ** 2))

        value, error_estimate, step_size, order = integrate(f, -1, 1, self._tolerance)

        self._xk, self._wk = tanh_sinh_nodes_and_weights(step_size, order)

    @property
    def r(self):
        return self._r

    @property
    def cutoff(self):
        return self._cutoff

    def integrate(self, z: Union[float, Sequence[float]], a: Union[float, Sequence[float]],
                  b: Union[float, Sequence[float]]):
        """
        Evaulate the integrals of the radial function at the evaluation points.

        Parameters
        ----------
        a: float
            Lower limit of integrals.
        b: float
            Upper limit of integrals.

        Returns
        -------
        1d array
            The evaulated integrals.
        """

        vr = np.zeros((len(z), self.r.shape[0]), np.float32)
        dvdr = np.zeros((len(z), self.r.shape[0]), np.float32)
        a = np.round(a - z, self._cache_key_decimals)
        b = np.round(b - z, self._cache_key_decimals)

        split = a * b < 0

        a, b = np.abs(a), np.abs(b)
        a, b = np.minimum(a, b), np.minimum(np.maximum(a, b), self.cutoff)

        for i, (ai, bi) in enumerate(zip(a, b)):
            if split[i]:  # split the integral
                values1, derivatives1 = self._do_integrate(0, ai)
                values2, derivatives2 = self._do_integrate(0, bi)
                result = (values1 + values2, derivatives1 + derivatives2)
            else:
                result = self._do_integrate(ai, bi)

            vr[i] = result[0]
            dvdr[i, :-1] = result[1]

        return vr, dvdr

    @cached_method('_cache')
    def _do_integrate(self, a, b):
        zm = (b - a) / 2.
        zp = (a + b) / 2.

        def f(z):
            return self._function(np.sqrt(self.r[:, None] ** 2 + (z * zm + zp) ** 2))

        values = np.sum(f(self._xk[None]) * self._wk[None], axis=1) * zm
        derivatives = np.diff(values) / np.diff(self.r)

        return values, derivatives


class FiniteProjectionIntegrals:

    def _generate_slices_finite(self, first_slice=0, last_slice=None, max_batch=1) -> Generator:
        xp = get_array_module_from_device(self._device)
        interpolate_radial_functions = get_device_function(xp, 'interpolate_radial_functions')

        array = None
        unique = np.unique(self.atoms.numbers)

        if (self._z_periodic) & (len(self.atoms) > 0):
            max_cutoff = max(self.get_integrator(number).cutoff for number in unique)
            atoms = pad_atoms(self.atoms, margin=max_cutoff, directions='z', in_place=False)
        else:
            atoms = self.atoms

        sliced_atoms = SlicedAtoms(atoms, self.slice_thickness)

        for start, end in generate_batches(last_slice - first_slice, max_batch=max_batch, start=first_slice):
            if array is None:
                array = xp.zeros((end - start,) + self.gpts, dtype=xp.float32)
            else:
                array[:] = 0.

            for number in unique:
                integrator = self.get_integrator(number)
                disc_indices = xp.asarray(self._get_radial_interpolation_points(number))
                chunk_atoms = sliced_atoms.get_subsliced_atoms(start, end, number, z_margin=integrator.cutoff)

                if len(chunk_atoms) == 0:
                    continue

                positions = np.zeros((0, 3), dtype=xp.float32)
                slice_entrances = np.zeros((0,), dtype=xp.float32)
                slice_exits = np.zeros((0,), dtype=xp.float32)
                run_length_enconding = np.zeros((end - start + 1,), dtype=xp.int32)

                for i, slice_idx in enumerate(range(start, end)):
                    slice_atoms = chunk_atoms.get_subsliced_atoms(slice_idx,
                                                                  padding=integrator.cutoff,
                                                                  z_margin=integrator.cutoff)

                    slice_positions = slice_atoms.positions
                    slice_entrance = slice_atoms.get_slice_entrance(slice_idx)
                    slice_exit = slice_atoms.get_slice_exit(slice_idx)

                    positions = np.vstack((positions, slice_positions))
                    slice_entrances = np.concatenate((slice_entrances, [slice_entrance] * len(slice_positions)))
                    slice_exits = np.concatenate((slice_exits, [slice_exit] * len(slice_positions)))

                    run_length_enconding[i + 1] = run_length_enconding[i] + len(slice_positions)

                vr, dvdr = integrator.integrate(positions[:, 2], slice_entrances, slice_exits)

                vr = xp.asarray(vr, dtype=xp.float32)
                dvdr = xp.asarray(dvdr, dtype=xp.float32)
                r = xp.asarray(integrator.r, dtype=xp.float32)
                sampling = xp.asarray(self.sampling, dtype=xp.float32)

                interpolate_radial_functions(array,
                                             run_length_enconding,
                                             disc_indices,
                                             positions,
                                             vr,
                                             r,
                                             dvdr,
                                             sampling)

            slice_thicknesses = [self.get_slice_thickness(i) for i in range(start, end)]

            yield start, end, PotentialArray(array[:end - start] / kappa,
                                             slice_thicknesses,
                                             extent=self.extent)

    def get_cutoff(self, number: int) -> float:
        """
        Cutoff distance for atomic number given an error tolerance.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        cutoff: float
            The potential cutoff.
        """

        try:
            return self._cutoffs[number]
        except KeyError:
            def f(r):
                return self.function(r, self.parameters[number]) - self.cutoff_tolerance

            self._cutoffs[number] = brentq(f, 1e-7, 1000)
            return self._cutoffs[number]

    def get_tapered_function(self, number: int) -> Callable:
        """
        Tapered potential function for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        callable
        """

        cutoff = self.get_cutoff(number)
        rolloff = .85 * cutoff

        def soft_function(r):
            result = np.zeros_like(r)
            valid = r < cutoff
            transition = valid * (r > rolloff)
            result[valid] = self._function(r[valid], self.parameters[number])
            result[transition] *= (np.cos(np.pi * (r[transition] - rolloff) / (cutoff - rolloff)) + 1.) / 2
            return result

        return soft_function

    def get_integrator(self, number: int) -> PotentialIntegrator:
        """
        Potential integrator for atomic number.

        Parameters
        ----------
        number: int
            Atomic number.

        Returns
        -------
        PotentialIntegrator object
        """

        try:
            return self._integrators[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            soft_function = self.get_tapered_function(number)
            inner_cutoff = np.min(self.sampling) / 2.

            num_points = int(np.ceil(cutoff / np.min(self.sampling) * 10.))
            r = np.geomspace(inner_cutoff, cutoff, num_points)
            max_interval = self.slice_thickness
            self._integrators[number] = PotentialIntegrator(soft_function, r, max_interval, cutoff)
            return self._integrators[number]

    def _get_radial_interpolation_points(self, number):
        """Internal function for the indices of the radial interpolation points."""
        try:
            return self._disc_indices[number]
        except KeyError:
            cutoff = self.get_cutoff(number)
            margin = np.int(np.ceil(cutoff / np.min(self.sampling)))
            rows, cols = _disc_meshgrid(margin)
            self._disc_indices[number] = np.hstack((rows[:, None], cols[:, None]))
            return self._disc_indices[number]