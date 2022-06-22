class CrystalPotential(AbstractPotential):
    """
    Crystal potential object

    The crystal potential may be used to represent a potential consisting of a repeating unit. This may allow
    calculations to be performed with lower memory and computational cost.

    The crystal potential has an additional function in conjunction with frozen phonon calculations. The number of
    frozen phonon configurations are not given by the FrozenPhonon objects, rather the ensemble of frozen phonon
    potentials represented by a potential with frozen phonons represent a collection of units, which will be assembled
    randomly to represent a random potential. The number of frozen phonon configurations should be given explicitely.
    This may save computational cost since a smaller number of units can be combined to a larger frozen phonon ensemble.

    Parameters
    ----------
    potential_unit : AbstractPotential
        The potential unit that repeated will create the full potential.
    repetitions : three int
        The repetitions of the potential in x, y and z.
    num_frozen_phonon_configs : int
        Number of frozen phonon configurations.
    """

    def __init__(self,
                 potential_unit: AbstractPotential,
                 repetitions: Tuple[int, int, int],
                 num_frozen_phonon_configs: int = 1,
                 exit_planes=None,
                 random_state=None, ):

        self._potential_unit = potential_unit
        self._repetitions = repetitions
        self._num_frozen_phonon_configs = num_frozen_phonon_configs
        self._random_state = random_state

        # if (potential_unit.num_frozen_phonon_configs == 1) & (num_frozen_phonon_configs > 1):
        #     warnings.warn('"num_frozen_phonon_configs" is greater than one, but the potential unit does not have'
        #                   'frozen phonons')
        #
        # if (potential_unit.num_frozen_phonon_configs > 1) & (num_frozen_phonon_configs == 1):
        #     warnings.warn('the potential unit has frozen phonons, but "num_frozen_phonon_configs" is set to 1')

        gpts = (self._potential_unit.gpts[0] * self.repetitions[0],
                self._potential_unit.gpts[1] * self.repetitions[1])
        extent = (self._potential_unit.extent[0] * self.repetitions[0],
                  self._potential_unit.extent[1] * self.repetitions[1])

        self._grid = Grid(extent=extent, gpts=gpts, sampling=self._potential_unit.sampling, lock_extent=True)

        super().__init__()

        self._exit_planes = validate_exit_planes(exit_planes, len(self))

    @property
    def ensemble_shape(self) -> Tuple[int, ...]:
        shape = ()
        shape += self.frozen_phonons.ensemble_shape
        shape += (self.num_exit_planes,)
        return shape

    @property
    def base_shape(self):
        return self.gpts

    @property
    def num_frozen_phonon_configs(self):
        return self._num_frozen_phonon_configs

    @property
    def random_state(self):
        return self._random_state

    @property
    def slice_thickness(self) -> np.ndarray:
        return np.tile(self._potential_unit.slice_thickness, self.repetitions[2])

    @property
    def exit_planes(self) -> Tuple[int]:
        return self._exit_planes

    @property
    def device(self):
        return self._potential_unit.device

    @property
    def potential_unit(self) -> AbstractPotential:
        return self._potential_unit

    @HasGridMixin.gpts.setter
    def gpts(self, gpts):
        if not ((gpts[0] % self.repetitions[0] == 0) and (gpts[1] % self.repetitions[0] == 0)):
            raise ValueError('gpts must be divisible by the number of potential repetitions')
        self.grid.gpts = gpts
        self._potential_unit.gpts = (gpts[0] // self._repetitions[0], gpts[1] // self._repetitions[1])

    @HasGridMixin.sampling.setter
    def sampling(self, sampling):
        self.sampling = sampling
        self._potential_unit.sampling = sampling

    @property
    def repetitions(self) -> Tuple[int, int, int]:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, repetitions: Tuple[int, int, int]):
        repetitions = tuple(repetitions)

        if len(repetitions) != 3:
            raise ValueError('repetitions must be sequence of length 3')

        self._repetitions = repetitions

    @property
    def num_slices(self) -> int:
        return self._potential_unit.num_slices * self.repetitions[2]

    @property
    def frozen_phonons(self) -> AbstractFrozenPhonons:
        return DummyFrozenPhonons()

    @property
    def num_frozen_phonons(self) -> int:
        return len(self.frozen_phonons)

    @property
    def ensemble_axes_metadata(self):
        axes_metadata = []
        axes_metadata += self.frozen_phonons.ensemble_axes_metadata
        axes_metadata += [ThicknessAxis(values=self.exit_thicknesses)]
        return axes_metadata

    def ensemble_blocks(self, chunks=(1, -1)):
        chunks = validate_chunks(self.ensemble_shape, chunks)
        random_state = dask.delayed(self.random_state)

        potential_unit_blocks = self.potential_unit.ensemble_blocks((-1, -1))
        potential_partial = self.potential_unit.ensemble_partial()

        blocks = []
        for i in range(self.num_frozen_phonon_configs):
            p = dask.delayed(potential_partial)(*potential_unit_blocks)
            p = da.from_delayed(p, shape=(1,), dtype=object)
            blocks.append(p)

        blocks = da.concatenate(blocks)

        return blocks, self._exit_plane_blocks(chunks[1:])

    def ensemble_partial(self):
        def crystal_potential(*args, **kwargs):
            potential_unit = args[0].item()
            exit_planes = args[1].item()

            arr = np.zeros((1,) * len(args), dtype=object)
            arr.itemset(CrystalPotential(potential_unit,
                                         exit_planes=exit_planes,
                                         **kwargs))
            return arr

        kwargs = {'repetitions': self.repetitions}
        return partial(crystal_potential, **kwargs)

    def generate_configurations(self):
        for i in range(self.num_frozen_phonon_configs):
            kwargs = self._copy_as_dict()
            kwargs['num_frozen_phonon_configs'] = 1
            yield CrystalPotential(**kwargs)

    def build(self,
              first_slice: int = 0,
              last_slice: int = None,
              chunks: int = 1,
              lazy: bool = None) -> 'PotentialArray':

        if last_slice is None:
            last_slice = len(self)

        xp = get_array_module(self.device)

        array = xp.zeros((len(self),) + self.gpts, dtype=xp.float32)

        for i, slic in enumerate(self.generate_slices(first_slice, last_slice)):
            array[i] = slic.array

        return PotentialArray(array, sampling=self.sampling, slice_thickness=self.slice_thickness)

    def generate_slices(self, first_slice: int = 0, last_slice: int = None):
        potentials = [potential.build() for potential in self.potential_unit.generate_configurations()]

        first_layer = first_slice // self._potential_unit.num_slices
        if last_slice is None:
            last_layer = self.repetitions[2]
        else:
            last_layer = last_slice // self._potential_unit.num_slices

        first_slice = first_slice % self._potential_unit.num_slices
        last_slice = None

        # configs = self._calculate_configs(energy, max_batch)

        # if len(configs) == 1:
        #    layers = configs * self.repetitions[2]
        # else:
        #    layers = [configs[np.random.randint(len(configs))] for _ in range(self.repetitions[2])]

        if self.random_state:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState()

        for i in range(self.repetitions[2]):

            j = random_state.randint(0, len(potentials))
            generator = potentials[j].generate_slices()

            for i in range(len(self.potential_unit)):
                yield next(generator).tile(self.repetitions[:2])

    def _copy_as_dict(self, copy_potential: bool = True):

        kwargs = {'repetitions': self.repetitions,
                  'num_frozen_phonon_configs': self.num_frozen_phonon_configs,
                  'exit_planes': self.exit_planes,
                  'random_state': self.random_state}

        if copy_potential:
            kwargs['potential_unit'] = self.potential_unit.copy()

        return kwargs

