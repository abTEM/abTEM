
class CrystalPotential(AbstractPotential, HasEventMixin):
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
                 num_frozen_phonon_configs: int = 1):

        self._potential_unit = potential_unit
        self.repetitions = repetitions
        self._num_frozen_phonon_configs = num_frozen_phonon_configs

        if (potential_unit.num_frozen_phonon_configs == 1) & (num_frozen_phonon_configs > 1):
            warnings.warn('"num_frozen_phonon_configs" is greater than one, but the potential unit does not have'
                          'frozen phonons')

        if (potential_unit.num_frozen_phonon_configs > 1) & (num_frozen_phonon_configs == 1):
            warnings.warn('the potential unit has frozen phonons, but "num_frozen_phonon_configs" is set to 1')

        self._cache = Cache(1)
        self._event = Event()

        gpts = (self._potential_unit.gpts[0] * self.repetitions[0],
                self._potential_unit.gpts[1] * self.repetitions[1])
        extent = (self._potential_unit.extent[0] * self.repetitions[0],
                  self._potential_unit.extent[1] * self.repetitions[1])

        self._grid = Grid(extent=extent, gpts=gpts, sampling=self._potential_unit.sampling, lock_extent=True)
        self._grid.observe(self._event.notify)
        self._event.observe(cache_clear_callback(self._cache))

        super().__init__()

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
    def num_frozen_phonon_configs(self):
        return self._num_frozen_phonon_configs

    def generate_frozen_phonon_potentials(self, pbar=False):
        for i in range(self.num_frozen_phonon_configs):
            yield self

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

    def get_slice_thickness(self, i) -> float:
        return self._potential_unit.get_slice_thickness(i)

    @cached_method('_cache')
    def _calculate_configs(self, energy, max_batch=1):
        potential_generators = self._potential_unit.generate_frozen_phonon_potentials(pbar=False)

        potential_configs = []
        for potential in potential_generators:

            if isinstance(potential, AbstractPotentialBuilder):
                potential = potential.build(max_batch=max_batch)
            elif not isinstance(potential, PotentialArray):
                raise RuntimeError()

            if energy is not None:
                potential = potential.as_transmission_function(energy=energy, max_batch=max_batch)

            potential = potential.tile(self.repetitions[:2])
            potential_configs.append(potential)

        return potential_configs

    def _generate_slices_base(self, first_slice=0, last_slice=None, max_batch=1, energy=None):

        first_layer = first_slice // self._potential_unit.num_slices
        if last_slice is None:
            last_layer = self.repetitions[2]
        else:
            last_layer = last_slice // self._potential_unit.num_slices

        first_slice = first_slice % self._potential_unit.num_slices
        last_slice = None

        configs = self._calculate_configs(energy, max_batch)

        if len(configs) == 1:
            layers = configs * self.repetitions[2]
        else:
            layers = [configs[np.random.randint(len(configs))] for _ in range(self.repetitions[2])]

        for layer_num, layer in enumerate(layers[first_layer:last_layer]):

            if layer_num == last_layer:
                last_slice = last_slice % self._potential_unit.num_slices

            for start, end, potential_slice in layer.generate_slices(first_slice=first_slice,
                                                                     last_slice=last_slice,
                                                                     max_batch=max_batch):
                yield layer_num + start, layer_num + end, potential_slice

                first_slice = 0

    def generate_slices(self, first_slice=0, last_slice=None, max_batch=1):
        return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch)

    def generate_transmission_functions(self, energy, first_slice=0, last_slice=None, max_batch=1):
        return self._generate_slices_base(first_slice=first_slice, last_slice=last_slice, max_batch=max_batch,
                                          energy=energy)