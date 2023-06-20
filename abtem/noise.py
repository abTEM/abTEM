"""Module for applying noise to measurements."""
from __future__ import annotations

import numpy as np

from abtem.core.axes import (
    NonLinearAxis,
    SampleAxis,
)
from abtem.core.backend import get_array_module
from abtem.distributions import _validate_distribution, BaseDistribution
from abtem.inelastic.phonons import _validate_seeds
from abtem.transform import EnsembleTransform


class NoiseTransform(EnsembleTransform):
    def __init__(
        self,
        dose: float | np.ndarray | BaseDistribution,
        samples: int = None,
        seeds: int | tuple[int, ...] = None,
    ):

        self._dose = _validate_distribution(dose)

        if samples is None and seeds is None:
            samples = 1

        if seeds is None and samples > 1:
            seeds = _validate_seeds(seeds, samples)
            seeds = _validate_distribution(seeds)

        self._seeds = seeds

        super().__init__(
            distributions=(
                "dose",
                "seeds",
            )
        )

    @property
    def dose(self):
        return self._dose

    @property
    def seeds(self):
        return self._seeds

    @property
    def samples(self):
        if hasattr(self.seeds, "__len__"):
            return len(self.seeds)
        else:
            return 1

    @property
    def ensemble_axes_metadata(self):
        ensemble_axes_metadata = []

        if isinstance(self.dose, BaseDistribution):
            ensemble_axes_metadata += [
                NonLinearAxis(label="Dose", values=tuple(self.dose.values), units="e")
            ]

        if isinstance(self.seeds, BaseDistribution):
            ensemble_axes_metadata += [SampleAxis()]

        return ensemble_axes_metadata

    @property
    def metadata(self):
        return {"units": "", "label": "electron counts"}

    def _calculate_new_array(self, array_object) -> np.ndarray | tuple[np.ndarray, ...]:

        array = array_object.array
        xp = get_array_module(array)

        if isinstance(self.seeds, BaseDistribution):
            array = xp.tile(array[None], (self.samples,) + (1,) * len(array.shape))

        if isinstance(self.dose, BaseDistribution):
            dose = xp.array(self.dose.values, dtype=xp.float32)
            array = array[None] * xp.expand_dims(
                dose, tuple(range(1, len(array.shape) + 1))
            )
        else:
            array = array * xp.array(self.dose, dtype=xp.float32)

        if isinstance(self.seeds, BaseDistribution):
            seed = sum(self.seeds.values)
        else:
            seed = self.seeds

        rng = xp.random.default_rng(seed=seed)

        randomized_seed = int(
            rng.integers(np.iinfo(np.int32).max)
        )  # fixes strange cupy bug

        rng = xp.random.RandomState(seed=randomized_seed)

        array = xp.clip(array, a_min=0.0, a_max=None)

        array = rng.poisson(array).astype(xp.float32)

        return array
