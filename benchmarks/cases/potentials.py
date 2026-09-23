"""Potential building."""

from abtem_bench.fixtures import silicon, srtio3
from abtem_bench.registry import Tier, Tolerance, case


@case(
    "potential.infinite",
    tags={"potential", "v1.1"},
    tiers={
        "quick": Tier(
            gpts=(256, 256),
            reps=(4, 4, 8),
            slice_thickness=2.0,
            structure="si",
            timeout=120,
        ),
        "standard": Tier(
            gpts=(1024, 1024),
            reps=(8, 8, 15),
            slice_thickness=2.0,
            structure="srtio3",
            timeout=600,
        ),
        "large": Tier(
            gpts=(4096, 4096),
            reps=(20, 20, 75),
            slice_thickness=2.0,
            structure="si",
            devices=("gpu",),
        ),
    },
    outputs=("potential",),
    tolerance=Tolerance(),
    warmup="cold_only",
)
def potential_infinite(p, device):
    """Build the projected potential (infinite projection, lobato) eagerly.

    A one-shot workload: the cold cost is what a user pays, so there is no
    warm-up and no repeat.
    """
    from abtem import Potential

    atoms = silicon(p.reps) if p.structure == "si" else srtio3(p.reps)
    potential = Potential(
        atoms,
        gpts=p.gpts,
        slice_thickness=p.slice_thickness,
        parametrization="lobato",
        projection="infinite",
        device=device,
    )

    def run():
        return potential.build(lazy=False)

    return run
