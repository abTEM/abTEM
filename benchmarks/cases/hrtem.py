"""Plane-wave multislice: exit wave and selected-area diffraction."""

from abtem_bench.fixtures import multislice_kwargs, silicon, srtio3
from abtem_bench.registry import Tier, Tolerance, Variant, case


@case(
    "hrtem.exitwave",
    tags={"hrtem", "multislice", "v1.1"},
    tiers={
        "quick": Tier(
            gpts=(256, 256),
            reps=(4, 4, 10),
            slice_thickness=2.0,
            structure="si",
            chunk=4,
            max_angle=60.0,
            timeout=120,
        ),
        "standard": Tier(
            gpts=(1024, 1024),
            reps=(8, 8, 40),
            slice_thickness=1.0,
            structure="srtio3",
            chunk=8,
            max_angle=60.0,
            timeout=900,
        ),
        "large": Tier(
            gpts=(4096, 4096),
            reps=(20, 20, 120),
            slice_thickness=2.0,
            structure="si",
            chunk=8,
            max_angle=60.0,
            devices=("gpu",),
        ),
    },
    variants={
        "order1": Variant(algorithm_order=1, compare_as="default"),
    },
    outputs=("exitwave", "saed"),
    tolerance=Tolerance(),
)
def hrtem_exitwave(p, device):
    """200 keV plane wave through the crystal; outputs the complex exit wave
    and its diffraction pattern with the direct beam blocked."""
    from abtem import PlaneWave, Potential

    atoms = silicon(p.reps) if p.structure == "si" else srtio3(p.reps)
    potential = Potential(
        atoms, gpts=p.gpts, slice_thickness=p.slice_thickness, device=device
    )
    wave = PlaneWave(gpts=p.gpts, extent=potential.extent, energy=200e3, device=device)
    kwargs = multislice_kwargs(p)

    def run():
        exit_wave = wave.multislice(potential, lazy=p.lazy, **kwargs)
        if p.lazy:
            exit_wave = exit_wave.compute()
        saed = exit_wave.diffraction_patterns(max_angle=p.max_angle, block_direct=True)
        return {"exitwave": exit_wave, "saed": saed}

    return run
