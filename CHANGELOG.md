# Changelog

The changes of each release, newest first. The changes of the next release go under a
heading `## Upcoming: <version>`, which is renamed to the version when it is released.

## Upcoming: 1.1.0

The `dev` branch is at version `1.1.0`. The entries below are merged into `dev` but not yet released.

Fixes that were also released in 1.0.11 are listed under that version as well.

Features:

- Energy ensemble support across the codebase: `PlaneWave`, `Probe`, `SMatrix`, `BlochWaves` and `CTF`
  accept a list of energies, and the resulting `EnergyAxis` propagates through indexing, angular sampling,
  unit conversion and diffraction-spot indexing ([PR #257](https://github.com/abTEM/abTEM/pull/257))
    - Bloch-wave rotation ensembles: `BlochWaves(..., energy=[...]).rotate(axis, angles)` with a
      distribution of angles computes every energy. Diffraction patterns and exit waves get an `EnergyAxis`
      after the rotation axes, the axis order of a multislice rotation series, and each energy and
      orientation keeps its own beams within the union beam set
      ([PR #521](https://github.com/abTEM/abTEM/pull/521))
- C-PRISM: `SMatrix(upsample=True)` reduces every probe from the complete plane-wave expansion of the
  aperture, so the interpolation factor only sets the number of multislice runs. Adds
  `CompressedSMatrixArray` and `GridScan.commensurate` ([PR #318](https://github.com/abTEM/abTEM/pull/318))
- Phonon-loss (thermal diffuse scattering) energy-loss workflow: `EnergyResolvedAtomsEnsemble`,
  `phonon_loss_diffraction_patterns`, `momentum_resolved_spectrum` and `MomentumResolvedSpectrum`,
  the `SpectralAnnularDetector` and `SpectralSlitDetector`, and detailed-balance thermal weighting that
  splits the classical TDS signal into loss and gain sides
  ([PR #324](https://github.com/abTEM/abTEM/pull/324), [PR #351](https://github.com/abTEM/abTEM/pull/351))
- `elastic_diffuse_diffraction_patterns` (also a `Waves` method) forms the total, elastic and diffuse
  intensity of any exit waves with a frozen-phonon axis, keeping every other ensemble axis. It has
  `unbiased=True` for the $(N - 1)/N$ bias of the diffuse part and `reduction_dtype="float64"` for diffuse
  signals much weaker than the total. `phonon_loss_diffraction_patterns` now builds on it and gains the
  same options. Its keyword `component` becomes `components`, and the components `"incoherent"`,
  `"coherent"` and `"tds"` become `"total"`, `"elastic"` and `"diffuse"`; the old names raise a
  `ValueError` naming the new one. `"all"` stacks `("total", "elastic", "diffuse")`, a tuple or list of names
  stacks them in the order given, and the metadata key `phonon_loss_component` becomes
  `frozen_phonon_component`. The loss/gain unfolding refuses a bool, negative or non-finite `temperature`;
  at zero the whole signal is on the loss side. `momentum_resolved_spectrum` takes the diffuse component from a
  stacked result and raises a `ValueError` for the total or elastic component, which contain the Bragg
  intensity ([PR #484](https://github.com/abTEM/abTEM/pull/484))
- Linear-scaling PRISM-EELS for core-loss simulations: `SMatrix.transition_potential_scan`, with
  single- and double-channel scattering and an optional windowed inelastic crop
  ([PR #289](https://github.com/abTEM/abTEM/pull/289))
- In addition to the Gaussian (G) distribution, now also implemented Lorentzian (L), Voigtian (convolution L * G) and
  pseudo-Voigtian (L + G) source-size distributions and filters ([PR #270](https://github.com/abTEM/abTEM/pull/270))
- The exact free-space propagator is now the default for Fourier multislice,
  `FourierMultislice(order="exact")`, replacing the paraxial approximation. Spatial frequencies beyond
  `k > 1 / lambda` are treated as evanescent rather than propagating; the paraxial propagator remains
  available as `order=1` ([PR #298](https://github.com/abTEM/abTEM/pull/298))
- Magnetic potentials and fields from collinear GPAW calculations: `gpaw_magnetic_fields` builds the
  electrostatic potential, vector potential and magnetic field from the same calculator(s) in one call,
  returning a `GPAWMagneticFields` bundle with `.tile()`, `.combined_potential()` and `.show()`;
  `rotate_field` now defaults to `"auto"` ([PR #326](https://github.com/abTEM/abTEM/pull/326))
- `GPAWParametrization` is now usable: it fits a Lobato-form IAM potential to the X-ray scattering factor
  of an all-electron GPAW calculation, with working ionization support and a regularized fit
  ([PR #329](https://github.com/abTEM/abTEM/pull/329))
- `Potential(sampling="auto", slice_thickness="auto")` automatically finds a grid commensurate with the atomic
  lattice — so translation-equivalent atoms discretize identically — near the usual default targets
  ($0.05 \ \mathrm{Å}$, $1 \ \mathrm{Å}$), preferring FFT-fast grid sizes wherever that is compatible with
  commensurability. New `grid.round-to-fast-fft` config key (`'auto'`/`True`/`False`), `Grid.round_to_fast_fft()`,
  and `is_fast_fft_size`/`next_fast_fft_size` helpers in `abtem.core.fft`
  ([PR #274](https://github.com/abTEM/abTEM/pull/274), [PR #347](https://github.com/abTEM/abTEM/pull/347))
- Significant improvements on simulating large potentials on GPU, alongside minor performance improvements
  ([PR #269](https://github.com/abTEM/abTEM/pull/269))
    - The potential is now built in chunks of contiguous slices instead of all at once, keeping peak VRAM
      bounded; new config key `potential.slice-chunk-size` (default `"auto"`)
    - Opt-in multi-GPU via the new config key `dask.multi-gpu` (requires `dask-cuda`)
    - `cupy.fft-cache-size`'s previous `0 MB` default silently disabled the cuFFT plan cache; changed to
      `-1` (unlimited) here, then to the device-relative `auto` default below once unbounded retention
      turned out to cost tens of GB on FFT-unfriendly grids
- **Bloch waves default to the non-paraxial wave equation, `use_wave_eq="exact"`**, the counterpart of the
  default exact multislice propagator, `FourierMultislice(order="exact")`
  ([PR #447](https://github.com/abTEM/abTEM/pull/447), [PR #448](https://github.com/abTEM/abTEM/pull/448)).
  `use_wave_eq=True` remains the paraxial counterpart of `order=1`, and `use_wave_eq=False` the standard
  (textbook) Bloch-wave equation, which was the default before
    - **Results computed with default arguments change.** For Si with `sg_max=0.5`, the R factor between the
      old and new defaults stays below 1 % at 100–300 keV, but reaches 5 % at 20 keV on zone axis and 24 %
      at 20 keV, 1000 Å thick and 7° off zone axis. Passing `use_wave_eq=False` does not restore the old
      numbers exactly, because of the metric fix under Bugfixes
    - `get_kinematical_diffraction_pattern` weights each reflection with the excitation error of the selected
      form, so kinematical patterns also change slightly off zone axis. Beam selection (`sg_max`) and
      diffraction-spot indexing keep the Ewald-sphere excitation error
    - With `"exact"`, beams with $\lambda |g_\perp| \geq 1$ cannot propagate; `BlochWaves` excludes them with
      a warning. This happens only at low energies with a large `g_max` (it needs
      $|g| \approx \sqrt{2}/\lambda$, about $3.6 \ \mathrm{Å^{-1}}$ at 1 keV)
- `BlochWaves.calculate_scattering_matrix(z, lazy=True)` returns a dask array; the method also no longer
  fails on GPU ([PR #448](https://github.com/abTEM/abTEM/pull/448))
- Bloch waves follow the `precision` setting like the rest of abTEM, so the default is now single
  precision; set `precision="float64"` for the previous double-precision results ([PR #449](https://github.com/abTEM/abTEM/pull/449)).
  Single precision is accurate to about $2 \times 10^{-4}$ relative for beams down to $10^{-3}$ of the
  strongest, even at 10–20 kÅ, and any single-precision Bloch-wave propagation emits a
  `BlochWavePrecisionWarning` recommending a double-precision check. The scattering matrix
  (`calculate_scattering_matrix`) is always exponentiated in double precision internally, as scaling and
  squaring breaks down in single precision for realistic beam counts
- Multi-energy `BlochWaves`: `excitation_errors()` returns one row per energy over the union beam set,
  `get_kinematical_diffraction_pattern()` returns patterns with an `EnergyAxis`, and the new
  `select_energy(energy)` returns the single-energy Bloch waves; `calculate_structure_matrix` and
  `calculate_scattering_matrix` still need a single energy and point to it ([PR #456](https://github.com/abTEM/abTEM/pull/456))
- Convergent-beam electron diffraction from Bloch waves: `BlochWaves.calculate_cbed_patterns` solves each
  plane wave of a probe and places its beams on the probe's grid. Without a scan it gives the
  position-averaged (PACBED) pattern; with one, the plane waves add coherently at each position, so
  overlapping discs interfere. Aberration, aperture and tilt series (precession included) and energy
  ensembles are supported, and the calculation is lazy. `BlochwaveEnsemble` takes a fixed
  `orientation_matrix` ([PR #254](https://github.com/abTEM/abTEM/pull/254))

Performance:

- The projection integrator is shared by reference across ensemble members instead of being deep-copied
  (and re-uploaded to the GPU) for each ([PR #350](https://github.com/abTEM/abTEM/pull/350))
- Removed a redundant potential rebuild on every scan chunk ([PR #340](https://github.com/abTEM/abTEM/pull/340))
- `CrystalPotential` builds its unit once per simulation instead of once per ensemble member and lazy scan
  block, unless every member reseeds its own frozen-phonon pool; a list of three `.gpw` files with six members
  is read 3 times instead of 18 ([PR #524](https://github.com/abTEM/abTEM/pull/524))
- The automatic batch sizes of PRISM count the exit planes: the plane waves per chunk of `SMatrix.build`
  hold every exit plane within `dask.chunk-size`, and the positions per reduction batch of
  `SMatrixArray.reduce`, `SMatrix.reduce` and `SMatrix.scan` hold the reduced waves of every exit plane
  and ensemble member. The peak memory of an eager scan with 9 exit planes on a 128 x 128 grid falls from
  9.2 GB to 1.3 GB; an integer `max_batch*`, and a potential with one exit plane and no ensemble, are
  batched as before. Where a batch changes, values move by float32 rounding of the reduction only
  ([PR #563](https://github.com/abTEM/abTEM/pull/563))

Dependencies:

- The `core-loss` extra is merged into a single `gpaw = ["hankel", "sympy"]` extra, and a new `all` extra
  installs every optional runtime dependency ([PR #329](https://github.com/abTEM/abTEM/pull/329))

Bugfixes:

- `RealSpaceMultislice(expansion_scope="full")` no longer subtracts the backscattered wave of each slice from the
  forward wave (Eq. 10 of Micron 190 (2025) 103778) unless `backscatter_correction=True` is given.
    - **Behaviour change:** the correction used to be always applied. The authors validated it at 30 and 200 keV
      with a converged series. Against an exact two-way solution of graphene it makes the strong beams less
      accurate between 1 and 5 keV (16 % against 7 % at 1 keV and order 1, 5.5 % against 1.2 % at 2 keV and order 2)
      and the transmitted intensity exceed 1 (1.014 at 1 keV, where the exact solution gives 0.9999); above about
      30 keV it leaves the strong beams unchanged but puts the beams at 120–377 mrad 1.5 % above the exact solution.
      `return_backscattered=True` still returns the backscattered wave
      ([PR #629](https://github.com/abTEM/abTEM/pull/629))
- `RealSpaceMultislice` applies the tilt of the wave function, as `FourierMultislice` does at orders 1 and 2: the
  beams are translated by `tan(tilt)` per unit depth. It ignored the tilt (of `Waves`, `Probe`, `PlaneWave` and tilt
  ensembles) without any message, in both expansion scopes and in the core-loss multislice
  ([Issue #627](https://github.com/abTEM/abTEM/issues/627), [PR #628](https://github.com/abTEM/abTEM/pull/628))
- `FresnelPropagator(order="exact")`, the default of `FourierMultislice`, propagates tilted beams exactly: it applied
  tilt as the first-order translation `tan(theta) * dz`, so the wave was propagated paraxially about the beam
  direction (a phase error of 0.015 rad per 1 Å slice at 100 mrad and 200 eV, 0.09 rad at 300 mrad). Results for
  tilted beams change, growing with the tilt; untilted beams and `order=1` and `2` are unchanged
  ([PR #626](https://github.com/abTEM/abTEM/pull/626))
- `block_direct=True` in `Waves.diffraction_patterns` blocked a fixed 1 mrad radius. It now blocks the
  bright-field disk plus a margin for a probe with a semiangle cutoff, as `DiffractionPatterns.block_direct()`
  does. For a plane wave, or a cutoff below half a pixel, it blocks the zero-frequency pixel only, now also
  with `fftshift=False` ([PR #484](https://github.com/abTEM/abTEM/pull/484))
- Lazy diffraction patterns of complex128 waves were declared float32, so later dask reductions
  accumulated in float32; they now keep the precision of the waves
  ([PR #484](https://github.com/abTEM/abTEM/pull/484))
- `GPAWPotential` for the new-style GPAW calculator API (GPAW 26+), and `GPAWPotential.from_file` on
  old-style restarted calculators ([PR #325](https://github.com/abTEM/abTEM/pull/325))
- `GPAWPotential` single-calculator `frozen_phonons` ensemble building
  ([PR #327](https://github.com/abTEM/abTEM/pull/327)), plus removal of dead and broken code from
  `GPAWPotential` and `GPAWParametrization` ([PR #328](https://github.com/abTEM/abTEM/pull/328))
- `FieldArray.tile()` for vector-valued fields, and unsupported `frozen_phonons`/`repetitions` on magnetic
  fields now raise instead of being silently ignored ([PR #326](https://github.com/abTEM/abTEM/pull/326))
- Silent corruption in eager multislice for potentials with two or more ensemble axes
  ([PR #333](https://github.com/abTEM/abTEM/pull/333))
- `numba` `TypingError` in quasi-dipole interpolation on some numba/numpy pairings
  ([PR #332](https://github.com/abTEM/abTEM/pull/332))
- Single-point `GridScan` failing when built lazily ([PR #342](https://github.com/abTEM/abTEM/pull/342))
- `LinearAxis` losing its offset under dask ensemble chunk partitioning
  ([PR #344](https://github.com/abTEM/abTEM/pull/344))
- Nondeterministic atom loss in `orthogonalize_cell`, and a hardened Gram-Schmidt fallback
  ([PR #345](https://github.com/abTEM/abTEM/pull/345))
- Repeated axis labels and colorbar overlap in exploded spectrum panels, and silently returned zeros for
  single-configuration TDS ([PR #351](https://github.com/abTEM/abTEM/pull/351))
- Azimuthal convention in `prism_coefficients`, which reflected azimuthally dependent aberrations in a PRISM
  reduction with a `CTF`, and exit planes not being remapped when slicing a `PotentialArray`
  ([PR #318](https://github.com/abTEM/abTEM/pull/318))
- `numpy` 2.5 test failures caused by an ASE deprecation warning
  ([PR #343](https://github.com/abTEM/abTEM/pull/343))
- `Probe.transition_potential_scan` raised for a scan split into more than one chunk (for example through
  `max_batch`) unless the scattering sites were passed explicitly
  ([PR #353](https://github.com/abTEM/abTEM/pull/353))
- Colorbars did not span multi-row exploded plots, and neighbouring panels could abut closely enough for
  their tick labels to collide ([PR #354](https://github.com/abTEM/abTEM/pull/354))
- Multi-GPU hardening, and the configuration fix found while chasing it
  ([PR #346](https://github.com/abTEM/abTEM/pull/346))
    - **The client's configuration now reaches `distributed` workers.** *ab*TEM resolves configuration
      inside each task, and worker processes start fresh and previously saw only the YAML defaults, so
      any distributed computation with a non-default configuration silently used the defaults instead —
      most consequentially `precision`, which meant `float64` runs were computed in `float32`.
      Distributed results obtained with a non-default configuration are worth repeating. Applies to any
      distributed client, CPU clusters included
    - `to_zarr()` on a lazy result honours `dask.multi-gpu`; it previously ignored the flag and ran the
      whole computation on a single device
    - `cupy.fft-cache-size` defaults to `auto` — 25 % of each device's memory, resolved per device —
      rather than unlimited. `-1` restores unlimited, `0 MB` disables the cache, and a size such as
      `512 MB` sets a fixed bound. A single plan larger than the bound runs uncached with a warning
      instead of raising
    - New config keys `dask.multi-gpu-rmm-pool` and `dask.multi-gpu-devices`, for an RMM memory pool per
      worker and for restricting the cluster to a subset of GPUs
    - Automatically sized scan batches are halved on grid sizes that force cuFFT's Bluestein fallback,
      which needs a much larger FFT workspace
    - Warnings replace silent fallbacks: multi-GPU requested but declined (with the reason), a missing
      `if __name__ == "__main__"` guard, and a grid size that forces the Bluestein fallback (naming the
      next fast size)
- Distribution weights (`gaussian`, `lorentzian`, `voigtian`, `pseudo_voigtian`, `from_values`) are now
  applied as probabilities when an ensemble is reduced
  ([PR #466](https://github.com/abTEM/abTEM/pull/466), [issue #305](https://github.com/abTEM/abTEM/issues/305)).
  They previously multiplied the wave amplitude, which gave wrong partial-coherence results:
    - **STEM probes ignored the weights.** Each ensemble member was normalized after weighting, so
      `Probe(defocus=gaussian(σ))` averaged its samples equally, as if the distribution were a top-hat.
      For a 200 kV, α = 25 mrad probe, the peak of the reduced intensity is 1.76× the old value at
      σ = 2.5 nm (25 Å) and 2.21× at σ = 5 nm (50 Å). STEM results with a defocus distribution are
      worth repeating
    - **HRTEM spreads were too narrow.** Weights on the amplitude entered the intensity squared, so a
      Gaussian of standard deviation σ acted as one of σ/√2, and Lorentzian and Voigt spreads gained
      $x^{-4}$ tails
    - The weights of tilt, energy and aperture distributions were ignored
    - Each ensemble member is now the unweighted wave function for its parameter value, the ensemble axis
      carries the weights, and `reduce_ensemble()` returns $\sum_i p_i I_i / \sum_i p_i$. Frozen-phonon
      results are unchanged. `reduce_ensemble(axis=...)` reduces an axis kept with `ensemble_mean=False`,
      and `.sum()`/`.mean()` over a weighted axis warn that they ignore the weights
    - The `.weights` of the distributions now sum to one (previously their squares did), and
      `normalize="intensity"`/`"amplitude"` are deprecated. A `gaussian(σ)` defocus spread corresponds to
      the 1/e focal-spread width `focal_spread = √2·σ` of the temporal envelope
- The Bloch-wave metric $M = \mathrm{diag}\big((1 + g_z/K)^{-1/2}\big)$ was applied inconsistently
  ([PR #448](https://github.com/abTEM/abTEM/pull/448)):
    - The eigendecomposition path (`calculate_diffraction_patterns`) and the matrix-exponential path
      (`calculate_scattering_matrix`) disagreed off zone axis, by up to $1.4 \times 10^{-4}$ in amplitude
      for Si 7° off [001] at 100 keV; they now agree to $10^{-13}$
    - The wave-equation forms (`use_wave_eq=True` and `"exact"`) carried the metric of the standard form,
      where it does not belong. Without it, `"exact"` agrees with the full Helmholtz equation to
      R = 0.03–1 % (previously up to 33 %), and better with multislice off zone axis: R against exact
      multislice drops from 6.7 % to 3.5 % for Si at 100 keV, 1000 Å, 7° off zone axis
    - `use_wave_eq=False` now solves the textbook Bloch-wave equation exactly, which moves its results by
      up to about 5 % R at 100 keV off zone axis. They move *away* from the full Helmholtz equation: the
      old, inconsistent metric approximated the $\gamma^2$ term that the textbook equation drops. The
      default `"exact"` form is the accurate choice
- Multi-energy `BlochWaves` methods that answered silently for the first energy only, and multi-energy
  `calculate_exit_waves(lazy=False)` / `calculate_diffraction_patterns(lazy=False)`, which returned a lazy
  result or raised ([PR #456](https://github.com/abTEM/abTEM/pull/456))
- `StructureFactor.get_projected_potential` ignored `sampling`, failed on an integer `gpts` and on a
  sequence `slice_thickness` ([PR #456](https://github.com/abTEM/abTEM/pull/456)), and `StructureFactorArray.get_projected_potential` ignored its
  `lazy` argument ([PR #487](https://github.com/abTEM/abTEM/pull/487))
- `IndexedDiffractionPatterns.sort()` always raised, and `positions_dict` returned three entries
  whatever the number of spots ([PR #456](https://github.com/abTEM/abTEM/pull/456))
- `index_diffraction_spots` on an energy ensemble indexed every member with the highest energy; each
  member now uses its own wavelength ([PR #446](https://github.com/abTEM/abTEM/pull/446))
- `BlochwaveEnsemble` rotations failed with SciPy 1.18, and eager (`lazy=False`) ensemble results gave
  every orientation the last orientation's pattern ([PR #450](https://github.com/abTEM/abTEM/pull/450))
- A-, B- and C-centred cells crashed `StructureFactor`, and their detection used the wrong centring
  vectors, which would have removed allowed reflections such as graphene's (100) ([PR #487](https://github.com/abTEM/abTEM/pull/487))
- A `CrystalPotential` without `seeds` is one crystal per simulation: every lazy scan block, potential chunk
  and backscattering pass sees the same mosaic of frozen-phonon configurations. The mosaic, and with
  `num_frozen_phonons` the member seeds, follow from the seeds of the unit's `FrozenPhonons`, so a seeded unit
  makes the crystal reproducible; two crystals made from one `FrozenPhonons` unit are the same crystal, and
  independent crystals need different `seeds`. `seeds` also accepts an int or a list, as documented
  ([PR #524](https://github.com/abTEM/abTEM/pull/524))
- `TransmissionFunction` could not be run through multislice, sliced or loaded with `from_zarr`, and
  `PotentialArray.transmission_function` dropped the potential's exit planes, ensemble axes and metadata
  ([PR #524](https://github.com/abTEM/abTEM/pull/524))
- Lazy results declare the dtype their blocks return, which is also the eager result's dtype. `Waves.dtype`
  is the dtype of the array instead of the configured precision, and lazy FFTs, transforms, detectors and
  measurements of 64-bit data under `float32` (or 32-bit data under `float64`) no longer declare, and
  accumulate reductions in, the configured precision. `Waves.multislice` and
  `Waves.transition_potential_multislice` cast waves in the other precision to the configured one before
  propagating, so every path returns the configured precision ([PR #524](https://github.com/abTEM/abTEM/pull/524))
- Backscattered and full-expansion real-space multislice, detector output declarations, and smaller fixes
  ([PR #499](https://github.com/abTEM/abTEM/pull/499))
    - Backscattered waves of an ensemble potential (frozen phonons, `CrystalPotential`) are now back-propagated
      per configuration, through that configuration's own slices; before, any ensemble potential raised
      `Wrong shapes`, or returned wrong backscattered waves when the number of configurations equalled the
      number of exit planes. Backscattering now requires `potential.exit_planes` to start with the entrance
      plane `-1`, and raises a `ValueError` otherwise (exit planes such as `(2, 5)` returned all zeros). The
      back-propagation holds one summed slice per exit plane instead of every slice of the configuration
    - The full-expansion multislice lost its correction term at every `potential_chunk_size` boundary, so
      transmitted and backscattered waves depended on the chunk size (backscattered waves by about 100 % of
      their maximum in the test case). It now gives the one-chunk result for every chunk size
    - The full-expansion series raised `NotConvergedError` for a wave that is exactly zero (identical
      consecutive slices, a slab followed by vacuum, an empty potential)
    - `PixelatedDetector(reciprocal_space=False)` and `WavesDetector(gpts=...)` now declare the shape and
      sampling of what they return, so scans, multislice, PRISM and lazy `detect` work with them;
      `PixelatedDetector(reciprocal_space=False, resample="uniform")` raises a `ValueError`
    - `SegmentedDetector(outer=None)` no longer crashes eagerly or integrates to a stale angle, and no radial
      detector writes the automatic outer angle onto the detector it is given
    - `build(first_slice, last_slice)` of a potential declares its own slice count and exit planes
    - `DiffractionPatterns.block_direct()` no longer grows an array-valued `semiangle_cutoff` in place, and
      GPAW's radial atom solvers no longer swap the process-wide `sys.stdout`
    - **Behaviour change:** an `AnnularDetector` with an automatic outer angle (`outer=None`) now integrates to
      `min(waves.cutoff_angles)` instead of its floor, the angle `show` draws and PRISM already used. Detected
      values change by up to 1.2e-3 relative (WSe2, 100 keV, `inner=50`), so results computed earlier shift
      slightly
- Changed results of `integrate_gradient` and `block_direct()`, and fixes to lazy `Waves.normalize()`, eager
  `SMatrix.build()`, `GPAWPotential` with a trajectory and arithmetic between CPU and GPU data
  ([PR #556](https://github.com/abTEM/abTEM/pull/556))
    - **Behaviour change:** `Images.integrate_gradient` shifts every image of an ensemble so that its own
      minimum is 0. An eager ensemble previously shared one constant, the minimum over the whole ensemble, and
      a lazy result was shifted by the minimum of each dask block, so it depended on the chunking. Only the
      offset of an ensemble member differs; a single image gives the same result. This applies to
      `DiffractionPatterns.integrated_center_of_mass` as well
    - **Behaviour change:** `DiffractionPatterns.block_direct()` without a `semiangle_cutoff` in the metadata
      blocks only the zero-angle pixel: the default radius is half the smaller angular sampling. It was the
      larger angular sampling, which also zeroed the neighbouring pixels (four, for similar samplings); for the
      pattern of a single unit cell those are the first-order reflections. Patterns with a `semiangle_cutoff`
      and calls with an explicit `radius` are unchanged. With `margin=True` and no `radius`, the margin (the
      larger angular sampling) is added to the new, smaller radius, so fewer pixels are blocked than before
    - Lazy `Waves.normalize()` raised `AttributeError`; it now gives the eager result for any chunking,
      whether the waves are stored in real or reciprocal space
    - Under `float64` the eager `SMatrix.build()` stored `complex64` plane waves; it now follows the configured
      precision, as the lazy build does. The eager S-matrix, on the device or on the host, is `complex128` and
      takes twice the memory it did; `float32` is unchanged
    - Eager `SMatrix.build()` with `store_on_host=True` raised `AttributeError` on the CPU device
    - A `GPAWPotential` of a single calculator with an `AtomsEnsemble` or `EnergyResolvedAtomsEnsemble` as
      `frozen_phonons` raises a `ValueError` at construction. It was accepted, and then raised `TypeError` in
      every build and multislice. One calculator takes `FrozenPhonons`; for a trajectory, pass one calculator
      per frame
    - Arithmetic between CPU and GPU data (an array object with CuPy data and a dask array of NumPy chunks or
      an array object with CPU data, or the reverse) gives the GPU result: the side on the host is moved to the
      device, in either operand order, and a lazy one block by block at compute time. It raised CuPy's
      `TypeError`, for two lazy operands only at compute time. NumPy scalars and Python numbers are handled as
      before
- Slice, grid and scan counts one too many when the step divides the extent exactly, precision-dependent and
  overlapping `DiffractionPatterns.azimuthal_average()` bins, and the license file missing from the wheel
  ([PR #564](https://github.com/abTEM/abTEM/pull/564))
    - **Behaviour change:** a `slice_thickness` or `sampling` that divides the depth or extent exactly gives
      exactly that many slices or grid points. Rounding noise in the quotient (`10.8 / 0.3` is
      `36.00000000000001`) used to add one: 37 slices of 0.2919 Å instead of 36 of 0.3 Å, `gpts=(37, 19)`
      instead of `(36, 18)`. This applies to `Potential`, `CrystalPotential`, `Grid`, `GridScan`,
      `Images.interpolate`, the line-profile `interpolate`, the detectors' `resample`, the grid of
      `Potential(sampling="auto")` and the target grid of `commensurate_gpts`. A quotient with a genuine
      remainder still rounds up. A repeated `GPAWPotential` with a transformed cell now matches the tiled
      one-cell potential, where its slices took one z sample more from the sixth slice on
    - **Behaviour change:** `LineScan(sampling=...)` with `endpoint=True` (the default) spaces its positions
      at or below the requested sampling, like `GridScan(endpoint=True)`: 10 Å at 0.3 Å gives 35 positions,
      not 34 positions 0.303 Å apart
    - A zero-length `LineScan` with a sampling, and an axis of zero extent in a `GridScan` with
      `endpoint=False`, have one position and simulate it, where a scan raised an error.
      `Images.interpolate_line` over a zero-length line gives the value at that point for `width=0` and
      raises `ValueError` for `width > 0`
    - **Behaviour change:** the step bins of `DiffractionPatterns.azimuthal_average()` are the half-open rings
      `c - width / 2 <= r < c + width / 2`. The default bins no longer overlap: every pixel counts in exactly
      one bin, and bin 0 is the direct beam alone. Each bin averages about half as many pixels as before.
      `width` is the full width of the ring, so `width=1.5` gives a ring of full width 1.5, where it gave 3.
      Which pixels a bin holds no longer depends on the precision or the pattern size, and a lazy result has
      the shape of the eager one. The Gaussian weighting is unchanged. A lazy call with an unknown
      `weighting_function` raises at the call, not at `compute()`
    - The wheel ships the `LICENSE` file
- PRISM with several exit planes, ensembles and single positions
  ([PR #563](https://github.com/abTEM/abTEM/pull/563)):
    - **Behaviour change:** PRISM `reduce` without a scan, and `reduce`/`scan` at a single `(x, y)`
      position, return no position axis, as multislice does. Before, the result kept a length-1 position
      axis, except for pixelated detectors on the default lazy path, so eager and lazy results disagreed;
      code that indexed that axis must drop the index. `scan(scan=None)` is a full `GridScan` and is
      unchanged
    - Elastic PRISM with a potential that has more than one exit plane: `SMatrix.build`, `reduce` and
      `scan` (eager and lazy) carry an exit-plane axis, placed after the potential ensemble axes as in
      multislice, instead of raising a broadcast or axes-metadata error. `upsample=True` with more than
      one exit plane raises `NotImplementedError`
    - Eager PRISM scans with a `WavesDetector` over `FrozenPhonons(ensemble_mean=True)` keep the exit
      waves of every configuration, as multislice and lazy PRISM do, instead of averaging them
    - Lazy and eager `SMatrix.transition_potential_scan` with `upsample=True` no longer compute the
      compressed S-matrix, so frozen phonons and several exit planes work; the dummy probes of an
      upsampled elastic scan size a default `PixelatedDetector` and `FlexibleAnnularDetector` from the
      downsampled grid on non-square grids
- Exports of reciprocal-space `Waves` and of `DiffractionPatterns(fftshift=False)` on axes that match the data,
  and the `to_quantem()` origin ([PR #572](https://github.com/abTEM/abTEM/pull/572))
    - **Behaviour change:** `to_data_array()`, `to_hyperspy()` and `to_quantem()` of a `Waves` with
      `reciprocal_space=True` give fftshifted data on `kx`/`ky` axes in 1/Å, where they gave the unshifted
      coefficients on real-space `x`/`y` axes in Å. The `to_quantem` dataset name is `"DiffractionPatterns"`.
      The waves' own `axes_metadata` and `to_zarr()` are unchanged
    - **Behaviour change:** the same exports of a `DiffractionPatterns` with `fftshift=False` give a shifted
      copy, so the data match the centred axes and the `DataArray` coordinates increase. Selection by
      coordinate value is unchanged; positional indexing is not. Data that are already centred need
      `fftshift=True`, or the exports shift them
    - **Behaviour change:** `to_quantem()` writes each axis's offset as its quantem `origin`, where it wrote 0,
      so a centred diffraction pattern has its zero frequency at k = 0, and scan axes, the radial and azimuthal
      axes of `PolarMeasurements` and offset axes cropped away from index 0 keep their offsets
    - `to_quantem()` of reciprocal-space `Waves` and of `DiffractionPatterns(fftshift=False)` leaves a lazy
      object lazy
- Core-loss PRISM detector outputs and frames
  ([PR #573](https://github.com/abTEM/abTEM/pull/573)):
    - **Behaviour change:** Core-loss PRISM at `interpolation=1` returns real-space outputs
      (`WavesDetector`, `PixelatedDetector(reciprocal_space=False)`) in the cell frame, as core-loss
      multislice and elastic PRISM do. Previously each position was rolled by half the window minus its
      pixel, so the image of a position did not sit at that position. Outputs in reciprocal space are
      unchanged, and so are those at `interpolation > 1`
    - **Behaviour change:** The results of the eager core-loss PRISM scan over a static potential carry
      `semiangle_cutoff` and `base_tilt_x`/`base_tilt_y` in their metadata, as the lazy and ensemble
      results and core-loss multislice do. `block_direct()` of such a diffraction pattern blocks the
      direct disk, where it blocked only the zero-angle pixel without them
    - Core-loss PRISM (`SMatrix.transition_potential_scan`, eager and lazy) with a downsampled (the
      default) or an upsampled S-matrix sizes the output of a `PixelatedDetector`, a
      `FlexibleAnnularDetector` and a `WavesDetector` from the waves the reduction detects instead of
      from the elastic probes. Eager scans over `FrozenPhonons` with a `PixelatedDetector` or
      `FlexibleAnnularDetector` no longer fail to broadcast, the shape a lazy scan declares equals the
      computed one, and a `WavesDetector` no longer raises `AttributeError: 'Waves' object has no
      attribute 'reduce_ensemble'`
- Monte Carlo plasmons: `MonteCarloPlasmons.draw_events` runs again, where it raised `TypeError` because
  `PlasmonScatteringEvents` was abstract ([PR #617](https://github.com/abTEM/abTEM/pull/617))
    - **Behaviour change:** the combined deflection of several plasmon excitations is the 2D vector sum of
      the individual deflections, where the polar angles were summed and the sum was also used as the
      azimuth
    - The `CoreLoss.local_potential` docstring gives `max_angle` in mrad, not degrees

Documentation:

- `sampling="auto"`/`slice_thickness="auto"` documented in detail in the potentials walkthrough, including a
  worked example of the commensurability artifact they remove; cross-referenced from the convergence appendix
  (manual commensurate sampling) and the performance-tips appendix (fast FFT sizes)
- New tutorial on phonon-loss spectroscopy: energy-resolved frozen phonons, the decomposition into total, elastic
  and diffuse intensity, the momentum-resolved spectrum $S(q, E)$, the spectral detectors and detailed-balance
  thermal weighting
- Energy ensembles documented in the wave-function walkthrough, with an energy series added to the
  multislice walkthrough
- PRISM-EELS added to the core-loss tutorial, compared against the equivalent multislice scan
- The exact free-space propagator is documented in the multislice walkthrough and the real-space multislice
  tutorial, which now selects the paraxial propagator explicitly where it compares algorithms at equal order
- The installation page documents the optional pip extras (`gpaw`, `extra`, `all`) and why the GPU packages
  are not among them
- The configuration reference is synchronized with the new and changed config keys
- The multiple-GPUs section of the parallelization walkthrough is expanded to cover the multi-GPU
  hardening fixes above, and the FFT plan-cache documentation is corrected to match the shipped `auto`
  default (it previously described a stale `0 MB` default that was never shipped)

Planned for this release (not yet merged):

- Support for skewed pixels (non-orthogonal x,y,z cell axes) ([PR #282](https://github.com/abTEM/abTEM/pull/282))
- Radially variable detector sensitivity ([PR #283](https://github.com/abTEM/abTEM/pull/283))
- Plasmons: fast `PhaseScramblePlasmons` for multislice and PRISM, and `MonteCarloPlasmons` for Bloch wave

## 1.0.11

A bugfix release on top of 1.0.10 with fixes ported from `dev`; no features and no new defaults.
Plain potentials, plane-wave and probe multislice with default settings, ensembles of unweighted
parameters, zarr round trips and the configuration defaults are bit-identical to 1.0.10 at the
default float32 precision (checked on 66 scenarios).

### Fixed: results that change

Magnitudes are from small test systems and vary with the data.

**4D-STEM and measurements**
- Centre of mass and iCOM are divided by the captured intensity ([#417](https://github.com/abTEM/abTEM/pull/417)); an all-zero pattern gives NaN instead of 0.
- Default diffraction patterns are unshifted: their coordinates, `azimuthal_average`, `bandlimit` and centre of mass now agree with that ([#463](https://github.com/abTEM/abTEM/pull/463); azimuthal averages differ by up to 80 %).
- Radial bins are exactly `step_size` wide ([#469](https://github.com/abTEM/abTEM/pull/469); default ADF/HAADF move by 0.2 to 0.8 %, `radial_binning` by 13 to 33 %).
- A `FlexibleAnnularDetector` without `outer` follows the waves of each call, not the first ones it saw ([#434](https://github.com/abTEM/abTEM/pull/434)).
- `PolarMeasurements.integrate` azimuthal limits ([#459](https://github.com/abTEM/abTEM/pull/459)), rad/deg conversions ([#458](https://github.com/abTEM/abTEM/pull/458), off by 1e6), `depth_profile` and `Images.coordinates` ([#461](https://github.com/abTEM/abTEM/pull/461)), `interpolate_line` and `2 / images` ([#470](https://github.com/abTEM/abTEM/pull/470)), periodic spline interpolation ([#506](https://github.com/abTEM/abTEM/pull/506)).
- `AnnularDetector(offset=...)` is applied ([#324](https://github.com/abTEM/abTEM/pull/324)); `get_detector_region` returns percent as a float ([#283](https://github.com/abTEM/abTEM/pull/283)).
- `block_direct=True` blocks the whole bright-field disk instead of 1 mrad ([#484](https://github.com/abTEM/abTEM/pull/484)); `block_direct()` without cutoff blocks only the zero-angle pixel ([#556](https://github.com/abTEM/abTEM/pull/556)).
- Reciprocal-space waves measure like real-space ones ([#565](https://github.com/abTEM/abTEM/pull/565), [#566](https://github.com/abTEM/abTEM/pull/566)); lazy `gaussian_source_size` is periodic like the eager one ([#485](https://github.com/abTEM/abTEM/pull/485)); `integrate_gradient` takes the minimum per image ([#556](https://github.com/abTEM/abTEM/pull/556)).
- Seeded Poisson and scan noise give another realisation for the same seed ([#464](https://github.com/abTEM/abTEM/pull/464), [#475](https://github.com/abTEM/abTEM/pull/475)). Float32 stays float32 in transfer functions, `integrate_gradient` and `center_of_mass` ([#604](https://github.com/abTEM/abTEM/pull/604)).
- `PixelatedDetector(reciprocal_space=False)` has real-space axes, `WavesDetector(gpts=...)` declares the right shape, and `block_direct` no longer grows its metadata ([#499](https://github.com/abTEM/abTEM/pull/499)).

**PRISM**
- Aberrated `SMatrix.reduce`/`scan` match `Probe` ([#318](https://github.com/abTEM/abTEM/pull/318); 17 to 75 %). An eager frozen-phonon reduce is averaged like the lazy one ([#576](https://github.com/abTEM/abTEM/pull/576)), and eager `SMatrix.scan`/`reduce` reduce the `CTF` ensemble axes ([#466](https://github.com/abTEM/abTEM/pull/466)). PRISM no longer leaves its outer angle on the caller's detector ([#491](https://github.com/abTEM/abTEM/pull/491)).

**Potentials and atoms**
- `orthogonalize_cell` followed by `Potential` no longer drops atoms near cell faces ([#345](https://github.com/abTEM/abTEM/pull/345), [#390](https://github.com/abTEM/abTEM/pull/390), [#555](https://github.com/abTEM/abTEM/pull/555); wurtzite GaN lost 2 of 8 atoms, projected potential +33 %). Non-periodic cells keep displaced atoms ([#460](https://github.com/abTEM/abTEM/pull/460), [#555](https://github.com/abTEM/abTEM/pull/555)).
- `CrystalPotential` honours `first_slice`/`last_slice` ([#269](https://github.com/abTEM/abTEM/pull/269)); `GPAWPotential(repetitions=...)` was wrong by about 100 % ([#494](https://github.com/abTEM/abTEM/pull/494)); `GPAWParametrization` applies `charge` ([#329](https://github.com/abTEM/abTEM/pull/329)).
- GPAW and Bloch slices are integrated centred on their slices, which shifts each slice by dz/2 (5 % for GPAW, 17 to 30 % for Bloch; whole-cell sums unchanged) ([#574](https://github.com/abTEM/abTEM/pull/574)); `ChargeDensityPotential` slices are sampled at midpoints ([#472](https://github.com/abTEM/abTEM/pull/472)).
- IAM magnetic fields change by ~1e-6 and their slices are labelled `ÅT`, `Å²T` ([#543](https://github.com/abTEM/abTEM/pull/543)); `projection="finite"` with sigmas for only some elements dropped the others ([#480](https://github.com/abTEM/abTEM/pull/480)); the GPAW magnetic `real_space` projection ignored `first_slice` ([#544](https://github.com/abTEM/abTEM/pull/544)); `wrapped_crop_2d` outside the cell ([#389](https://github.com/abTEM/abTEM/pull/389)); eager multislice of potentials with two or more ensemble axes scrambled the results ([#333](https://github.com/abTEM/abTEM/pull/333)).

**Bloch waves**
- Non-centrosymmetric structures were computed for the negated structure ([#379](https://github.com/abTEM/abTEM/pull/379); wurtzite exit waves change by up to 1.27).
- The symmetrising metric is applied consistently ([#448](https://github.com/abTEM/abTEM/pull/448)): the eigendecomposition and matrix-exponential paths now agree to rounding. For Si at 100 keV, default intensities change by R = 1.5e-3 to 2.2e-3 on the zone axis and 3e-4 to 1.3e-3 at a 7° tilt (R = Σ|ΔI| / ΣI).
- Eager rotation ensembles ([#450](https://github.com/abTEM/abTEM/pull/450)), A/B/C centring ([#487](https://github.com/abTEM/abTEM/pull/487)) and CuPy scattering matrices ([#449](https://github.com/abTEM/abTEM/pull/449)).

**HRTEM and CTF**
- `set_aberrations('scherzer')` C10 sign and ordering ([#471](https://github.com/abTEM/abTEM/pull/471)); `apply_ctf` no longer writes the waves' energy into the CTF ([#489](https://github.com/abTEM/abTEM/pull/489)).

**Core-loss**
- Continuum normalisation ([#364](https://github.com/abTEM/abTEM/pull/364)), the (2l+1) double count ([#368](https://github.com/abTEM/abTEM/pull/368); L2,3 ÷3, M4,5 ÷5), site threshold and a spurious 1.02 factor ([#468](https://github.com/abTEM/abTEM/pull/468); K +3 to 4 %, L2,3 −22 to −27 %) and the vxc vacuum tail (−4 % at 3 eV). Single-channel frozen-phonon core loss was `num_configs` times too large or zero ([#387](https://github.com/abTEM/abTEM/pull/387)).

**Backscattering**
- Back-propagation is corrected ([#562](https://github.com/abTEM/abTEM/pull/562)) and ensembles work ([#499](https://github.com/abTEM/abTEM/pull/499)).

**Weighted distributions**
- Distribution weights are probabilities applied when the ensemble is reduced ([#466](https://github.com/abTEM/abTEM/pull/466)). In 1.0.10 a STEM focal spread was effectively a top-hat, an HRTEM Gaussian spread had a standard deviation of σ/√2, and tilt and aperture weights were ignored. The peak of the reduced intensity of a 200 kV, 25 mrad probe with `gaussian(σ, num_samples=11)` defocus is 1.76 times the 1.0.10 value at σ = 25 Å and 2.21 times at 50 Å. Frozen-phonon results are unchanged.
- New: `OrdinalAxis.weights`/`from_distribution` and `reduce_ensemble(axis=)`. `gaussian` weights sum to one; `normalize='intensity'` and `'amplitude'` are deprecated; `.sum()`/`.mean()` over a weighted axis warn; negative or wrong-length weights raise `ValueError`. Bloch rotation spreads and noise axes carry no weights, as before. Zarr files with a weighted axis cannot be read by 1.0.10.

**Distributed computations**
- The client's configuration reaches the workers of a `distributed` client, also for `to_tiff` and `to_zarr(compute=False)` ([#346](https://github.com/abTEM/abTEM/pull/346)). Workers used to see only the YAML defaults, so for example `precision='float64'` runs were computed in `float32`. Distributed results obtained with a non-default configuration are worth repeating.

**Other**
- FFTW `fftn` over three axes transformed the wrong axes and strided input with `overwrite_x=True` raised ([#558](https://github.com/abTEM/abTEM/pull/558)); a lazy `fft2_convolve` had the wrong shape for long kernels ([#577](https://github.com/abTEM/abTEM/pull/577)).
- `Waves.phase_shift` relabelled reciprocal-space waves and widened complex64 ([#550](https://github.com/abTEM/abTEM/pull/550)), as did `tile(renormalize=True)` ([#414](https://github.com/abTEM/abTEM/pull/414)); 1D `interpolate` lost float64/complex dtypes ([#427](https://github.com/abTEM/abTEM/pull/427)); `bandlimit` and `block_direct(radius=0)` kept the direct beam on some grids ([#506](https://github.com/abTEM/abTEM/pull/506)).
- `StructureFactorArray.get_projected_potential` honours `sampling=` and an integer `gpts` and `IndexedDiffractionPatterns.positions_dict` is complete ([#456](https://github.com/abTEM/abTEM/pull/456)).

### Fixed: errors and crashes

- No longer raising: `Images.scan_noise`, `apply_ctf` at a second energy, `number - Images`, lazy `gaussian_filter` with a constant boundary, lazy `normalize`, single-point `GridScan`, noisy near-orthogonal `Potential` cells, tilted eager Bloch exit waves with scalar thickness, `IndexedDiffractionPatterns.sort()`, core loss on 16 or 17 pixel grids, `AnnularDetector` over a `LineScan` with an extra ensemble axis, GPAW 26 calculators, frozen-phonon `GPAWPotential`, ensemble backscattering, A/B/C centring, graphene repeated along c, and numba typing on some numba/numpy pairings.
- `TransitionPotentialArray.filter_sites` with `Atoms` of several elements and a threshold no longer raises `IndexError`; it returns the kept atoms of the element of the transition potential, as `Atoms` for `Atoms` input and as positions for positions ([#375](https://github.com/abTEM/abTEM/pull/375)).
- `show(overlay=...)` of an ensemble that has no base axes, such as the integrated core-loss measurements of the core-loss tutorial, gave more line labels than lines, and an ensemble of complex lines fewer. Matplotlib 3.11 raises an error for this ("label must be scalar or have the same length as the input data"); older versions took the whole list as the label. The lines are now labelled by the axes that are overlaid.
- `excitations_weights` fails no longer on NumPy 2 (`np.math`). On CuPy 14.2 a lazy FFT in a dask worker thread no longer fails with `CUFFT_EXEC_FAILED` (see `cupy.fft-cache-size`).

### Changed: behaviour

- abTEM reads `~/.config/abtem` and `$ABTEM_CONFIG` as well as dask's configuration files ([#416](https://github.com/abTEM/abTEM/pull/416)); 1.0.10 ignored them. A leftover `cupy.fft-cache-size: 0 MB` there now disables the cuFFT plan cache.
- `cupy.fft-cache-size`, ignored in 1.0.10, is read and defaults to `auto` (25 % of the device memory per thread); `0 MB` disables the cache, `-1` removes the bound, and a plan larger than the bound runs uncached with a warning.
- `LineScan.angle` reports degrees ([#470](https://github.com/abTEM/abTEM/pull/470)). `SegmentedDetector(to_cpu=...)` defaults to `True` as documented ([#371](https://github.com/abTEM/abTEM/pull/371)). Exploded `show()` grids use 0.25 between panels ([#354](https://github.com/abTEM/abTEM/pull/354)).
- New errors instead of wrong results: backscattering with exit planes that do not start at the entrance plane or are not after every slice ([#562](https://github.com/abTEM/abTEM/pull/562), [#499](https://github.com/abTEM/abTEM/pull/499)); a built `TransitionPotentialArray` at another energy or grid ([#386](https://github.com/abTEM/abTEM/pull/386)); `PixelatedDetector(resample='uniform')` without reciprocal space ([#499](https://github.com/abTEM/abTEM/pull/499)); `transition_potential_scan` with `expansion_scope="full"` ([#506](https://github.com/abTEM/abTEM/pull/506)).
- Zarr arrays over 512 MiB use a multi-chunk layout and a failed write removes the partial file ([#363](https://github.com/abTEM/abTEM/pull/363), [#369](https://github.com/abTEM/abTEM/pull/369)). `GPAWParametrization` for Pm, Sm, Eu and Pu falls back to an unpolarised atom with a warning ([#467](https://github.com/abTEM/abTEM/pull/467)). Importing `abtem` no longer fails if `cupyx.scipy.ndimage` cannot be loaded ([#381](https://github.com/abTEM/abTEM/pull/381)).

## 1.0.10

Features:

- Expanded real-space multislice with propagator- and fully-corrected algorithms, and backscattered waves ([PR #236](https://github.com/abTEM/abTEM/pull/236))
    - Related internal function name change: standard multislice is now properly called `FourierMultislice`
- Updated `BullseyeAperture` to use smoothed aperture edges/corners ([PR #266](https://github.com/abTEM/abTEM/pull/266))
- Logarithmic scale display for `DiffractionPatterns` and images ([PR #303](https://github.com/abTEM/abTEM/pull/303))
- Finite-projection integrals 7x faster on CPU ([PR #309](https://github.com/abTEM/abTEM/pull/309))

Documentation:

- Expanded tutorial on real-space multislice
- Depth-profile visualization of potentials in the walkthrough
- Single-file Zarr zip storage, logarithmic display scaling, soft Bullseye apertures, anisotropic
  Debye-Waller factors and B-factor conversion helpers
- All published notebooks verified to run with this release

Dependencies:

- NumPy 2.0 or newer is now required ([PR #245](https://github.com/abTEM/abTEM/pull/245))
- GitHub actions based on `uv` and now cover more versions (including Python 3.14) ([PR #308](https://github.com/abTEM/abTEM/pull/308))
- New branching structure: `dev` for development, `main` for releases (only via PRs from `dev`)
- Support for Zarr 3 `ZipStore` (requiring `zarr>=3.1`)
    - Added Zstandard compression of the arrays in the ZipStore at default level 4 (hat tip: quantEM) ([PR #252](https://github.com/abTEM/abTEM/pull/252))
- Deprecated `[gpu]` optional dependency (as just specifying `cupy` will not install the correct CUDA version)
- Moved `testing`, `docs`, and `dev` from optional dependencies to groups
- Narrow Dask version exclusion to !=2025.12.*,!=2026.1.0,!=2026.1.1 ([PR #285](https://github.com/abTEM/abTEM/pull/285))
- Declared `sympy` as an optional dependency (`core-loss` extra), required for core-loss EELS form
  factors ([PR #320](https://github.com/abTEM/abTEM/pull/320))

Bugfixes:

- Frozen-phonon ensemble handling ([PR #267](https://github.com/abTEM/abTEM/pull/267) & [PR #292](https://github.com/abTEM/abTEM/pull/292))
  - May also have resulted in incorrect behavior with `ensemble_mean = False` for e.g. defocus distributions
- `CrystalPotential` with frozen phonons bugs (especially bad on – luckily rare – eager compute) ([PR #306](https://github.com/abTEM/abTEM/pull/306))
- Minor bugs, unsafe patterns, and dead code ([PR #265](https://github.com/abTEM/abTEM/pull/265))
- Anisotropic Debye-Waller factors for Bloch waves ([PR #271](https://github.com/abTEM/abTEM/pull/271))
  - Added helper functions to convert between crystallographic B-factors and thermal sigmas
- Added missing `.calculate_exit_waves` for `BlochwaveEnsemble` ([PR #294](https://github.com/abTEM/abTEM/pull/294))
- Silent atom drop when `z`-position lands in `SliceIndexedAtoms` blind spot ([PR #273](https://github.com/abTEM/abTEM/pull/273))
- Early-exit bug in `orthogonalize_cell` ([PR #291](https://github.com/abTEM/abTEM/pull/291))
- Fixed broken tutorial workflow for core-loss filtered imaging ([PR #284](https://github.com/abTEM/abTEM/pull/284))
  - Minor performance improvements for `transition_potential_scan` ([PR #286](https://github.com/abTEM/abTEM/pull/286))
- `Bullseye` aperture: `ring_width` and `spoke_width` are now validated, rejecting values that previously
  produced a silently wrong (solid disk) aperture; docstring corrected to describe the actual fractional
  units introduced by the soft-edge redesign ([PR #319](https://github.com/abTEM/abTEM/pull/319))
- Unified the task-level progress bar config key on `diagnostics.task_progress` (Bloch-wave code paths
  previously read a different, non-functional key) ([PR #321](https://github.com/abTEM/abTEM/pull/321))

## 1.0.9

Dependencies:

- Support for `scipy>=1.7` and `cupy>=12`.
- Restricted Dask versions (`>=2022.12.1,!=2025.12.*,!=2026.1.*`) to avoid an issue with Numba in the latest ones

## 1.0.8

Starting the changelog with version 1.0.8.

Features:

- Fully featured Bloch-wave simulations
- Simple real-space multislice algorithm
- Core-loss filtered imaging
- Structured illumination (custom apertures and phase plates)

Documentation:

- Updated and fixed example gallery
- Appendix on convergence
- Expanded tutorial on orthogonal periodic supercells

Bugfixes:

- Numerous small bugfixes and improvements
