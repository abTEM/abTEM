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
- C-PRISM: `SMatrix(upsample=True)` reduces every probe from the complete plane-wave expansion of the
  aperture, so the interpolation factor only sets the number of multislice runs. Adds
  `CompressedSMatrixArray` and `GridScan.commensurate` ([PR #318](https://github.com/abTEM/abTEM/pull/318))
- Phonon-loss (thermal diffuse scattering) energy-loss workflow: `EnergyResolvedAtomsEnsemble`,
  `phonon_loss_diffraction_patterns`, `momentum_resolved_spectrum` and `MomentumResolvedSpectrum`,
  the `SpectralAnnularDetector` and `SpectralSlitDetector`, and detailed-balance thermal weighting that
  splits the classical TDS signal into loss and gain sides
  ([PR #324](https://github.com/abTEM/abTEM/pull/324), [PR #351](https://github.com/abTEM/abTEM/pull/351))
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

Performance:

- The projection integrator is shared by reference across ensemble members instead of being deep-copied
  (and re-uploaded to the GPU) for each ([PR #350](https://github.com/abTEM/abTEM/pull/350))
- Removed a redundant potential rebuild on every scan chunk ([PR #340](https://github.com/abTEM/abTEM/pull/340))

Dependencies:

- The `core-loss` extra is merged into a single `gpaw = ["hankel", "sympy"]` extra, and a new `all` extra
  installs every optional runtime dependency ([PR #329](https://github.com/abTEM/abTEM/pull/329))

Bugfixes:

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

Documentation:

- `sampling="auto"`/`slice_thickness="auto"` documented in detail in the potentials walkthrough, including a
  worked example of the commensurability artifact they remove; cross-referenced from the convergence appendix
  (manual commensurate sampling) and the performance-tips appendix (fast FFT sizes)
- New tutorial on phonon-loss spectroscopy: energy-resolved frozen phonons, the TDS decomposition, the
  momentum-resolved spectrum $S(q, E)$, the spectral detectors and detailed-balance thermal weighting
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
- CBED patterns for Bloch waves ([PR #254](https://github.com/abTEM/abTEM/pull/254))

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
