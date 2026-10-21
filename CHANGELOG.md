# Changelog

The changes of each release, newest first. The changes of the next release go under a
heading `## Upcoming: <version>`, which is renamed to the version when it is released.

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
