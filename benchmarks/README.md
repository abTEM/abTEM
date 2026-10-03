# abtem-bench: regression benchmarks for abTEM

Runs a fixed matrix of user-facing abTEM workloads against a checkout and records, per case, the result arrays, wall time and peak memory, in a format that does not depend on the abtem version. Two bundles are then compared: results (bit-identical, within tolerance, drift, shape change), speed and memory. Design and milestones: [abtem-benchmarks/design](https://github.com/abTEM/abtem-benchmarks/tree/main/design); discussion: [#380](https://github.com/abTEM/abTEM/discussions/380).

## Invariants

- The harness (`abtem_bench/`) and the cases (`abtem_bench/cases/`) always come from the checkout you invoke; only the `abtem` package is swapped per ref, via a git worktree under `.worktrees/bench/<sha>` placed on `PYTHONPATH` behind the harness directory. Both refs run byte-identical case code, and every bundle records a hash of the case sources.
- Each case id runs in a fresh subprocess (`python -P -m abtem_bench.worker`). The worker prints and records which abtem it imported and aborts if it is not the requested worktree.
- Peak host memory is the worker's own `VmHWM` from `/proc/self/status`, read after the case finishes, and the record names it (`memory.rss_meter = "VmHWM"`). `ru_maxrss` from `os.wait4` is recorded too (`peak_rss_wait4_bytes`) but not compared: Linux carries the runner's high-water mark into the child, so it never reads below the runner's own peak. A record without `rss_meter` is read as holding the `wait4` figure (harness versions before 0.2.0); when the two sides of a pair used different meters, `compare` shows no `rss` ratio, flags no memory, and says so in the row's note. On GPU cases a background thread samples the CuPy pool and the driver's device usage; CPU cases never initialise the GPU. No meter runs code on the computation.
- A case may declare `requires=`, a check on the imported abtem; a ref that fails it records `UNSUPPORTED` instead of failing the run. No shipped case needs one: v1.0.10 runs all of them.

## Running

From the repository root, no install needed:

```
export PYTHONPATH=$PWD/benchmarks
python -P -m abtem_bench list --tier quick --device cpu
python -P -m abtem_bench self-check --ref dev --tier quick --device cpu --preset accuracy --out .bench-out/selfcheck
python -P -m abtem_bench capture --ref v1.0.10 --tier quick --device cpu --preset accuracy --out .bench-out/v1.0.10
python -P -m abtem_bench capture --ref dev --tier quick --device cpu --preset accuracy --out .bench-out/dev
python -P -m abtem_bench compare .bench-out/v1.0.10 .bench-out/dev --noise .bench-out/selfcheck --md report.md --fail-on drift,shape
```

On a machine whose runtime has no `git` (a container image that only carries Python), resolve the refs first where git exists, then run inside:

```
python3 -P -m abtem_bench.prepare --ref origin/dev --ref v1.0.10   # host: creates .worktrees/bench/<sha> and index.json
python -P -m abtem_bench capture --ref origin/dev ...              # container: resolves the label from the index
```

`prepare` needs only the standard library. Without git and without a prepared index, the runner stops with a message naming the ref to prepare.

Workers run with the same Python as the CLI, so that environment needs abtem's dependencies (abtem itself comes from each ref's worktree). `uv pip install -e benchmarks` into that environment installs the `abtem-bench` console script instead of the `PYTHONPATH` line. The `-P` flag matters when running from a checkout: without it the current directory shadows `PYTHONPATH`.

Case ids are `name[variant]@tier/device`; `--only` takes globs on ids or names, with brackets matched literally. Tiers are `quick` (seconds per case, the CI tier), `standard` (minutes) and `large` (GPU stress sizes). A case times out after its tier's declared `timeout`, else 120 s on `quick` and 900 s on `standard`. A case is skipped (`SKIPPED-MEMORY`) when less than 8 GB of memory is available when it starts.

`capture` refuses an output directory that already holds files unless `--overwrite` is given, which deletes it first, but only when it is a bundle (it has a top-level `manifest.json`): a non-empty directory that is not a bundle is refused with or without `--overwrite`, and the current directory and its parents are never deleted. `self-check` writes the bundles `a/` and `b/` under its `--out` directory and applies the same rule to each of them; other files in `--out` are left alone. `capture` and `self-check` exit 1 when any captured case ended `ERROR`, `OOM`, `TIMEOUT` or `SKIPPED-MEMORY` (listed on stderr; the bundle is still written). An `--only` pattern that matches no case id is reported as a warning on stderr, in `list`, `capture` and `self-check`. `--repeats` sets the number of warm repeats per worker; `--rounds N` runs the whole case list N times, the refs interleaved per case, and merges the timings of all rounds into one record. The ref worktrees under `.worktrees/bench/` stay for the next run; remove them with `git worktree remove`.

## Presets

`accuracy` pins `precision: float64`, `fft: fftw` with `FFTW_ESTIMATE`, one FFTW and BLAS thread, the synchronous dask scheduler, explicit chunk sizes, fixed dask chunk sizes, `grid.round-to-fast-fft: false`, a fixed cuFFT cache size, and the thread environment, then dumps the fully resolved abTEM and dask configuration into the record. On CPU the expectation is bit-identical output across processes and days on the same machine; the self-check verifies that. `speed` keeps `FFTW_MEASURE` so the cold cost users pay is represented.

## Reading a report

One row per paired case id. `verdict` is the worst over the case's outputs: `IDENTICAL` (bit-for-bit), `OK` (within the case tolerance), `DRIFT` (beyond tolerance), `ACCEPTED` (drift covered by `accepted_changes.toml`), `SHAPE` (shape, dtype or axes changed), or a run status (`UNSUPPORTED`, `ERROR`, `OOM`, `TIMEOUT`, `SKIPPED-MEMORY`; the note says which side failed). `ONLY-A` and `ONLY-B` are ids present in only the reference or only the candidate. Attribution rows pair a candidate variant declaring `compare_as` (`x[order1]`) with the reference default (`x`); they are informational and never fail a gate.

`rel` is the largest relative error over elements above the case's `above_rel` fraction of the reference maximum, `intensity` the largest relative change of the integrated intensity; both name the output when a case has several. Any NaN in these counts as beyond tolerance. Complex outputs are compared on the modulus of the difference, `|candidate - reference|`, and their intensity is the sum of `|x|²`; real outputs use the plain sum. Axes compare numbers to a relative 1e-9 and an absolute 1e-12, with NaN equal to NaN, and everything else exactly. An axis field that defines the grid (`type`, `sampling`, `offset`, `values`, `units`, `endpoint`) present with a value on one side only is a `SHAPE` mismatch; labels, internal `_` fields and other one-sided fields are named in the note (all of them in the JSON report's `axes_info`) but do not change the verdict. An output's metadata is compared by the same rules (ignoring `label`, `units`, `tex_label`, `tex_units` and `_` fields) and its differences are named in the note and recorded in full as `metadata_info`, but never change the verdict. `time`, `cold`, `rss` and `vram` are candidate over reference ratios (warm median, first call, peak resident memory, peak CuPy pool usage).

Speed is flagged beyond `max(--speed-threshold, 3 x speed floor)`, or marked `short` when the two medians differ by less than `--min-delta` seconds (default 0.05). Memory and VRAM are flagged beyond `max(--memory-threshold, 3 x floor)`, and only against a noise floor (`--noise`, a self-check directory or its `noise.json`): one process's peak memory is not reproducible enough to judge without one. With a noise file, the report counts the case ids that have no memory floor in it (`memory not judged for N case ids ...`). The self-check's accuracy spread masks elements below each case's own `above_rel`, as `compare` does. The floor also widens each accuracy tolerance to three times the spread the self-check measured, which matters on GPU, where float64 is not bit-reproducible. `auto` variants are never flagged. `compare` refuses bundles with different case code (`--allow-case-mismatch`), different presets (`--allow-preset-mismatch`) or no common case id.

`--fail-on` takes a comma-separated list of gates: `drift` and `shape` (rows of that verdict), `error` (a failed run on the candidate side, including an id only the candidate holds), `missing` (`ONLY-A` rows, and ids the candidate reports `UNSUPPORTED` where the reference has a result), `speed[:N%]` and `memory[:N%]` (an increase beyond the flag threshold, or beyond `N%`; `memory` needs `--noise`). A `--fail-on` that names no gate is refused; leaving it out means no gates. Exit code 0 means every gate passed, 1 that a gate failed (for `capture` and `self-check`: that a case failed; for `self-check` of the `accuracy` preset on CPU: that the two captures are not bit-identical), 2 an input error or a crash (unknown gate, missing bundle, refused comparison, invalid `accepted_changes.toml`, corrupt bundle file, malformed `--noise` file). The JSON report (`--json`) is strict JSON: non-finite numbers are the strings `"inf"`, `"-inf"` and `"nan"`.

## Accepted changes

A `DRIFT` fails `--fail-on drift` unless an entry in `accepted_changes.toml` covers it. Add the entry in the same pull request as the change:

```toml
[[accepted]]
case = "hrtem.exitwave@quick/*"  # glob over case ids or names; must match a registered id
since = "v1.0.10"                # applies only when the reference bundle is this ref
reason = "One line: what changed and why the new result is right."
pr = 298
max_abs_norm = 6e-3              # required: max|diff| / max|reference| per output
max_intensity = 5e-4             # optional bounds per output: |intensity|, rel (max_rel)
```

`since` matches the reference bundle's ref label, its `git describe`, or a prefix of at least seven hex digits of its sha, so an entry never hides drift against a later reference. Every entry must set `max_abs_norm`, the largest elementwise difference relative to the largest reference element: the integrated intensity is unchanged by a shift, a flip or a phase scramble of the output, so a bound on it alone would accept a corrupted result. Where the accepted drift is itself as large as such a corruption, as for the #298 HRTEM and CBED entries at the standard tier, no bound can tell them apart; the `[order1]` variants, which bound `max_abs_norm` at 6e-7 to 7e-6, guard every code path but the exact propagator itself. `max_rel` and `max_intensity` are optional. A drift beyond any bound of any matching entry stays `DRIFT`, and an output with non-finite values that differ between the two sides is never accepted, whatever the bounds. Unknown keys are refused, and `accepted` must be an array of tables (`[[accepted]]`). An explicit `--accepted <path>` must exist; when the default file is absent, nothing is accepted and the report says so (`accepted_path` and `accepted_exists` in the JSON report). The report renders the entries that apply as the changelog with the drift measured for each, and lists as stale the entries that cover a case both bundles hold but match no drift.

## Adding a case

Decorate a function `(params, device) -> run` with `@case` in a module under `abtem_bench/cases/`. Setup goes in the function body; `run()` is the only timed region and returns the output objects (an `ArrayObject`, a list matching `outputs`, or a dict). Declare all three tiers with explicit `gpts`, `max_batch` and chunk sizes; never pass `sampling=` (the registry rejects it). Import abtem inside the function. Use `fixtures.multislice_kwargs(params)` to forward the propagator order and chunk size only where the ref accepts them.

## Checks

```
uvx ruff check benchmarks/abtem_bench benchmarks/tests abtem/core/testing.py
uvx mypy --follow-imports=silent --ignore-missing-imports benchmarks/abtem_bench
PYTHONPATH=benchmarks python -P -m pytest benchmarks/tests
```

CI runs the harness tests on Linux.
