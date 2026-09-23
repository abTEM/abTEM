# abtem-bench: regression benchmarks for abTEM

Runs a fixed matrix of user-facing abTEM workloads against a checkout and records, per case, the result arrays, wall time and peak memory, in a format that does not depend on the abtem version. Two bundles are then compared: results (bit-identical, within tolerance, drift, shape change), speed and memory. Design and milestones: [abtem-benchmarks/design](https://github.com/abTEM/abtem-benchmarks/tree/main/design); discussion: [#380](https://github.com/abTEM/abTEM/discussions/380).

## Invariants

- The harness (`abtem_bench/`) and the cases (`cases/`) always come from the checkout you invoke; only the `abtem` package is swapped per ref, via a git worktree under `.worktrees/bench/<sha>` placed on `PYTHONPATH` behind the harness directory. Both refs run byte-identical case code, and every bundle records a hash of the case sources.
- Each case id runs in a fresh subprocess (`python -P -m abtem_bench.worker`). The worker prints and records which abtem it imported and aborts if it is not the requested worktree.
- Peak host memory is `ru_maxrss` of that subprocess from `os.wait4`. VRAM is sampled from the CuPy pool and the driver. No meter runs code on the computation.
- A case whose API is missing on a ref records `UNSUPPORTED` instead of failing the run.

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

`uv pip install -e benchmarks` (in a throwaway environment) installs the `abtem-bench` console script instead. The `-P` flag matters when running from a checkout: without it the current directory shadows `PYTHONPATH`.

Case ids are `name[variant]@tier/device`; `--only` takes globs on ids or names. Tiers are `quick` (seconds per case, the CI tier), `standard` (minutes) and `large` (GPU stress sizes).

## Presets

`accuracy` pins `precision: float64`, `fft: fftw` with `FFTW_ESTIMATE`, one FFTW and BLAS thread, the synchronous dask scheduler, explicit chunk sizes, fixed dask chunk sizes, `grid.round-to-fast-fft: false`, a fixed cuFFT cache size, and the thread environment, then dumps the fully resolved abTEM and dask configuration into the record. On CPU the expectation is bit-identical output across processes and days on the same machine; the self-check verifies that. `speed` keeps `FFTW_MEASURE` so the cold cost users pay is represented.

## Reading a report

One row per paired case id. `verdict` is the worst over the case's outputs: `IDENTICAL` (bit-for-bit), `OK` (within the case tolerance), `DRIFT` (beyond tolerance), `ACCEPTED` (drift listed in `accepted_changes.toml`), `SHAPE` (shape, dtype or axes changed), or the run status (`UNSUPPORTED`, `ERROR`, `OOM`, `TIMEOUT`). `rel` is the largest relative error over elements above the case's `above_rel` fraction of the reference maximum; `intensity` the relative change of the integrated intensity; `time`, `cold` and `rss` are candidate over reference ratios (warm median, first call, peak resident memory). Speed and memory flags fire beyond `max(threshold, 3 x noise floor)` from a self-check; `auto` variants are never flagged.

A `DRIFT` fails `--fail-on drift` unless an entry in `accepted_changes.toml` matches the case; add the entry in the same pull request as the change, with the PR number and a one-line reason. The report renders matched entries as the changelog and lists entries that no longer match anything.

## Adding a case

Decorate a function `(params, device) -> run` with `@case` in a module under `cases/`. Setup goes in the function body; `run()` is the only timed region and returns the output objects (an `ArrayObject`, a list matching `outputs`, or a dict). Declare all three tiers with explicit `gpts`, `max_batch` and chunk sizes; never pass `sampling=` (the registry rejects it). Import abtem inside the function. Use `fixtures.multislice_kwargs(params)` to forward the propagator order and chunk size only where the ref accepts them.

## Checks

```
uvx ruff check benchmarks/abtem_bench benchmarks/cases abtem/core/testing.py
uvx mypy --follow-imports=silent --ignore-missing-imports benchmarks/abtem_bench benchmarks/cases
pytest benchmarks/tests
```
