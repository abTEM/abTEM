# GPU testing

GitHub CI has no GPU runner, so the GPU code paths — CuPy raw kernels and their runtime compilation (NVRTC on CUDA, hipRTC on ROCm), `cupyx` submodules, device-specific numerics — are exercised only on real hardware. Two bugs shipped in 2026-09 exactly this way: a kernel whose compiler options NVRTC rejected (invisible on the ROCm box it was developed on, [#376](https://github.com/abTEM/abTEM/pull/376) follow-up) and Thrust-backed sorts that segfault on some HIP builds. This directory holds the runner that would have caught both.

## What "GPU-validated" means

Two invocations, both green:

```bash
export OMP_NUM_THREADS=1
python -m pytest test/test_realspace_multislice.py -q -k "StencilNumericalAccuracy or rejects"
python -m pytest test/ -q -k gpu -m "not multigpu"
```

The first force-compiles the raw GPU kernels and compares them against the scipy reference — it fails fast on compiler/toolchain problems. The second runs every GPU-parameterized test (the `--runslow`-gated ones stay skipped). Passing on one platform does not imply the other: CUDA/NVRTC and ROCm/hipRTC have caught disjoint bugs, so changes to GPU code should be validated on whichever platforms are available, ideally both.

## Running

`run_gpu_tests.sh` wraps both invocations with logging, a `status.tsv` history, and optional failure email.

**Developer, testing the current checkout** (installs nothing, never touches git state; uses the active environment):

```bash
tools/gpu/run_gpu_tests.sh
```

On ROCm the second invocation differs; see [ROCm](#rocm) below. The test dependencies come from the `test` dependency group (`uv pip install -e . --group test` or `pip install --group test .` with pip ≥ 25.1) plus a cupy matching your platform (`cupy-cuda12x`, or a ROCm build).

**Unattended / cron** (`ABTEM_CI_ROOT` enables managed mode: the script maintains its own clone and venv there, hard-tracking `origin/dev`, and self-heals if a scratch purge deletes them):

```bash
ABTEM_CI_ROOT=$SCRATCH/abtem-gpu-ci \
ABTEM_CI_MODULES="cudatoolkit/12.9" \
ABTEM_CI_MAILTO=you@example.org \
tools/gpu/run_gpu_tests.sh
```

Example: weekly on NERSC Perlmutter via `scrontab -e` (times are local; `-q shared --gpus 1 -c 32` charges ~¼ node):

```text
#SCRON -A <account>
#SCRON -q shared
#SCRON -C gpu
#SCRON --gpus 1
#SCRON -c 32
#SCRON -t 00:40:00
#SCRON -J abtem-gpu-tests
#SCRON --mail-type=FAIL,TIMEOUT
#SCRON --mail-user=<you@example.org>
0 6 * * 1 ABTEM_CI_ROOT=$SCRATCH/abtem-gpu-ci ABTEM_CI_MODULES=cudatoolkit/12.9 ABTEM_CI_MAILTO=<you@example.org> $SCRATCH/abtem-gpu-ci/run_gpu_tests.sh
```

(Copy the script itself somewhere stable first — in managed mode it clones the repo, so it cannot run from inside the clone it manages. Managed mode clones anonymously over HTTPS by default, since compute nodes often cannot use the SSH key that works on login nodes; set `ABTEM_CI_REPO_URL` for a private fork. On Perlmutter, note that `/global/common` is read-only from compute nodes: keep `ABTEM_CI_ROOT` on `$SCRATCH`.)

**Reusing an existing venv** (e.g. a dev machine where cupy is a custom ROCm build): `ABTEM_CI_VENV=<venv>` activates it and installs nothing; the checkout under test is put on `PYTHONPATH`, so an editable abtem install in that venv is never re-pointed.

**Compute nodes without internet:** `ABTEM_CI_OFFLINE=1` (managed mode) tests the clone under `ABTEM_CI_ROOT` as it is: no clone, fetch or reset, and the branch tested is the one checked out there (the run fails if the clone does not exist). Update the clone from a node with internet access first. The dependency install runs with `UV_OFFLINE=1` unless the caller set it, so uv installs from its cache only and fails when the cache lacks a package; `ABTEM_CI_VENV` pointing at an existing environment skips all installs. In-place mode never touches git state, so the knob changes nothing there.

**Multi-GPU:** `ABTEM_CI_MULTIGPU=1` adds invocations running the `multigpu`-marked tests. The knob takes `1`, `0` or empty; any other value (`yes`, `true`, `2`) fails the run at the start. **NVIDIA/CUDA only** — abTEM's multi-GPU path is built on `dask-cuda`, which has no ROCm equivalent, so this flag does not apply on a ROCm machine. Setting it is a contract. A preflight first checks that `dask_cuda` imports and that at least 2 GPUs are visible; when it fails, the multi-GPU tests are not run and the run fails after the single-GPU sweep, with the reason in the `multigpu[...]` part of the `status.tsv` summary. The tests run as two pytest calls. `test/test_multigpu.py` runs alone and **fails** the run when pytest fails, when no test passes, or when its tally reports any skipped test, wherever the skip was raised. The remaining `multigpu` tests (the one in `test/test_measure.py`, which needs only 2 GPUs) run in a second call that must return 0; skips at collection time, such as the `gpaw` import skip in `test_gpaw_atom_solver_output.py`, do not fail it. `status.tsv` shows both tallies. The run needs a ≥ 2-GPU allocation (e.g. `--gpus 2` in the scrontab resources). Managed mode installs `dask-cuda` and `cuda-bindings>=12,<13` from PyPI into its venv when `ABTEM_CI_MULTIGPU=1` (`dask-cuda` does not declare `cuda-bindings`) and supports only the default `ABTEM_CI_CUPY_PKG` (`cupy-cuda12x`), because the `cuda-bindings` pin is the 12.x line; `ABTEM_CI_VENV` / in-place mode must already provide them. When scheduling a multi-GPU job alongside a single-GPU one, give it its own `ABTEM_CI_ROOT`; runs that do share a root are serialized by a lock (a run that starts while another is still holding the same `ABTEM_CI_ROOT` skips that trigger rather than corrupting the shared clone).

**Deselecting tests:** `ABTEM_CI_DESELECT` is a pytest `-k` expression of tests to leave out of the full sweep, on every backend (`ABTEM_CI_DESELECT="charge_density"`): a machine whose GPU hangs on some tests deselects them here. The `status.tsv` summary ends with `; deselected: <expression>`, so a deselection is never silent.

**Provenance:** the commit column of `status.tsv` gets a `-dirty` suffix when tracked files differ from `HEAD`.

### ROCm

On a ROCm GPU the script changes the second invocation:

- **One pytest process per test file**, with `-rfE` (failed and errored tests are listed) and abort detection: a file whose process is killed by a signal (exit status ≥ 128, e.g. a GPU abort) is reported as aborted, costs only that file, and the other files still run and report their failures. The sweep fails when no test passed in any file. `status.tsv` records the numbers of passed and of failed or errored tests, and the aborted files, instead of one pytest tally.

All knobs are documented in the header of `run_gpu_tests.sh`.

## Toward CI integration

The eventual proper solution is a GitHub Actions self-hosted runner on a GPU machine, wired into PR checks. Until such a machine is designated, this script is the bridge: scheduled runs on maintainers' hardware (a weekly Perlmutter A100 run and a weekly local ROCm run exist as of 2026-09), plus manual runs before merging GPU-touching changes.
