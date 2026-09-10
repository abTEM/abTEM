# GPU testing

GitHub CI has no GPU runner, so the GPU code paths — CuPy raw kernels and their runtime compilation (NVRTC on CUDA, hipRTC on ROCm), `cupyx` submodules, device-specific numerics — are exercised only on real hardware. Two bugs shipped in 2026-09 exactly this way: a kernel whose compiler options NVRTC rejected (invisible on the ROCm box it was developed on, [#376](https://github.com/abTEM/abTEM/pull/376) follow-up) and Thrust-backed sorts that segfault on some HIP builds. This directory holds the runner that would have caught both.

## What "GPU-validated" means

Two invocations, both green:

```bash
export OMP_NUM_THREADS=1
python -m pytest test/test_realspace_multislice.py -q -k "StencilNumericalAccuracy or rejects"
python -m pytest test/ -q -k gpu -m "not multigpu"
```

The first force-compiles the raw GPU kernels and compares them against the scipy reference — it fails fast on compiler/toolchain problems. The second runs every GPU-parameterized test (~540 as of 2026-09; the `--runslow`-gated ones stay skipped). Passing on one platform does not imply the other: CUDA/NVRTC and ROCm/hipRTC have caught disjoint bugs, so changes to GPU code should be validated on whichever platforms are available, ideally both.

## Running

`run_gpu_tests.sh` wraps both invocations with logging, a `status.tsv` history, and optional failure email.

**Developer, testing the current checkout** (installs nothing, never touches git state; uses the active environment):

```bash
tools/gpu/run_gpu_tests.sh
```

The test dependencies come from the `test` dependency group (`uv pip install -e . --group test` or `pip install --group test .` with pip ≥ 25.1) plus a cupy matching your platform (`cupy-cuda12x`, or a ROCm build).

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

(Copy the script itself somewhere stable first — in managed mode it clones the repo, so it cannot run from inside the clone it manages. On Perlmutter, note that `/global/common` is read-only from compute nodes: keep `ABTEM_CI_ROOT` on `$SCRATCH`.)

**Reusing an existing venv** (e.g. a dev machine where cupy is a custom ROCm build): `ABTEM_CI_VENV=<venv>` activates it and installs nothing; the checkout under test is put on `PYTHONPATH`, so an editable abtem install in that venv is never re-pointed.

**Multi-GPU:** `ABTEM_CI_MULTIGPU=1` adds a third invocation running the `multigpu`-marked tests. They require ≥ 2 visible GPUs plus `dask-cuda` (and skip themselves otherwise), so on a cluster this means requesting at least a 2-GPU allocation (e.g. `--gpus 2` instead of `--gpus 1` in the scrontab resources).

All knobs are documented in the header of `run_gpu_tests.sh`.

## Toward CI integration

The eventual proper solution is a GitHub Actions self-hosted runner on a GPU machine, wired into PR checks. Until such a machine is designated, this script is the bridge: scheduled runs on maintainers' hardware (a weekly Perlmutter A100 run and a weekly local ROCm run exist as of 2026-09), plus manual runs before merging GPU-touching changes.
