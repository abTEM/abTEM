#!/bin/bash
# GPU test runner for abTEM.
#
# GitHub CI has no GPU runner, so the GPU code paths (CuPy kernels, their
# NVRTC/hipRTC compilation, cupyx submodules) are validated by running this
# script on a machine with a GPU. It performs the two invocations that define
# "GPU-validated":
#
#   1. the Laplacian-stencil reference tests, which force-compile the raw GPU
#      kernels and compare them against the scipy reference
#   2. the full GPU-parameterized suite (excluding multi-GPU tests)
#
# On ROCm the second invocation runs one pytest process per test file; on a
# gfx1151 GPU it also skips the GPU ChargeDensityPotential tests (see
# ABTEM_CI_ROCM_CHARGE_DENSITY).
#
# Modes
# -----
# in-place (default):
#     Tests the checkout this script lives in, using the active Python
#     environment (or ABTEM_CI_VENV). Installs nothing, never touches git
#     state. For developers: run it before merging changes to GPU code.
#
# managed (ABTEM_CI_ROOT set):
#     Maintains its own clone and venv under ABTEM_CI_ROOT and hard-tracks
#     origin/ABTEM_CI_BRANCH; refreshes dependencies on every run. For
#     unattended use (cron / scrontab); self-heals if the clone or venv is
#     deleted (e.g. by a scratch purge).
#
# Environment knobs
# -----------------
#   ABTEM_CI_ROOT      enable managed mode; working dir for clone/venv/logs
#   ABTEM_CI_BRANCH    branch to test in managed mode (default: dev)
#   ABTEM_CI_VENV      existing venv to activate; in managed mode this skips
#                      venv creation and all installs and runs the clone via
#                      PYTHONPATH (an editable install in that venv is never
#                      re-pointed)
#   ABTEM_CI_CUPY_PKG  cupy package for the managed venv (default: cupy-cuda12x;
#                      irrelevant when ABTEM_CI_VENV is set)
#   ABTEM_CI_MODULES   space-separated environment modules to load, e.g.
#                      "cudatoolkit/12.9" on NERSC Perlmutter (default: none)
#   ABTEM_CI_MAILTO    address to email on failure; requires a working
#                      mail/mailx/sendmail on the node (default: no mail)
#   ABTEM_CI_MULTIGPU  set to 1 to also run the multigpu-marked tests
#                      (default: off). NVIDIA/CUDA only: abTEM's multi-GPU path
#                      is built on dask-cuda, which has no ROCm equivalent, so
#                      this flag is not usable on a ROCm machine. This is a
#                      contract: the run FAILS when pytest fails, when no
#                      multigpu test passes, or when a test of
#                      test/test_multigpu.py is skipped (see multigpu_gate). A
#                      preflight checks that dask_cuda imports and that >= 2
#                      GPUs are visible; when it fails, the multi-GPU tests are
#                      not run and the run fails after the single-GPU sweep.
#                      Managed mode installs dask-cuda and cuda-bindings into
#                      its venv; with ABTEM_CI_VENV or in-place mode the
#                      environment must already provide them
#   ABTEM_CI_REPO_URL  clone URL for managed mode (default: anonymous HTTPS —
#                      the runner only reads, and cluster compute nodes often
#                      cannot use the SSH key that works on login nodes; set a
#                      git@ URL for a private fork)
#   ABTEM_CI_OFFLINE   set to 1 in managed mode to test the clone under
#                      ABTEM_CI_ROOT as it is: no clone, fetch or reset, and the
#                      branch tested is the one checked out there. For compute
#                      nodes without internet access; the clone must already
#                      exist, and uv installs from its cache only (UV_OFFLINE=1
#                      unless set), so the install fails unless the cache holds
#                      every package or ABTEM_CI_VENV is set. Ignored in
#                      in-place mode
#   ABTEM_CI_ROCM_CHARGE_DENSITY
#                      set to 1 to keep the GPU ChargeDensityPotential tests on a
#                      gfx1151 GPU or one of unknown architecture (default:
#                      skipped there; see the sweep below)
#
# Exit status is non-zero when either test invocation fails. Each run appends
# one line to status.tsv in the log directory:
#   date <tab> mode/branch <tab> commit <tab> PASS|FAIL <tab> summary <tab> log

set -uo pipefail

BRANCH="${ABTEM_CI_BRANCH:-dev}"
STAMP="$(date -u +%Y-%m-%d_%H%M)"
export OMP_NUM_THREADS=1

if [ -n "${ABTEM_CI_ROOT:-}" ]; then
    MODE="managed/${BRANCH}"
    CI_ROOT="${ABTEM_CI_ROOT}"
    REPO="${CI_ROOT}/abTEM"
    LOGS="${CI_ROOT}/logs"
else
    MODE="in-place"
    # resolve symlinks so REPO is the repo the script physically lives in, not
    # wherever a convenience symlink to it sits
    _self="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")"
    REPO="$(cd "$(dirname "${_self}")/../.." && pwd)"
    LOGS="${REPO}/.gpu-test-logs"
fi
STATUS="${LOGS}/status.tsv"
mkdir -p "${LOGS}"

# Managed mode maintains one clone and venv under CI_ROOT, so two runs sharing
# it must not overlap: the second's `git reset --hard` / reinstall would race
# the first's in-flight checkout. If a previous run is still going (queued or
# hung past the next trigger), skip this one rather than corrupt it. In-place
# mode shares no such state, so it is not locked.
if [ "${MODE}" != "in-place" ]; then
    exec 200>"${CI_ROOT}/.run.lock"
    if ! flock -n 200; then
        echo "another run holds ${CI_ROOT}/.run.lock; skipping this trigger" >&2
        exit 0
    fi
fi

LOG="${LOGS}/${STAMP}.log"
exec > >(tee "${LOG}") 2>&1

record() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${STAMP}" "${MODE}" "${SHA:-unknown}" "$1" "$2" "${LOG}" >> "${STATUS}"
}

send_failure_mail() {
    [ -n "${ABTEM_CI_MAILTO:-}" ] || return 0
    local reason="$1"
    local report="${LOGS}/${STAMP}.report.txt"
    {
        echo "abTEM GPU tests FAILED on $(hostname) (job ${SLURM_JOB_ID:-none})"
        echo "date:    ${STAMP} (UTC)"
        echo "mode:    ${MODE} @ ${SHA:-unknown}"
        echo "reason:  ${reason}"
        echo "log:     ${LOG}"
        echo
        echo "==== pytest short test summary (if any) ===="
        sed -n '/short test summary info/,$p' "${LOG}" | head -60
        echo
        echo "==== last 150 log lines ===="
        tail -150 "${LOG}"
    } > "${report}"
    local subject="[abtem-gpu-tests] FAIL ${MODE}@${SHA:-unknown} ${STAMP}"
    if command -v mail >/dev/null 2>&1; then
        mail -s "${subject}" "${ABTEM_CI_MAILTO}" < "${report}"
    elif command -v mailx >/dev/null 2>&1; then
        mailx -s "${subject}" "${ABTEM_CI_MAILTO}" < "${report}"
    elif command -v sendmail >/dev/null 2>&1; then
        { echo "To: ${ABTEM_CI_MAILTO}"; echo "Subject: ${subject}"; echo; cat "${report}"; } \
            | sendmail -t
    else
        echo "WARNING: ABTEM_CI_MAILTO set but no mailer found; report at ${report}" >&2
    fi
}

fail() { record FAIL "$1"; send_failure_mail "$1"; echo "FAILED: $1"; exit 1; }

for m in ${ABTEM_CI_MODULES:-}; do
    module load "$m" || fail "module load $m failed"
done

# --- repo --------------------------------------------------------------------
if [ "${MODE}" != "in-place" ] && [ "${ABTEM_CI_OFFLINE:-0}" = 1 ]; then
    [ -d "${REPO}/.git" ] || fail "ABTEM_CI_OFFLINE=1 but ${REPO} is not a git checkout"
    cd "${REPO}" || fail "cd ${REPO} failed"
    export UV_OFFLINE="${UV_OFFLINE:-1}"
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    MODE="managed/${BRANCH}"
elif [ "${MODE}" != "in-place" ]; then
    if [ ! -d "${REPO}/.git" ]; then
        git clone --branch "${BRANCH}" \
            "${ABTEM_CI_REPO_URL:-https://github.com/abTEM/abTEM.git}" "${REPO}" \
            || fail "git clone failed"
    fi
    cd "${REPO}"
    git fetch origin "${BRANCH}" || fail "git fetch failed"
    git checkout -q "${BRANCH}" && git reset --hard -q "origin/${BRANCH}" \
        || fail "git checkout failed"
else
    cd "${REPO}"
fi
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "== abTEM ${MODE} @ ${SHA} on $(hostname) =="

# --- python environment ------------------------------------------------------
if [ -n "${ABTEM_CI_VENV:-}" ]; then
    [ -x "${ABTEM_CI_VENV}/bin/python" ] || fail "ABTEM_CI_VENV has no python"
    source "${ABTEM_CI_VENV}/bin/activate"
    export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
elif [ "${MODE}" != "in-place" ]; then
    VENV="${CI_ROOT}/venv"
    if [ ! -x "${VENV}/bin/python" ]; then
        uv venv "${VENV}" || fail "uv venv failed"
    fi
    source "${VENV}/bin/activate"
    # One resolve pass over the union of all constraints, so a conflict fails
    # loudly here instead of being silently decided by whichever of two
    # installs ran last. Multi-GPU adds dask-cuda and cuda-bindings (cuda-core
    # does `from cuda import bindings`; dask-cuda does not declare it). The
    # 12.x line matches the default cupy-cuda12x. The preflight below turns any
    # remaining dask_cuda import failure into a red run.
    EXTRA_PKGS=()
    [ "${ABTEM_CI_MULTIGPU:-}" = 1 ] && EXTRA_PKGS=(dask-cuda "cuda-bindings>=12,<13")
    uv pip install -e . --group test "${ABTEM_CI_CUPY_PKG:-cupy-cuda12x}" \
        ${EXTRA_PKGS[@]+"${EXTRA_PKGS[@]}"} \
        || fail "dependency install failed"
fi
# in-place without ABTEM_CI_VENV: use whatever python is active, but make sure
# this checkout wins over any installed abtem
[ "${MODE}" = "in-place" ] && export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'EOF' || fail "sanity import failed"
import abtem, cupy
props = cupy.cuda.runtime.getDeviceProperties(0)
print("abtem:", abtem.__file__)
print("cupy:", cupy.__version__, "device:", props["name"].decode())
EOF

# A failed multi-GPU preflight fails the run at the end, after the single-GPU
# sweep has run.
MULTI_RC=0
MULTI_NOTE=""
preflight_failed() {
    echo "ABTEM_CI_MULTIGPU needs dask-cuda and >= 2 GPUs"
    MULTI_RC=1
    MULTI_NOTE="preflight failed: needs dask-cuda and >= 2 GPUs"
}
if [ "${ABTEM_CI_MULTIGPU:-}" = 1 ]; then
    python - <<'EOF' || preflight_failed
import cupy, dask_cuda
n = cupy.cuda.runtime.getDeviceCount()
print("dask-cuda:", dask_cuda.__version__, "GPUs:", n)
assert n >= 2
EOF
fi

# --- the test phases ---------------------------------------------------------
# Each phase's pytest output is tee'd to its own tally file so status.tsv can
# record a per-phase summary. Grepping the whole log for the last "N passed"
# line instead would report only the small multi-GPU tally when
# ABTEM_CI_MULTIGPU=1, hiding the main sweep.
_tally() { grep -E '[0-9]+ (passed|failed|skipped|error)' "$1" 2>/dev/null | tail -1; }

STENCIL_TALLY="${LOGS}/${STAMP}.stencil-tally"
SWEEP_TALLY="${LOGS}/${STAMP}.sweep-tally"
MULTI_TALLY="${LOGS}/${STAMP}.multigpu-tally"

echo "== stencil reference tests =="
python -m pytest test/test_realspace_multislice.py -q -p no:cacheprovider \
    -k "StencilNumericalAccuracy or rejects" | tee "${STENCIL_TALLY}"
STENCIL_RC=${PIPESTATUS[0]}

echo "== full GPU sweep =="
SWEEP_K="gpu"
SWEEP_NOTE=""
if python -c "import sys, cupy.cuda.runtime as r; sys.exit(0 if r.is_hip else 1)"; then
    # On a gfx1151 (Strix Halo) integrated GPU, building a repeated cell as a
    # ChargeDensityPotential on the GPU has hung the device until the driver
    # resets it (MODE2 reset), which aborts the pytest process, so the rest of
    # the sweep never runs, and can black out a display driven by the same GPU.
    # Every test that builds one is skipped there (by test name or by
    # parametrisation id, hence both spellings). An architecture that cannot be
    # read counts as gfx1151, so a hang costs a skip rather than the sweep.
    # ABTEM_CI_ROCM_CHARGE_DENSITY=1 keeps the tests.
    ROCM_ARCH="$(python -c "
import cupy.cuda.runtime as r
try:
    a = r.getDeviceProperties(0)['gcnArchName']
    print(a.decode() if isinstance(a, bytes) else a)
except Exception:
    print('unknown')
" 2>/dev/null)"
    ROCM_ARCH="${ROCM_ARCH:-unknown}"
    SKIP_NOTE=""
    case "${ROCM_ARCH}" in
        gfx1151* | unknown)
            if [ "${ABTEM_CI_ROCM_CHARGE_DENSITY:-0}" != 1 ]; then
                SWEEP_K="gpu and not (charge_density or chargedensity)"
                SKIP_NOTE="; ChargeDensityPotential tests skipped"
                echo "ROCm (arch ${ROCM_ARCH}): skipping the GPU ChargeDensityPotential tests"
            fi ;;
    esac
    # One process per test file: a GPU abort (exit status >= 128, a signal)
    # then costs only its own file, and the other files still report their
    # failures (-rfE lists failed and errored tests).
    SWEEP_RC=0
    ABORTED=""
    for test_file in test/test_*.py; do
        echo "== ${test_file} =="
        python -m pytest "${test_file}" -q -rfE -p no:cacheprovider \
            -k "${SWEEP_K}" -m "not multigpu" | tee -a "${SWEEP_TALLY}"
        rc=${PIPESTATUS[0]}
        [ "${rc}" -eq 5 ] && continue  # no test selected in this file
        [ "${rc}" -ne 0 ] && SWEEP_RC=${rc}
        [ "${rc}" -ge 128 ] && ABORTED="${ABORTED} ${test_file}"
    done
    SWEEP_NOTE="$(grep -cE '^(FAILED|ERROR) test/' "${SWEEP_TALLY}") failed or errored tests${SKIP_NOTE}"
    if [ -n "${ABORTED}" ]; then
        echo "ABORTED (killed by a signal) in:${ABORTED}"
        SWEEP_NOTE="aborted in:${ABORTED}; ${SWEEP_NOTE}"
    fi
else
    python -m pytest test/ -q -p no:cacheprovider -k "${SWEEP_K}" -m "not multigpu" \
        | tee "${SWEEP_TALLY}"
    SWEEP_RC=${PIPESTATUS[0]}
fi

# Gate for the multi-GPU run, from pytest's exit status and its output (-rs).
# Exit status 0 also covers a run where requires_multigpu skipped everything
# (no dask-cuda, too few GPUs), and the multigpu test of test/test_measure.py
# needs only 2 GPUs, so a pass anywhere does not show that dask-cuda works: with
# a broken dask-cuda import the run ends with one pass and six skips. Require a
# pass and no skip in test/test_multigpu.py. Skips in other modules are
# collection-time (a module that imports gpaw) and say nothing about the
# multigpu tests. Prints the reason; returns pytest's status, or 1 (also for
# status 5, nothing collected).
multigpu_gate() {
    local rc="$1" out="$2"
    { [ "${rc}" -eq 0 ] || [ "${rc}" -eq 5 ]; } || return "${rc}"
    if ! grep -Eq '[0-9]+ passed' "${out}"; then
        echo "ERROR: no multigpu test passed (all skipped or none collected)"
        return 1
    fi
    if grep -Eq '^SKIPPED .*test/test_multigpu\.py' "${out}"; then
        echo "ERROR: a test of test/test_multigpu.py was skipped (check the" \
             "GPU count and that 'import dask_cuda' works in this environment)"
        return 1
    fi
}

if [ "${ABTEM_CI_MULTIGPU:-}" = 1 ] && [ "${MULTI_RC}" -eq 0 ]; then
    echo "== multi-GPU tests =="
    python -m pytest test/ -q -rfs -p no:cacheprovider -m multigpu \
        | tee "${MULTI_TALLY}"
    MULTI_RC=${PIPESTATUS[0]}
    multigpu_gate "${MULTI_RC}" "${MULTI_TALLY}"
    MULTI_RC=$?
fi

SWEEP_SUMMARY="${SWEEP_NOTE:-$(_tally "${SWEEP_TALLY}")}"
SUMMARY="stencil[$(_tally "${STENCIL_TALLY}")] sweep[${SWEEP_SUMMARY}]"
[ "${ABTEM_CI_MULTIGPU:-}" = 1 ] \
    && SUMMARY="${SUMMARY} multigpu[${MULTI_NOTE:-$(_tally "${MULTI_TALLY}")}]"
rm -f "${STENCIL_TALLY}" "${SWEEP_TALLY}" "${MULTI_TALLY}"
RCS="stencil_rc=${STENCIL_RC} sweep_rc=${SWEEP_RC} multigpu_rc=${MULTI_RC}"
if [ "${STENCIL_RC}" -eq 0 ] && [ "${SWEEP_RC}" -eq 0 ] && [ "${MULTI_RC}" -eq 0 ]; then
    record PASS "${SUMMARY}"
    echo "== all green =="
else
    record FAIL "${RCS}; ${SUMMARY}"
    send_failure_mail "${RCS}; ${SUMMARY}"
    echo "== FAILURES — see ${LOG} =="
    exit 1
fi
