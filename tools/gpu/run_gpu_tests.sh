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
#   ABTEM_CI_MULTIGPU  set non-empty to also run the multigpu-marked tests;
#                      they need >= 2 visible GPUs and dask-cuda and skip
#                      themselves otherwise (default: off)
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
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    LOGS="${REPO}/.gpu-test-logs"
fi
STATUS="${LOGS}/status.tsv"
mkdir -p "${LOGS}"
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
if [ "${MODE}" != "in-place" ]; then
    if [ ! -d "${REPO}/.git" ]; then
        git clone --branch "${BRANCH}" git@github.com:abTEM/abTEM.git "${REPO}" \
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
    uv pip install -e . --group test "${ABTEM_CI_CUPY_PKG:-cupy-cuda12x}" \
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

# --- the two runs ------------------------------------------------------------
echo "== stencil reference tests =="
python -m pytest test/test_realspace_multislice.py -q -p no:cacheprovider \
    -k "StencilNumericalAccuracy or rejects"
STENCIL_RC=$?

echo "== full GPU sweep =="
python -m pytest test/ -q -p no:cacheprovider -k gpu -m "not multigpu"
SWEEP_RC=$?

MULTI_RC=0
if [ -n "${ABTEM_CI_MULTIGPU:-}" ]; then
    # the multigpu-marked tests skip themselves unless >= 2 GPUs and dask-cuda
    # are present, so this invocation is safe on any machine; exit code 5
    # (nothing collected) is treated as success
    echo "== multi-GPU tests =="
    python -m pytest test/ -q -p no:cacheprovider -m multigpu
    MULTI_RC=$?
    [ "${MULTI_RC}" -eq 5 ] && MULTI_RC=0
fi

SUMMARY="$(grep -E '[0-9]+ (passed|failed)' "${LOG}" | tail -1)"
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
