#!/usr/bin/env bash
#
# TiraStore Multi-Node Concurrency Test — Launcher
#
# Launches N workers via separate srun commands (each srun = separate
# Slurm job on a different node) and then verifies the results.
#
# Usage:
#   bash launch_test.sh [NUM_WORKERS] [RECORDS_PER_WORKER] [CONTESTED_RECORDS]
#
# Example:
#   bash launch_test.sh 8 50 20
#
# Prerequisites:
#   - tirastore must be installed (pip install -e .)
#   - The shared directory must be on Lustre (accessible from all nodes)
#   - Adjust PARTITION and any srun flags for your cluster

set -euo pipefail

# ---- Configuration ----
NUM_WORKERS="${1:-8}"
RECORDS_PER_WORKER="${2:-50}"
CONTESTED_RECORDS="${3:-20}"

# Where to put the test DB and logs — CHANGE THIS to a Lustre path
TEST_DIR="${TIRASTORE_TEST_DIR:-$(pwd)/tirastore_multinode_test_$(date +%Y%m%d_%H%M%S)}"

# Slurm partition — CHANGE THIS to match your cluster
PARTITION="${SLURM_PARTITION:-batch}"

# Path to this repo (for locating scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/test_worker.py"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_results.py"

DB_PATH="${TEST_DIR}/test_store.db"
LOG_DIR="${TEST_DIR}/logs"
SRUN_LOG_DIR="${TEST_DIR}/srun_logs"

echo "============================================================"
echo "  TiraStore Multi-Node Concurrency Test"
echo "============================================================"
echo "  Workers:             ${NUM_WORKERS}"
echo "  Records/worker:      ${RECORDS_PER_WORKER}"
echo "  Contested records:   ${CONTESTED_RECORDS}"
echo "  Test directory:      ${TEST_DIR}"
echo "  Partition:           ${PARTITION}"
echo "============================================================"
echo ""

# ---- Setup ----
mkdir -p "${LOG_DIR}" "${SRUN_LOG_DIR}"

# ---- Launch workers ----
echo "Launching ${NUM_WORKERS} workers via separate srun commands..."
echo ""

PIDS=()
for (( w=0; w<NUM_WORKERS; w++ )); do
    srun_log="${SRUN_LOG_DIR}/worker_${w}.out"
    echo "  Starting worker ${w} ..."

    srun --partition="${PARTITION}" \
         --ntasks=1 \
         --cpus-per-task=1 \
         --exclusive \
         --job-name="tirastore_test_w${w}" \
         python "${WORKER_SCRIPT}" \
             --db "${DB_PATH}" \
             --worker-id "${w}" \
             --num-workers "${NUM_WORKERS}" \
             --records-per-worker "${RECORDS_PER_WORKER}" \
             --contested-records "${CONTESTED_RECORDS}" \
             --output-dir "${LOG_DIR}" \
         > "${srun_log}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All ${NUM_WORKERS} srun commands launched. Waiting for completion..."
echo "  (PIDs: ${PIDS[*]})"
echo ""

# ---- Wait for all workers ----
FAILED=0
for (( w=0; w<NUM_WORKERS; w++ )); do
    pid=${PIDS[$w]}
    if wait "${pid}"; then
        echo "  Worker ${w} (PID ${pid}) — finished OK"
    else
        echo "  Worker ${w} (PID ${pid}) — FAILED (exit code $?)"
        echo "    See: ${SRUN_LOG_DIR}/worker_${w}.out"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "${FAILED}" -gt 0 ]; then
    echo "WARNING: ${FAILED} worker(s) failed. Check srun logs in:"
    echo "  ${SRUN_LOG_DIR}/"
    echo ""
fi

# ---- Verify ----
echo "Running verification..."
echo ""

python "${VERIFY_SCRIPT}" \
    --db "${DB_PATH}" \
    --output-dir "${LOG_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --records-per-worker "${RECORDS_PER_WORKER}" \
    --contested-records "${CONTESTED_RECORDS}"
