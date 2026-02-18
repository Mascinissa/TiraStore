#!/usr/bin/env python3
"""TiraStore multi-node concurrency test — worker script.

Each worker (launched via a separate srun) performs:

1. PHASE 1 — Unique writes: write records that only this worker owns.
2. PHASE 2 — Contested writes: all workers attempt to write the SAME records.
   Only the first writer should succeed (overwrite=False).
3. PHASE 3 — Cross-read: read back own records AND records from other workers.
4. PHASE 4 — Overwrite test: one designated worker overwrites a contested record.

Results are written to a per-worker JSON log file for the verifier to inspect.

Usage:
    python test_worker.py --db /path/to/test.db \
                          --worker-id 3 \
                          --num-workers 8 \
                          --records-per-worker 50 \
                          --output-dir /path/to/logs
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def make_program_source(worker_id: int, record_idx: int) -> str:
    """Generate a deterministic fake Tiramisu source for a unique record."""
    return f"void prog_w{worker_id}_r{record_idx}() {{ /* worker {worker_id} record {record_idx} */ }}"


def make_schedule(worker_id: int, record_idx: int) -> str:
    return f"L0_w{worker_id},L1_r{record_idx}"


def make_contested_source(record_idx: int) -> str:
    """All workers use the same source for contested records."""
    return f"void contested_prog_r{record_idx}() {{ /* shared */ }}"


def make_contested_schedule(record_idx: int) -> str:
    return f"L0_shared,L1_r{record_idx}"


def main():
    parser = argparse.ArgumentParser(description="TiraStore multi-node test worker")
    parser.add_argument("--db", required=True, help="Path to shared TiraStore DB")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--records-per-worker", type=int, default=50)
    parser.add_argument("--contested-records", type=int, default=20)
    parser.add_argument("--output-dir", required=True, help="Directory for log files")
    args = parser.parse_args()

    # Import here so path issues surface clearly
    from tirastore import TiraStore

    log = {
        "worker_id": args.worker_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", "N/A"),
        "phases": {},
        "errors": [],
        "start_time": time.time(),
    }

    try:
        store = TiraStore(
            args.db,
            source_project=f"multinode_test_worker_{args.worker_id}",
            cpu_model="test_cpu",
            slurm_cpus="1",
        )

        # ---- PHASE 1: Unique writes ----
        phase1 = {"writes": 0, "already_existed": 0, "errors": []}
        for i in range(args.records_per_worker):
            name = f"unique_w{args.worker_id}_r{i}"
            src = make_program_source(args.worker_id, i)
            sched = make_schedule(args.worker_id, i)
            times = [0.01 * (i + 1) + args.worker_id * 0.001]
            try:
                wrote = store.record(name, src, sched, is_legal=True, execution_times=times)
                if wrote:
                    phase1["writes"] += 1
                else:
                    phase1["already_existed"] += 1
            except Exception as e:
                phase1["errors"].append(f"record {i}: {e}")
        log["phases"]["unique_writes"] = phase1

        # ---- PHASE 2: Contested writes (all workers write same records) ----
        phase2 = {"writes": 0, "already_existed": 0, "errors": []}
        for i in range(args.contested_records):
            name = f"contested_r{i}"
            src = make_contested_source(i)
            sched = make_contested_schedule(i)
            times = [0.1 + args.worker_id * 0.0001]
            try:
                wrote = store.record(name, src, sched, is_legal=True, execution_times=times)
                if wrote:
                    phase2["writes"] += 1
                else:
                    phase2["already_existed"] += 1
            except Exception as e:
                phase2["errors"].append(f"contested {i}: {e}")
        log["phases"]["contested_writes"] = phase2

        # Small sleep to let other workers finish writing
        time.sleep(2)

        # ---- PHASE 3: Cross-reads ----
        phase3 = {"own_found": 0, "own_missing": 0, "contested_found": 0,
                   "contested_missing": 0, "other_found": 0, "other_missing": 0,
                   "errors": []}

        # Read own records
        for i in range(args.records_per_worker):
            name = f"unique_w{args.worker_id}_r{i}"
            src = make_program_source(args.worker_id, i)
            sched = make_schedule(args.worker_id, i)
            try:
                result = store.lookup(name, src, sched)
                if result is not None:
                    phase3["own_found"] += 1
                else:
                    phase3["own_missing"] += 1
            except Exception as e:
                phase3["errors"].append(f"own read {i}: {e}")

        # Read contested records
        for i in range(args.contested_records):
            name = f"contested_r{i}"
            src = make_contested_source(i)
            sched = make_contested_schedule(i)
            try:
                result = store.lookup(name, src, sched)
                if result is not None:
                    phase3["contested_found"] += 1
                else:
                    phase3["contested_missing"] += 1
            except Exception as e:
                phase3["errors"].append(f"contested read {i}: {e}")

        # Read a sample of records from other workers
        for other_w in range(args.num_workers):
            if other_w == args.worker_id:
                continue
            # Only check first 5 from each other worker to keep it fast
            for i in range(min(5, args.records_per_worker)):
                name = f"unique_w{other_w}_r{i}"
                src = make_program_source(other_w, i)
                sched = make_schedule(other_w, i)
                try:
                    result = store.lookup(name, src, sched)
                    if result is not None:
                        phase3["other_found"] += 1
                    else:
                        phase3["other_missing"] += 1
                except Exception as e:
                    phase3["errors"].append(f"other read w{other_w}_r{i}: {e}")

        log["phases"]["cross_reads"] = phase3

        # ---- PHASE 4: Overwrite test (only worker 0) ----
        phase4 = {"overwrite_attempted": False, "overwrite_succeeded": False, "errors": []}
        if args.worker_id == 0 and args.contested_records > 0:
            phase4["overwrite_attempted"] = True
            name = "contested_r0"
            src = make_contested_source(0)
            sched = make_contested_schedule(0)
            try:
                wrote = store.record(
                    name, src, sched,
                    is_legal=True,
                    execution_times=[9.999],
                    overwrite=True,
                )
                phase4["overwrite_succeeded"] = wrote
            except Exception as e:
                phase4["errors"].append(f"overwrite: {e}")
        log["phases"]["overwrite_test"] = phase4

        # Final stats
        log["final_count"] = store.count()

    except Exception as e:
        log["errors"].append(traceback.format_exc())

    log["end_time"] = time.time()
    log["duration_s"] = log["end_time"] - log["start_time"]

    # Write log
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"worker_{args.worker_id}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[Worker {args.worker_id}] Done on {log['hostname']} in {log['duration_s']:.1f}s — "
          f"unique_writes={phase1['writes']}, contested_writes={phase2['writes']}")


if __name__ == "__main__":
    main()
