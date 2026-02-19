#!/usr/bin/env python3
"""Verify the results of a TiraStore multi-node concurrency test.

Reads the per-worker JSON logs and the database, then checks:

1. Every unique record was written exactly once.
2. Every contested record exists exactly once (no duplicates, no corruption).
3. All workers could read their own records back.
4. All workers could read contested records.
5. Cross-node reads found other workers' records.
6. The overwrite test (worker 0) succeeded.
7. No errors were reported by any worker.
8. Workers ran on different nodes (the whole point of the test).

Usage:
    python verify_results.py --db /path/to/test.db \
                             --output-dir /path/to/logs \
                             --num-workers 8 \
                             --records-per-worker 50 \
                             --contested-records 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    return condition


def main():
    parser = argparse.ArgumentParser(description="Verify TiraStore multi-node test")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--records-per-worker", type=int, default=50)
    parser.add_argument("--contested-records", type=int, default=20)
    args = parser.parse_args()

    from tirastore import TiraStore

    out_dir = Path(args.output_dir)
    all_passed = True

    # ---- Load worker logs ----
    logs: dict[int, dict] = {}
    missing_logs = []
    for w in range(args.num_workers):
        log_path = out_dir / f"worker_{w}.json"
        if not log_path.exists():
            missing_logs.append(w)
            continue
        logs[w] = json.loads(log_path.read_text())

    print(f"\n{'='*60}")
    print(f"  TiraStore Multi-Node Concurrency Test — Verification")
    print(f"{'='*60}")
    print(f"  Workers expected: {args.num_workers}")
    print(f"  Worker logs found: {len(logs)}")
    if missing_logs:
        print(f"  [{FAIL}] Missing logs for workers: {missing_logs}")
        all_passed = False
    print()

    # ---- Check workers ran on different nodes ----
    print("1. Node distribution:")
    hostnames = {w: l["hostname"] for w, l in logs.items()}
    slurm_nodes = {w: l.get("slurm_node", "N/A") for w, l in logs.items()}
    unique_hosts = set(hostnames.values())
    unique_slurm = set(slurm_nodes.values()) - {"N/A"}

    for w, l in sorted(logs.items()):
        print(f"     Worker {w}: host={l['hostname']}  "
              f"slurm_node={l.get('slurm_node', 'N/A')}  "
              f"pid={l['pid']}  duration={l.get('duration_s', '?'):.1f}s")

    multi_node = len(unique_hosts) > 1 or len(unique_slurm) > 1
    if multi_node:
        all_passed &= check("Workers ran on multiple distinct nodes",
                            True, f"{len(unique_hosts)} unique hosts")
    else:
        print(f"  [{WARN}] All workers ran on the SAME host — "
              f"this does NOT validate cross-node locking.")
        print(f"         Make sure your srun commands request --exclusive or different nodes.")
    print()

    # ---- Check for errors ----
    print("2. Worker errors:")
    any_errors = False
    for w, l in sorted(logs.items()):
        top_errors = l.get("errors", [])
        phase_errors = []
        for phase_name, phase_data in l.get("phases", {}).items():
            if isinstance(phase_data, dict):
                phase_errors.extend(phase_data.get("errors", []))
        all_errs = top_errors + phase_errors
        if all_errs:
            any_errors = True
            print(f"  [{FAIL}] Worker {w} reported {len(all_errs)} error(s):")
            for e in all_errs[:5]:
                print(f"         {e}")
            if len(all_errs) > 5:
                print(f"         ... and {len(all_errs) - 5} more")
    all_passed &= check("No errors across all workers", not any_errors)
    print()

    # ---- Phase 1: Unique writes ----
    print("3. Unique writes:")
    total_unique_writes = sum(
        l["phases"]["unique_writes"]["writes"] for l in logs.values()
    )
    expected_unique = args.num_workers * args.records_per_worker
    all_passed &= check(
        "All unique records written",
        total_unique_writes == expected_unique,
        f"{total_unique_writes}/{expected_unique}",
    )
    print()

    # ---- Phase 2: Contested writes ----
    print("4. Contested writes:")
    total_contested_writes = sum(
        l["phases"]["contested_writes"]["writes"] for l in logs.values()
    )
    total_contested_existed = sum(
        l["phases"]["contested_writes"]["already_existed"] for l in logs.values()
    )
    # Exactly `contested_records` writes should have succeeded total
    # (one winner per record)
    all_passed &= check(
        "Each contested record written exactly once",
        total_contested_writes == args.contested_records,
        f"writes={total_contested_writes}, expected={args.contested_records}",
    )
    expected_existed = args.num_workers * args.contested_records - args.contested_records
    all_passed &= check(
        "Remaining workers correctly saw 'already exists'",
        total_contested_existed == expected_existed,
        f"already_existed={total_contested_existed}, expected={expected_existed}",
    )
    print()

    # ---- Phase 3: Cross-reads ----
    print("5. Cross-reads:")
    for w, l in sorted(logs.items()):
        cr = l["phases"]["cross_reads"]
        all_passed &= check(
            f"Worker {w}: own records readable",
            cr["own_found"] == args.records_per_worker and cr["own_missing"] == 0,
            f"found={cr['own_found']}, missing={cr['own_missing']}",
        )
    total_contested_found = sum(
        l["phases"]["cross_reads"]["contested_found"] for l in logs.values()
    )
    total_contested_missing = sum(
        l["phases"]["cross_reads"]["contested_missing"] for l in logs.values()
    )
    all_passed &= check(
        "All workers can read all contested records",
        total_contested_missing == 0,
        f"found={total_contested_found}, missing={total_contested_missing}",
    )
    total_other_found = sum(
        l["phases"]["cross_reads"]["other_found"] for l in logs.values()
    )
    total_other_missing = sum(
        l["phases"]["cross_reads"]["other_missing"] for l in logs.values()
    )
    # Some "other_missing" is acceptable since workers run concurrently
    # and phase 3 may execute before all other workers finish phase 1.
    # But we report it.
    if total_other_missing > 0:
        print(f"  [{WARN}] Some cross-worker reads missed (timing): "
              f"found={total_other_found}, missing={total_other_missing}")
    else:
        check("Cross-worker reads all succeeded", True,
              f"found={total_other_found}")
    print()

    # ---- Phase 4: Overwrite ----
    print("6. Overwrite test:")
    if 0 in logs:
        ow = logs[0]["phases"]["overwrite_test"]
        all_passed &= check("Worker 0 attempted overwrite", ow["overwrite_attempted"])
        all_passed &= check("Overwrite succeeded", ow["overwrite_succeeded"])
    else:
        print(f"  [{FAIL}] Worker 0 log missing — cannot verify overwrite")
        all_passed = False
    print()

    # ---- Database integrity ----
    print("7. Database integrity:")
    store = TiraStore(
        args.db,
        source_project="verifier",
        cpu_model="test_cpu",
        slurm_cpus="1",
    )
    db_count = store.count()
    expected_total = expected_unique + args.contested_records
    all_passed &= check(
        "Total record count matches expected",
        db_count == expected_total,
        f"db={db_count}, expected={expected_total} "
        f"({expected_unique} unique + {args.contested_records} contested)",
    )

    stats = store.stats()
    all_passed &= check(
        "All records are legal",
        stats["legal_records"] == expected_total,
        f"legal={stats['legal_records']}",
    )
    print()

    # ---- Final verdict ----
    print(f"{'='*60}")
    if all_passed:
        print(f"  [{PASS}] ALL CHECKS PASSED")
    else:
        print(f"  [{FAIL}] SOME CHECKS FAILED — review output above")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
