#!/usr/bin/env python3
"""Explicitly launch the approved 3 x 20k refinement and reuse the 0% archive.

No GPU work without --launch/--worker. The detached worker keeps checkpointing
48-shot batches and verifies all four equal-budget conditions after completion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np

import run_experiment as runner
from analyze_results import row

HERE = runner.HERE
CAMPAIGN = HERE / "results/campaign_20k_20261020"
COMPANION = runner.DEFAULT_COMPANION.resolve()
ARCHIVE = COMPANION / "a100_followup_downloads/n119-a100-followup-20260726-v2/merged/a16_augmented_20k/shotwise/n119_a16/shotwise/lambda_1p0-1p0.npz"
ARCHIVE_SHA = "fc3cd6e2a4c6f2d767b960ffb3dc1b51744932a2374d79e90d95f5a6e7d452ee"
OLD_CONFIG = COMPANION / "configs/n119_a16_augmented.json"
OLD_CONFIG_SHA = "58ed5efaaadc0884c6630d6e29dcc7686a90cc71fcd970732ca466d58b910495"
PERCENTS = [.2, .3, .4]


def now():
    return datetime.now(timezone.utc).isoformat()


def reference():
    assert runner.sha(ARCHIVE) == ARCHIVE_SHA
    assert runner.sha(OLD_CONFIG) == OLD_CONFIG_SHA
    old = json.loads(OLD_CONFIG.read_text())
    assert old["workspace_reset"] is True and old["global_seed"] == 42
    assert old["shots_per_lambda"] == 20000
    config = runner.load_configuration()
    with np.load(ARCHIVE, allow_pickle=False) as z:
        for k, v in {"N": 119, "a": 16, "order": 6, "measurement_bits": 14,
                     "shots": 20000, "two_qubit_gate_count": 49686,
                     "folding_seed": 42}.items():
            assert int(z[k]) == v
        assert str(z["ops_cache_sha256"]) == config["operation_cache_sha256"]
        assert str(z["circuit_variant"]) == "gate_count_matched_common_opt1"
        assert str(z["folding_pattern_scope"]) == "per_rate_lambda_qpe_round_shot"
        indices = np.flatnonzero(z["rates"] == .00005)
        scales = np.flatnonzero(z["lambdas"] == 1.)
        assert len(indices) == len(scales) == 1
        i, j = int(indices[0]), int(scales[0])
        assert np.all(z["folded_gate_counts_by_round"][j] == 0)
        outcomes = z["outcomes"][i, j].copy()
        counts = z["counts"][i, j].astype(np.int64)
        assert outcomes.shape == (20000,) and outcomes.dtype == np.uint16
        assert counts.shape == (16384,) and counts.sum() == 20000
        np.testing.assert_array_equal(counts, np.bincount(outcomes, minlength=16384))
        info = {"source": str(ARCHIVE), "source_sha256": ARCHIVE_SHA,
                "source_config": str(OLD_CONFIG), "source_config_sha256": OLD_CONFIG_SHA,
                "source_rate_index": i, "source_lambda_index": j,
                "source_fold_plan_sha256": str(z["fold_plan_sha256"][i, j]),
                "N": 119, "a": 16, "order": 6, "measurement_bits": 14,
                "gate_error_probability": .00005, "lambda": 1.,
                "reset_error_probability": 0., "shots": 20000,
                "operation_cache_sha256": config["operation_cache_sha256"],
                "reused_existing_shots": True, "new_shots": 0,
                "note": "Exact saved ideal-arithmetic-reset 20k sample; not the 160k extension. No claim of bitwise pairing across old/new hardware."}
    return outcomes, counts, info


def engine_arguments():
    return ["--execute", "--shots-per-condition", "20000", "--chunk-shots", "48",
            "--reset-error-percent", *[str(x) for x in PERCENTS],
            "--output-dir", str(CAMPAIGN / "new_shots")]


def finalize():
    """Independent recount and hash check before reporting the four point metrics."""
    folder = CAMPAIGN / "new_shots"
    manifest = json.loads((folder / "manifest.json").read_text())
    conf = manifest["configuration"]
    assert manifest["status"] == "complete" and len(manifest["conditions"]) == 3
    assert conf["shots_per_condition"] == 20000 and conf["chunk_shots"] == 48
    assert conf["reset_error_percent"] == PERCENTS
    import hashlib
    assert hashlib.sha256(runner.canonical(conf).encode()).hexdigest() == manifest["run_identity"]
    for name, digest in conf["source_sha256"].items():
        assert runner.sha(HERE / name) == digest
    _, original, _ = reference()
    with np.load(CAMPAIGN / "reference/reset_0percent_20000.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(original, z["counts"])
    rows = [row("Ideal_reset_reused", 0., 20000, original)]
    paths = set()
    for _, condition in sorted(manifest["conditions"].items(), key=lambda t: t[1]["reset_error_probability"]):
        assert condition["shots"] == 20000 and condition["status"] == "complete"
        aggregate = folder / condition["aggregate"]
        assert aggregate.resolve().is_relative_to(folder)
        assert runner.sha(aggregate) == condition["sha256"]
        counts = np.zeros(16384, dtype=np.int64)
        faults = np.zeros((14, 9), dtype=np.int64)
        for offset in range(0, 20000, 48):
            size = min(48, 20000-offset)
            path = aggregate.parent / f"shots_{offset:09d}_{size:06d}.npz"
            key = str(path.relative_to(folder))
            paths.add(key)
            assert runner.sha(path) == manifest["chunks"][key]["sha256"]
            data = runner.verify_chunk(path, identity=manifest["run_identity"],
                pm=condition["reset_error_probability"], offset=offset, size=size, config=conf["experiment"])
            counts += data["counts"]
            faults += data["reset_fault_counts_by_round_qubit"]
        with np.load(aggregate, allow_pickle=False) as z:
            assert str(z["run_identity"]) == manifest["run_identity"]
            assert int(z["shots"]) == 20000
            assert float(z["reset_error_probability"]) == condition["reset_error_probability"]
            np.testing.assert_array_equal(counts, z["counts"])
            np.testing.assert_array_equal(faults, z["reset_fault_counts_by_round_qubit"])
        rows.append(row("Reset_with_assignment_error", condition["reset_error_percent"], 20000, counts))
    assert paths == set(manifest["chunks"]) and len(paths) == 1251
    assert len(rows) == 4 and sum(r["shots"] for r in rows) == 80000
    target = CAMPAIGN / "point_metrics.csv"
    with target.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    runner.atomic_json(CAMPAIGN / "verification.json", {
        "status": "passed", "verified_utc": now(), "equal_shots_per_condition": 20000,
        "conditions": 4, "new_trajectories": 60000, "reused_trajectories": 20000,
        "verified_new_chunks": len(paths), "run_manifest_sha256": runner.sha(folder / "manifest.json"),
        "metrics_csv_sha256": runner.sha(target), "baseline_source_sha256": ARCHIVE_SHA,
        "confidence_intervals": None, "bootstrap_performed": False, "zne_fitting_performed": False})
    print(json.dumps(rows, indent=2), flush=True)


def worker():
    started = time.monotonic()
    try:
        snapshot = json.loads((CAMPAIGN / "source_snapshot/sha256.json").read_text())
        for name, digest in snapshot.items():
            assert runner.sha(HERE / name) == digest
        runner.atomic_json(CAMPAIGN / "status.json", {"status": "running", "pid": os.getpid(), "started_utc": now()})
        rc = runner.main(engine_arguments())
        assert rc == 0
        finalize()
        runner.atomic_json(CAMPAIGN / "status.json", {"status": "complete", "pid": os.getpid(),
            "finished_utc": now(), "elapsed_seconds": time.monotonic()-started})
    except BaseException as exc:
        runner.atomic_json(CAMPAIGN / "status.json", {"status": "failed", "pid": os.getpid(),
            "failed_utc": now(), "error": repr(exc), "elapsed_seconds": time.monotonic()-started})
        traceback.print_exc()
        return 1
    return 0


def launch():
    outcomes, counts, info = reference()
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount() >= 1
    cp.cuda.Device(0).use()
    assert (cp.ones(1) + 1).get()[0] == 2
    runner.read_cache(COMPANION / runner.load_configuration()["operation_cache_relative"], runner.load_configuration())
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    (CAMPAIGN / "reference").mkdir()
    runner.atomic_npz(CAMPAIGN / "reference/reset_0percent_20000.npz", outcomes=outcomes,
        counts=counts, shots=np.int64(20000), reset_error_probability=np.float64(0),
        gate_error_probability=np.float64(.00005), noise_scale=np.float64(1),
        source_sha256=np.asarray(ARCHIVE_SHA))
    info["extracted_file_sha256"] = runner.sha(CAMPAIGN / "reference/reset_0percent_20000.npz")
    info["point_metrics"] = row("Ideal_reset_reused", 0., 20000, counts)
    runner.atomic_json(CAMPAIGN / "reference/provenance.json", info)
    snapshot = CAMPAIGN / "source_snapshot"
    snapshot.mkdir()
    signatures = {}
    for p in [HERE / "config.json", *sorted(HERE.glob("*.py")), *sorted((HERE / "vendor").glob("*.py"))]:
        relative = p.relative_to(HERE)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        signatures[str(relative)] = runner.sha(p)
    runner.atomic_json(snapshot / "sha256.json", signatures)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"]
    runner.atomic_json(CAMPAIGN / "campaign.json", {"created_utc": now(), "N": 119, "a": 16,
        "order": 6, "p2_probability": .00005, "p2_percent": .005, "lambda": 1,
        "reset_error_percent": [0., *PERCENTS], "shots_per_condition": 20000,
        "chunk_shots": 48, "new_trajectories": 60000, "reused_trajectories": 20000,
        "worker_command": command, "engine_arguments": engine_arguments(),
        "source_snapshot": "source_snapshot/sha256.json", "bootstrap": False, "zne_fits": False})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUPY_CACHE_DIR="/tmp/qst-reset-cupy-cache-20261020")
    with (CAMPAIGN / "run.log").open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=runner.PAPER, env=env, start_new_session=True)
    runner.atomic_json(CAMPAIGN / "process.json", {"pid": process.pid, "launched_utc": now(), "command": command})
    print(json.dumps({"pid": process.pid, "campaign": str(CAMPAIGN), "baseline": info["point_metrics"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--launch", action="store_true")
    group.add_argument("--worker", action="store_true")
    group.add_argument("--validate-reference", action="store_true")
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.worker:
        raise SystemExit(worker())
    else:
        _, counts, info = reference()
        print(json.dumps({"provenance": info, "point_metrics": row("Ideal_reset_reused", 0., 20000, counts)}, indent=2))
