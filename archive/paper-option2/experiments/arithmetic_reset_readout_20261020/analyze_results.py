#!/usr/bin/env python3
"""Validate completed experiment counts, then compute the two unchanged metrics.

No simulation, ZNE fit, bootstrap or confidence intervals. By default reject
an unequal-shot comparison with the saved160k Raw/ideal-Reset references.
Timing-only outputs are not accepted as scientific experiment results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from run_experiment import HERE, PAPER, canonical, sha, verify_chunk, load_configuration, atomic_json
from vendor.metrics import distribution_metrics

BASELINE = PAPER/"analysis_results/c1_all_zero_fixed_20261018/bootstrap_10000/equal_shot/n119_a16/rate_5em05/parent_estimates.npz"
BASELINE_SHA = "740b136904547321750b428208f73171bff223fb364af1b22985871c52b77031"


def row(label, percent, shots, counts_or_prob, empirical=True):
    p = np.asarray(counts_or_prob, dtype=np.float64)
    if empirical:
        assert np.all(p >= 0) and p.sum() == shots and np.all(p == np.floor(p))
        p = p/shots
    assert p.shape == (16384,) and np.isfinite(p).all() and abs(p.sum()-1) < 1e-12
    m = distribution_metrics(p, 14, 6)
    return {"route": label, "reset_error_percent": percent, "shots": shots,
            "tvd": m.tvd, "normalized_peak_background_gap": m.normalized_grouped_peak_margin,
            "weakest_group_probability": m.minimum_group_probability,
            "largest_individual_background": m.maximum_outside_group_probability,
            "probability_sum": float(p.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-unequal-shot-pilot", action="store_true")
    parser.add_argument("--baseline-parent", type=Path, default=BASELINE)
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    manifest = json.loads((folder/"manifest.json").read_text())
    assert manifest["status"] == "complete", "Incomplete run"
    conf = manifest["configuration"]
    assert hashlib.sha256(canonical(conf).encode()).hexdigest() == manifest["run_identity"]
    assert conf["experiment"] == load_configuration()
    assert len(manifest["conditions"]) == len(conf["reset_error_probabilities"])
    for name, digest in conf["source_sha256"].items():
        assert sha(HERE/name) == digest, "Run source differs from this implementation"
    shots = conf["shots_per_condition"]
    if shots != 160000 and not args.allow_unequal_shot_pilot:
        parser.error("Pilot has unequal shots vs saved160k baselines; use --allow-unequal-shot-pilot only for exploratory output")
    assert sha(args.baseline_parent) == BASELINE_SHA, "Wrong saved reference parent"
    with np.load(args.baseline_parent, allow_pickle=False) as f:
        results = [row("Raw_saved_reference", None, 160000, f["raw_full_original"], False),
                   row("Ideal_reset_saved_reference", 0., 160000, f["reset_reference"], False)]
    expected_chunk_paths = set()
    for _, condition in sorted(manifest["conditions"].items(), key=lambda item: item[1]["reset_error_probability"]):
        aggregate = folder/condition["aggregate"]
        assert aggregate.resolve().is_relative_to(folder)
        assert sha(aggregate) == condition["sha256"] and condition["shots"] == shots
        total = np.zeros(16384, dtype=np.int64)
        faults = np.zeros((14, 9), dtype=np.int64)
        pm = condition["reset_error_probability"]
        for offset in range(0, shots, conf["chunk_shots"]):
            size = min(conf["chunk_shots"], shots-offset)
            path = aggregate.parent/f"shots_{offset:09d}_{size:06d}.npz"
            key = str(path.relative_to(folder))
            assert key not in expected_chunk_paths
            expected_chunk_paths.add(key)
            assert sha(path) == manifest["chunks"][key]["sha256"]
            data = verify_chunk(path, identity=manifest["run_identity"], pm=pm, offset=offset,
                                size=size, config=conf["experiment"])
            total += data["counts"]; faults += data["reset_fault_counts_by_round_qubit"]
        with np.load(aggregate, allow_pickle=False) as f:
            assert str(f["run_identity"]) == manifest["run_identity"] and int(f["shots"]) == shots
            assert float(f["reset_error_probability"]) == pm
            np.testing.assert_array_equal(f["counts"], total)
            np.testing.assert_array_equal(f["reset_fault_counts_by_round_qubit"], faults)
        results.append(row("Reset_with_assignment_error", condition["reset_error_percent"], shots, total))
    assert set(manifest["chunks"]) == expected_chunk_paths
    csv_path = folder/"point_metrics.csv"
    if csv_path.exists() or (folder/"point_metrics_metadata.json").exists():
        raise FileExistsError("Metrics already exist; preserve earlier output")
    with csv_path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    metadata = {"status": "verified_point_estimates", "run_identity": manifest["run_identity"],
                "baseline_sha256": BASELINE_SHA, "input_manifest_sha256": sha(folder/"manifest.json"),
                "metrics_csv_sha256": sha(csv_path), "equal_shot_comparison": shots == 160000,
                "confidence_intervals": None, "new_shots_generated_by_analysis": 0,
                "warning": None if shots == 160000 else "Exploratory pilot only: finite-shot TVD/gap comparisons use unequal shot budgets."}
    atomic_json(folder/"point_metrics_metadata.json", metadata)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
