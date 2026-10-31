#!/usr/bin/env python3
"""Summarize only completed C1 follow-up artifacts; never fit or resample."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_results/c1_followup_20261018/bootstrap_10000"
CASES = ("n119_a13", "n119_a50", "n119_a16")
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
PRIMARY = "full_lambda_minus2_ridge_1e-8"
GATE = "equal_gate_full_lambda_minus2_ridge_1e-8"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def classification(low, high, tolerance=1e-10):
    return "positive" if low > tolerance else "negative" if high < -tolerance else "unresolved"


def main():
    summaries, contrasts, manifests = [], [], []
    runner_sha = sha(ROOT / "analysis/run_c1_followup_20261018.py")
    for case in CASES:
        directory = OUT / case
        manifest = json.loads((directory / "manifest.json").read_text())
        start = json.loads((directory / "started_manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["final_10000_contract"]
        assert manifest["configuration"]["replicates"] == 10000
        assert manifest["identity"] == start["identity"]
        recorded = next(source["sha256"] for source in start["provenance"]["sources"]
                        if Path(source["path"]).name == "run_c1_followup_20261018.py")
        assert recorded == runner_sha, "Completed source differs from current frozen runner"
        for name in ("points.csv", "summary.csv", "paired_sensitivity_contrasts.csv"):
            assert sha(directory / name) == manifest["artifacts"][name]
        with (directory / "summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 24 and all(row["variant"] == "C1" and int(row["replicates"]) == 10000 for row in rows)
        summaries.extend(rows)
        with (directory / "paired_sensitivity_contrasts.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 32 and all(int(row["replicates"]) == 10000 for row in rows)
        contrasts.extend(rows)
        manifests.append({"instance": case, "identity": manifest["identity"],
                          "manifest_sha256": sha(directory / "manifest.json"),
                          "rate": manifest["configuration"]["rate"],
                          "replicates": manifest["configuration"]["replicates"]})

    write_csv(OUT / "summary_all_cases.csv", summaries)
    write_csv(OUT / "paired_sensitivity_all_cases.csv", contrasts)
    budget_rows = []
    for case in CASES:
        for setting in (PRIMARY, GATE):
            for method in METHODS:
                row = next(r for r in summaries if (r["instance"], r["setting"], r["method"]) == (case, setting, method))
                item = {k: row[k] for k in ("instance", "setting", "method", "per_scale_shots", "zne_circuit_shots")}
                for metric, quantity in (("tvd", "reset_minus_zne_tvd"), ("gap", "zne_minus_reset_normalized_grouped_peak_margin")):
                    low, high = float(row[quantity + "_ci_low"]), float(row[quantity + "_ci_high"])
                    item[metric + "_improvement_point"] = float(row["point_" + quantity])
                    item[metric + "_improvement_ci_low"] = low
                    item[metric + "_improvement_ci_high"] = high
                    item[metric + "_interval_sign"] = classification(low, high)
                budget_rows.append(item)
    write_csv(OUT / "budget_improvements_all_cases.csv", budget_rows)
    report = {"status": "complete", "scope": "three cases at p2=0.01 percent only",
              "cases": manifests, "runner_sha256": runner_sha,
              "setting_model_summary_rows": len(summaries), "paired_sensitivity_rows": len(contrasts),
              "budget_comparison_rows": len(budget_rows), "new_quantum_shots": 0,
              "sign_classification_tolerance": 1e-10,
              "sign_classification_note": "Numerical display tolerance only; unresolved is not equivalence. Pointwise intervals are not simultaneous claims.",
              "cross_budget_paired_inference": False,
              "budget_stream_note": "Budget-specific RNG objects retain the same historical seed hierarchy; statistical independence between budgets is not claimed."}
    (OUT / "aggregate_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    for row in budget_rows:
        print(row["instance"], "gate" if row["setting"] == GATE else "equal-shot", row["method"],
              "TVD", row["tvd_interval_sign"], "gap", row["gap_interval_sign"])
    print("Special sensitivity, order-six dual, uniform weights:")
    for row in contrasts:
        if row["instance"] == "n119_a16" and row["method"] == "dual_exponential_zne" and row["setting"] == "full_uniform_ridge_1e-8":
            print(row["metric"], row["point"], row["ci_low"], row["ci_high"])


if __name__ == "__main__":
    main()
