#!/usr/bin/env python3
"""Independently verify S6 CSVs, masks, provenance, and all 222 saved-parent points.

Does not import the exporter or any fitting/simulation/bootstrap implementation.
The decoder is rebuilt by enumerating every reduced rational with denominator
1..119 and selecting the nearest neighbor using exact integer comparisons.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from functools import cmp_to_key
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
M, N = 16384, 119
CONFIG_ID = "c1_beta2_all_ridges_zero_20261018"
CONFIG_SHA = "f549613646e2e6d89807da92322878b540b49a805288ec55b5bcf0254b28d31f"
CASES = {"n119_a13": (13, 4), "n119_a50": (50, 2), "n119_a16": (16, 6)}
ZNE = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
ROUTES = ("raw_160k", "reset_160k", *ZNE)
COLUMNS = ("instance", "a", "order", "rate", "rate_percent", "route", "p_exact", "success_percent", "tvd",
           "normalized_gap", "nonzero_support", "singleton_y", "singleton_flag", "source_parent_sha256")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rational_truth_table() -> tuple[np.ndarray, int]:
    """Exhaustive rational enumeration; no continued-fraction implementation."""
    candidates = [(p, q) for q in range(1, N + 1) for p in range(q + 1) if math.gcd(p, q) == 1]

    def compare(left: tuple[int, int], right: tuple[int, int]) -> int:
        value = left[0] * right[1] - right[0] * left[1]
        return (value > 0) - (value < 0)

    candidates.sort(key=cmp_to_key(compare))
    assert len(candidates) == len(set(candidates))
    assert candidates[0] == (0, 1) and candidates[-1] == (1, 1)
    q_table = np.zeros(M, dtype=np.int16)
    upper = 1
    for y in range(M):
        while candidates[upper][0] * M < y * candidates[upper][1]:
            upper += 1
        p_left, q_left = candidates[upper - 1]
        p_right, q_right = candidates[upper]
        left_error = abs(p_left * M - y * q_left)
        right_error = abs(p_right * M - y * q_right)
        left_comparison, right_comparison = left_error * q_right, right_error * q_left
        if left_comparison < right_comparison:
            q_table[y] = q_left
        elif left_comparison > right_comparison:
            q_table[y] = q_right
        else:
            # At equal denominators the left rational is smaller.
            q_table[y] = min(q_left, q_right)
    return q_table, len(candidates)


def independent_ideal(order: int) -> np.ndarray:
    """Exact-QPE closed form using small integer phase residues.

    This differs from the exporter's sinc implementation: each eigenphase
    contributes sin^2(pi M delta)/(M^2 sin^2(pi delta))/r.  Integer residues avoid
    large-argument trigonometric cancellation.  Exact phase-grid coincidences
    are set to their analytic limiting value 1/r.
    """
    y = np.arange(M, dtype=np.int64)
    result = np.zeros(M)
    for phase in range(order):
        delta_numerator = M * phase - order * y
        on_grid = delta_numerator == 0
        if (M * phase) % order == 0:
            result[on_grid] += 1 / order
            continue
        residual = (delta_numerator + order * M // 2) % (order * M) - order * M // 2
        numerator = math.sin(math.pi * ((M * phase) % order) / order) ** 2
        denominator = np.sin(math.pi * residual / (order * M)) ** 2
        result += numerator / (order * M * M * denominator)
    assert abs(float(result.sum()) - 1) < 5e-13
    return result / result.sum()


def earlier_convergent_mask(a: int) -> np.ndarray:
    accepted = np.zeros(M, dtype=bool)
    for y in range(M):
        n, d = y, M
        terms = []
        while d:
            quotient, remainder = divmod(n, d)
            terms.append(quotient)
            n, d = d, remainder
        last2, last1 = 1, 0
        for term in terms:
            current = term * last1 + last2
            last2, last1 = last1, current
            if 1 < current < N and pow(a, current, N) == 1:
                accepted[y] = True
                break
    return accepted


def independent_metrics(vector: np.ndarray, ideal: np.ndarray, order: int) -> tuple[float, float]:
    # Explicit integer floor/ceiling construction, independent of exporter helpers.
    bins = []
    for phase in range(order):
        lower = M * phase // order
        upper = (M * phase + order - 1) // order
        bins.append((lower,) if lower == upper else (lower, upper))
    group_union = set(y for group in bins for y in group)
    outside = np.asarray([y for y in range(M) if y not in group_union])
    observed_margin = min(math.fsum(float(vector[y]) for y in group) for group in bins) - float(max(vector[outside]))
    ideal_margin = min(math.fsum(float(ideal[y]) for y in group) for group in bins) - float(max(ideal[outside]))
    return math.fsum(abs(float(a) - float(b)) for a, b in zip(vector, ideal)) / 2, observed_margin / ideal_margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--write-report", type=Path, help="Write a new verification JSON; existing files are never overwritten.")
    args = parser.parse_args()
    data = args.data_dir
    provenance_path = data / "provenance.json"
    provenance = load_json(provenance_path)
    assert provenance["configuration_id"] == CONFIG_ID and provenance["configuration_sha256"] == CONFIG_SHA
    assert provenance["equal_shot_conditions"] == 37 and provenance["route_points"] == 222
    assert provenance["routes"] == list(ROUTES) and provenance["excluded_gate_matched_conditions"] == 3
    assert all(provenance[key] == 0 for key in ("new_quantum_shots", "new_qpu_jobs", "new_gpu_simulations", "new_zne_fits", "new_bootstrap_replicates"))
    assert provenance["confidence_intervals_computed"] is False and provenance["hyperparameters_retuned"] is False
    assert provenance["decoder"]["candidate_count_per_outcome"] == 1
    assert provenance["decoder"]["additional_convergents_or_multiples_searched"] is False
    assert provenance["decoder"]["divisor_reduction"] is False
    assert digest(HERE / "export_order_data.py") == provenance["export_script_sha256"]
    for name, expected in provenance["input_sha256"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        assert digest(path) == expected, f"Input checksum mismatch: {path}"
    for name, expected in provenance["output_sha256"].items():
        assert digest(data / name) == expected, f"Output checksum mismatch: {name}"
    config_path = ROOT / "analysis/c1_all_zero_fixed_20261018.json"
    assert digest(config_path) == CONFIG_SHA
    config = load_json(config_path)
    assert config["configuration_id"] == CONFIG_ID and config["estimator"] == "C1" and config["weight_exponent"] == 2
    assert set(config["ridges"]) == set(ZNE) and not any(config["ridges"].values())
    assert config["dimension"] == M and config["direct_shots"] == 160000
    aggregate_path = ROOT / provenance["inventory_source"]
    assert digest(aggregate_path) == provenance["inventory_source_sha256"]
    aggregate = load_json(aggregate_path)
    assert aggregate["configuration_id"] == CONFIG_ID and aggregate["approved_config_sha256"] == CONFIG_SHA
    assert aggregate["status"] == "complete" and aggregate["completed_equal_shot_jobs"] == 37
    inventory = [item for item in aggregate["inventory"] if item["budget"] == "equal_shot"]
    origins = {item["identity"]: item for item in aggregate["verified_job_origins"]}
    parents = {(x["instance"], x["rate"]): x for x in provenance["parents"]}
    assert len(inventory) == len(parents) == 37
    assert {k: sum(item["instance"] == k for item in inventory) for k in CASES} == {
        "n119_a13": 11, "n119_a50": 11, "n119_a16": 15}
    rows = load_csv(data / "success_all_rates.csv")
    assert len(rows) == 222 and tuple(rows[0]) == COLUMNS
    lookup = {(row["instance"], float(row["rate"]), row["route"]): row for row in rows}
    assert len(lookup) == 222
    assert set(lookup) == {(item["instance"], item["rate"], route) for item in inventory for route in ROUTES}
    references = {row["instance"]: row for row in load_csv(data / "references.csv")}
    assert set(references) == set(CASES)
    q, candidate_count = rational_truth_table()
    assert q[0] == 1 and q[M // 2] == 2 and q[M // 4] == 4 and q[M - 1] == 1
    assert q[2731] == q[2735] == 6
    masks = {}
    ideal_max_error = 0.0
    with np.load(data / "decoder_masks.npz", allow_pickle=False) as z:
        assert np.array_equal(z["q_single_candidate"], q)
        assert np.array_equal(z["outcome_y"], np.arange(M))
        for instance, (a, order) in CASES.items():
            assert pow(a, order, N) == 1 and all(pow(a, k, N) != 1 for k in range(1, order))
            exact = q == order
            modular = np.asarray([pow(a, int(candidate), N) == 1 for candidate in q])
            legacy = earlier_convergent_mask(a)
            independent_target = independent_ideal(order)
            assert np.array_equal(z[instance + "__exact"], exact)
            assert np.array_equal(z[instance + "__modular_accept"], modular)
            assert np.array_equal(z[instance + "__legacy_diagnostic"], legacy)
            error = float(np.max(np.abs(z[instance + "__ideal"] - independent_target)))
            ideal_max_error = max(ideal_max_error, error)
            assert error < 2e-12
            masks[instance] = exact
            ref = references[instance]
            assert (int(ref["N"]), int(ref["a"]), int(ref["order"]), int(ref["bits"]), int(ref["total_outcomes"])) == (119, a, order, 14, M)
            assert int(ref["strict_exact_uniform_count"]) == int(exact.sum()) == {2: 69, 4: 70, 6: 48}[order]
            assert int(ref["modular_accept_uniform_count"]) == int(modular.sum())
            assert int(ref["legacy_uniform_count"]) == int(legacy.sum())
            for key, value in (("p_exact_uniform", exact.mean()), ("success_percent_uniform", exact.mean() * 100),
                               ("p_modular_accept_uniform", modular.mean()), ("p_legacy_uniform", legacy.mean()),
                               ("p_exact_ideal", independent_target[exact].sum()),
                               ("success_percent_ideal", independent_target[exact].sum() * 100)):
                assert abs(float(ref[key]) - value) < 2e-11, (instance, key, ref[key], value)
    locked_path = aggregate_path.parent / "absolute_locked_metrics.csv"
    assert digest(locked_path) == aggregate["output_csv_sha256"][locked_path.name]
    locked = {(row["budget"], row["instance"], float(row["rate"]), row["route"], row["metric"]): float(row["point"])
              for row in load_csv(locked_path)}
    assert len(locked) == 474
    max_probability_error, max_tvd_error, max_gap_error, max_dot_error = 0.0, 0.0, 0.0, 0.0
    accepted_singletons = {}
    scalar_count = 0
    for item in inventory:
        instance, rate = item["instance"], item["rate"]
        a, order = CASES[instance]
        parent = parents[(instance, rate)]
        directory = ROOT / item["jobdir"]
        assert item["status"] == "complete_and_verified" and parent["identity"] == item["identity"]
        assert ROOT / parent["parent_path"] == directory / "parent_estimates.npz"
        manifest_path = directory / "manifest.json"
        assert digest(manifest_path) == parent["manifest_sha256"] == origins[item["identity"]]["manifest_sha256"]
        manifest = load_json(manifest_path)
        cfg = manifest["configuration"]
        payload = json.dumps({"configuration": cfg, "provenance": manifest["provenance"]}, indent=2, sort_keys=True, allow_nan=False) + "\n"
        assert hashlib.sha256(payload.encode()).hexdigest() == item["identity"]
        assert manifest["status"] == "complete" and manifest["identity"] == item["identity"]
        assert cfg["configuration_id"] == CONFIG_ID and cfg["weight_exponent"] == 2 and cfg["ridges"] == config["ridges"]
        assert cfg["budget"] == "equal_shot" and cfg["instance"] == instance and cfg["order"] == order and cfg["rate"] == rate
        assert cfg["routes"] == list(ZNE) and cfg["fit_lambdas"] == config["effective_scales"]
        point_path = directory / "point_manifest.json"
        assert digest(point_path) == parent["point_manifest_sha256"]
        point = load_json(point_path)
        assert point["identity"] == item["identity"] and point["status"] == "points_complete"
        parent_path = directory / "parent_estimates.npz"
        assert digest(parent_path) == parent["parent_sha256"] == manifest["artifacts"][parent_path.name] == point["artifacts"][parent_path.name]
        with np.load(parent_path, allow_pickle=False) as z:
            assert str(z["identity"]) == item["identity"] and json.loads(str(z["configuration_json"])) == cfg
            ideal = z["target"]
            ideal_error = float(np.max(np.abs(ideal - independent_ideal(order))))
            ideal_max_error = max(ideal_max_error, ideal_error)
            assert ideal_error < 2e-12
            for route in ROUTES:
                row = lookup[(instance, rate, route)]
                assert int(row["a"]) == a and int(row["order"]) == order
                assert float(row["rate_percent"]) == rate * 100
                assert row["source_parent_sha256"] == parent["parent_sha256"]
                vector_name = "raw_full_original" if route == "raw_160k" else "reset_reference" if route == "reset_160k" else "after__" + route
                vector = z[vector_name]
                assert vector.shape == (M,) and np.isfinite(vector).all() and min(vector) >= 0
                assert abs(math.fsum(float(value) for value in vector) - 1) < 2e-12
                probability = math.fsum(float(vector[y]) for y in range(M) if q[y] == order)
                error = abs(probability - float(row["p_exact"]))
                max_probability_error = max(max_probability_error, error)
                assert error < 2e-14
                assert abs(float(row["success_percent"]) - probability * 100) < 2e-12
                dot_error = abs(probability - float(np.dot(vector, masks[instance].astype(float))))
                max_dot_error = max(max_dot_error, dot_error)
                assert dot_error < 2e-14
                tvd, gap = independent_metrics(vector, ideal, order)
                tvd_error, gap_error = abs(tvd - float(row["tvd"])), abs(gap - float(row["normalized_gap"]))
                max_tvd_error, max_gap_error = max(max_tvd_error, tvd_error), max(max_gap_error, gap_error)
                assert tvd_error < 5e-12 and gap_error < 5e-12
                assert abs(tvd - locked[("equal_shot", instance, rate, route, "tvd")]) < 5e-12
                assert abs(gap - locked[("equal_shot", instance, rate, route, "normalized_grouped_peak_margin")]) < 5e-12
                scalar_count += 2
                support = [y for y in range(M) if vector[y] > 0]
                assert int(row["nonzero_support"]) == len(support)
                assert row["singleton_flag"] == str(len(support) == 1)
                assert row["singleton_y"] == (str(support[0]) if len(support) == 1 else "")
                if len(support) == 1 and probability == 1:
                    key = instance, rate, route
                    accepted_singletons[key] = (row, support[0], float(ideal[support[0]]))
    assert scalar_count == 444
    expected_singletons = {("n119_a13", .0003, "dual_exponential_zne"): 4096,
                           ("n119_a16", .00014, "dual_exponential_zne"): 2731,
                           ("n119_a16", .00016, "dual_exponential_zne"): 2731,
                           ("n119_a16", .00018, "dual_exponential_zne"): 2735}
    assert {key: value[1] for key, value in accepted_singletons.items()} == expected_singletons
    singleton_rows = load_csv(data / "singleton_cases.csv")
    assert len(singleton_rows) == 4
    for row in singleton_rows:
        key = row["instance"], float(row["rate"]), row["route"]
        source, y, target_at_y = accepted_singletons[key]
        assert {name: row[name] for name in COLUMNS} == source
        assert int(row["decoded_q"]) == int(q[y]) == int(row["order"])
        assert float(row["ideal_probability_at_y"]) == target_at_y
    report = {"status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "verification_script_sha256": digest(Path(__file__)), "provenance_sha256": digest(provenance_path),
              "independent_from_exporter": True, "source_hashes_checked": len(provenance["input_sha256"]),
              "output_hashes_checked": len(provenance["output_sha256"]), "parents_verified": 37,
              "route_points_verified": len(rows), "references_verified": 3, "accepted_singletons_verified": 4,
              "decoder_outcomes_verified": M, "enumerated_reduced_rational_candidates": candidate_count,
              "decoder_verification": "All reduced rational numbers in [0,1] with denominators 1..119; exact nearest-neighbor comparisons and denominator tie break",
              "independent_ideal_formula": "Mixed eigenphase sin-squared finite-QPE expression with exact integer phase residues",
              "independent_ideal_max_bin_residual": ideal_max_error,
              "p_exact_max_absolute_residual": max_probability_error, "sum_vs_dot_max_absolute_residual": max_dot_error,
              "tvd_max_absolute_residual": max_tvd_error, "normalized_gap_max_absolute_residual": max_gap_error,
              "locked_metric_scalars_verified": scalar_count, "strict_uniform_counts": {"r2": 69, "r4": 70, "r6": 48},
              "new_quantum_shots": 0, "new_gpu_simulations": 0, "new_zne_fits": 0, "new_bootstrap_replicates": 0,
              "old_success_csv_read": False}
    if args.write_report:
        with args.write_report.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
