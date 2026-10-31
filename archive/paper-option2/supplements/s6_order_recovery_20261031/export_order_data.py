#!/usr/bin/env python3
"""Export strict single-candidate order recovery from locked saved parents.

No simulator, estimator, fit, shot archive, or bootstrap implementation is
imported or run.  The sole result inventory is the 37 equal-shot conditions in
the completed all-zero-ridge aggregate.  Earlier decoder masses are not inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "analysis_results/c1_all_zero_fixed_20261018"
CONFIG_PATH = ROOT / "analysis/c1_all_zero_fixed_20261018.json"
CONFIG_ID = "c1_beta2_all_ridges_zero_20261018"
CONFIG_SHA = "f549613646e2e6d89807da92322878b540b49a805288ec55b5bcf0254b28d31f"
N, BITS, M = 119, 14, 16384
INSTANCES = {"n119_a13": (13, 4), "n119_a50": (50, 2), "n119_a16": (16, 6)}
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
ROUTES = ("raw_160k", "reset_160k", *METHODS)
COLUMNS = ("instance", "a", "order", "rate", "rate_percent", "route", "p_exact", "success_percent",
           "tvd", "normalized_gap", "nonzero_support", "singleton_y", "singleton_flag", "source_parent_sha256")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nearest_rational_denominators() -> np.ndarray:
    """Exhaustively enumerate d=1..119, without continued fractions.

    Distances are compared by integer cross multiplication.  Equal-distance
    candidates prefer the smaller reduced denominator, then the lower rational,
    matching Fraction.limit_denominator's specified tie rule.  Both neighboring
    numerators are enumerated for every denominator, including the endpoint 1.
    """
    y = np.arange(M, dtype=np.int64)
    best_error = y.copy()  # initial rational 0/1
    best_d = np.ones(M, dtype=np.int64)
    best_q = np.ones(M, dtype=np.int64)
    best_p = np.zeros(M, dtype=np.int64)
    for d in range(1, N + 1):
        lower = y * d // M
        for numerator in (lower, lower + 1):
            error = np.abs(numerator * M - y * d)
            divisor = np.gcd(numerator, d)
            q = d // divisor
            p = numerator // divisor
            left, right = error * best_d, best_error * d
            better_tie = (q < best_q) | ((q == best_q) & (p < best_p))
            update = (left < right) | ((left == right) & better_tie)
            best_error[update], best_d[update] = error[update], d
            best_q[update], best_p[update] = q[update], p[update]
    return best_q.astype(np.int16)


def ideal_distribution(order: int) -> np.ndarray:
    """Locked finite-register ideal QPE reference, not a fitted distribution."""
    if M % order == 0:
        result = np.zeros(M)
        result[np.arange(order) * (M // order)] = 1 / order
        return result
    delta = np.arange(order)[:, None] / order - np.arange(M)[None, :] / M
    amplitude = np.sinc(M * delta) / np.sinc(delta)
    result = np.maximum(np.mean(amplitude * amplitude, axis=0), 0)
    return result / result.sum()


def phase_groups(order: int) -> tuple[np.ndarray, ...]:
    result = []
    for s in range(order):
        y, remainder = divmod(M * s, order)
        result.append(np.asarray([y] if remainder == 0 else [y, y + 1]))
    return tuple(result)


def metrics(values: np.ndarray, target: np.ndarray, order: int) -> tuple[float, float]:
    groups = phase_groups(order)
    outside = np.ones(M, dtype=bool)
    outside[np.concatenate(groups)] = False
    gap = min(float(values[g].sum()) for g in groups) - float(values[outside].max())
    ideal_gap = min(float(target[g].sum()) for g in groups) - float(target[outside].max())
    return float(np.abs(values - target).sum() / 2), gap / ideal_gap


def legacy_mask(a: int) -> np.ndarray:
    """Diagnostic only: earlier any-verified-convergent decoder.

    For these three primitive true orders, every modularly verified exponent
    reduces to that order.  This local truth table is checked bit-for-bit against
    the checksum-validated earlier decoder mask before reporting its baseline.
    It is never used to calculate strict success in any parent distribution.
    """
    result = np.zeros(M, dtype=bool)
    for y in range(M):
        numerator, denominator = y, M
        q_prev, q = 1, 0
        while denominator:
            term, remainder = divmod(numerator, denominator)
            q_prev, q = q, term * q + q_prev
            if 1 < q < N and pow(a, q, N) == 1:
                result[y] = True
                break
            numerator, denominator = denominator, remainder
    return result


def calculate() -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    checked: dict[str, str] = {}

    def check(path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if key not in checked:
            checked[key] = sha(path)
        actual = checked[key]
        assert expected is None or actual == expected, f"SHA256 mismatch: {path}"
        return actual

    config = read_json(CONFIG_PATH)
    check(CONFIG_PATH, CONFIG_SHA)
    assert config["configuration_id"] == CONFIG_ID
    assert config["estimator"] == "C1" and config["weight_exponent"] == 2
    assert set(config["ridges"]) == set(METHODS) and all(v == 0 for v in config["ridges"].values())
    assert config["dimension"] == M and config["new_quantum_shots"] == 0
    aggregate_path = BASE / "aggregate/summary.json"
    aggregate = read_json(aggregate_path)
    check(aggregate_path)
    assert aggregate["status"] == "complete" and aggregate["completed_jobs"] == 40
    assert aggregate["configuration_id"] == CONFIG_ID and aggregate["approved_config_sha256"] == CONFIG_SHA
    assert aggregate["completed_equal_shot_jobs"] == 37 and aggregate["completed_gate_matched_jobs"] == 3
    inventory = [x for x in aggregate["inventory"] if x["budget"] == "equal_shot"]
    assert len(inventory) == 37 and len(aggregate["inventory"]) == 40
    assert len({(x["instance"], x["rate"]) for x in inventory}) == 37
    assert {k: sum(x["instance"] == k for x in inventory) for k in INSTANCES} == {
        "n119_a13": 11, "n119_a50": 11, "n119_a16": 15}
    locked_path = BASE / "aggregate/absolute_locked_metrics.csv"
    check(locked_path, aggregate["output_csv_sha256"][locked_path.name])
    locked = {(x["budget"], x["instance"], float(x["rate"]), x["route"], x["metric"]): float(x["point"])
              for x in read_csv(locked_path)}
    assert len(locked) == 474
    q = np.asarray([Fraction(y, M).limit_denominator(N).denominator for y in range(M)], dtype=np.int16)
    assert np.array_equal(q, nearest_rational_denominators())
    assert q[0] == 1 and q[M // 2] == 2 and q[M // 4] == 4
    assert q[2731] == q[2735] == 6 and q[M - 1] == 1
    legacy_dir = ROOT / "analysis_results/order_success_noridge_20261019"
    legacy_summary_path = legacy_dir / "summary.json"
    legacy_summary = read_json(legacy_summary_path)
    check(legacy_summary_path)
    legacy_masks_path = legacy_dir / "decoder_masks.npz"
    check(legacy_masks_path, legacy_summary["output_sha256"][legacy_masks_path.name])
    assert legacy_summary["configuration_id"] == CONFIG_ID
    arrays = {"q_single_candidate": q, "outcome_y": np.arange(M, dtype=np.int32)}
    references = []
    with np.load(legacy_masks_path, allow_pickle=False) as old:
        for instance, (a, order) in INSTANCES.items():
            assert pow(a, order, N) == 1 and all(pow(a, k, N) != 1 for k in range(1, order))
            exact = q == order
            modular = np.asarray([pow(a, int(candidate), N) == 1 for candidate in q])
            legacy = legacy_mask(a)
            assert np.array_equal(legacy, old[instance + "__order"])
            target = ideal_distribution(order)
            assert np.array_equal(target, old[instance + "__ideal"])
            arrays.update({instance + "__exact": exact, instance + "__modular_accept": modular,
                           instance + "__legacy_diagnostic": legacy, instance + "__ideal": target})
            expected_count = {2: 69, 4: 70, 6: 48}[order]
            assert int(exact.sum()) == expected_count
            ideal_mass = float(target[exact].sum())
            assert abs(ideal_mass - {2: .5, 4: .5, 6: .32911339185035715}[order]) < 5e-13
            references.append({"instance": instance, "N": N, "a": a, "order": order, "bits": BITS,
                               "total_outcomes": M, "strict_exact_uniform_count": int(exact.sum()),
                               "p_exact_uniform": float(exact.mean()), "success_percent_uniform": float(exact.mean() * 100),
                               "p_exact_ideal": ideal_mass, "success_percent_ideal": ideal_mass * 100,
                               "modular_accept_uniform_count": int(modular.sum()),
                               "p_modular_accept_uniform": float(modular.mean()),
                               "legacy_uniform_count": int(legacy.sum()), "p_legacy_uniform": float(legacy.mean())})
    rows, parents = [], []
    origins = {x["identity"]: x for x in aggregate["verified_job_origins"]}
    metric_max_error, normalization_max_error, dot_max_error = 0.0, 0.0, 0.0
    for item in inventory:
        assert item["status"] == "complete_and_verified"
        directory = ROOT / item["jobdir"]
        manifest_path = directory / "manifest.json"
        check(manifest_path, origins[item["identity"]]["manifest_sha256"])
        manifest = read_json(manifest_path)
        assert manifest["identity"] == item["identity"] and manifest["status"] == "complete"
        identity_payload = {"configuration": manifest["configuration"], "provenance": manifest["provenance"]}
        assert hashlib.sha256(json_text(identity_payload).encode()).hexdigest() == item["identity"]
        cfg = manifest["configuration"]
        instance, rate = item["instance"], item["rate"]
        a, order = INSTANCES[instance]
        assert cfg["configuration_id"] == CONFIG_ID and cfg["weight_exponent"] == 2
        assert cfg["ridges"] == config["ridges"] and cfg["routes"] == list(METHODS)
        assert cfg["order"] == order and cfg["instance"] == instance and cfg["rate"] == rate
        assert cfg["budget"] == "equal_shot" and cfg["fit_lambdas"] == config["effective_scales"]
        assert cfg["fitting_floor"] == 1 / M and cfg["direct_shots"] == 160000
        check(Path(manifest["provenance"]["approved_config"]["path"]), CONFIG_SHA)
        # Validate saved code provenance without importing or executing fitting code.
        for source in manifest["provenance"]["sources"]:
            check(Path(source["path"]), source["sha256"])
        point_path = directory / "point_manifest.json"
        point = read_json(point_path)
        check(point_path)
        assert point["identity"] == item["identity"] and point["status"] == "points_complete"
        parent_path = directory / "parent_estimates.npz"
        expected_sha = manifest["artifacts"][parent_path.name]
        assert expected_sha == point["artifacts"][parent_path.name]
        parent_sha = check(parent_path, expected_sha)
        parents.append({"instance": instance, "rate": rate, "budget": "equal_shot", "identity": item["identity"],
                        "parent_path": str(parent_path.relative_to(ROOT)), "parent_sha256": parent_sha,
                        "manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
                        "point_manifest_path": str(point_path.relative_to(ROOT)), "point_manifest_sha256": sha(point_path)})
        with np.load(parent_path, allow_pickle=False) as z:
            assert str(z["identity"]) == item["identity"] and json.loads(str(z["configuration_json"])) == cfg
            target = z["target"]
            assert np.array_equal(target, arrays[instance + "__ideal"])
            assert np.array_equal(z["fit_lambdas"], np.asarray(config["effective_scales"]))
            for route in ROUTES:
                key = {"raw_160k": "raw_full_original", "reset_160k": "reset_reference"}.get(route, "after__" + route)
                values = z[key]
                assert values.shape == (M,) and np.isfinite(values).all() and np.min(values) >= 0
                normalization_error = abs(float(values.sum()) - 1)
                normalization_max_error = max(normalization_max_error, normalization_error)
                assert normalization_error < 2e-12
                mask = arrays[instance + "__exact"]
                probability = float(values[mask].sum())
                dot_error = abs(probability - float(np.dot(values, mask.astype(float))))
                dot_max_error = max(dot_max_error, dot_error)
                assert dot_error < 2e-15 and -1e-14 <= probability <= 1 + 1e-14
                tvd, gap = metrics(values, target, order)
                for metric, value in (("tvd", tvd), ("normalized_grouped_peak_margin", gap)):
                    error = abs(value - locked[("equal_shot", instance, rate, route, metric)])
                    metric_max_error = max(metric_max_error, error)
                    assert error < 5e-12
                support = np.flatnonzero(values > 0)
                singleton = len(support) == 1
                rows.append(dict(zip(COLUMNS, (instance, a, order, rate, rate * 100, route, probability, probability * 100,
                                              tvd, gap, len(support), int(support[0]) if singleton else "", singleton, parent_sha))))
    rows.sort(key=lambda x: (x["order"], x["rate"], ROUTES.index(x["route"])))
    references.sort(key=lambda x: x["order"])
    assert len(rows) == 222 and len({(x["instance"], x["rate"], x["route"]) for x in rows}) == 222
    singletons = [{**x, "decoded_q": int(q[x["singleton_y"]]),
                   "ideal_probability_at_y": float(arrays[x["instance"] + "__ideal"][x["singleton_y"]])}
                  for x in rows if x["singleton_flag"] and abs(x["p_exact"] - 1) < 1e-12]
    expected = {("n119_a13", .0003, 4096), ("n119_a16", .00014, 2731),
                ("n119_a16", .00016, 2731), ("n119_a16", .00018, 2735)}
    assert {(x["instance"], x["rate"], x["singleton_y"]) for x in singletons} == expected
    assert len(singletons) == 4 and all(x["route"] == "dual_exponential_zne" for x in singletons)
    provenance = {"analysis_id": "s6_strict_single_candidate_exact_order_20261031", "configuration_id": CONFIG_ID,
                  "configuration_sha256": CONFIG_SHA, "inventory_source": str(aggregate_path.relative_to(ROOT)),
                  "inventory_source_sha256": sha(aggregate_path), "equal_shot_conditions": 37,
                  "route_points": len(rows), "routes": list(ROUTES), "excluded_gate_matched_conditions": 3,
                  "decoder": {"rule": "q(y)=Fraction(y,16384).limit_denominator(119).denominator; success iff q(y)==r",
                              "denominator_cap_inclusive": 119, "candidate_count_per_outcome": 1,
                              "additional_convergents_or_multiples_searched": False, "divisor_reduction": False,
                              "all_16384_outcomes_match_independent_denominator_enumeration": True,
                              "bit_order": "Integer y directly indexes the saved distribution; no bit reversal",
                              "tie_rule": "Nearest rational; smaller reduced denominator at ties, then lower rational"},
                  "metrics": {"p_exact": "sum_y p(y) 1[q(y)=r]", "nonzero_support": "number of entries strictly greater than zero",
                              "normalized_gap": "minimum phase-group mass minus maximum outside-bin mass, divided by the ideal counterpart",
                              "tvd": "one half of the L1 distance to the saved ideal target",
                              "sum_vs_dot_max_absolute_error": dot_max_error,
                              "probability_sum_max_absolute_error": normalization_max_error,
                              "matched_existing_tvd_gap_scalars": 444, "existing_metrics_max_absolute_error": metric_max_error},
                  "diagnostic_references": {"modular_accept": "Same single q(y), accepted iff a**q(y) mod 119 == 1; not exact-order success",
                                            "legacy": "Any modularly verified CF convergent with 1<q<119, followed by order reduction; diagnostic only",
                                            "legacy_masks_bitwise_match_checksum_verified_earlier_pack": True,
                                            "old_success_csv_used": False, "used_in_success_curves": False},
                  "parents": parents, "input_sha256": checked,
                  "new_quantum_shots": 0, "new_qpu_jobs": 0, "new_gpu_simulations": 0,
                  "new_zne_fits": 0, "new_bootstrap_replicates": 0, "confidence_intervals_computed": False,
                  "hyperparameters_retuned": False, "selection_disclosure": config["selection_disclosure"],
                  "warnings": ["ZNE success values are decoder-accepted mass in saved reconstructed distributions, not independently measured mitigated-shot frequencies.",
                               "An ideal-QPE success reference is not a universal upper bound on arbitrary distributions.",
                               "A concentrated decoder-accepted point mass can yield 100% strict success with poor TVD or nonpositive normalized gap.",
                               "No new full-Shor, factorization, uncertainty, or independent-validation claim is made."]}
    return rows, references, singletons, arrays, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    rows, references, singletons, arrays, provenance = calculate()
    tables = {"success_all_rates.csv": rows, "references.csv": references, "singleton_cases.csv": singletons}
    if args.verify_only:
        old = read_json(args.output_dir / "provenance.json")
        for name, expected in old["output_sha256"].items():
            assert sha(args.output_dir / name) == expected
        for name, table in tables.items():
            actual = read_csv(args.output_dir / name)
            assert len(actual) == len(table)
            for observed, expected in zip(actual, table):
                assert observed == {k: str(v) for k, v in expected.items()}, (name, observed, expected)
        with np.load(args.output_dir / "decoder_masks.npz", allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            assert all(np.array_equal(z[key], values) for key, values in arrays.items())
        assert all(old[key] == value for key, value in provenance.items())
        print("PASS: exporter reproducibility check: 37 parents, 222 strict route points, 3 references, 4 accepted singletons.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [*(args.output_dir / name for name in tables), args.output_dir / "decoder_masks.npz", args.output_dir / "provenance.json"]
    if any(path.exists() for path in expected_files):
        raise FileExistsError("Refusing to overwrite a data pack: use --verify-only or a new --output-dir.")
    for name, table in tables.items():
        write_csv(args.output_dir / name, table)
    np.savez_compressed(args.output_dir / "decoder_masks.npz", **arrays)
    provenance.update({"created_utc": datetime.now(timezone.utc).isoformat(), "export_script_sha256": sha(Path(__file__)),
                       "output_sha256": {p.name: sha(p) for p in expected_files if p.name != "provenance.json"}})
    with (args.output_dir / "provenance.json").open("x", encoding="utf-8") as handle:
        handle.write(json_text(provenance))
    print("PASS: exported 37 checksum-validated parents × 6 routes = 222 points; no shots, fits, or bootstrap.")
    for row in references:
        print(f"r={row['order']}: strict uniform count={row['strict_exact_uniform_count']}, ideal={row['success_percent_ideal']:.12f}%")
    for row in singletons:
        print(f"Singleton: {row['instance']} {row['rate_percent']:g}% y={row['singleton_y']} p_exact={row['p_exact']:.12g} TVD={row['tvd']:.12g} gap={row['normalized_gap']:.12g}")


if __name__ == "__main__":
    main()
