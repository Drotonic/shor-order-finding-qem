#!/usr/bin/env python3
"""Independently aggregate approved mixed-ridge C1 bootstrap artifacts.

Uses only saved NumPy arrays and standard-library metadata. No fitting modules,
bootstrap samplers, simulator, or plotting code are imported. An incomplete
40-job inventory is always labelled partial, never final.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results" / "c1_primary_fixed_20261018"
APPROVED = ROOT / "analysis" / "c1_primary_fixed_20261018.json"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
CONTROLS = ("dual_exponential_zne_ridge0_control", "dual_exponential_zne_ridge1e-8_control")
METRICS = ("tvd", "normalized_grouped_peak_margin")
CASES = ("n119_a13", "n119_a50", "n119_a16")
COMMON_RATES = (1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4)
EXTRA_RATES = (1.2e-4, 1.4e-4, 1.6e-4, 1.8e-4)
TOL = 1e-10
HASH_CACHE = {}


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256(path):
    path = Path(path)
    stamp = path.stat()
    key = (str(path.resolve()), stamp.st_size, stamp.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                digest.update(chunk)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def path_label(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def rate_key(rate):
    return format(float(rate), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def expected_jobs():
    jobs = []
    for instance in CASES:
        rates = sorted(COMMON_RATES + (EXTRA_RATES if instance == "n119_a16" else ()))
        jobs.extend({"budget": "equal_shot", "instance": instance, "rate": rate} for rate in rates)
        jobs.append({"budget": "gate_matched", "instance": instance, "rate": .0001})
    return jobs


def unpack_string(value):
    return str(np.asarray(value).item())


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def verify_entry(entry, *, context):
    check(isinstance(entry, dict) and "path" in entry and "sha256" in entry,
          f"Missing declared path/hash: {context}")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = ROOT / path
    check(path.is_file(), f"Missing hashed file: {context}: {path}")
    check(sha256(path) == entry["sha256"], f"Hash mismatch: {context}: {path}")


def verify_provenance(provenance, approved_sha):
    for name in ("approved_config", "frozen_point_pack", "replay_reference"):
        verify_entry(provenance[name], context=name)
    check(provenance["approved_config"]["sha256"] == approved_sha,
          "Job uses another approved configuration snapshot")
    frozen = provenance["frozen_point_pack"]
    check(sha256(Path(frozen["points_path"])) == frozen["points_sha256"], "Frozen point-pack CSV hash mismatch")
    check(isinstance(provenance["sources"], list) and provenance["sources"], "No source provenance")
    check(isinstance(provenance["inputs"], list) and provenance["inputs"], "No saved-count provenance")
    for role in ("sources", "inputs"):
        for index, entry in enumerate(provenance[role]):
            verify_entry(entry, context=f"{role}[{index}]")
    source_names = {Path(entry["path"]).name for entry in provenance["sources"]}
    check("run_c1_primary_bootstrap_20261018.py" in source_names,
          "New approved-primary runner is absent from source identity")


def assert_close(actual, expected, *, message, atol=2e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True), message)
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]), initial=0))


def distribution_summary(samples, point, *, strict=True):
    samples = np.asarray(samples, dtype=float).reshape(-1)
    finite = samples[np.isfinite(samples)]
    if strict:
        check(len(samples) > 0 and len(finite) == len(samples) and np.isfinite(point),
              "A locked-metric sample or parent center is nonfinite")
    result = {"point": float(point) if np.isfinite(point) else None,
              "replicates": len(samples), "finite_replicates": len(finite),
              "nonfinite_replicates": int(len(samples) - len(finite)),
              "numerical_tolerance": TOL}
    if len(finite) == 0:
        result.update({key: None for key in ("ci_low", "ci_high", "bootstrap_mean", "bootstrap_median", "minimum", "maximum",
                                            "q001", "q01", "q99", "q999")})
        return result
    qs = np.quantile(finite, (.001, .01, .025, .5, .975, .99, .999))
    result.update(ci_low=float(qs[2]), ci_high=float(qs[4]), bootstrap_mean=float(np.mean(finite)),
                  bootstrap_median=float(qs[3]), minimum=float(np.min(finite)), maximum=float(np.max(finite)),
                  q001=float(qs[0]), q01=float(qs[1]), q99=float(qs[5]), q999=float(qs[6]),
                  exact_negative_count=int(np.count_nonzero(finite < 0)),
                  exact_zero_count=int(np.count_nonzero(finite == 0)),
                  exact_positive_count=int(np.count_nonzero(finite > 0)),
                  numerical_negative_count=int(np.count_nonzero(finite < -TOL)),
                  numerical_zero_count=int(np.count_nonzero(np.abs(finite) <= TOL)),
                  numerical_positive_count=int(np.count_nonzero(finite > TOL)),
                  below_central_interval_count=int(np.count_nonzero(finite < qs[2])),
                  above_central_interval_count=int(np.count_nonzero(finite > qs[4])),
                  point_outside_interval=bool(point < qs[2] or point > qs[4]))
    result["pointwise_interval_sign"] = "positive" if qs[2] > TOL else "negative" if qs[4] < -TOL else "unresolved"
    return result


def independent_probability_metrics(probability, target, order):
    probability, target = np.asarray(probability), np.asarray(target)
    check(probability.shape == target.shape == (16384,), "Unexpected output dimension")
    check(np.min(probability) >= -1e-14 and abs(np.sum(probability) - 1) < 1e-11,
          "Parent projected distribution is not a normalized nonnegative vector")
    groups = [sorted({(16384 * s) // order, (16384 * s + order - 1) // order}) for s in range(order)]
    outside = np.ones(16384, dtype=bool)
    for group in groups:
        outside[group] = False
    def margin(p):
        return min(float(np.sum(p[group])) for group in groups) - float(np.max(p[outside]))
    denominator = margin(target)
    if order == 6:
        check(abs(denominator - .13535877187470172) < 1e-12, "Locked r=6 ideal normalization changed")
    check(denominator > 0, "Nonpositive ideal group-gap normalization")
    return np.asarray([.5 * np.sum(np.abs(probability - target)), margin(probability) / denominator])


def audit_completed_job(jobdir, job, approved_sha, *, expected_replicates):
    manifest_path = jobdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(manifest["status"] == "complete", f"Noncomplete manifest supplied as complete: {jobdir}")
    check(manifest["actual_bootstrap_started"] is True and manifest["actual_bootstrap_completed"] is True
          and manifest["new_quantum_shots"] == 0 and manifest["new_noisy_circuit_simulations"] == 0
          and manifest["no_old_interval_reuse"] is True, "Completed cache does not satisfy actual saved-data-only execution contract")
    configuration, provenance = manifest["configuration"], manifest["provenance"]
    identity = hashlib.sha256(json_text({"configuration": configuration, "provenance": provenance}).encode("utf-8")).hexdigest()
    check(identity == manifest["identity"], "Configuration/provenance identity digest mismatch")
    for key in ("budget", "instance", "rate"):
        check(configuration[key] == job[key], f"Runtime configuration {key} differs from planned job")
    check(int(configuration["replicates"]) == expected_replicates, "Unexpected replicate count")
    approved = json.loads(APPROVED.read_text(encoding="utf-8"))
    check(configuration["configuration_id"] == approved["configuration_id"] and configuration["weight_exponent"] == 2
          and configuration["ridges"] == approved["ridges"] and configuration["dimension"] == 16384
          and configuration["fitting_floor"] == 1 / 16384, "Runtime estimator differs from approved fixed scientific settings")
    check(configuration["fit_lambdas"] == approved["effective_scales"]
          and configuration["single_gamma_grid"] == approved["single_gamma_grid"]
          and configuration["dual_gamma_grid"] == approved["dual_gamma_grid"], "Runtime scale/rate search changed")
    per_scale = 20000 if job["budget"] == "equal_shot" else 11188
    check(configuration["per_scale_shots"] == per_scale and configuration["zne_circuit_shots"] == per_scale * 8
          and configuration["direct_shots"] == 160000 and configuration["reset_extension_shots"] == 160000 - per_scale,
          "Runtime shot-budget contract changed")
    check(configuration["seed"] == (20260722 if job["budget"] == "equal_shot" else 20260804)
          and configuration["batch_size"] == 25 and configuration["native_argmax"] is True
          and configuration["observed_support_reselected"] is True and configuration["gamma_refitted_each_replicate"] is True
          and configuration["no_cross_budget_paired_interval"] is True, "RNG/refitting/pairing contract changed")
    check(configuration["paired_reset_improvement_signs"]["positive_favors"] == "model"
          and configuration["paired_control_improvement_signs"]["positive_favors"] == "control"
          and configuration["control_reference_route"] == "dual_exponential_zne", "Paired improvement orientation changed")
    if expected_replicates == 10000:
        check(manifest["final_10000_contract"] is True, "Final 10000 contract not satisfied")
    verify_provenance(provenance, approved_sha)
    for filename, declared in manifest["artifacts"].items():
        path = jobdir / filename
        digest = declared["sha256"] if isinstance(declared, dict) else declared
        check(path.is_relative_to(jobdir) and path.is_file() and sha256(path) == digest,
              f"Declared output hash mismatch: {path}")
    started = json.loads((jobdir / "started_manifest.json").read_text(encoding="utf-8"))
    check(started["identity"] == identity and started["configuration"] == configuration and started["provenance"] == provenance,
          "Started and completed job identities differ")
    parent_path, cache_path = jobdir / "parent_estimates.npz", jobdir / "scalar_bootstrap_draws.npz"
    check(parent_path.name in manifest["artifacts"] and cache_path.name in manifest["artifacts"],
          "Parent or final scalar cache is not covered by output manifest")
    with np.load(parent_path, allow_pickle=False) as handle:
        parent = {key: handle[key] for key in handle.files}
    with np.load(cache_path, allow_pickle=False) as handle:
        cache = {key: handle[key] for key in handle.files}
    # Link each exact segment of the final cache to its immutable checkpoint
    # payload digest, without importing the numerical producer or fitting code.
    previous_hash, previous_rng, expected_start = None, None, 0
    replay_totals = {"checked": 0, "bitwise_equal": 0, "max_abs_error": 0., "atol": 1e-11}
    for item in manifest["checkpoint_files"]:
        check(Path(item["path"]).name == item["path"], "Checkpoint name escapes its declared directory")
        path = jobdir / "batches" / item["path"]
        check(sha256(path) == item["sha256"], f"Checkpoint file hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as checkpoint:
            metadata = json.loads(unpack_string(checkpoint["checkpoint_json"]))
        first, stop = int(metadata["start"]), int(metadata["stop"])
        check(metadata["identity"] == identity and first == expected_start and first == item["start"] and stop == item["stop"]
              and first < stop <= expected_replicates and stop - first <= 25 and metadata["full_draw_batch_size"] == 25,
              "Checkpoint identity/contiguity/batch size mismatch")
        check(metadata["previous_checkpoint_sha256"] == previous_hash, "Broken checkpoint predecessor hash")
        if previous_rng is not None:
            check(metadata["rng_before"] == previous_rng, "Checkpoint RNG chain is discontinuous")
        for key, digest in metadata["array_hashes"].items():
            check(key in cache, f"Checkpoint array absent from final cache: {key}")
            segment = cache[key][first:stop] if key in ("support_sizes", "packed_observed_support") else cache[key][..., first:stop]
            check(array_sha(segment) == digest, f"Final scalar cache differs from checkpoint payload: {key}, {first}:{stop}")
        replay_totals["checked"] += metadata["replay"]["checked"]
        replay_totals["bitwise_equal"] += metadata["replay"]["bitwise_equal"]
        replay_totals["max_abs_error"] = max(replay_totals["max_abs_error"], metadata["replay"]["max_abs_error"])
        previous_hash, previous_rng, expected_start = item["sha256"], metadata["rng_after"], stop
    check(expected_start == expected_replicates and replay_totals == manifest["replay"], "Incomplete checkpoint/replay reduction")
    check(replay_totals["max_abs_error"] <= 1e-11, "Declared frozen replay failed")
    for archive in (parent, cache):
        check(unpack_string(archive["identity"]) == identity, "NPZ identity differs from completed manifest")
        check(json.loads(unpack_string(archive["configuration_json"])) == configuration,
              "NPZ runtime configuration differs from completed manifest")
    for key in ("routes", "quantities", "direct_routes", "direct_metrics"):
        check(np.array_equal(parent[key], cache[key]), f"Parent/bootstrap axis mismatch: {key}")
    routes = [str(x) for x in cache["routes"]]
    quantities = [str(x) for x in cache["quantities"]]
    direct_routes = [str(x) for x in cache["direct_routes"]]
    direct_metrics = [str(x) for x in cache["direct_metrics"]]
    controls_expected = job["budget"] == "equal_shot" and job["rate"] == .0001
    check(set(routes) == set(METHODS + (CONTROLS if controls_expected else ())) and len(routes) == len(set(routes)),
          "Wrong primary/control estimator scope")
    check(direct_metrics == list(METRICS) and all(metric in quantities for metric in METRICS), "Locked scalar metric axes changed")
    check(routes == configuration["routes"] and quantities == configuration["quantities"]
          and direct_routes == configuration["direct_routes"] and direct_metrics == configuration["direct_metrics"],
          "NPZ scalar axes differ from signed configuration identity")
    check(int(cache["replicates"]) == expected_replicates and int(cache["seed"]) == configuration["seed"]
          and int(cache["batch_size"]) == 25, "NPZ seed/replicate/batch metadata changed")
    required_direct = {"raw_160k", "reset_160k", "raw_20k", "reset_20k"} if job["budget"] == "equal_shot" else {"reset_160k"}
    check(set(direct_routes) == required_direct and len(direct_routes) == len(set(direct_routes)), "Wrong direct-baseline scope")
    scientific_direct = ["raw_160k", "reset_160k"] if job["budget"] == "equal_shot" else ["reset_160k"]
    check(configuration["scientific_direct_routes"] == scientific_direct, "Replay-only 20k routes are mislabelled as scientific comparison budgets")
    specs = {spec["route"]: spec for spec in configuration["route_specifications"]}
    check(set(specs) == set(routes), "Missing route-specific fitting identity")
    for route in routes:
        ridge = 0. if route == CONTROLS[0] else 1e-8 if route == CONTROLS[1] else approved["ridges"][route]
        method = "dual_exponential_zne" if route in CONTROLS else route
        check(specs[route]["method"] == method and specs[route]["ridge"] == ridge, "Primary/control ridge map changed")
    b = expected_replicates
    check(cache["scalar_draws"].shape == (len(routes), len(quantities), b) and parent["scalar_draws"].shape == (len(routes), len(quantities), 1),
          "Incomplete scalar route cache")
    check(cache["direct_draws"].shape == (len(direct_routes), 2, b) and parent["direct_draws"].shape == (len(direct_routes), 2, 1),
          "Incomplete direct-baseline scalar cache")
    check(cache["packed_observed_support"].shape == (b, 2048) and cache["support_sizes"].shape == (b,), "Incomplete resampled-support records")
    check(np.all((cache["support_sizes"] >= 1) & (cache["support_sizes"] <= 16384)), "Invalid observed-support size")
    check(cache["selected_indices"].shape == (len(routes), b) and cache["selected_parameters"].shape == (len(routes), 2, b),
          "Incomplete adaptive-fit selection records")
    rindex, qindex = {route: i for i, route in enumerate(routes)}, {name: i for i, name in enumerate(quantities)}
    metric_draws = cache["scalar_draws"][:, [qindex[metric] for metric in METRICS], :]
    parent_metrics = parent["scalar_draws"][:, [qindex[metric] for metric in METRICS], 0]
    check(np.isfinite(metric_draws).all() and np.isfinite(parent_metrics).all() and np.isfinite(cache["direct_draws"]).all(),
          "Nonfinite locked performance metric")
    order = {"n119_a13": 4, "n119_a50": 2, "n119_a16": 6}[job["instance"]]
    parent_metric_residual = 0.
    for route in routes:
        actual = independent_probability_metrics(parent[f"after__{route}"][0] if parent[f"after__{route}"].ndim == 2 else parent[f"after__{route}"], parent["target"], order)
        parent_metric_residual = max(parent_metric_residual, assert_close(actual, parent_metrics[rindex[route]],
                                                                         message="Parent vectors do not reproduce saved locked metrics", atol=1e-11))
    reset_index = direct_routes.index("reset_160k")
    reset_draws = cache["direct_draws"][reset_index]
    expected_reset = np.stack((reset_draws[0] - metric_draws[:, 0, :], metric_draws[:, 1, :] - reset_draws[1]), axis=1)
    paired_residual = assert_close(cache["paired_reset_improvements"], expected_reset,
                                   message="Saved Reset comparisons are not actual same-replicate differences")
    control_routes = [route for route in routes if route in CONTROLS]
    check(cache["paired_control_improvements"].shape == (len(control_routes), 2, b), "Control paired-cache axis mismatch")
    if controls_expected:
        check(cache["control_routes"].astype(str).tolist() == control_routes, "Control comparison axis changed")
        primary = metric_draws[rindex["dual_exponential_zne"]]
        expected_control = np.asarray([np.stack((primary[0] - metric_draws[rindex[route], 0], metric_draws[rindex[route], 1] - primary[1]))
                                       for route in control_routes])
        paired_residual = max(paired_residual, assert_close(cache["paired_control_improvements"], expected_control,
             message="Dual-control differences are not actual same-replicate comparisons"))
    return {"manifest": manifest, "configuration": configuration, "identity": identity,
            "parent": parent, "cache": cache, "routes": routes, "quantities": quantities,
            "direct_routes": direct_routes, "rindex": rindex, "qindex": qindex,
            "control_routes": control_routes, "parent_metrics": parent_metrics, "metric_draws": metric_draws,
            "parent_metric_residual": parent_metric_residual, "paired_residual": paired_residual,
            "replay": replay_totals,
            "manifest_path": path_label(manifest_path), "manifest_sha256": sha256(manifest_path)}


def row_context(job, *, identity):
    return {"configuration_id": "c1_beta2_dual_ridge1e-6_only_20261018", **job,
            "rate_percent": 100 * job["rate"], "identity": identity}


def aggregate_job(job, data):
    context = row_context(job, identity=data["identity"])
    scalar, direct = data["cache"]["scalar_draws"], data["cache"]["direct_draws"]
    parent_scalar, parent_direct = data["parent"]["scalar_draws"][..., 0], data["parent"]["direct_draws"][..., 0]
    all_scalars, absolute, paired, controls = [], [], [], []
    for route_index, route in enumerate(data["routes"]):
        for q_index, quantity in enumerate(data["quantities"]):
            stats = distribution_summary(scalar[route_index, q_index], parent_scalar[route_index, q_index], strict=quantity in METRICS)
            record = {**context, "route": route, "quantity": quantity, "role": "dual_ridge_control" if route in CONTROLS else "primary_C1", **stats}
            all_scalars.append(record)
            if quantity in METRICS:
                absolute.append({**record, "metric": quantity, "circuit_shots": 160000 if job["budget"] == "equal_shot" else 89504})
    for route_index, route in enumerate(data["direct_routes"]):
        for metric_index, metric in enumerate(METRICS):
            stats = distribution_summary(direct[route_index, metric_index], parent_direct[route_index, metric_index])
            record = {**context, "route": route, "quantity": metric,
                      "role": "context_only_20k" if route.endswith("_20k") else "direct_160k", **stats}
            all_scalars.append(record)
            if route.endswith("_160k"):
                absolute.append({**record, "metric": metric, "circuit_shots": 160000})
    reset_i = data["direct_routes"].index("reset_160k")
    reset_point = parent_direct[reset_i]
    for route_index, route in enumerate(data["routes"]):
        target_point = data["parent_metrics"][route_index]
        parent_difference = (reset_point[0] - target_point[0], target_point[1] - reset_point[1])
        for metric_index, metric in enumerate(METRICS):
            paired.append({**context, "comparison": f"{route}_vs_Reset", "route": route, "reference": "reset_160k", "metric": metric,
                           "positive_favors": route, "pairing": "same stored replicate, refit support/rate/projection",
                           **distribution_summary(data["cache"]["paired_reset_improvements"][route_index, metric_index], parent_difference[metric_index])})
    if "raw_160k" in data["direct_routes"]:
        raw_i = data["direct_routes"].index("raw_160k")
        differences = (direct[raw_i, 0] - direct[reset_i, 0], direct[reset_i, 1] - direct[raw_i, 1])
        points = (parent_direct[raw_i, 0] - reset_point[0], reset_point[1] - parent_direct[raw_i, 1])
        for metric_index, metric in enumerate(METRICS):
            paired.append({**context, "comparison": "Reset_vs_Raw", "route": "reset_160k", "reference": "raw_160k", "metric": metric,
                           "positive_favors": "reset_160k", "pairing": "same stored bootstrap replicate; original independent Raw stream",
                           **distribution_summary(differences[metric_index], points[metric_index])})
    primary_i = data["rindex"]["dual_exponential_zne"]
    for control_index, route in enumerate(data["control_routes"]):
        control_i = data["rindex"][route]
        parent_difference = (data["parent_metrics"][primary_i, 0] - data["parent_metrics"][control_i, 0],
                             data["parent_metrics"][control_i, 1] - data["parent_metrics"][primary_i, 1])
        for metric_index, metric in enumerate(METRICS):
            samples = data["cache"]["paired_control_improvements"][control_index, metric_index]
            record = {**context, "comparison": f"{route}_vs_primary_dual", "route": route, "reference": "dual_exponential_zne", "metric": metric,
                      "positive_favors": route, "pairing": "same actual scale counts and observed support, distinct fixed ridge",
                      **distribution_summary(samples, parent_difference[metric_index])}
            primary_stats = distribution_summary(-samples, -parent_difference[metric_index])
            controls.append({**record, **{f"primary_improvement_{key}": value for key, value in primary_stats.items()}})
            paired.append(record)
    support = {**context, **distribution_summary(data["cache"]["support_sizes"],
               int(np.count_nonzero(np.sum(data["parent"]["scale_parents"], axis=0))))}
    identity_row = {**context, "manifest": data["manifest_path"], "manifest_sha256": data["manifest_sha256"],
                    "parent_vector_metric_max_residual": data["parent_metric_residual"],
                    "paired_cache_max_residual": data["paired_residual"], "replicates": scalar.shape[-1],
                    "replay_checked": data["replay"]["checked"], "replay_bitwise_equal": data["replay"]["bitwise_equal"],
                    "replay_max_abs_error": data["replay"]["max_abs_error"]}
    return all_scalars, absolute, paired, controls, support, identity_row


def write_csv(path, records):
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ("status",))
        writer.writeheader()
        writer.writerows(records)


def fmt(value):
    return f"{float(value):.6g}"


def findings_text(summary, absolute, paired, controls):
    lines = ["승인된 β=2 / dual-only ridge=1e−6 C1 bootstrap 집계", "=" * 62,
             f"상태: {summary['status']} — 완료·검증 {summary['completed_jobs']}/40 조건.",
             f"완료 조건당 재표본 수: {summary['replicates_per_completed_job']}회.",
             "중심값은 원래 저장 부모 분포의 점추정이며 bootstrap 평균/중앙값으로 교체하지 않았다.",
             "구간은 저장 경험분포와 사후 선택된 고정 설정에 조건부인 pointwise 95% percentile 구간이다.",
             "새 양자회로 실행이나 noisy shots 생성, 독립 물리 검증, post-selection coverage 보장은 아니다.", ""]
    if summary["smoke_only"]:
        lines.extend(("이 산출물은 처리 코드의 smoke 검증용이며 최종 10,000회 분석 결과가 아니다.",
                      "CSV의 구간·꼬리 통계는 구현 비교에만 사용하며 과학적 결과 해석은 보류한다.",
                      "publication_ready=False. 원고·최종 보고서에 이 구간을 사용하지 않는다.", ""))
        return "\n".join(lines)
    def find(records, **keys):
        return next((row for row in records if all(row.get(key) == value for key, value in keys.items())), None)
    for budget in ("equal_shot", "gate_matched"):
        lines.append("[등샷]" if budget == "equal_shot" else "[noisy-gate-application 맞춤 재표본화: ZNE 89,504 vs 직접 Reset 160,000 shots]")
        for instance in CASES:
            result = find(absolute, budget=budget, instance=instance, rate=.0001, route="dual_exponential_zne", metric="tvd")
            if result is None:
                lines.append(f"{instance}: 해당 조건 완료 cache가 아직 없어 수치를 집계하지 않았다.")
                continue
            for metric in METRICS:
                result = find(absolute, budget=budget, instance=instance, rate=.0001, route="dual_exponential_zne", metric=metric)
                change = find(paired, budget=budget, instance=instance, rate=.0001, route="dual_exponential_zne", reference="reset_160k", metric=metric)
                lines.append(f"{instance} {metric}: dual={fmt(result['point'])} [{fmt(result['ci_low'])}, {fmt(result['ci_high'])}]; "
                             f"Reset 대비 개선량={fmt(change['point'])} [{fmt(change['ci_low'])}, {fmt(change['ci_high'])}].")
        lines.append("")
    lines.append("[r=6, p2=0.01% dual ridge 대조: 양수이면 primary ridge=1e−6에 유리]")
    for route in CONTROLS:
        for metric in METRICS:
            row = find(controls, instance="n119_a16", rate=.0001, route=route, metric=metric)
            if row is None:
                lines.append(f"{route} / {metric}: 완료 자료 대기.")
            else:
                prefix = "primary_improvement_"
                lines.append(f"{route} / {metric}: primary 개선량={fmt(row[prefix+'point'])} "
                             f"[{fmt(row[prefix+'ci_low'])}, {fmt(row[prefix+'ci_high'])}]; 실제 재표본 범위 "
                             f"[{fmt(row[prefix+'minimum'])}, {fmt(row[prefix+'maximum'])}], "
                             f"수치적 음수/0/양수={row[prefix+'numerical_negative_count']}/{row[prefix+'numerical_zero_count']}/{row[prefix+'numerical_positive_count']}.")
    lines.extend(("", "0을 포함하는 구간은 동등성 증명이 아니며 [0,0]도 드문 큰 꼬리가 없다는 뜻이 아니다.",
                  "부호 집계의 1e−10은 반올림 수준 수치 분류용이고 유의수준 또는 실질적 효과 기준이 아니다.",
                  "등샷과 gate-matched 결과 사이에 paired 구간을 만들지 않았다. 모든 원래 분석·원고는 별도로 보존했다.", ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=BASE / "bootstrap_10000")
    parser.add_argument("--output-dir", type=Path, default=BASE / "aggregate")
    parser.add_argument("--expected-replicates", type=int, default=10000)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    check(ROOT.name == "paper-option2" and subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "option2-metric-locked-revision",
        "Wrong isolated worktree or branch")
    artifacts, output = args.artifacts.resolve(), args.output_dir.resolve()
    smoke_input = args.allow_smoke and args.expected_replicates < 10000 and artifacts.is_relative_to(Path("/tmp")) and artifacts != Path("/tmp")
    check((artifacts.is_relative_to(BASE) or smoke_input) and output.is_relative_to(BASE) and output != BASE,
          "Only the new c1_primary_fixed tree is in scope; explicit nonfinal smoke input may use a /tmp child directory")
    check(args.expected_replicates == 10000 or args.allow_smoke, "Nonfinal replicate counts require --allow-smoke")
    approved = json.loads(APPROVED.read_text(encoding="utf-8"))
    check(approved["configuration_id"] == "c1_beta2_dual_ridge1e-6_only_20261018" and approved["estimator"] == "C1"
          and approved["weight_exponent"] == 2 and approved["ridges"] == dict(zip(METHODS, (0., 0., 0., 1e-6))),
          "Approved scientific configuration differs from requested primary")
    approved_sha = sha256(APPROVED)
    all_scalars, absolute, paired, controls, supports, identities, inventory = [], [], [], [], [], [], []
    for job in expected_jobs():
        jobdir = artifacts / job["budget"] / job["instance"] / f"rate_{rate_key(job['rate'])}"
        manifest_path = jobdir / "manifest.json"
        if not manifest_path.is_file():
            inventory.append({**job, "status": "incomplete_or_not_started", "jobdir": path_label(jobdir)})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            inventory.append({**job, "status": str(manifest.get("status", "incomplete")), "jobdir": path_label(jobdir)})
            continue
        data = audit_completed_job(jobdir, job, approved_sha, expected_replicates=args.expected_replicates)
        result = aggregate_job(job, data)
        all_scalars.extend(result[0]); absolute.extend(result[1]); paired.extend(result[2]); controls.extend(result[3])
        supports.append(result[4]); identities.append(result[5])
        inventory.append({**job, "status": "complete_and_verified", "identity": data["identity"],
                          "jobdir": path_label(jobdir)})
        print(f"Verified {job['budget']}/{job['instance']}/{job['rate']:.6g}: {args.expected_replicates} actual replicates", flush=True)
        del data
    complete = len(identities) == 40 and args.expected_replicates == 10000
    summary = {"schema_version": 1, "configuration_id": approved["configuration_id"], "approved_config_sha256": approved_sha,
        "status": "complete" if complete else "partial", "publication_ready": complete,
        "completed_jobs": len(identities), "expected_jobs": 40,
        "completed_equal_shot_jobs": sum(row["budget"] == "equal_shot" for row in identities),
        "completed_gate_matched_jobs": sum(row["budget"] == "gate_matched" for row in identities),
        "replicates_per_completed_job": args.expected_replicates,
        "completed_conditional_replicates": args.expected_replicates * len(identities),
        "smoke_only": args.expected_replicates != 10000,
        "absolute_locked_metric_rows": len(absolute), "paired_comparison_rows": len(paired),
        "all_scalar_summary_rows": len(all_scalars), "dual_control_rows": len(controls),
        "parent_vector_metric_max_residual": max((row["parent_vector_metric_max_residual"] for row in identities), default=0),
        "paired_cache_max_residual": max((row["paired_cache_max_residual"] for row in identities), default=0),
        "replay_checked": sum(row["replay_checked"] for row in identities),
        "replay_bitwise_equal": sum(row["replay_bitwise_equal"] for row in identities),
        "replay_max_abs_error": max((row["replay_max_abs_error"] for row in identities), default=0),
        "inventory": inventory, "verified_job_manifests": identities,
        "generator_sha256": sha256(Path(__file__)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "new_quantum_shots": 0, "metric_definitions_changed": False,
        "statistical_scope": "Conditional on empirical parents and retrospectively selected fixed settings; pointwise percentile, no coverage or simultaneous-inference guarantee",
        "centers": "original saved-parent scalar points; never bootstrap means or medians",
        "cross_budget_paired_interval": False,
        "sign_tolerance": {"value": TOL, "purpose": "numerical display classification only, not alpha/effect size/equivalence margin"}}
    output.mkdir(parents=True, exist_ok=True)
    tables = {"all_scalar_summaries.csv": all_scalars, "absolute_locked_metrics.csv": absolute,
              "paired_comparisons_all.csv": paired,
              "paired_reset_improvements.csv": [row for row in paired if row["reference"] == "reset_160k"],
              "paired_raw_reset_improvements.csv": [row for row in paired if row["comparison"] == "Reset_vs_Raw"],
              "paired_dual_controls.csv": controls,
              "primary_rate_results.csv": [row for row in absolute if row["rate"] == .0001],
              "support_summaries.csv": supports, "verified_job_identities.csv": identities, "inventory.csv": inventory}
    for filename, records in tables.items():
        write_csv(output / filename, records)
    summary["output_csv_sha256"] = {filename: sha256(output / filename) for filename in tables}
    (output / "summary.json").write_text(json_text(summary), encoding="utf-8")
    (output / "findings_ko.txt").write_text(findings_text(summary, absolute, paired, controls), encoding="utf-8")
    print(f"Aggregate status={summary['status']}: {len(identities)}/40 complete jobs, {len(absolute)} absolute metric rows, {len(paired)} paired rows.", flush=True)
    if args.require_complete and not complete:
        raise SystemExit("Incomplete production inventory: outputs are explicitly partial, not final.")


if __name__ == "__main__":
    main()
