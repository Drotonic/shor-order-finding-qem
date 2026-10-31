#!/usr/bin/env python3
"""Aggregate fixed-beta C1 all-zero-ridge results with explicit exact reuse.

Imports only pinned, pure summary/audit helpers. No fitting code or sampler is
called. Copied unchanged methods and existing controls remain labelled as reuse,
and finalized paired records are never reported as new noisy shots or new draws.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results" / "c1_all_zero_fixed_20261018"
APPROVED = ROOT / "analysis" / "c1_all_zero_fixed_20261018.json"
HELPER_PATH = ROOT / "analysis" / "summarize_c1_primary_bootstrap_20261018.py"
HELPER_SHA = "8f4ba37e1d0e30ffe076bbdf706bc63dd718a0fbe5a61719f9cbc2a670277fbd"
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError("Pinned immutable statistical/origin-audit helper changed")
_spec = importlib.util.spec_from_file_location("c1_all_zero_pinned_summary_helpers", HELPER_PATH)
helper = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helper
_spec.loader.exec_module(helper)

METHODS, METRICS, CASES, TOL = helper.METHODS, helper.METRICS, helper.CASES, helper.TOL
check, sha256, json_text = helper.check, helper.sha256, helper.json_text
distribution_summary, assert_close = helper.distribution_summary, helper.assert_close


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def exact_array(actual, expected, message):
    check(helper.array_sha(actual) == helper.array_sha(expected), message)


def context(job, configuration_id, identity):
    return {"configuration_id": configuration_id, **job, "rate_percent": 100 * job["rate"], "identity": identity}


def validate_configuration(configuration, approved, job, expected_replicates):
    for key, value in job.items():
        check(configuration[key] == value, f"New job {key} differs from planned scope")
    check(configuration["configuration_id"] == approved["configuration_id"]
          and configuration["ridges"] == dict(zip(METHODS, (0., 0., 0., 0.)))
          and configuration["weight_exponent"] == 2 and configuration["dimension"] == 16384
          and configuration["fitting_floor"] == 1 / 16384, "All-zero scientific profile mismatch")
    check(configuration["fit_lambdas"] == approved["effective_scales"]
          and configuration["single_gamma_grid"] == approved["single_gamma_grid"]
          and configuration["dual_gamma_grid"] == approved["dual_gamma_grid"], "Scale/gamma grids changed")
    per_scale = 20000 if job["budget"] == "equal_shot" else 11188
    check(configuration["per_scale_shots"] == per_scale and configuration["zne_circuit_shots"] == per_scale * 8
          and configuration["direct_shots"] == 160000 and configuration["replicates"] == expected_replicates
          and configuration["batch_size"] == 25
          and configuration["seed"] == (20260722 if job["budget"] == "equal_shot" else 20260804),
          "Shot/RNG/replicate contract changed")
    check(configuration["native_argmax"] is True and configuration["observed_support_reselected"] is True
          and configuration["gamma_refitted_each_replicate"] is True, "Retuning or support/rate-refitting contract changed")
    check(configuration["paired_new_dual_vs_mixed_signs"]["positive_favors"] == "all_zero_dual"
          and configuration["new_independent_bootstrap_resamples"] == 0
          and configuration["resamples_exactly_coupled_to_completed_source"] is True
          and configuration["no_cross_budget_paired_interval"] is True, "Same-draw contrast/reuse contract changed")
    specifications = configuration["route_specifications"]
    check(len(specifications) == 4 and {row["route"] for row in specifications} == set(METHODS)
          and all(row["method"] == row["route"] and row["ridge"] == 0. for row in specifications),
          "Route-specific scientific settings differ from the all-zero map")


def verify_declared_files(jobdir, manifest):
    for filename, declared in manifest["artifacts"].items():
        digest = declared["sha256"] if isinstance(declared, dict) else declared
        path = jobdir / filename
        check(path.is_relative_to(jobdir) and path.is_file() and sha256(path) == digest, f"New output hash mismatch: {path}")


def audit_completed_job(jobdir, job, approved, approved_sha, *, expected_replicates):
    manifest_path = jobdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check(manifest["status"] == "complete", "Incomplete new job")
    configuration, provenance = manifest["configuration"], manifest["provenance"]
    validate_configuration(configuration, approved, job, expected_replicates)
    identity = hashlib.sha256(json_text({"configuration": configuration, "provenance": provenance}).encode()).hexdigest()
    check(identity == manifest["identity"], "New configuration/source identity digest mismatch")
    if expected_replicates == 10000:
        check(manifest["final_10000_contract"] is True, "New final-10000 contract incomplete")
    check(manifest["new_quantum_shots"] == 0 and manifest["new_noisy_circuit_simulations"] == 0
          and manifest["new_independent_bootstrap_resamples"] == 0
          and manifest["old_interval_endpoints_copied"] is False
          and manifest["intervals_to_be_aggregated_from_actual_paired_scalar_arrays"] is True,
          "Unexpected quantum-shot, independent-sample, or old-interval-copy claim")
    helper.verify_entry(provenance["approved_config"], context="all-zero approved configuration")
    check(provenance["approved_config"]["sha256"] == approved_sha, "Wrong all-zero configuration snapshot")
    for entry in provenance["sources"]:
        helper.verify_entry(entry, context="new all-zero producer source")
    if "inputs" in provenance:
        for entry in provenance["inputs"]:
            helper.verify_entry(entry, context="saved-count input")
    if "frozen_point_pack" in provenance:
        helper.verify_entry(provenance["frozen_point_pack"], context="all-zero frozen point manifest")
        pack = provenance["frozen_point_pack"]
        if "points_path" in pack:
            check(sha256(Path(pack["points_path"])) == pack["points_sha256"], "All-zero point CSV hash mismatch")
    source_entry = provenance["source_completed_job"]
    helper.verify_entry(source_entry, context="immutable completed mixed-ridge source")
    source_manifest_path = Path(source_entry["path"])
    if not source_manifest_path.is_absolute():
        source_manifest_path = ROOT / source_manifest_path
    expected_source = ROOT / approved["previous_results"] / job["budget"] / job["instance"] / f"rate_{helper.rate_key(job['rate'])}" / "manifest.json"
    check(source_manifest_path.resolve() == expected_source.resolve(), "Mixed source is not the approved same-condition origin")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    check(source_manifest["identity"] == source_entry["identity"], "Claimed mixed-source identity mismatch")
    check(source_entry["configuration_id"] == source_manifest["configuration"]["configuration_id"] == approved["previous_profile"],
          "Mixed-source profile is mislabelled")
    # The old validator is intentionally used ONLY for the old mixed profile,
    # never for validation of this new all-zero configuration.
    source = helper.audit_completed_job(source_manifest_path.parent, job, sha256(helper.APPROVED), expected_replicates=10000)
    check(source["identity"] == source_entry["identity"], "Mixed source did not pass its own immutable profile audit")
    verify_declared_files(jobdir, manifest)
    started = json.loads((jobdir / "started_manifest.json").read_text(encoding="utf-8"))
    check(started["identity"] == identity and started["configuration"] == configuration and started["provenance"] == provenance,
          "New started/completed identities disagree")
    parent = load_archive(jobdir / "parent_estimates.npz")
    cache = load_archive(jobdir / "scalar_bootstrap_draws.npz")
    for archive in (parent, cache):
        check(helper.unpack_string(archive["identity"]) == identity
              and json.loads(helper.unpack_string(archive["configuration_json"])) == configuration,
              "New NPZ identity/configuration mismatch")
        check(helper.unpack_string(archive["source_job_identity"]) == source["identity"], "NPZ mixed-source identity mismatch")
    routes = [str(value) for value in cache["routes"]]
    quantities = [str(value) for value in cache["quantities"]]
    direct_routes = [str(value) for value in cache["direct_routes"]]
    check(routes == list(METHODS) and quantities == source["quantities"] and direct_routes == source["direct_routes"]
          and cache["direct_metrics"].astype(str).tolist() == list(METRICS), "New/source scalar axes are incompatible")
    check(routes == configuration["routes"] and quantities == configuration["quantities"]
          and direct_routes == configuration["direct_routes"]
          and int(cache["replicates"]) == expected_replicates and int(cache["batch_size"]) == 25
          and int(cache["seed"]) == configuration["seed"] and len(cache["control_routes"]) == 0,
          "New NPZ axes or sampling metadata differ from frozen configuration")
    for key in ("routes", "quantities", "direct_routes", "direct_metrics"):
        exact_array(parent[key], cache[key], f"New parent/cache axis mismatch: {key}")
    b = expected_replicates
    check(cache["scalar_draws"].shape == (4, len(quantities), b)
          and parent["scalar_draws"].shape == (4, len(quantities), 1)
          and cache["direct_draws"].shape == (len(direct_routes), 2, b)
          and cache["selected_indices"].shape == (4, b)
          and cache["selected_parameters"].shape == (4, 2, b)
          and cache["packed_observed_support"].shape == (b, 2048)
          and cache["support_sizes"].shape == (b,), "New full-cache shape/inventory mismatch")
    qindex = {value: i for i, value in enumerate(quantities)}
    for index, route in enumerate(METHODS[:3]):
        si = source["rindex"][route]
        for key in ("scalar_draws", "selected_indices", "selected_parameters"):
            exact_array(cache[key][index], source["cache"][key][si, ..., :b], f"Unchanged method was not reused exactly: {route}/{key}")
        exact_array(parent["scalar_draws"][index], source["parent"]["scalar_draws"][si], f"Unchanged parent was relabelled/refitted: {route}")
        for prefix in ("before__", "after__"):
            exact_array(parent[prefix + route], source["parent"][prefix + route], f"Unchanged full parent vector differs: {route}")
    for key in ("direct_draws",):
        exact_array(cache[key], source["cache"][key][..., :b], "Direct baseline results changed instead of exact reuse")
        exact_array(parent[key], source["parent"][key], "Direct parent centers changed")
    check(configuration["direct_route_origins"] == {route: "exact_reuse_completed_mixed_profile" for route in direct_routes},
          "Direct-route reuse origins are mislabelled")
    check(configuration["scientific_direct_routes"] == (["raw_160k", "reset_160k"] if job["budget"] == "equal_shot" else ["reset_160k"]),
          "Replay-only 20k context is incorrectly presented as a new scientific budget")
    for key in ("support_sizes", "packed_observed_support"):
        exact_array(cache[key], source["cache"][key][:b], "New/source empirical supports do not match actual same draws")
    mixed_index = source["rindex"]["dual_exponential_zne"]
    for new_key, old_key in (("source_mixed_dual_scalar_draws", "scalar_draws"),
                             ("source_mixed_dual_selected_indices", "selected_indices"),
                             ("source_mixed_dual_selected_parameters", "selected_parameters")):
        exact_array(cache[new_key], source["cache"][old_key][mixed_index, ..., :b], "Stored mixed-dual reference is not exact source data")
    control_reuse = job["budget"] == "equal_shot" and job["rate"] == .0001
    for index, route in enumerate(METHODS):
        expected_origin = {"operation": "exact_reuse_completed_mixed_profile", "source_route": route,
                           "source_ridge": 0., "new_ridge": 0.}
        if index == 3:
            expected_origin = {"operation": "exact_reuse_same_draw_zero_ridge_control" if control_reuse else "new_dual_refit_on_replayed_source_counts",
                               "source_route": helper.CONTROLS[0] if control_reuse else route,
                               "source_ridge": 0. if control_reuse else 1e-6, "new_ridge": 0.}
        check(configuration["route_origins"][route] == expected_origin and manifest["route_origins"][route] == expected_origin,
              "Primary estimator reuse/refit origin mislabelled")
    if control_reuse:
        control_index = source["rindex"][helper.CONTROLS[0]]
        for key in ("scalar_draws", "selected_indices", "selected_parameters"):
            exact_array(cache[key][3], source["cache"][key][control_index, ..., :b], "Existing zero-ridge control was not reused exactly")
        exact_array(parent["scalar_draws"][3], source["parent"]["scalar_draws"][control_index], "Existing zero-ridge parent control changed")
        for prefix in ("before__", "after__"):
            exact_array(parent[prefix + METHODS[3]], source["parent"][prefix + helper.CONTROLS[0]], "Copied full zero-ridge parent control changed")
    for key in ("target", "scale_parents", "reset_reference", "fit_lambdas", "ideal_gap_denominator"):
        exact_array(parent[key], source["parent"][key], f"Same saved-parent reference changed: {key}")
    for prefix in ("before", "after"):
        exact_array(parent[f"source_mixed_dual_{prefix}"], source["parent"][f"{prefix}__dual_exponential_zne"], "Stored mixed-dual parent vector is not source data")
    metric_indices = [qindex[name] for name in METRICS]
    scalar_metrics = cache["scalar_draws"][:, metric_indices]
    parent_metrics = parent["scalar_draws"][:, metric_indices, 0]
    mixed_metrics = cache["source_mixed_dual_scalar_draws"][metric_indices]
    mixed_parent_metrics = source["parent_metrics"][mixed_index]
    reset_index = direct_routes.index("reset_160k")
    reset = cache["direct_draws"][reset_index]
    expected_reset = np.stack((reset[0] - scalar_metrics[:, 0], scalar_metrics[:, 1] - reset[1]), axis=1)
    paired_residual = assert_close(cache["paired_reset_improvements"], expected_reset, message="New Reset contrasts are not actual paired samples")
    expected_mixed = np.stack((mixed_metrics[0] - scalar_metrics[3, 0], scalar_metrics[3, 1] - mixed_metrics[1]))
    paired_residual = max(paired_residual, assert_close(cache["paired_new_dual_vs_mixed"], expected_mixed,
          message="Zero-vs-mixed contrast has wrong orientation or independent samples"))
    check(np.isfinite(scalar_metrics).all() and cache["paired_control_improvements"].shape == (0, 2, b), "Unexpected nonfinite metrics/legacy control axis")
    order = {"n119_a13": 4, "n119_a50": 2, "n119_a16": 6}[job["instance"]]
    parent_residual = 0.
    for index, route in enumerate(METHODS):
        actual = helper.independent_probability_metrics(parent[f"after__{route}"], parent["target"], order)
        parent_residual = max(parent_residual, assert_close(actual, parent_metrics[index], message="New parent vector/locked metrics disagree", atol=1e-11))
    golden_rows = read_csv(BASE / "point_estimates.csv")
    for index, route in enumerate(METHODS):
        rows = [row for row in golden_rows if row["instance"] == job["instance"] and float(row["rate"]) == job["rate"] and row["method"] == route]
        check(len(rows) == 1 and rows[0]["configuration_id"] == approved["configuration_id"], "Missing frozen all-zero parent center")
        assert_close(parent_metrics[index], np.asarray([float(rows[0][metric]) for metric in METRICS]),
                     message="Bootstrap parent center differs from original frozen all-zero points", atol=1e-11)
        check(int(parent["selected_indices"][index, 0]) == int(rows[0]["selected_gamma_index"]), "Frozen native gamma selection changed")
    # Exact checkpoint-to-final-cache and source-draw coupling is checked below.
    audit_checkpoint_chain(jobdir, manifest, cache, identity, source, b)
    return {"manifest": manifest, "configuration": configuration, "identity": identity, "parent": parent, "cache": cache,
            "routes": routes, "quantities": quantities, "direct_routes": direct_routes, "qindex": qindex,
            "metric_draws": scalar_metrics, "parent_metrics": parent_metrics,
            "mixed_metric_draws": mixed_metrics, "mixed_parent_metrics": mixed_parent_metrics,
            "source_identity": source["identity"], "source_manifest": helper.path_label(source_manifest_path),
            "source_manifest_sha256": sha256(source_manifest_path), "control_reuse": control_reuse,
            "new_dual_fit_replicates": 0 if control_reuse else b,
            "reused_dual_control_replicates": b if control_reuse else 0,
            "parent_metric_residual": parent_residual, "paired_residual": paired_residual,
            "manifest_path": helper.path_label(manifest_path), "manifest_sha256": sha256(manifest_path)}


def audit_checkpoint_chain(jobdir, manifest, cache, identity, source, b):
    """Validate exact final segments and source checkpoint references.

    Source-coupling metadata is deliberately checked independently of the
    producer. The old mixed source already passed its own full checkpoint audit.
    """
    previous, expected = None, 0
    new_fits, reused_zero, independently_replayed = 0, 0, 0
    control_reuse = bool(manifest["configuration"]["reused_dual_zero_replicates_expected"])
    for item in manifest["checkpoint_files"]:
        check(Path(item["path"]).name == item["path"], "Escaping new checkpoint path")
        path = jobdir / "batches" / item["path"]
        check(sha256(path) == item["sha256"], "New checkpoint file hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(helper.unpack_string(archive["checkpoint_json"]))
        first, stop = int(meta["start"]), int(meta["stop"])
        check(meta["identity"] == identity and first == expected and first == item["start"] and stop == item["stop"]
              and first < stop <= b and stop - first <= 25 and meta["previous_checkpoint_sha256"] == previous,
              "New checkpoint identity/order/coverage mismatch")
        for key, digest in meta["array_hashes"].items():
            check(key in cache, f"New checkpoint key missing from final cache: {key}")
            segment = cache[key][first:stop] if key in ("support_sizes", "packed_observed_support") else cache[key][..., first:stop]
            check(helper.array_sha(segment) == digest, f"New final array differs from its checkpoint: {key}")
        source_record = meta["source_checkpoint"]
        source_path = Path(source_record["path"])
        check(sha256(source_path) == source_record["sha256"] and source_record["identity"] == source["identity"],
              "Wrong immutable mixed-source checkpoint reference")
        with np.load(source_path, allow_pickle=False) as archive:
            source_meta = json.loads(helper.unpack_string(archive["checkpoint_json"]))
        check(source_meta["identity"] == source["identity"] and source_meta["start"] == first and source_meta["stop"] >= stop,
              "New/source checkpoint replicate windows do not align")
        expected_source = source["manifest"]["checkpoint_files"][first // 25]
        check(source_record["sha256"] == expected_source["sha256"] and source_record["start"] == first
              and source_record["stop"] == source_meta["stop"], "Source checkpoint does not match canonical source window")
        check(meta["rng_before"] == source_meta["rng_before"] and meta["rng_after"] == source_meta["rng_after"]
              and meta["draw_counts"] == source_meta["draw_counts"], "New dual fit is not coupled to exact original count/RNG record")
        check(meta["new_independent_bootstrap_resamples"] == 0
              and meta["count_hashes_independently_replayed"] == (not control_reuse)
              and meta["new_dual_refits"] == (0 if control_reuse else stop - first)
              and meta["reused_dual_zero_replicates"] == (stop - first if control_reuse else 0),
              "Checkpoint refit/reuse/sample accounting is incorrect")
        new_fits += meta["new_dual_refits"]
        reused_zero += meta["reused_dual_zero_replicates"]
        independently_replayed += (stop - first) if meta["count_hashes_independently_replayed"] else 0
        previous, expected = item["sha256"], stop
    check(expected == b, "New checkpoint chain does not cover all declared replicates")
    check(new_fits == manifest["new_dual_refits"] == manifest["configuration"]["new_dual_refits_expected"]
          and reused_zero == manifest["reused_dual_zero_replicates"] == manifest["configuration"]["reused_dual_zero_replicates_expected"]
          and manifest["unchanged_c1_model_replicates_reused"] == 3 * b
          and independently_replayed == manifest["replay"]["count_replayed_replicates"],
          "Final refit/reuse accounting does not match exact checkpoint reductions")


def aggregate_job(job, data, configuration_id):
    ctx = context(job, configuration_id, data["identity"])
    parent, cache = data["parent"], data["cache"]
    absolute, all_scalars, paired, contrasts, mixed_reference = [], [], [], [], []
    for ri, route in enumerate(METHODS):
        origin = "exact_reuse_unchanged_method" if ri < 3 else "exact_reuse_existing_zero_control" if data["control_reuse"] else "new_dual_zero_refit_on_identical_draws"
        for qi, quantity in enumerate(data["quantities"]):
            row = {**ctx, "route": route, "quantity": quantity, "origin": origin,
                   "source_mixed_identity": data["source_identity"],
                   **distribution_summary(cache["scalar_draws"][ri, qi], parent["scalar_draws"][ri, qi, 0], strict=quantity in METRICS)}
            all_scalars.append(row)
            if quantity in METRICS:
                absolute.append({**row, "metric": quantity, "circuit_shots": data["configuration"]["zne_circuit_shots"]})
    for ri, route in enumerate(data["direct_routes"]):
        for mi, metric in enumerate(METRICS):
            row = {**ctx, "route": route, "quantity": metric, "origin": "exact_reuse_direct_baseline",
                   "source_mixed_identity": data["source_identity"],
                   "role": "context_only_20k" if route.endswith("_20k") else "scientific_direct_160k",
                   **distribution_summary(cache["direct_draws"][ri, mi], parent["direct_draws"][ri, mi, 0])}
            all_scalars.append(row)
            if route.endswith("_160k"):
                absolute.append({**row, "metric": metric, "circuit_shots": 160000})
    reset_i = data["direct_routes"].index("reset_160k")
    reset_point = parent["direct_draws"][reset_i, :, 0]
    for ri, route in enumerate(METHODS):
        points = (reset_point[0] - data["parent_metrics"][ri, 0], data["parent_metrics"][ri, 1] - reset_point[1])
        for mi, metric in enumerate(METRICS):
            paired.append({**ctx, "route": route, "reference": "reset_160k", "comparison": f"{route}_vs_Reset",
                           "metric": metric, "positive_favors": route, "pairing": "actual same-replicate saved-count draws",
                           **distribution_summary(cache["paired_reset_improvements"][ri, mi], points[mi])})
    if "raw_160k" in data["direct_routes"]:
        raw_i = data["direct_routes"].index("raw_160k")
        samples = (cache["direct_draws"][raw_i, 0] - cache["direct_draws"][reset_i, 0],
                   cache["direct_draws"][reset_i, 1] - cache["direct_draws"][raw_i, 1])
        points = (parent["direct_draws"][raw_i, 0, 0] - reset_point[0], reset_point[1] - parent["direct_draws"][raw_i, 1, 0])
        for mi, metric in enumerate(METRICS):
            paired.append({**ctx, "route": "reset_160k", "reference": "raw_160k", "comparison": "Reset_vs_Raw",
                           "metric": metric, "positive_favors": "reset_160k", "pairing": "unchanged original independent Raw stream",
                           **distribution_summary(samples[mi], points[mi])})
    points = (data["mixed_parent_metrics"][0] - data["parent_metrics"][3, 0],
              data["parent_metrics"][3, 1] - data["mixed_parent_metrics"][1])
    for mi, metric in enumerate(METRICS):
        samples = cache["paired_new_dual_vs_mixed"][mi]
        row = {**ctx, "route": "dual_exponential_zne", "reference": "previous_mixed_dual_ridge1e-6",
               "comparison": "all_zero_dual_vs_previous_mixed_dual", "metric": metric,
               "positive_favors": "all_zero_dual", "pairing": "hash-verified identical source replicate windows and observed supports",
               "source_mixed_identity": data["source_identity"], **distribution_summary(samples, points[mi])}
        inverse = distribution_summary(-samples, -points[mi])
        contrasts.append({**row, **{f"mixed_improvement_{key}": value for key, value in inverse.items()}})
        paired.append(row)
        mixed_reference.append({**ctx, "route": "previous_mixed_dual_ridge1e-6", "metric": metric,
                                "source_mixed_identity": data["source_identity"],
                                **distribution_summary(data["mixed_metric_draws"][mi], data["mixed_parent_metrics"][mi])})
    support = {**ctx, **distribution_summary(cache["support_sizes"], np.count_nonzero(np.sum(parent["scale_parents"], axis=0)))}
    audit = {**ctx, "manifest": data["manifest_path"], "manifest_sha256": data["manifest_sha256"],
             "source_mixed_manifest": data["source_manifest"], "source_mixed_manifest_sha256": data["source_manifest_sha256"],
             "source_mixed_identity": data["source_identity"], "replicates": cache["scalar_draws"].shape[-1],
             "new_dual_fit_replicates": data["new_dual_fit_replicates"],
             "reused_dual_control_replicates": data["reused_dual_control_replicates"],
             "reused_unchanged_model_replicates": 3 * cache["scalar_draws"].shape[-1],
             "parent_metric_max_residual": data["parent_metric_residual"], "paired_max_residual": data["paired_residual"]}
    return all_scalars, absolute, paired, contrasts, mixed_reference, support, audit


def findings_text(summary, absolute, paired, contrasts):
    lines = ["β=2 / 모든 ridge=0 C1 bootstrap 집계", "=" * 55,
             f"상태: {summary['status']} — 완료·검증 {summary['completed_jobs']}/40 조건, 조건당 {summary['replicates_per_job']}회.",
             f"완성된 paired 결과기록 {summary['finalized_paired_replicate_records']}개 중 새 dual 적합은 {summary['new_dual_fit_replicates']}회, 기존 dual0 control 재사용은 {summary['reused_dual_control_replicates']}회다.",
             "나머지 세 모델·직접 baseline·관측 support는 원래 같은 입력의 결과를 정확히 재사용했다.",
             "모든 중심값은 원래 부모 데이터의 점추정이다. 구간은 저장 경험분포와 사후 지정된 고정 설정에 조건부이며 coverage 보장이 아니다.", ""]
    if summary["smoke_only"]:
        return "\n".join(lines + ["SMOKE 구현 검증용: 최종 10,000회 결과가 아니므로 과학적 해석과 원고 사용을 보류한다.", ""])
    def find(rows, **keys):
        return next((row for row in rows if all(row.get(key) == value for key, value in keys.items())), None)
    for budget in ("equal_shot", "gate_matched"):
        lines.append("[등샷]" if budget == "equal_shot" else "[gate-matched 재표본화: ZNE89,504 vs Reset160,000 shots]")
        for method in METHODS:
            for metric in METRICS:
                result = find(absolute, budget=budget, instance="n119_a16", rate=.0001, route=method, metric=metric)
                if result is None:
                    continue
                gain = find(paired, budget=budget, instance="n119_a16", rate=.0001, route=method, reference="reset_160k", metric=metric)
                lines.append(f"r6 {method}/{metric}: 점={helper.fmt(result['point'])}, 95%[{helper.fmt(result['ci_low'])},{helper.fmt(result['ci_high'])}]; "
                             f"Reset 대비 개선량 {helper.fmt(gain['point'])}, 95%[{helper.fmt(gain['ci_low'])},{helper.fmt(gain['ci_high'])}].")
        for metric in METRICS:
            row = find(contrasts, budget=budget, instance="n119_a16", rate=.0001, metric=metric)
            if row is not None:
                lines.append(f"r6 dual0−이전mixed 비교({metric}, 양수=all-zero에 유리): {helper.fmt(row['point'])}, "
                             f"95%[{helper.fmt(row['ci_low'])},{helper.fmt(row['ci_high'])}], 실제 범위[{helper.fmt(row['minimum'])},{helper.fmt(row['maximum'])}].")
        lines.append("")
    lines += ["[0,0] 구간을 동등성·안정성·양의 peak 분리로 해석하지 않는다. CSV에 exact/numerical 부호, 꼬리, 극값을 보존했다.",
              "재표본 빈도는 실제 장비의 failure 확률이 아니다. 두 예산 사이의 paired 구간은 만들지 않았다.",
              "새 noisy shots·물리 오류율·성능 지표·복구 분석을 추가하지 않았고 기존 분석과 원고는 변경하지 않았다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=BASE / "bootstrap_10000")
    parser.add_argument("--output-dir", type=Path, default=BASE / "aggregate")
    parser.add_argument("--expected-replicates", type=int, default=10000)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    check(ROOT.name == "paper-option2" and subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "option2-metric-locked-revision", "Wrong isolated worktree or branch")
    artifacts, output = args.artifacts.resolve(), args.output_dir.resolve()
    smoke_input = args.allow_smoke and args.expected_replicates < 10000 and artifacts.is_relative_to(Path("/tmp")) and artifacts != Path("/tmp")
    check((artifacts.is_relative_to(BASE) or smoke_input) and output.is_relative_to(BASE) and output != BASE, "Output/input isolation failure")
    check(args.expected_replicates == 10000 or args.allow_smoke, "Explicit smoke opt-in required")
    approved = json.loads(APPROVED.read_text(encoding="utf-8"))
    check(approved["configuration_id"] == "c1_beta2_all_ridges_zero_20261018" and approved["estimator"] == "C1"
          and approved["weight_exponent"] == 2 and approved["ridges"] == dict(zip(METHODS, (0., 0., 0., 0.))),
          "Wrong approved all-zero profile")
    approved_sha = sha256(APPROVED)
    all_scalars, absolute, paired, contrasts, mixed, supports, audits, inventory = [], [], [], [], [], [], [], []
    for job in helper.expected_jobs():
        jobdir = artifacts / job["budget"] / job["instance"] / f"rate_{helper.rate_key(job['rate'])}"
        manifest_path = jobdir / "manifest.json"
        if not manifest_path.is_file():
            inventory.append({**job, "status": "incomplete_or_not_started", "jobdir": helper.path_label(jobdir)})
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            inventory.append({**job, "status": str(manifest.get("status", "incomplete")), "jobdir": helper.path_label(jobdir)})
            continue
        data = audit_completed_job(jobdir, job, approved, approved_sha, expected_replicates=args.expected_replicates)
        result = aggregate_job(job, data, approved["configuration_id"])
        for target, values in zip((all_scalars, absolute, paired, contrasts, mixed), result[:5]):
            target.extend(values)
        supports.append(result[5]); audits.append(result[6])
        inventory.append({**job, "status": "complete_and_verified", "identity": data["identity"], "jobdir": helper.path_label(jobdir)})
        print(f"Verified all-zero {job['budget']}/{job['instance']}/{job['rate']:g}, B={args.expected_replicates}", flush=True)
        del data
    complete = len(audits) == 40 and args.expected_replicates == 10000
    summary = {"schema_version": 1, "configuration_id": approved["configuration_id"], "approved_config_sha256": approved_sha,
        "status": "complete" if complete else "partial", "publication_ready": complete, "completed_jobs": len(audits), "expected_jobs": 40,
        "completed_equal_shot_jobs": sum(row["budget"] == "equal_shot" for row in audits),
        "completed_gate_matched_jobs": sum(row["budget"] == "gate_matched" for row in audits),
        "replicates_per_job": args.expected_replicates, "finalized_paired_replicate_records": len(audits) * args.expected_replicates,
        "new_dual_fit_replicates": sum(row["new_dual_fit_replicates"] for row in audits),
        "reused_dual_control_replicates": sum(row["reused_dual_control_replicates"] for row in audits),
        "reused_unchanged_model_replicates": sum(row["reused_unchanged_model_replicates"] for row in audits),
        "smoke_only": args.expected_replicates != 10000, "absolute_locked_metric_rows": len(absolute),
        "paired_comparison_rows": len(paired), "dual_vs_mixed_rows": len(contrasts),
        "parent_metric_max_residual": max((row["parent_metric_max_residual"] for row in audits), default=0),
        "paired_max_residual": max((row["paired_max_residual"] for row in audits), default=0),
        "inventory": inventory, "verified_job_origins": audits,
        "generator_sha256": sha256(Path(__file__)), "pinned_statistical_helper_sha256": HELPER_SHA,
        "created_utc": datetime.now(timezone.utc).isoformat(), "new_quantum_shots": 0, "new_noisy_circuit_simulations": 0,
        "new_random_bootstrap_draws_claim": False, "metric_definitions_changed": False, "cross_budget_paired_interval": False,
        "new_independent_bootstrap_resamples": 0,
        "interval_scope": "Conditional pointwise paired-percentile intervals; requested after inspecting previous results; no post-selection coverage guarantee",
        "centers": "saved-parent plug-in points, never bootstrap means/medians", "sign_tolerance": TOL,
        "zero_interval_warning": "Central95% [0,0] can hide nonzero tails and does not imply equivalence or positive gap"}
    output.mkdir(parents=True, exist_ok=True)
    tables = {"all_scalar_summaries.csv": all_scalars, "absolute_locked_metrics.csv": absolute,
              "paired_comparisons_all.csv": paired,
              "paired_reset_improvements.csv": [row for row in paired if row["reference"] == "reset_160k"],
              "paired_raw_reset_improvements.csv": [row for row in paired if row["comparison"] == "Reset_vs_Raw"],
              "paired_all_zero_dual_vs_mixed.csv": contrasts, "mixed_dual_reference_metrics.csv": mixed,
              "primary_rate_results.csv": [row for row in absolute if row["rate"] == .0001],
              "support_summaries.csv": supports, "verified_job_origins.csv": audits, "inventory.csv": inventory}
    for filename, rows in tables.items():
        helper.write_csv(output / filename, rows)
    summary["output_csv_sha256"] = {filename: sha256(output / filename) for filename in tables}
    (output / "summary.json").write_text(json_text(summary), encoding="utf-8")
    (output / "findings_ko.txt").write_text(findings_text(summary, absolute, paired, contrasts), encoding="utf-8")
    print(f"All-zero aggregate {summary['status']}: {len(audits)}/40 jobs; new dual fits={summary['new_dual_fit_replicates']}; exact control reuse={summary['reused_dual_control_replicates']}", flush=True)
    if args.require_complete and not complete:
        raise SystemExit("All-zero inventory incomplete; written outputs are explicitly partial.")


if __name__ == "__main__":
    main()
