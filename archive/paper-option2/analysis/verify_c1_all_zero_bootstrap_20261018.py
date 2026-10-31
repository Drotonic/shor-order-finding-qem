#!/usr/bin/env python3
"""Independent cache-reuse and native-inference audit of the all-zero C1 run.

Read-only, apart from explicitly requested separate smoke runner invocations.
This verifier itself creates no samples, circuits, intervals or output files;
exact source-checkpoint count replay is solely a reproducibility diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/c1_all_zero_fixed_20261018"
CONFIG = ROOT / "analysis/c1_all_zero_fixed_20261018.json"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
SOURCE = ROOT / "analysis_results/c1_primary_fixed_20261018/bootstrap_10000"
REFERENCE_SHA = "8ab632e828e1d11cf6289ef7573e440f81ede21f058216921cd8d42394a4eb76"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def equal(actual, expected, message, atol=0, rtol=0):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), message)


def bitwise(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C"), message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def startup():
    prior_path = ROOT / "analysis/verify_c1_primary_bootstrap_20261018.py"
    check(sha(prior_path) == REFERENCE_SHA, "Prior independent verifier was modified")
    old = module("allzero_verify_prior_reference", prior_path)
    estimators, bvr, core, reference, mixed = old.modules_and_config()
    approved = json.loads(CONFIG.read_text())
    check(approved["configuration_id"] == "c1_beta2_all_ridges_zero_20261018" and approved["ridges"] == dict.fromkeys(METHODS, 0.) and approved["weight_exponent"] == 2., "New profile is not the approved all-zero C1")
    for name in ("effective_scales", "single_gamma_grid", "dual_gamma_grid", "shots_per_scale", "direct_shots", "dimension", "exponential_offset", "metrics"):
        check(approved[name] == mixed[name], "New profile changed non-ridge scientific field: " + name)
    check(approved["new_quantum_shots"] == 0 and approved["metric_definitions_changed"] is False and approved["canonical_manuscript_updated"] is False, "All-zero profile invents data/metrics/manuscript migration")
    pack = json.loads((BASE / "manifest.json").read_text())
    check(pack["configuration"] == approved and pack["point_rows"] == 148 and pack["case_count"] == 37, "New golden point-pack identity/coverage wrong")
    for path, digest in pack["sources"].items():
        check(sha(ROOT / path) == digest, "New golden source changed")
    for filename, digest in pack["outputs"].items():
        check(sha(BASE / filename) == digest, "New golden table changed")
    golden_rows = rows(BASE / "point_estimates.csv")
    golden = {(r["method"], r["case_id"]): r for r in golden_rows}
    check(len(golden_rows) == len(golden) == 148 and all(float(r["ridge"]) == 0. and float(r["weight_exponent"]) == 2. and r["bootstrap_status_at_point_freeze"] == "pending" for r in golden_rows), "Golden rows contain old ridge or pre-existing interval claim")
    return old, estimators, bvr, core, reference, approved, golden


def verify_toys(old, estimators, core, reference):
    n = 0
    for dimension in (16, 31):
        for support in (np.array([0, 3, dimension - 1]), np.arange(dimension)):
            scales = np.array([1., 1.35, 1.8, 2.4])
            p = np.zeros((4, dimension))
            i, j = np.indices((4, len(support)))
            frequencies = .1 + ((17 * i + 7 * j + dimension + 3) % 29) / 23
            frequencies[0, 1] = 0
            p[:, support] = frequencies / frequencies.sum(axis=1, keepdims=True)
            fixed, families = old.independent_suite(estimators, scales, toy=True, dual_ridge=0.)
            for method in METHODS:
                native = core.infer_model(method, "C1", fixed_coefficients=fixed, exponential_families=families, support=core.ObservedBatch.from_dense(p[None]))
                before, after, selected = reference.independent_restricted(p, scales, method=method, weight_exponent=2., ridge=0., family=families.get(method))
                equal(native["before"].materialize()[0], before, "All-zero C1 toy not explicit observed-only least squares", atol=5e-11)
                equal(native["after"].materialize()[0], after, "All-zero C1 toy not independent Euclidean simplex", atol=5e-11)
                if selected >= 0:
                    check(int(native["selection"]["selected_index"][0]) == selected, "All-zero C1 toy gamma not actual native optimum")
                n += 1
    print(f"PASS {n} all-zero direct-LS/support/simplex toys, fixed1/D offset and cross-scale zeros", flush=True)


def svd_selected_diagnostic(grid, fit, parent, old):
    """Stable selected-candidate comparison, never a replacement estimator."""
    weights = np.asarray(parent["lambdas"]) ** -2
    weights /= weights.mean()
    max_before = max_after = max_metric = max_condition = 0.
    for b, ids in enumerate(fit["support_ids"]):
        parameters = fit["selected_parameters"][b]
        design = np.exp(-np.asarray(parent["lambdas"])[:, None] * parameters[None])
        weighted = np.sqrt(weights)[:, None] * design
        centered = grid[b][:, ids] - 1 / grid.shape[-1]
        operator = np.linalg.lstsq(weighted, np.eye(len(weights)), rcond=None)[0]
        h = operator.sum(axis=0) * np.sqrt(weights)
        before = 1 / grid.shape[-1] + h @ centered
        after = old.simplex_sorted(before)
        full_after = np.zeros(grid.shape[-1])
        full_after[ids] = after
        original = fit["after"][b]
        max_before = max(max_before, float(np.max(abs(before - fit["before"][b, ids]))))
        max_after = max(max_after, float(np.max(abs(full_after - original))))
        max_metric = max(max_metric, float(np.max(abs(old.metric_rows(full_after, parent["target"], parent["order"]) - old.metric_rows(original, parent["target"], parent["order"])))))
        max_condition = max(max_condition, float(np.linalg.cond(weighted)))
    return {"native_vs_selected_candidate_SVD_before_max": max_before, "native_vs_selected_candidate_SVD_after_max": max_after,
            "native_vs_selected_candidate_SVD_metric_max": max_metric, "selected_weighted_design_condition_max": max_condition,
            "replaces_native_solver": False, "rescans_gamma_using_SVD": False}


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_configuration(c, approved, source_c):
    check(c["configuration_id"] == approved["configuration_id"] and c["ridges"] == dict.fromkeys(METHODS, 0.) and c["weight_exponent"] == 2., "Delta job is not all-zero/beta2")
    check(c["routes"] == list(METHODS) and c["source_mixed_dual_ridge"] == 1e-6, "All-zero primary/control reference scope changed")
    for field in ("budget", "instance", "rate", "order", "dimension", "fit_lambdas", "fitting_floor", "per_scale_shots", "zne_circuit_shots", "direct_shots", "reset_extension_shots", "seed", "batch_size", "single_gamma_grid", "dual_gamma_grid", "direct_routes", "scientific_direct_routes", "direct_metrics", "quantities"):
        check(c[field] == source_c[field], "Delta changed non-ridge source contract: " + field)
    check(c["dimension"] == 16384 and c["fitting_floor"] == 1 / 16384 and c["batch_size"] == 25 and c["native_argmax"] and c["observed_support_reselected"] and c["gamma_refitted_each_replicate"], "Delta estimator offset/native/support/gamma contract changed")
    check(c["new_independent_bootstrap_resamples"] == 0 and c["resamples_exactly_coupled_to_completed_source"] is True and c["no_cross_budget_paired_interval"] is True, "Delta invents independent draws or cross-budget pairing")
    check(c["paired_new_dual_vs_mixed_signs"] == {"tvd": "mixed dual ridge1e-6 minus all-zero dual", "normalized_grouped_peak_margin": "all-zero dual minus mixed dual ridge1e-6", "positive_favors": "all_zero_dual"}, "All-zero versus mixed comparison signs changed")
    check("post-selection" in c["coverage"] and "no independent validation" in c["coverage"], "Data-selected profile falsely labeled independent evidence")
    reused = c["budget"] == "equal_shot" and c["rate"] == .0001
    for method in METHODS[:3]:
        check(c["route_origins"][method] == {"operation": "exact_reuse_completed_mixed_profile", "source_route": method, "source_ridge": 0., "new_ridge": 0.}, "Unchanged method source identity mislabeled")
    check(c["route_origins"][METHODS[3]] == {"operation": "exact_reuse_same_draw_zero_ridge_control" if reused else "new_dual_refit_on_replayed_source_counts", "source_route": "dual_exponential_zne_ridge0_control" if reused else METHODS[3], "source_ridge": 0. if reused else 1e-6, "new_ridge": 0.}, "Dual reuse/refit source identity wrong")
    check(c["new_dual_refits_expected"] == (0 if reused else c["replicates"]) and c["reused_dual_zero_replicates_expected"] == (c["replicates"] if reused else 0), "Dual evaluation/reuse count wrong")
    return reused


def verify_job(folder, state, *, require_final=False):
    old, estimators, bvr, core, reference, approved, golden = state
    started = json.loads((folder / "started_manifest.json").read_text())
    c, provenance = started["configuration"], started["provenance"]
    identity = hashlib.sha256((json.dumps({"configuration": c, "provenance": provenance}, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    check(started["identity"] == identity, "Delta identity does not hash actual configuration/provenance")
    source_entry = provenance["source_completed_job"]
    source_manifest_path = Path(source_entry["path"])
    source_dir = source_manifest_path.parent
    check(source_dir.is_relative_to(SOURCE) and sha(source_manifest_path) == source_entry["sha256"], "Delta references another/changed old profile")
    source_m = json.loads(source_manifest_path.read_text())
    source_c = source_m["configuration"]
    check(source_m["status"] == "complete" and source_m["final_10000_contract"] is True and source_m["identity"] == source_entry["identity"] and source_c["ridges"] == dict(zip(METHODS, (0., 0., 0., 1e-6))) and source_c["replicates"] == 10000, "Delta reused incomplete/misconfigured source")
    reused = check_configuration(c, approved, source_c)
    for entry in provenance["sources"] + provenance["inputs"] + [provenance["approved_config"], provenance["frozen_point_pack"], source_entry]:
        check(sha(entry["path"]) == entry["sha256"], "Delta source/input/approved profile changed")
    for filename, digest in source_m["artifacts"].items():
        check(sha(source_dir / filename) == digest, "Old source artifacts changed")
    m = json.loads((folder / "manifest.json").read_text())
    check(m["status"] == "complete" and m["identity"] == identity and m["configuration"] == c and m["provenance"] == provenance and m["actual_bootstrap_completed"] is True, "Delta job not complete or identity changed")
    n = c["replicates"]
    check(m["new_independent_bootstrap_resamples"] == m["new_quantum_shots"] == m["new_noisy_circuit_simulations"] == 0 and m["old_interval_endpoints_copied"] is False and m["intervals_to_be_aggregated_from_actual_paired_scalar_arrays"] is True, "Delta misrepresents old scalar reuse as new independent shots/intervals")
    check(m["unchanged_c1_model_replicates_reused"] == 3 * n and m["new_dual_refits"] == (0 if reused else n) and m["reused_dual_zero_replicates"] == (n if reused else 0), "Delta effective evaluation/reuse totals wrong")
    check(not require_final or (n == 10000 and not c["smoke"] and m["final_10000_contract"]), "Smoke delta mistaken for final10000")
    for filename, digest in m["artifacts"].items():
        check(sha(folder / filename) == digest, "Delta output hash changed")
    parent, cache = read_npz(folder / "parent_estimates.npz"), read_npz(folder / "scalar_bootstrap_draws.npz")
    source_parent, source_cache = read_npz(source_dir / "parent_estimates.npz"), read_npz(source_dir / "scalar_bootstrap_draws.npz")
    for payload in (parent, cache):
        check(str(payload["identity"]) == identity and str(payload["source_job_identity"]) == source_m["identity"] and json.loads(str(payload["configuration_json"])) == c, "Delta NPZ identity/source/configuration mismatch")
    check(cache["scalar_draws"].shape == (4, 16, n) and cache["routes"].astype(str).tolist() == list(METHODS) and cache["quantities"].astype(str).tolist() == c["quantities"], "Delta primary scalar schema changed")
    check(int(cache["seed"]) == c["seed"] and int(cache["replicates"]) == n and int(cache["batch_size"]) == 25 and len(cache["control_routes"]) == 0, "Delta seed/batch/primary-axis metadata wrong")
    check(np.isfinite(cache["scalar_draws"]).all() and np.isfinite(cache["direct_draws"]).all(), "Delta drops/nonfinite metric or diagnostic")
    for ri, method in enumerate(METHODS):
        indices, parameters = cache["selected_indices"][ri], cache["selected_parameters"][ri]
        if ri < 2:
            equal(indices, np.full(n, -1), "Polynomial assigned adaptive gamma")
            check(np.isnan(parameters).all(), "Polynomial gamma sentinel incorrect")
        elif ri == 2:
            check(np.all((indices >= 0) & (indices < 200)), "Single native candidate index out of bounds")
            equal(parameters[0], np.linspace(.01, 10, 200)[indices], "Single parameters differ from native grid indices")
            check(np.isnan(parameters[1]).all(), "Single estimator has two rates")
        else:
            check(np.all((indices >= 0) & (indices < 19900)), "Zero-dual candidate index out of bounds")
            first, second = np.triu_indices(200, 1)
            grid = np.linspace(0., 10., 200)
            equal(parameters, np.stack([grid[first[indices]], grid[second[indices]]]), "Zero-dual parameters differ from strict increasing native pairs")
    def compare_reuse(new, source, count):
        for key in ("scalar_draws", "selected_indices", "selected_parameters"):
            bitwise(new[key][:3], source[key][:3, ..., :count], "Three unchanged models not bitwise source-identical: " + key)
        for key in ("direct_draws", "support_sizes", "packed_observed_support"):
            expected = source[key][:count] if key in ("support_sizes", "packed_observed_support") else source[key][..., :count]
            bitwise(new[key], expected, "Direct/support array changed during reuse: " + key)
        for new_key, old_key in (("source_mixed_dual_scalar_draws", "scalar_draws"), ("source_mixed_dual_selected_indices", "selected_indices"), ("source_mixed_dual_selected_parameters", "selected_parameters")):
            bitwise(new[new_key], source[old_key][3, ..., :count], "Mixed-dual paired reference altered")
        if reused:
            ix = source_c["routes"].index("dual_exponential_zne_ridge0_control")
            for key in ("scalar_draws", "selected_indices", "selected_parameters"):
                bitwise(new[key][3], source[key][ix, ..., :count], "Existing zero-control not reused bitwise: " + key)
    compare_reuse(cache, source_cache, n)
    compare_reuse(parent, source_parent, 1)
    for method in METHODS[:3]:
        for prefix in ("before__", "after__"):
            bitwise(parent[prefix + method], source_parent[prefix + method], "Unchanged parent vector altered")
    for prefix, source_prefix in (("source_mixed_dual_before", "before__"), ("source_mixed_dual_after", "after__")):
        bitwise(parent[prefix], source_parent[source_prefix + METHODS[3]], "Parent mixed-dual reference changed")
    if reused:
        for prefix in ("before__", "after__"):
            bitwise(parent[prefix + METHODS[3]], source_parent[prefix + "dual_exponential_zne_ridge0_control"], "Zero-control parent vector not reused exactly")
    for key in ("target", "scale_parents", "reset_reference", "reset_extension_parent", "fit_lambdas", "raw_full_original", "reset_full_original"):
        bitwise(parent[key], source_parent[key], "Delta changed actual input/budget center: " + key)
    actual_parent = old.original_parent_data(bvr, c["instance"], c["rate"])
    zero_spec = {"method": METHODS[3], "ridge": 0.}
    parent_fit, parent_scalar = old.route_reference(actual_parent["grid"][None], parent["reset_reference"][None], actual_parent, zero_spec, estimators)
    equal(parent["before__" + METHODS[3]], parent_fit["before"][0], "All-zero parent native fit incorrect", atol=5e-10, rtol=2e-12)
    equal(parent["after__" + METHODS[3]], parent_fit["after"][0], "All-zero parent independent simplex incorrect", atol=1e-10)
    equal(parent["selected_indices"][3], parent_fit["selected_indices"], "All-zero parent native gamma differs")
    for qi, quantity in enumerate(c["quantities"]):
        equal(parent["scalar_draws"][3, qi], parent_scalar[quantity], "All-zero parent16 diagnostics differ: " + quantity, atol=8e-10, rtol=2e-11)
    case_id = c["instance"] + "__rate_" + old.rate_key(c["rate"])
    for ri, method in enumerate(METHODS):
        gold = golden[method, case_id]
        equal(parent["scalar_draws"][ri, :2, 0], [float(gold["tvd"]), float(gold["normalized_grouped_peak_margin"])], "New golden148 point replay failed", atol=1e-11)
        check(int(parent["selected_indices"][ri, 0]) == int(gold["selected_gamma_index"]), "New golden native gamma winner differs")
    for arrays in (parent, cache):
        reset = arrays["direct_draws"][c["direct_routes"].index("reset_160k")]
        expected_reset = np.stack([reset[0] - arrays["scalar_draws"][:, 0], arrays["scalar_draws"][:, 1] - reset[1]], axis=1)
        bitwise(arrays["paired_reset_improvements"], expected_reset, "All-zero Reset improvements not actual paired metric differences")
        expected_mixed = np.stack([arrays["source_mixed_dual_scalar_draws"][0] - arrays["scalar_draws"][3, 0], arrays["scalar_draws"][3, 1] - arrays["source_mixed_dual_scalar_draws"][1]])
        bitwise(arrays["paired_new_dual_vs_mixed"], expected_mixed, "Same-budget zero-versus-mixed sign/pairing incorrect")
        check(arrays["paired_control_improvements"].shape == (0, 2, arrays["scalar_draws"].shape[-1]), "Legacy controls retained as new primary routes")
    previous_hash, next_start = None, 0
    for batch_index, entry in enumerate(m["checkpoint_files"]):
        batch_path = folder / "batches" / entry["path"]
        check(sha(batch_path) == entry["sha256"], "New delta checkpoint hash changed")
        batch = read_npz(batch_path)
        meta = json.loads(str(batch.pop("checkpoint_json")))
        check(meta["identity"] == identity and meta["start"] == next_start and meta["stop"] > next_start and meta["previous_checkpoint_sha256"] == previous_hash, "Delta checkpoint chain gap/overlap")
        source_record = source_m["checkpoint_files"][meta["start"] // 25]
        source_path = source_dir / "batches" / source_record["path"]
        check(meta["source_checkpoint"] == {"path": str(source_path), "sha256": source_record["sha256"], "identity": source_m["identity"], "start": meta["start"], "stop": meta["start"] + 25} and sha(source_path) == source_record["sha256"], "Delta checkpoint not coupled to exact source checkpoint")
        source_batch = read_npz(source_path)
        source_meta = json.loads(str(source_batch.pop("checkpoint_json")))
        for key in ("rng_before", "rng_after", "draw_counts"):
            check(meta[key] == source_meta[key], "Delta changed actual source RNG/count identity")
        check(meta["count_hashes_independently_replayed"] == (not reused) and meta["new_independent_bootstrap_resamples"] == 0 and meta["new_dual_refits"] == (0 if reused else meta["stop"] - meta["start"]) and meta["reused_dual_zero_replicates"] == (meta["stop"] - meta["start"] if reused else 0), "Checkpoint refit/reuse provenance wrong")
        compare_reuse(batch, source_batch, meta["stop"] - meta["start"])
        for key, digest in meta["array_hashes"].items():
            check(old.array_sha(batch[key]) == digest, "Delta checkpoint numeric payload changed")
            expected = cache[key][meta["start"]:meta["stop"]] if key in ("support_sizes", "packed_observed_support") else cache[key][..., meta["start"]:meta["stop"]]
            bitwise(batch[key], expected, "Delta aggregate differs from checkpoint concatenation")
        if batch_index == 0:
            rngs = old.exact_seed_generators(c["seed"], c["instance"], c["rate"])
            if c["budget"] == "gate_matched":
                rngs = rngs[:2]
            check([r.bit_generator.state for r in rngs] == source_meta["rng_before"], "Original first source seed hierarchy wrong")
            drawn = old.independently_draw(actual_parent, c["per_scale_shots"], rngs, raw=c["budget"] == "equal_shot")
            check(old.array_sha(drawn["scale_counts"]) == meta["draw_counts"]["scale_counts_sha256"] and old.array_sha(drawn["reset_extension_counts"]) == meta["draw_counts"]["extension_counts_sha256"] and [r.bit_generator.state for r in rngs] == meta["rng_after"], "Independent exact-source count/RNG replay differs")
            if c["budget"] == "equal_shot":
                check(old.array_sha(drawn["raw_initial_counts"]) == meta["draw_counts"]["raw_20k_counts_sha256"] and old.array_sha(drawn["raw_extension_counts"]) == meta["draw_counts"]["raw_extension_counts_sha256"], "Original independent Raw child-stream count identities differ")
            mask = np.any(drawn["grid"] > 0, axis=1)
            equal(np.unpackbits(batch["packed_observed_support"], axis=1, count=16384, bitorder="little").astype(bool), mask[:meta["stop"]], "Reused support is not actual source resample support")
            fit, scalar = old.route_reference(drawn["grid"], drawn["reset"], actual_parent, zero_spec, estimators)
            equal(batch["selected_indices"][3], fit["selected_indices"][:meta["stop"]], "New/reused dual-zero native gamma incorrect")
            equal(batch["selected_parameters"][3], fit["selected_parameters"][:meta["stop"]].T, "New/reused dual-zero native grid parameters incorrect")
            for qi, quantity in enumerate(c["quantities"]):
                equal(batch["scalar_draws"][3, qi], scalar[quantity][:meta["stop"]], "Independent dual-zero first25 scalar/projection differs: " + quantity, atol=8e-10 if qi > 1 else 6e-11, rtol=2e-11 if qi > 1 else 0)
            print("SVD diagnostic " + c["budget"] + "/" + c["instance"] + "/" + str(c["rate"]) + " " + json.dumps(svd_selected_diagnostic(drawn["grid"], fit, actual_parent, old), sort_keys=True), flush=True)
        next_start, previous_hash = meta["stop"], entry["sha256"]
    check(next_start == n and len(m["checkpoint_files"]) == (n + 24) // 25, "Incomplete delta checkpoint coverage")
    print(f"PASS {c['budget']}/{c['instance']}/{c['rate']:g}: all{n} exact source-reuse/checkpoint/paired differences; independent native zero-dual first25 and golden parents; {'dual-control reused' if reused else 'only zero-dual refitted'}", flush=True)
    return c


def verify_resume(left, right):
    def inventory(root):
        return {p.parent.relative_to(root): p.parent for p in root.rglob("started_manifest.json")}
    a, b = inventory(left), inventory(right)
    check(a.keys() == b.keys() and len(a) > 0, "Resume job inventories differ")
    count = 0
    for job in a:
        for filename in ("parent_estimates.npz", "scalar_bootstrap_draws.npz"):
            x, y = read_npz(a[job] / filename), read_npz(b[job] / filename)
            check(x.keys() == y.keys(), "Resume array inventory differs")
            for key in x:
                bitwise(x[key], y[key], "Resume differs at raw-byte level, including signedzero/NaN payload: " + key)
                count += 1
        pa, pb = sorted((a[job] / "batches").glob("batch_*.npz")), sorted((b[job] / "batches").glob("batch_*.npz"))
        check(len(pa) == len(pb) == 2, "Resume50 did not preserve two25 checkpoints")
        for x, y in zip(pa, pb):
            dx, dy = read_npz(x), read_npz(y)
            check(dx.keys() == dy.keys(), "Resume checkpoint inventory differs")
            for key in dx:
                bitwise(dx[key], dy[key], "Resume checkpoint RNG/count/array bytes differ")
    print(f"PASS {len(a)} reuse/refit resume branches: {count} parent/cache arrays raw-byte identical for uninterrupted50 versus25→50, including signedzeros/NaN payloads and checkpoint RNG/count identity", flush=True)


def verify_aggregate(directory, state, *, require_final=False):
    old, estimators, bvr, core, reference, approved, golden = state
    summary = json.loads((directory / "summary.json").read_text())
    check(summary["configuration_id"] == approved["configuration_id"] and summary["approved_config_sha256"] == sha(CONFIG), "Aggregate profile/configuration mismatch")
    check(summary["generator_sha256"] == sha(ROOT / "analysis/summarize_c1_all_zero_bootstrap_20261018.py") and summary["pinned_statistical_helper_sha256"] == "8f4ba37e1d0e30ffe076bbdf706bc63dd718a0fbe5a61719f9cbc2a670277fbd", "Aggregate statistical source changed")
    for filename, digest in summary["output_csv_sha256"].items():
        check(sha(directory / filename) == digest, "All-zero aggregate table changed/raced: " + filename)
    final = summary["completed_jobs"] == 40 and summary["replicates_per_job"] == 10000
    check(summary["publication_ready"] == final and summary["status"] == ("complete" if final else "partial") and (not require_final or final), "Partial/smoke aggregate mislabeled final")
    check(summary["new_quantum_shots"] == summary["new_noisy_circuit_simulations"] == 0 and summary["new_random_bootstrap_draws_claim"] is False and summary["metric_definitions_changed"] is False and summary["cross_budget_paired_interval"] is False, "Aggregate implies independent draws/newdata or altered metrics")
    scalars = rows(directory / "all_scalar_summaries.csv")
    absolute = rows(directory / "absolute_locked_metrics.csv")
    paired = rows(directory / "paired_comparisons_all.csv")
    contrasts = rows(directory / "paired_all_zero_dual_vs_mixed.csv")
    mixed = rows(directory / "mixed_dual_reference_metrics.csv")
    support = rows(directory / "support_summaries.csv")
    audits = rows(directory / "verified_job_origins.csv")
    check(len(audits) == len(support) == summary["completed_jobs"], "Aggregate job/support coverage changed")
    counts = dict(scalar=0, absolute=0, paired=0, mixed=0, contrast=0, newfits=0, reusedzero=0, reusedunchanged=0, replicate_records=0)
    seen = set()
    for job in audits:
        key = job["budget"], job["instance"], float(job["rate"])
        check(key not in seen, "Aggregate duplicate all-zero condition")
        seen.add(key)
        path = Path(job["manifest"])
        path = path if path.is_absolute() else ROOT / path
        check(sha(path) == job["manifest_sha256"], "Aggregate links changed all-zero job")
        manifest = json.loads(path.read_text())
        c = manifest["configuration"]
        check(manifest["status"] == "complete" and manifest["identity"] == job["identity"] and c["configuration_id"] == approved["configuration_id"], "Aggregate uses wrong/incomplete condition")
        parent, cache = read_npz(path.with_name("parent_estimates.npz")), read_npz(path.with_name("scalar_bootstrap_draws.npz"))
        n = c["replicates"]
        select = lambda table: [r for r in table if (r["budget"], r["instance"], float(r["rate"])) == key]
        ss = select(scalars)
        lookup = {(r["route"], r["quantity"]): r for r in ss}
        check(len(lookup) == len(ss) == 4 * 16 + len(c["direct_routes"]) * 2, "Missing/duplicated all-zero scalar summaries")
        reuse = c["budget"] == "equal_shot" and c["rate"] == .0001
        for ri, route in enumerate(METHODS):
            expected_origin = "exact_reuse_unchanged_method" if ri < 3 else "exact_reuse_existing_zero_control" if reuse else "new_dual_zero_refit_on_identical_draws"
            for qi, quantity in enumerate(c["quantities"]):
                row = lookup[route, quantity]
                check(row["origin"] == expected_origin and row["source_mixed_identity"] == str(cache["source_job_identity"]), "Summary mislabels reused/refitted route identity")
                old.verify_summary_row(row, cache["scalar_draws"][ri, qi], parent["scalar_draws"][ri, qi, 0])
                counts["scalar"] += 1
        for ri, route in enumerate(c["direct_routes"]):
            for qi, metric in enumerate(c["direct_metrics"]):
                row = lookup[route, metric]
                check(row["origin"] == "exact_reuse_direct_baseline" and row["role"] == ("context_only_20k" if route.endswith("_20k") else "scientific_direct_160k"), "Direct-reuse scientific budget label wrong")
                old.verify_summary_row(row, cache["direct_draws"][ri, qi], parent["direct_draws"][ri, qi, 0])
                counts["scalar"] += 1
        aa = select(absolute)
        expected_abs = {(r, q) for r in METHODS for q in c["direct_metrics"]} | {(r, q) for r in c["direct_routes"] if r.endswith("_160k") for q in c["direct_metrics"]}
        check(len(aa) == len(expected_abs) and {(r["route"], r["metric"]) for r in aa} == expected_abs, "Absolute metric scope wrong")
        for row in aa:
            for field, value in lookup[row["route"], row["metric"]].items():
                check(row[field] == value, "Absolute metric not exact scalar subset")
            check(int(row["circuit_shots"]) == (160000 if row["route"] in c["direct_routes"] else c["zne_circuit_shots"]), "Absolute metric shot label wrong")
        counts["absolute"] += len(aa)
        reset_i = c["direct_routes"].index("reset_160k")
        pp = select(paired)
        check(len(pp) == 10 + (2 if "raw_160k" in c["direct_routes"] else 0), "Missing all-zero paired comparison")
        for row in pp:
            qi = c["direct_metrics"].index(row["metric"])
            if row["reference"] == "reset_160k":
                ri = list(METHODS).index(row["route"])
                sample, point = cache["paired_reset_improvements"][ri, qi], parent["paired_reset_improvements"][ri, qi, 0]
                expected_favors = row["route"]
            elif row["reference"] == "raw_160k":
                raw_i = c["direct_routes"].index("raw_160k")
                sample = cache["direct_draws"][raw_i, qi] - cache["direct_draws"][reset_i, qi]
                point = parent["direct_draws"][raw_i, qi, 0] - parent["direct_draws"][reset_i, qi, 0]
                if qi == 1:
                    sample, point = -sample, -point
                expected_favors = "reset_160k"
            else:
                check(row["reference"] == "previous_mixed_dual_ridge1e-6" and row["route"] == METHODS[3], "Unknown/cross-budget reference in all-zero paired table")
                sample, point = cache["paired_new_dual_vs_mixed"][qi], parent["paired_new_dual_vs_mixed"][qi, 0]
                expected_favors = "all_zero_dual"
            check(row["positive_favors"] == expected_favors, "Paired improvement label wrong")
            old.verify_summary_row(row, sample, point)
            counts["paired"] += 1
        cc, mm = select(contrasts), select(mixed)
        check(len(cc) == len(mm) == 2, "Zero/mixed contrasts/reference scope missing")
        for row in cc:
            qi = c["direct_metrics"].index(row["metric"])
            sample, point = cache["paired_new_dual_vs_mixed"][qi], parent["paired_new_dual_vs_mixed"][qi, 0]
            old.verify_summary_row(row, sample, point)
            old.verify_summary_row(row, -sample, -point, prefix="mixed_improvement_")
            counts["contrast"] += 1
        for row in mm:
            qi = c["direct_metrics"].index(row["metric"])
            old.verify_summary_row(row, cache["source_mixed_dual_scalar_draws"][qi], parent["source_mixed_dual_scalar_draws"][qi, 0])
            counts["mixed"] += 1
        sr = select(support)
        check(len(sr) == 1, "Support summary missing")
        old.verify_summary_row(sr[0], cache["support_sizes"], np.count_nonzero(np.any(parent["scale_parents"] > 0, axis=0)))
        counts["newfits"] += manifest["new_dual_refits"]
        counts["reusedzero"] += manifest["reused_dual_zero_replicates"]
        counts["reusedunchanged"] += manifest["unchanged_c1_model_replicates_reused"]
        counts["replicate_records"] += n
    check(counts["scalar"] == len(scalars) and counts["absolute"] == len(absolute) == summary["absolute_locked_metric_rows"] and counts["paired"] == len(paired) == summary["paired_comparison_rows"] and counts["contrast"] == len(contrasts) == summary["dual_vs_mixed_rows"] and counts["mixed"] == len(mixed), "All-zero aggregate row accounting mismatch")
    check(counts["newfits"] == summary["new_dual_fit_replicates"] and counts["reusedzero"] == summary["reused_dual_control_replicates"] and counts["reusedunchanged"] == summary["reused_unchanged_model_replicates"] and counts["replicate_records"] == summary["finalized_paired_replicate_records"], "Reused versus newly evaluated counts mislabeled")
    for filename, expected in (("paired_reset_improvements.csv", [r for r in paired if r["reference"] == "reset_160k"]), ("paired_raw_reset_improvements.csv", [r for r in paired if r["reference"] == "raw_160k"]), ("primary_rate_results.csv", [r for r in absolute if float(r["rate"]) == .0001])):
        # DictWriter's union header adds empty inapplicable metadata columns to
        # the all-comparison table; a subset need not repeat those blank fields.
        nonblank = lambda table: [{k: v for k, v in row.items() if v != ""} for row in table]
        check(nonblank(rows(directory / filename)) == nonblank(expected), "All-zero subset CSV changed: " + filename)
    if require_final:
        check(len(seen) == 40 and counts["scalar"] == 2862 and counts["absolute"] == counts["paired"] == 474 and counts["contrast"] == counts["mixed"] == 80 and counts["newfits"] == 370000 and counts["reusedzero"] == 30000 and counts["reusedunchanged"] == 1200000 and counts["replicate_records"] == 400000, "Final all-zero40-job/result/reuse accounting mismatch")
    print(f"PASS independent all-zero aggregate: {len(seen)} jobs/{len(scalars)} scalar rows/{len(paired)} paired rows, all parent centers/quantiles/means/tails/signs and both zero/mixed comparison directions; status={summary['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume-comparison", type=Path, nargs=2)
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    check(not (args.allow_partial and args.require_final), "Partial and final contracts cannot be combined")
    state = startup()
    old, estimators, bvr, core, reference, approved, golden = state
    verify_toys(old, estimators, core, reference)
    if args.artifacts:
        root = args.artifacts
        started_files = [root / "started_manifest.json"] if (root / "started_manifest.json").is_file() else sorted(root.rglob("started_manifest.json"))
        seen = set()
        for p in started_files:
            if args.allow_partial and not (p.parent / "manifest.json").is_file():
                continue
            c = verify_job(p.parent, state, require_final=args.require_final)
            key = c["budget"], c["instance"], float(c["rate"])
            check(key not in seen, "Duplicate all-zero job")
            seen.add(key)
        check(seen, "No completed all-zero job verified")
        if args.require_final:
            expected = {("equal_shot", r["instance"], float(r["rate"])) for r in golden.values()} | {("gate_matched", case, .0001) for case in old.CASES}
            check(seen == expected and len(seen) == 40, "Final all-zero scope missing37+3 jobs")
    if args.resume_comparison:
        verify_resume(*args.resume_comparison)
    if args.aggregate:
        verify_aggregate(args.aggregate, state, require_final=args.require_final)
    check(not args.require_final or args.artifacts is not None, "Final verification requires actual40-job artifacts")
    if not (args.artifacts or args.resume_comparison or args.aggregate):
        print("PASS all-zero configuration/golden/reference toys only; no bootstrap artifacts requested.", flush=True)
    else:
        print("PASS all-zero delta bootstrap verification; no production launch or source/data writes.", flush=True)


if __name__ == "__main__":
    main()
