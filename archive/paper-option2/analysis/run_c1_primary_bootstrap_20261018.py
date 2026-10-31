#!/usr/bin/env python3
"""Checkpointed saved-count bootstrap for the explicitly approved C1 primary.

One invocation owns ONE budget/instance/rate job. Production orchestration is
external. The original constructors, C1 core, counts and prior results remain
read-only. No circuit code, new quantum trajectories or noisy shots are run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/c1_primary_fixed_20261018"
APPROVED_CONFIG = ROOT / "analysis/c1_primary_fixed_20261018.json"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
CONTROL_ROUTES = ("dual_exponential_zne_ridge0_control", "dual_exponential_zne_ridge1e-8_control")
METRICS = ("tvd", "normalized_grouped_peak_margin")
QUANTITIES = (*METRICS, "negative_mass_before_projection", "negative_coordinate_fraction", "normalization_defect",
              "l1_projection_correction", "l2_projection_correction", "postprojection_support_size",
              "projected_change_from_scale_one_l2", "projection_correction_to_change_ratio",
              "normal_condition_number", "intercept_l1_norm", "intercept_l2_norm", "intercept_l2_norm_squared",
              "full_model_l2_projection_correction", "full_model_normalization_defect")
BATCH = 25
REPLAY_ATOL = 1e-11
EXPECTED_BVR_SHA = "792445cfd99777c2e6eddb5dc781ec30c58b9dc8bf8650d931757d60a717c541"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", required=True, choices=("equal_shot", "gate_matched"))
    parser.add_argument("--instance", required=True, choices=("n119_a13", "n119_a50", "n119_a16"))
    parser.add_argument("--rate", required=True, type=float, help="physical probability, not percent")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.smoke:
        if not 0 < args.replicates <= 100:
            parser.error("--smoke requires1..100 replicates")
    elif args.replicates != 10000:
        parser.error("Production requires10000 replicates; use --smoke for <=100")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    if args.budget == "gate_matched" and args.rate != .0001:
        parser.error("Approved gate-matched scope is p2=.0001 only")
    if args.output_dir is None:
        args.output_dir = BASE / (f"smoke_B{args.replicates}" if args.smoke else "bootstrap_10000")
    args.output_dir = args.output_dir.resolve()
    allowed = args.output_dir.is_relative_to(BASE.resolve()) and args.output_dir != BASE.resolve()
    temporary_smoke = args.smoke and args.output_dir.is_relative_to(Path("/tmp")) and args.output_dir != Path("/tmp")
    if not (allowed or temporary_smoke):
        parser.error("Outputs must descend c1_primary_fixed_20261018; /tmp subdirectories allowed only for smoke")
    return args


def rate_key(rate):
    return format(float(rate), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_suite(lambdas, controls, approved, estimators):
    fixed = {method: estimators.polynomial_intercept_coefficients(lambdas, 1 if method == METHODS[0] else 2,
              ridge=approved["ridges"][method], weight_exponent=2.) for method in METHODS[:2]}
    specifications = [{"route": method, "method": method, "ridge": approved["ridges"][method], "role": "approved_primary"} for method in METHODS]
    if controls:
        specifications += [{"route": route, "method": METHODS[3], "ridge": ridge, "role": "same_draw_dual_control"}
                           for route, ridge in zip(CONTROL_ROUTES, (0., 1e-8))]
    exponential = {}
    for spec in specifications:
        if spec["method"] in METHODS[2:]:
            exponential[spec["route"]] = estimators.build_exponential_grid_family(lambdas,
                components=1 if spec["method"] == METHODS[2] else 2, ridge=spec["ridge"], weight_exponent=2.,
                single_grid=np.linspace(**{"start": .01, "stop": 10., "num": 200}), dual_grid=np.linspace(0., 10., 200))
    polynomial_condition = {}
    weights = estimators.objective_weights(lambdas, exponent=2.)
    for method in METHODS[:2]:
        design = np.vander(lambdas, N=2 if method == METHODS[0] else 3, increasing=True)
        normal = design.T @ (weights[:, None] * design) + approved["ridges"][method] * np.eye(design.shape[1])
        polynomial_condition[method] = float(np.linalg.cond(normal))
    return fixed, exponential, specifications, polynomial_condition


def evaluate(grid, direct, suite, *, target, groups, denominator, bvr, core, keep_vectors=False):
    fixed, exponential, specs, polynomial_condition = suite
    mask = np.any(grid > 0, axis=1)
    indices = [np.flatnonzero(row) for row in mask]
    observed = [grid[b][:, index] for b, index in enumerate(indices)]
    result = core.analyze_compressed(observed, indices, grid.shape[-1], fixed, exponential, variants=("C1",))["C1"]
    assert result["fit_arithmetic"] == "observed_only" and result["fitting_floor"] == 1 / 16384
    routes = [spec["route"] for spec in specs]
    scalar = np.empty((len(routes), len(QUANTITIES), grid.shape[0]))
    selected = np.full((len(routes), grid.shape[0]), -1, dtype=np.int64)
    parameters = np.full((len(routes), 2, grid.shape[0]), np.nan)
    vectors = {}
    for ri, route in enumerate(routes):
        before, after = result["before"][route].materialize(), result["after"][route].materialize()
        if np.any(before[~mask] != 0) or np.any(after[~mask] != 0):
            raise AssertionError("C1 probability escaped the actual resampled observed union")
        if np.min(after) < 0 or np.max(np.abs(after.sum(axis=1) - 1)) > 1e-10:
            raise FloatingPointError("invalid projected C1 probability distribution")
        post = bvr._post_metrics(after, target, groups, denominator)
        diagnostic = result["diagnostics"][route]
        if route in fixed:
            cond = np.full(grid.shape[0], polynomial_condition[route])
            h1 = np.full(grid.shape[0], np.abs(fixed[route]).sum())
            h2sq = np.full(grid.shape[0], np.square(fixed[route]).sum())
        else:
            selection = result["selection"][route]
            selected[ri] = selection["selected_index"]
            parameters[ri, :selection["selected_parameters"].shape[1]] = selection["selected_parameters"].T
            family = exponential[route]
            cond = family.normal_condition_numbers[selected[ri]]
            h1 = family.intercept_l1_norms[selected[ri]]
            h2sq = family.intercept_l2_norms_squared[selected[ri]]
        change = np.linalg.norm(after - grid[:, 0], axis=1)
        values = {**{m: post[m] for m in METRICS},
            "negative_mass_before_projection": diagnostic["negative_mass_before_projection"],
            "negative_coordinate_fraction": diagnostic["negative_entry_fraction_before_projection"],
            "normalization_defect": diagnostic["normalization_error_before_projection"],
            "l1_projection_correction": diagnostic["l1_projection_correction"], "l2_projection_correction": diagnostic["l2_projection_correction"],
            "postprojection_support_size": diagnostic["support_size_after_projection"], "projected_change_from_scale_one_l2": change,
            "projection_correction_to_change_ratio": diagnostic["l2_projection_correction"] / np.maximum(change, np.finfo(float).eps),
            "normal_condition_number": cond, "intercept_l1_norm": h1, "intercept_l2_norm": np.sqrt(h2sq), "intercept_l2_norm_squared": h2sq,
            "full_model_l2_projection_correction": diagnostic["full_model_l2_projection_correction"],
            "full_model_normalization_defect": diagnostic["full_model_normalization_error_before_projection"]}
        scalar[ri] = np.stack([values[q] for q in QUANTITIES])
        if keep_vectors:
            vectors["before__" + route], vectors["after__" + route] = before[0], after[0]
    direct_routes = list(direct)
    direct_scalar = np.stack([np.stack([bvr._post_metrics(direct[r], target, groups, denominator)[m] for m in METRICS]) for r in direct_routes])
    reset = direct_scalar[direct_routes.index("reset_160k")]
    gains = scalar[:, :2].copy()
    gains[:, 0] = reset[0] - scalar[:, 0]
    gains[:, 1] = scalar[:, 1] - reset[1]
    contrasts = np.empty((max(0, len(routes) - 4), 2, grid.shape[0]))
    for ci in range(len(contrasts)):
        contrasts[ci, 0] = scalar[3, 0] - scalar[ci + 4, 0]
        contrasts[ci, 1] = scalar[ci + 4, 1] - scalar[3, 1]
    arrays = {"scalar_draws": scalar, "direct_draws": direct_scalar, "selected_indices": selected,
              "selected_parameters": parameters, "support_sizes": mask.sum(axis=1),
              "packed_observed_support": np.packbits(mask, axis=1, bitorder="little"),
              "paired_reset_improvements": gains, "paired_control_improvements": contrasts}
    if not np.all(np.isfinite(scalar)) or not np.all(np.isfinite(direct_scalar)):
        raise FloatingPointError("nonfinite scalar metric/diagnostic")
    return arrays, vectors


def reference_cache(args, io):
    if args.budget == "equal_shot":
        path = ROOT / "analysis_results/equal_shot_realized" / args.instance / "realized/bootstrap_cache.npz"
        with np.load(path, allow_pickle=False) as z:
            assert int(z["seed"]) == 20260722 and int(z["batch_size"]) == 25 and int(z["replicates"]) == 10000
            ri = np.flatnonzero(z["rates"] == args.rate)
            assert len(ri) == 1
            routes, metrics = z["routes"].astype(str).tolist(), z["metrics"].astype(str).tolist()
            all_values = z["postprojection_draws"][int(ri[0])]
            references = {route: np.stack([all_values[routes.index(route), metrics.index(m)] for m in METRICS])
                          for route in ("raw_160k", "reset_160k", "raw_20k", "reset_20k", METHODS[0])}
    else:
        path = ROOT / "analysis_results/c1_followup_20261018/bootstrap_10000" / args.instance / "scalar_bootstrap_draws.npz"
        complete = json.loads(path.with_name("manifest.json").read_text())
        assert complete["final_10000_contract"] and complete["artifacts"][path.name] == io.sha(path)
        with np.load(path, allow_pickle=False) as z:
            assert int(z["seed"]) == 20260804 and int(z["replicates"]) == 10000
            setting = z["settings"].astype(str).tolist().index("equal_gate_full_lambda_minus2_ridge_1e-8")
            mi = z["methods"].astype(str).tolist().index(METHODS[0])
            quantities = z["quantities"].astype(str).tolist()
            references = {"reset_160k": z["reset_draws"][1],
                METHODS[0]: np.stack([z["scalar_draws"][setting, mi, quantities.index(m)] for m in METRICS])}
    return references, {"path": str(path), "sha256": io.sha(path)}


def replay(arrays, references, direct_routes, start, stop):
    checked, exact, maximum = 0, 0, 0.
    for route, source in references.items():
        actual = arrays["scalar_draws"][0, :2] if route == METHODS[0] else arrays["direct_draws"][direct_routes.index(route)]
        expected = source[:, start:stop]
        delta = np.abs(actual - expected)
        if not np.all(np.isfinite(delta)) or np.max(delta) > REPLAY_ATOL:
            raise AssertionError(f"frozen replay failed: {route} batch{start}:{stop},max={np.max(delta)}")
        checked += delta.size
        exact += int(np.count_nonzero(actual == expected))
        maximum = max(maximum, float(delta.max()))
    return {"checked": checked, "bitwise_equal": exact, "max_abs_error": maximum, "atol": REPLAY_ATOL}


def run(args):
    if ROOT.name != "paper-option2" or subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "option2-metric-locked-revision":
        raise RuntimeError("wrong isolated worktree or branch")
    assert (ROOT / "codex_revision_1_2/prompts/02_option2_metric_locked_no_decoder.txt").is_file()
    import run_c1_followup_20261018 as io
    assert io.sha(ROOT / "shor_error_mitigation/src/shor_qem/bias_variance_revision.py") == EXPECTED_BVR_SHA
    sys.path.insert(0, str(ROOT / "shor_error_mitigation/src"))
    from shor_qem import bias_variance_revision as bvr, estimators, metrics
    import sparse_zne_core_20261018 as core
    approved = json.loads(APPROVED_CONFIG.read_text())
    assert approved["configuration_id"] == "c1_beta2_dual_ridge1e-6_only_20261018"
    assert approved["weight_exponent"] == 2 and approved["ridges"] == dict(zip(METHODS, (0., 0., 0., 1e-6)))
    assert approved["exponential_offset"] == 1 / 16384 and approved["bootstrap_plan"]["batch_size"] == BATCH
    data = bvr._prepare_data(bvr.INSTANCE_SPECS[args.instance], np.array([args.rate]), "realized")
    assert data["lambda_one"] == 0 and np.array_equal(data["fit_lambdas"], approved["effective_scales"])
    grid, extension = data["reset"][0], data["reset_extension"][0]
    controls = args.budget == "equal_shot" and args.rate == .0001
    suite = make_suite(data["fit_lambdas"], controls, approved, estimators)
    routes = [spec["route"] for spec in suite[2]]
    per_scale = 20000 if args.budget == "equal_shot" else 11188
    seed = 20260722 if args.budget == "equal_shot" else 20260804
    direct_routes = ["raw_160k", "reset_160k", "raw_20k", "reset_20k"] if args.budget == "equal_shot" else ["reset_160k"]
    target = bvr.ideal_distribution(14, data["config"]["order"])
    groups = bvr.main_lobe_groups(14, data["config"]["order"])
    denominator = bvr.distribution_metrics(target, 14, data["config"]["order"]).grouped_peak_margin
    common = dict(target=target, groups=groups, denominator=denominator, bvr=bvr, core=core)
    refs, replay_source = reference_cache(args, io)
    point_manifest = json.loads((BASE / "manifest.json").read_text())
    for name, digest in point_manifest["outputs"].items():
        assert io.sha(BASE / name) == digest
    assert point_manifest["sources"][str(APPROVED_CONFIG.relative_to(ROOT))] == io.sha(APPROVED_CONFIG)
    golden = {r["method"]: r for r in load_csv(BASE / "point_estimates.csv")
              if r["instance"] == args.instance and float(r["rate"]) == args.rate}
    assert len(golden) == 4
    sources = [Path(__file__), Path(io.__file__), Path(core.__file__), Path(bvr.__file__), Path(estimators.__file__), Path(metrics.__file__),
               ROOT / "shor_error_mitigation/src/shor_qem/bootstrap_analysis.py"]
    provenance = {"approved_config": {"path": str(APPROVED_CONFIG), "sha256": io.sha(APPROVED_CONFIG)},
        "frozen_point_pack": {"path": str(BASE / "manifest.json"), "sha256": io.sha(BASE / "manifest.json"),
                              "points_path": str(BASE / "point_estimates.csv"), "points_sha256": io.sha(BASE / "point_estimates.csv")},
        "sources": [{"path": str(p.resolve()), "sha256": io.sha(p)} for p in sources],
        "inputs": bvr._manifest_rows(bvr.INSTANCE_SPECS[args.instance]), "replay_reference": replay_source,
        "python": sys.version, "numpy": np.__version__}
    configuration = {"configuration_id": approved["configuration_id"], "budget": args.budget, "instance": args.instance,
        "rate": args.rate, "order": data["config"]["order"], "dimension": 16384, "weight_exponent": 2., "ridges": approved["ridges"],
        "routes": routes, "route_specifications": suite[2], "quantities": list(QUANTITIES), "direct_routes": direct_routes, "direct_metrics": list(METRICS),
        "scientific_direct_routes": ["raw_160k", "reset_160k"] if args.budget == "equal_shot" else ["reset_160k"],
        "paired_reset_improvement_signs": {"tvd": "Reset minus model", "normalized_grouped_peak_margin": "model minus Reset", "positive_favors": "model"},
        "paired_control_improvement_signs": {"tvd": "primary dual minus control", "normalized_grouped_peak_margin": "control minus primary dual", "positive_favors": "control"},
        "control_reference_route": "dual_exponential_zne",
        "fit_lambdas": data["fit_lambdas"].tolist(), "fitting_floor": 1 / 16384, "per_scale_shots": per_scale,
        "zne_circuit_shots": per_scale * 8, "direct_shots": 160000, "reset_extension_shots": 160000 - per_scale,
        "seed": seed, "batch_size": BATCH, "replicates": args.replicates, "smoke": args.smoke,
        "single_gamma_grid": approved["single_gamma_grid"], "dual_gamma_grid": approved["dual_gamma_grid"],
        "native_argmax": True, "observed_support_reselected": True, "gamma_refitted_each_replicate": True,
        "packed_support_bitorder": "little", "prevector_convention": "native_C1_zero_lift; full_model diagnostics separately named",
        "no_cross_budget_paired_interval": True, "coverage": "conditional on empirical parents and postselected fixed settings; no post-selection coverage guarantee"}
    identity = hashlib.sha256(io.json_text({"configuration": configuration, "provenance": provenance}).encode("utf-8")).hexdigest()
    output = args.output_dir / args.budget / args.instance / ("rate_" + rate_key(args.rate))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start_path = output / "started_manifest.json"
        if start_path.exists():
            if not args.resume or json.loads(start_path.read_text())["identity"] != identity:
                raise RuntimeError("existing job requires --resume and identical inputs/source/config")
        elif any(p.name != "run.lock" for p in output.iterdir()):
            raise RuntimeError("unrecognized output directory; never overwrite existing data")
        else:
            io.atomic_json(start_path, {"identity": identity, "configuration": configuration, "provenance": provenance,
                                      "status": "started", "actual_bootstrap_started": False, "started_utc": datetime.now(timezone.utc).isoformat()})
        complete_path = output / "manifest.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            assert complete["identity"] == identity and complete["status"] == "complete"
            for name, digest in complete["artifacts"].items():
                assert io.sha(output / name) == digest
            for item in complete["checkpoint_files"]:
                assert io.sha(output / "batches" / item["path"]) == item["sha256"]
            print(f"Verified complete: {output}", flush=True)
            return
        reset_reference = (per_scale * grid[0] + (160000 - per_scale) * extension) / 160000
        parent_direct = {"reset_160k": reset_reference[None]}
        if args.budget == "equal_shot":
            parent_direct = {"raw_160k": data["raw_full"], "reset_160k": reset_reference[None],
                             "raw_20k": data["raw_initial"], "reset_20k": grid[None, 0]}
        if not (output / "point_manifest.json").exists():
            parent, vectors = evaluate(grid[None], parent_direct, suite, keep_vectors=True, **common)
            differences = {method: {metric: abs(float(parent["scalar_draws"][mi, qi, 0]) - float(golden[method][metric]))
                                    for qi, metric in enumerate(METRICS)} for mi, method in enumerate(METHODS)}
            assert max(v for d in differences.values() for v in d.values()) <= REPLAY_ATOL
            io.atomic_npz(output / "parent_estimates.npz", {**parent, **vectors, "routes": np.array(routes), "quantities": np.array(QUANTITIES),
                "direct_routes": np.array(direct_routes), "direct_metrics": np.array(METRICS), "target": target, "scale_parents": grid,
                "reset_reference": reset_reference, "raw_full_original": data["raw_full"][0], "reset_full_original": data["reset_full"][0],
                "reset_extension_parent": extension, "fit_lambdas": data["fit_lambdas"], "ideal_gap_denominator": denominator,
                "identity": np.asarray(identity), "configuration_json": np.asarray(io.json_text(configuration))})
            point_rows = [{"configuration_id": approved["configuration_id"], "budget": args.budget, "instance": args.instance, "rate": args.rate,
                          "route": route, "quantity": quantity, "point": float(parent["scalar_draws"][ri, qi, 0])}
                          for ri, route in enumerate(routes) for qi, quantity in enumerate(QUANTITIES)]
            point_rows += [{"configuration_id": approved["configuration_id"], "budget": args.budget, "instance": args.instance, "rate": args.rate,
                            "route": route, "quantity": metric, "point": float(parent["direct_draws"][ri, qi, 0])}
                           for ri, route in enumerate(direct_routes) for qi, metric in enumerate(METRICS)]
            io.write_csv(output / "points.csv", point_rows)
            io.atomic_json(output / "point_manifest.json", {"identity": identity, "status": "points_complete", "golden_metric_errors": differences,
                "artifacts": {name: io.sha(output / name) for name in ("parent_estimates.npz", "points.csv")}, "actual_bootstrap_started": False})
        else:
            point_meta = json.loads((output / "point_manifest.json").read_text())
            assert point_meta["identity"] == identity
            for name, digest in point_meta["artifacts"].items():
                assert io.sha(output / name) == digest
        seeds = bvr._bootstrap_seed_streams(seed, args.instance, args.rate)
        rngs = [np.random.default_rng(s) for s in (seeds if args.budget == "equal_shot" else seeds[:2])]
        batches = output / "batches"
        batches.mkdir(exist_ok=True)
        completed, previous_hash = 0, None
        for path in sorted(batches.glob("batch_*.npz")):
            _, meta = io.checkpoint_read(path, identity, completed)
            assert meta["rng_before"] == [rng.bit_generator.state for rng in rngs]
            assert meta["previous_checkpoint_sha256"] == previous_hash and meta["stop"] <= args.replicates
            completed = meta["stop"]
            for rng, state in zip(rngs, meta["rng_after"]):
                rng.bit_generator.state = state
            previous_hash = io.sha(path)
        if (output / "progress.json").exists():
            progress = json.loads((output / "progress.json").read_text())
            if progress.get("last_checkpoint"):
                assert io.sha(batches / progress["last_checkpoint"]) == progress["last_checkpoint_sha256"]
        new_batches, started = 0, time.perf_counter()
        for first in range(completed, args.replicates, BATCH):
            stop = min(first + BATCH, args.replicates)
            before_states = [rng.bit_generator.state for rng in rngs]
            if args.budget == "equal_shot":
                paired = bvr._draw_paired_batch(reset_parents=grid, reset_extension_parent=extension,
                    raw_initial_parent=data["raw_initial"][0], raw_extension_parent=data["raw_extension"][0],
                    reset_rng=rngs[0], reset_extension_rng=rngs[1], raw_rng=rngs[2], raw_extension_rng=rngs[3], size=BATCH, lambda_one_index=0)
                sampled = paired["reset_grid"]
                direct = {route: paired[route] for route in direct_routes}
                draw_audit = {"scale_counts_sha256": io.array_sha(np.rint(sampled * 20000)),
                    "extension_counts_sha256": io.array_sha(paired["reset_extension_counts"]),
                    "raw_20k_counts_sha256": io.array_sha(paired["raw_20k_counts"]), "raw_extension_counts_sha256": io.array_sha(paired["raw_extension_counts"])}
            else:
                sampled, reset, draw_audit = io.draw_budget_batch(grid, extension, per_scale, rngs)
                direct = {"reset_160k": reset}
            arrays, _ = evaluate(sampled, direct, suite, **common)
            size = stop - first
            arrays = {k: a[:size] if k in ("support_sizes", "packed_observed_support") else a[..., :size] for k, a in arrays.items()}
            replay_record = replay(arrays, refs, direct_routes, first, stop)
            path = batches / f"batch_{first:06d}_{stop:06d}.npz"
            io.checkpoint_write(path, arrays, {"identity": identity, "start": first, "stop": stop, "full_draw_batch_size": BATCH,
                "rng_before": before_states, "rng_after": [rng.bit_generator.state for rng in rngs], "draw_counts": draw_audit,
                "previous_checkpoint_sha256": previous_hash, "replay": replay_record})
            io.checkpoint_read(path, identity, first)
            previous_hash = io.sha(path)
            completed, new_batches = stop, new_batches + 1
            io.atomic_json(output / "progress.json", {"identity": identity, "status": "running" if stop < args.replicates else "bootstrap_draws_complete",
                "completed": stop, "total": args.replicates, "last_checkpoint": path.name, "last_checkpoint_sha256": previous_hash,
                "actual_bootstrap_started": True, "actual_bootstrap_completed": stop == args.replicates,
                "seconds_this_invocation": time.perf_counter() - started, "updated_utc": datetime.now(timezone.utc).isoformat()})
            if stop % 500 == 0 or new_batches == 1 or stop == args.replicates:
                print(f"[{args.budget}/{args.instance}/{args.rate:g}] {stop}/{args.replicates}", flush=True)
            if args.max_batches is not None and new_batches >= args.max_batches:
                break
        if completed != args.replicates:
            print(f"Checkpointed partial job {completed}/{args.replicates}; no final interval claim", flush=True)
            return
        pieces, checkpoint_files, expected_start, previous_hash = [], [], 0, None
        replay_totals = {"checked": 0, "bitwise_equal": 0, "max_abs_error": 0., "atol": REPLAY_ATOL}
        for path in sorted(batches.glob("batch_*.npz")):
            arrays, meta = io.checkpoint_read(path, identity, expected_start)
            assert meta["previous_checkpoint_sha256"] == previous_hash
            previous_hash, expected_start = io.sha(path), meta["stop"]
            pieces.append(arrays)
            checkpoint_files.append({"path": path.name, "sha256": previous_hash, "start": meta["start"], "stop": meta["stop"]})
            replay_totals["checked"] += meta["replay"]["checked"]
            replay_totals["bitwise_equal"] += meta["replay"]["bitwise_equal"]
            replay_totals["max_abs_error"] = max(replay_totals["max_abs_error"], meta["replay"]["max_abs_error"])
        assert expected_start == args.replicates
        aggregate = {k: np.concatenate([piece[k] for piece in pieces], axis=0 if k in ("support_sizes", "packed_observed_support") else -1) for k in pieces[0]}
        io.atomic_npz(output / "scalar_bootstrap_draws.npz", {**aggregate, "routes": np.array(routes), "quantities": np.array(QUANTITIES),
            "direct_routes": np.array(direct_routes), "direct_metrics": np.array(METRICS), "control_routes": np.array(routes[4:]),
            "scalar_axis_order": np.array(["route", "quantity", "replicate"]), "support_axis_order": np.array(["replicate", "packed_byte"]),
            "selected_parameter_axis_order": np.array(["route", "parameter", "replicate"]),
            "seed": seed, "batch_size": BATCH, "replicates": args.replicates, "identity": np.asarray(identity),
            "configuration_json": np.asarray(io.json_text(configuration))})
        io.atomic_json(output / "replay.json", replay_totals)
        artifacts = {name: io.sha(output / name) for name in ("parent_estimates.npz", "points.csv", "point_manifest.json", "scalar_bootstrap_draws.npz", "replay.json")}
        io.atomic_json(complete_path, {"identity": identity, "status": "complete", "configuration": configuration, "provenance": provenance,
            "artifacts": artifacts, "checkpoint_files": checkpoint_files, "replay": replay_totals,
            "actual_bootstrap_started": True, "actual_bootstrap_completed": True,
            "final_10000_contract": args.replicates == 10000 and not args.smoke,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds_this_invocation": time.perf_counter() - started,
            "new_batches_this_invocation": new_batches, "new_bootstrap_replicates_this_invocation": sum(p["stop"] - p["start"] for p in checkpoint_files[-new_batches:]) if new_batches else 0,
            "new_quantum_shots": 0, "new_noisy_circuit_simulations": 0, "no_old_interval_reuse": True})
        print(f"Complete {output}; replay maxerror={replay_totals['max_abs_error']:.3g}", flush=True)


if __name__ == "__main__":
    run(arguments())
