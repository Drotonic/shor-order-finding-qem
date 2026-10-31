#!/usr/bin/env python3
"""All-zero-ridge C1 bootstrap, coupled exactly to completed mixed-profile draws.

The unchanged three C1 models and direct routes are copied bitwise. The three
representative equal-shot dual-zero controls are also copied bitwise. Only the
other dual-zero fits are evaluated, on replayed and hash-checked old counts.
No circuit simulation, new noisy shots, or new independent resamples occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/c1_all_zero_fixed_20261018"
APPROVED_CONFIG = ROOT / "analysis/c1_all_zero_fixed_20261018.json"
SOURCE_BASE = ROOT / "analysis_results/c1_primary_fixed_20261018/bootstrap_10000"
SOURCE_RUNNER_SHA = "32fd620f67423059a2037ed1538e4710ab5880b8e500d8841674807126d04e57"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
DUAL = METHODS[-1]
ZERO_CONTROL = "dual_exponential_zne_ridge0_control"
BATCH = 25
ATOL = 1e-11
SAMPLE_AXIS_ZERO = ("support_sizes", "packed_observed_support")


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
            parser.error("--smoke requires 1..100 replicates")
    elif args.replicates != 10000:
        parser.error("Production requires 10000 replicates; use --smoke for <=100")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    if args.budget == "gate_matched" and args.rate != .0001:
        parser.error("Gate-matched scope is p2=.0001 only")
    args.output_dir = (args.output_dir or BASE / (f"smoke_B{args.replicates}" if args.smoke else "bootstrap_10000")).resolve()
    allowed = args.output_dir.is_relative_to(BASE.resolve()) and args.output_dir != BASE.resolve()
    temporary = args.smoke and args.output_dir.is_relative_to(Path("/tmp")) and args.output_dir != Path("/tmp")
    if not (allowed or temporary):
        parser.error("Outputs must descend c1_all_zero_fixed_20261018; /tmp subdirectories allowed only for smoke")
    return args


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_files(directory, entries, io):
    for name, digest in entries.items():
        if io.sha(directory / name) != digest:
            raise AssertionError(f"Artifact hash mismatch: {directory / name}")


def load_source(args, old, io):
    directory = SOURCE_BASE / args.budget / args.instance / ("rate_" + old.rate_key(args.rate))
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    config = manifest["configuration"]
    assert manifest["status"] == "complete" and manifest["final_10000_contract"]
    assert manifest["actual_bootstrap_completed"] and config["replicates"] == 10000 and not config["smoke"]
    assert config["configuration_id"] == "c1_beta2_dual_ridge1e-6_only_20261018"
    assert config["budget"] == args.budget and config["instance"] == args.instance and config["rate"] == args.rate
    assert config["ridges"] == dict(zip(METHODS, (0., 0., 0., 1e-6)))
    assert config["weight_exponent"] == 2. and config["batch_size"] == BATCH
    assert config["seed"] == (20260722 if args.budget == "equal_shot" else 20260804)
    assert config["routes"][:4] == list(METHODS)
    assert manifest["identity"] == hashlib.sha256(io.json_text({"configuration": config, "provenance": manifest["provenance"]}).encode()).hexdigest()
    verify_files(directory, manifest["artifacts"], io)
    for record in manifest["provenance"]["sources"]:
        assert io.sha(Path(record["path"])) == record["sha256"]
    for name in ("approved_config", "frozen_point_pack", "replay_reference"):
        record = manifest["provenance"][name]
        assert io.sha(Path(record["path"])) == record["sha256"]
    checkpoints = manifest["checkpoint_files"]
    assert len(checkpoints) == 400
    previous_stop = 0
    for record in checkpoints:
        assert record["start"] == previous_stop and record["stop"] - record["start"] == BATCH
        assert io.sha(directory / "batches" / record["path"]) == record["sha256"]
        previous_stop = record["stop"]
    assert previous_stop == 10000
    parent = read_npz(directory / "parent_estimates.npz")
    assert str(parent["identity"]) == manifest["identity"]
    assert parent["routes"].astype(str).tolist() == config["routes"]
    return directory, manifest, parent


def origins(reuse_control):
    result = {method: {"operation": "exact_reuse_completed_mixed_profile", "source_route": method,
                       "source_ridge": 0., "new_ridge": 0.} for method in METHODS[:3]}
    result[DUAL] = {"operation": "exact_reuse_same_draw_zero_ridge_control" if reuse_control else "new_dual_refit_on_replayed_source_counts",
                    "source_route": ZERO_CONTROL if reuse_control else DUAL,
                    "source_ridge": 0. if reuse_control else 1e-6, "new_ridge": 0.}
    return result


def combine(source, dual, source_routes, direct_routes):
    """Copy unchanged native arrays and retain an explicit mixed-dual reference."""
    source_dual = source_routes.index(DUAL)
    result = {key: source[key].copy() for key in ("direct_draws", "support_sizes", "packed_observed_support")}
    for key in ("scalar_draws", "selected_indices", "selected_parameters"):
        result[key] = np.concatenate((source[key][:3], dual[key]), axis=0)
    for key in SAMPLE_AXIS_ZERO:
        assert np.array_equal(result[key], dual[key])
    reset = result["direct_draws"][direct_routes.index("reset_160k")]
    scalar = result["scalar_draws"]
    gain = scalar[:, :2].copy()
    gain[:, 0], gain[:, 1] = reset[0] - scalar[:, 0], scalar[:, 1] - reset[1]
    result["paired_reset_improvements"] = gain
    result["paired_control_improvements"] = np.empty((0, 2, scalar.shape[-1]))
    result["source_mixed_dual_scalar_draws"] = source["scalar_draws"][source_dual].copy()
    result["source_mixed_dual_selected_indices"] = source["selected_indices"][source_dual].copy()
    result["source_mixed_dual_selected_parameters"] = source["selected_parameters"][source_dual].copy()
    result["paired_new_dual_vs_mixed"] = np.stack((source["scalar_draws"][source_dual, 0] - scalar[3, 0],
                                                   scalar[3, 1] - source["scalar_draws"][source_dual, 1]))
    assert np.all(np.isfinite(scalar)) and np.all(np.isfinite(result["direct_draws"]))
    return result


def copied_dual(source, source_routes):
    index = source_routes.index(ZERO_CONTROL)
    return {key: source[key][index:index + 1].copy() for key in ("scalar_draws", "selected_indices", "selected_parameters")} | {
        key: source[key].copy() for key in SAMPLE_AXIS_ZERO}


def reuse_audit(arrays, source, source_routes, reuse_control):
    def same_bits(actual, expected):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()

    checked = 0
    for key in ("scalar_draws", "selected_indices", "selected_parameters"):
        expected, actual = source[key][:3], arrays[key][:3]
        same_bits(actual, expected)
        checked += actual.size
    for key in ("direct_draws", *SAMPLE_AXIS_ZERO):
        same_bits(arrays[key], source[key])
        checked += arrays[key].size
    if reuse_control:
        index = source_routes.index(ZERO_CONTROL)
        for key in ("scalar_draws", "selected_indices", "selected_parameters"):
            same_bits(arrays[key][3], source[key][index])
            checked += arrays[key][3].size
    return {"checked": checked, "bitwise_equal": checked, "max_abs_error": 0., "atol": 0.,
            "dual_zero_control_reused": reuse_control}


def sliced(arrays, size):
    return {key: value[:size] if key in SAMPLE_AXIS_ZERO else value[..., :size] for key, value in arrays.items()}


def replay_counts(meta, data, direct_routes, budget, bvr, io):
    rngs = [np.random.default_rng(0) for _ in meta["rng_before"]]
    for rng, state in zip(rngs, meta["rng_before"]):
        rng.bit_generator.state = state
    grid, extension = data["reset"][0], data["reset_extension"][0]
    if budget == "equal_shot":
        paired = bvr._draw_paired_batch(reset_parents=grid, reset_extension_parent=extension,
            raw_initial_parent=data["raw_initial"][0], raw_extension_parent=data["raw_extension"][0],
            reset_rng=rngs[0], reset_extension_rng=rngs[1], raw_rng=rngs[2], raw_extension_rng=rngs[3],
            size=BATCH, lambda_one_index=0)
        sampled = paired["reset_grid"]
        direct = {route: paired[route] for route in direct_routes}
        audit = {"scale_counts_sha256": io.array_sha(np.rint(sampled * 20000)),
            "extension_counts_sha256": io.array_sha(paired["reset_extension_counts"]),
            "raw_20k_counts_sha256": io.array_sha(paired["raw_20k_counts"]),
            "raw_extension_counts_sha256": io.array_sha(paired["raw_extension_counts"])}
    else:
        sampled, reset, audit = io.draw_budget_batch(grid, extension, 11188, rngs)
        direct = {"reset_160k": reset}
    assert audit == meta["draw_counts"], "Actual replayed integer counts differ from completed source"
    assert [rng.bit_generator.state for rng in rngs] == meta["rng_after"]
    return sampled, direct


def run(args):
    if ROOT.name != "paper-option2" or subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "option2-metric-locked-revision":
        raise RuntimeError("wrong isolated worktree or branch")
    import run_c1_followup_20261018 as io
    import run_c1_primary_bootstrap_20261018 as old
    assert io.sha(Path(old.__file__)) == SOURCE_RUNNER_SHA
    assert io.sha(ROOT / "shor_error_mitigation/src/shor_qem/bias_variance_revision.py") == old.EXPECTED_BVR_SHA
    sys.path.insert(0, str(ROOT / "shor_error_mitigation/src"))
    from shor_qem import bias_variance_revision as bvr, estimators, metrics
    import sparse_zne_core_20261018 as core
    approved = json.loads(APPROVED_CONFIG.read_text())
    assert approved["configuration_id"] == "c1_beta2_all_ridges_zero_20261018"
    assert approved["weight_exponent"] == 2. and approved["ridges"] == dict.fromkeys(METHODS, 0.)
    assert approved["exponential_offset"] == 1 / 16384 and approved["bootstrap_plan"]["batch_size"] == BATCH
    data = bvr._prepare_data(bvr.INSTANCE_SPECS[args.instance], np.array([args.rate]), "realized")
    assert data["lambda_one"] == 0 and np.array_equal(data["fit_lambdas"], approved["effective_scales"])
    source_dir, source_manifest, source_parent = load_source(args, old, io)
    source_config = source_manifest["configuration"]
    source_routes, direct_routes = source_config["routes"], source_config["direct_routes"]
    assert source_parent["quantities"].astype(str).tolist() == list(old.QUANTITIES)
    assert np.array_equal(source_parent["scale_parents"], data["reset"][0])
    reuse_control = args.budget == "equal_shot" and args.rate == .0001
    assert (ZERO_CONTROL in source_routes) == reuse_control
    if reuse_control:
        control_spec = source_config["route_specifications"][source_routes.index(ZERO_CONTROL)]
        assert control_spec["route"] == ZERO_CONTROL and control_spec["method"] == DUAL and control_spec["ridge"] == 0.
    point_manifest = json.loads((BASE / "manifest.json").read_text())
    verify_files(BASE, point_manifest["outputs"], io)
    for name, digest in point_manifest["sources"].items():
        assert io.sha(ROOT / name) == digest
    assert point_manifest["configuration_id"] == approved["configuration_id"]
    golden = {row["method"]: row for row in old.load_csv(BASE / "point_estimates.csv")
              if row["instance"] == args.instance and float(row["rate"]) == args.rate}
    assert set(golden) == set(METHODS)
    sources = [Path(__file__), Path(old.__file__), Path(io.__file__), Path(core.__file__), Path(bvr.__file__),
               Path(estimators.__file__), Path(metrics.__file__), ROOT / "shor_error_mitigation/src/shor_qem/bootstrap_analysis.py"]
    provenance = {"approved_config": {"path": str(APPROVED_CONFIG), "sha256": io.sha(APPROVED_CONFIG)},
        "frozen_point_pack": {"path": str(BASE / "manifest.json"), "sha256": io.sha(BASE / "manifest.json"),
                              "points_path": str(BASE / "point_estimates.csv"), "points_sha256": io.sha(BASE / "point_estimates.csv")},
        "source_completed_job": {"path": str(source_dir / "manifest.json"), "sha256": io.sha(source_dir / "manifest.json"),
                                 "identity": source_manifest["identity"], "configuration_id": source_config["configuration_id"]},
        "sources": [{"path": str(path.resolve()), "sha256": io.sha(path)} for path in sources],
        "inputs": bvr._manifest_rows(bvr.INSTANCE_SPECS[args.instance]), "python": sys.version, "numpy": np.__version__}
    configuration = {key: value for key, value in source_config.items() if key not in
                     ("configuration_id", "ridges", "routes", "route_specifications", "replicates", "smoke",
                      "paired_control_improvement_signs", "control_reference_route")}
    configuration.update({"configuration_id": approved["configuration_id"], "ridges": approved["ridges"], "routes": list(METHODS),
        "route_specifications": [{"route": method, "method": method, "ridge": 0., "role": "approved_all_zero_primary"} for method in METHODS],
        "replicates": args.replicates, "smoke": args.smoke, "route_origins": origins(reuse_control),
        "direct_route_origins": {route: "exact_reuse_completed_mixed_profile" for route in direct_routes},
        "support_origin": "exact_reuse_actual_source_replicate_support; independently replay-checked for new dual fits",
        "source_mixed_dual_ridge": 1e-6,
        "paired_new_dual_vs_mixed_signs": {"tvd": "mixed dual ridge1e-6 minus all-zero dual", "normalized_grouped_peak_margin": "all-zero dual minus mixed dual ridge1e-6", "positive_favors": "all_zero_dual"},
        "new_independent_bootstrap_resamples": 0, "resamples_exactly_coupled_to_completed_source": True,
        "new_dual_refits_expected": 0 if reuse_control else args.replicates,
        "reused_dual_zero_replicates_expected": args.replicates if reuse_control else 0,
        "coverage": "conditional empirical-parent, pointwise post-selection analysis; no independent validation or post-selection coverage guarantee"})
    identity = hashlib.sha256(io.json_text({"configuration": configuration, "provenance": provenance}).encode()).hexdigest()
    output = args.output_dir / args.budget / args.instance / ("rate_" + old.rate_key(args.rate))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start_path = output / "started_manifest.json"
        if start_path.exists():
            if not args.resume or json.loads(start_path.read_text())["identity"] != identity:
                raise RuntimeError("existing job requires --resume and identical sources/config")
        elif any(path.name != "run.lock" for path in output.iterdir()):
            raise RuntimeError("unrecognized output directory; never overwrite existing data")
        else:
            io.atomic_json(start_path, {"identity": identity, "configuration": configuration, "provenance": provenance,
                "status": "started", "actual_bootstrap_started": False, "started_utc": datetime.now(timezone.utc).isoformat()})
        complete_path = output / "manifest.json"
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            assert complete["identity"] == identity and complete["status"] == "complete"
            verify_files(output, complete["artifacts"], io)
            for record in complete["checkpoint_files"]:
                assert io.sha(output / "batches" / record["path"]) == record["sha256"]
            print(f"Verified complete: {output}", flush=True)
            return
        suite = None
        if not reuse_control:
            family = estimators.build_exponential_grid_family(data["fit_lambdas"], components=2, ridge=0., weight_exponent=2.,
                         single_grid=np.linspace(.01, 10., 200), dual_grid=np.linspace(0., 10., 200))
            suite = ({}, {DUAL: family}, [{"route": DUAL, "method": DUAL, "ridge": 0., "role": "approved_all_zero_primary"}], {})
        target = source_parent["target"]
        assert np.array_equal(target, bvr.ideal_distribution(14, data["config"]["order"]))
        common = dict(target=target, groups=bvr.main_lobe_groups(14, data["config"]["order"]),
                      denominator=float(source_parent["ideal_gap_denominator"]), bvr=bvr, core=core)
        parent_direct = {"reset_160k": source_parent["reset_reference"][None]}
        if args.budget == "equal_shot":
            parent_direct = {"raw_160k": source_parent["raw_full_original"][None], "reset_160k": source_parent["reset_reference"][None],
                             "raw_20k": data["raw_initial"], "reset_20k": source_parent["scale_parents"][None, 0]}
        if not (output / "point_manifest.json").exists():
            if reuse_control:
                dual_parent = copied_dual(source_parent, source_routes)
                new_vectors = {prefix + DUAL: source_parent[prefix + ZERO_CONTROL].copy() for prefix in ("before__", "after__")}
            else:
                dual_parent, new_vectors = old.evaluate(source_parent["scale_parents"][None], parent_direct, suite, keep_vectors=True, **common)
            parent = combine(source_parent, dual_parent, source_routes, direct_routes)
            errors = {method: {metric: abs(float(parent["scalar_draws"][mi, qi, 0]) - float(golden[method][metric]))
                              for qi, metric in enumerate(old.METRICS)} for mi, method in enumerate(METHODS)}
            assert max(value for row in errors.values() for value in row.values()) <= ATOL
            assert all(int(parent["selected_indices"][mi, 0]) == int(golden[method]["selected_gamma_index"]) for mi, method in enumerate(METHODS))
            vectors = {prefix + method: source_parent[prefix + method].copy() for method in METHODS[:3] for prefix in ("before__", "after__")}
            vectors.update(new_vectors)
            vectors.update({"source_mixed_dual_before": source_parent["before__" + DUAL].copy(), "source_mixed_dual_after": source_parent["after__" + DUAL].copy()})
            retained = {key: source_parent[key] for key in ("target", "scale_parents", "reset_reference", "raw_full_original", "reset_full_original", "reset_extension_parent", "fit_lambdas", "ideal_gap_denominator")}
            io.atomic_npz(output / "parent_estimates.npz", {**parent, **vectors, **retained, "routes": np.array(METHODS),
                "quantities": np.array(old.QUANTITIES), "direct_routes": np.array(direct_routes), "direct_metrics": np.array(old.METRICS),
                "identity": np.asarray(identity), "source_job_identity": np.asarray(source_manifest["identity"]), "configuration_json": np.asarray(io.json_text(configuration))})
            rows = [{"configuration_id": approved["configuration_id"], "budget": args.budget, "instance": args.instance, "rate": args.rate,
                     "route": method, "quantity": quantity, "point": float(parent["scalar_draws"][mi, qi, 0])}
                    for mi, method in enumerate(METHODS) for qi, quantity in enumerate(old.QUANTITIES)]
            rows += [{"configuration_id": approved["configuration_id"], "budget": args.budget, "instance": args.instance, "rate": args.rate,
                      "route": route, "quantity": metric, "point": float(parent["direct_draws"][ri, qi, 0])}
                     for ri, route in enumerate(direct_routes) for qi, metric in enumerate(old.METRICS)]
            io.write_csv(output / "points.csv", rows)
            io.atomic_json(output / "point_manifest.json", {"identity": identity, "status": "points_complete", "golden_metric_errors": errors,
                "artifacts": {name: io.sha(output / name) for name in ("parent_estimates.npz", "points.csv")},
                "reuse": reuse_audit(parent, source_parent, source_routes, reuse_control), "actual_bootstrap_started": False})
        else:
            point_meta = json.loads((output / "point_manifest.json").read_text())
            assert point_meta["identity"] == identity
            verify_files(output, point_meta["artifacts"], io)
        batches = output / "batches"
        batches.mkdir(exist_ok=True)
        completed, previous_hash = 0, None
        for path in sorted(batches.glob("batch_*.npz")):
            _, meta = io.checkpoint_read(path, identity, completed)
            assert meta["previous_checkpoint_sha256"] == previous_hash and meta["stop"] <= args.replicates
            source_record = source_manifest["checkpoint_files"][meta["start"] // BATCH]
            assert meta["source_checkpoint"]["sha256"] == source_record["sha256"]
            completed, previous_hash = meta["stop"], io.sha(path)
        if (output / "progress.json").exists():
            progress = json.loads((output / "progress.json").read_text())
            if progress.get("last_checkpoint"):
                assert io.sha(batches / progress["last_checkpoint"]) == progress["last_checkpoint_sha256"]
        new_batches, started = 0, time.perf_counter()
        for first in range(completed, args.replicates, BATCH):
            stop = min(first + BATCH, args.replicates)
            source_record = source_manifest["checkpoint_files"][first // BATCH]
            source_path = source_dir / "batches" / source_record["path"]
            source, meta = io.checkpoint_read(source_path, source_manifest["identity"], first)
            assert meta["stop"] == first + BATCH and io.sha(source_path) == source_record["sha256"]
            expected_previous = source_manifest["checkpoint_files"][first // BATCH - 1]["sha256"] if first else None
            assert meta["previous_checkpoint_sha256"] == expected_previous
            if reuse_control:
                dual = copied_dual(source, source_routes)
            else:
                sampled, direct = replay_counts(meta, data, direct_routes, args.budget, bvr, io)
                dual, _ = old.evaluate(sampled, direct, suite, **common)
            arrays = sliced(combine(source, dual, source_routes, direct_routes), stop - first)
            reuse = reuse_audit(arrays, sliced(source, stop - first), source_routes, reuse_control)
            path = batches / f"batch_{first:06d}_{stop:06d}.npz"
            source_info = {"path": str(source_path), "sha256": source_record["sha256"], "identity": source_manifest["identity"], "start": first, "stop": first + BATCH}
            io.checkpoint_write(path, arrays, {"identity": identity, "start": first, "stop": stop, "full_draw_batch_size": BATCH,
                "rng_before": meta["rng_before"], "rng_after": meta["rng_after"], "draw_counts": meta["draw_counts"],
                "source_checkpoint": source_info, "previous_checkpoint_sha256": previous_hash, "replay": reuse,
                "count_hashes_independently_replayed": not reuse_control,
                "new_dual_refits": 0 if reuse_control else stop - first,
                "reused_dual_zero_replicates": stop - first if reuse_control else 0,
                "new_independent_bootstrap_resamples": 0})
            io.checkpoint_read(path, identity, first)
            completed, previous_hash, new_batches = stop, io.sha(path), new_batches + 1
            io.atomic_json(output / "progress.json", {"identity": identity, "status": "running" if stop < args.replicates else "bootstrap_arrays_complete",
                "completed": stop, "total": args.replicates, "last_checkpoint": path.name, "last_checkpoint_sha256": previous_hash,
                "actual_bootstrap_started": True, "actual_bootstrap_completed": stop == args.replicates,
                "new_independent_bootstrap_resamples": 0, "updated_utc": datetime.now(timezone.utc).isoformat()})
            if stop % 500 == 0 or new_batches == 1 or stop == args.replicates:
                print(f"[{args.budget}/{args.instance}/{args.rate:g}] {stop}/{args.replicates}; dual={'reuse' if reuse_control else 'refit'}", flush=True)
            if args.max_batches is not None and new_batches >= args.max_batches:
                break
        if completed != args.replicates:
            print(f"Checkpointed partial job {completed}/{args.replicates}; no final interval claim", flush=True)
            return
        pieces, checkpoint_files, expected_start, previous_hash = [], [], 0, None
        audit = {"checked": 0, "bitwise_equal": 0, "max_abs_error": 0., "atol": 0., "new_dual_refits": 0, "reused_dual_zero_replicates": 0,
                 "new_independent_bootstrap_resamples": 0, "count_replayed_replicates": 0}
        for path in sorted(batches.glob("batch_*.npz")):
            arrays, meta = io.checkpoint_read(path, identity, expected_start)
            assert meta["previous_checkpoint_sha256"] == previous_hash
            previous_hash, expected_start = io.sha(path), meta["stop"]
            pieces.append(arrays)
            checkpoint_files.append({"path": path.name, "sha256": previous_hash, "start": meta["start"], "stop": meta["stop"]})
            for key in ("checked", "bitwise_equal"):
                audit[key] += meta["replay"][key]
            for key in ("new_dual_refits", "reused_dual_zero_replicates"):
                audit[key] += meta[key]
            if meta["count_hashes_independently_replayed"]:
                audit["count_replayed_replicates"] += meta["stop"] - meta["start"]
        assert expected_start == args.replicates
        aggregate = {key: np.concatenate([piece[key] for piece in pieces], axis=0 if key in SAMPLE_AXIS_ZERO else -1) for key in pieces[0]}
        io.atomic_npz(output / "scalar_bootstrap_draws.npz", {**aggregate, "routes": np.array(METHODS), "quantities": np.array(old.QUANTITIES),
            "direct_routes": np.array(direct_routes), "direct_metrics": np.array(old.METRICS), "control_routes": np.array([], dtype="U1"),
            "scalar_axis_order": np.array(["route", "quantity", "replicate"]), "support_axis_order": np.array(["replicate", "packed_byte"]),
            "selected_parameter_axis_order": np.array(["route", "parameter", "replicate"]),
            "seed": configuration["seed"], "batch_size": BATCH, "replicates": args.replicates,
            "identity": np.asarray(identity), "source_job_identity": np.asarray(source_manifest["identity"]), "configuration_json": np.asarray(io.json_text(configuration))})
        io.atomic_json(output / "replay.json", audit)
        artifacts = {name: io.sha(output / name) for name in ("parent_estimates.npz", "points.csv", "point_manifest.json", "scalar_bootstrap_draws.npz", "replay.json")}
        io.atomic_json(complete_path, {"identity": identity, "status": "complete", "configuration": configuration, "provenance": provenance,
            "artifacts": artifacts, "checkpoint_files": checkpoint_files, "replay": audit, "route_origins": configuration["route_origins"],
            "actual_bootstrap_started": True, "actual_bootstrap_completed": True, "final_10000_contract": args.replicates == 10000 and not args.smoke,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds_this_invocation": time.perf_counter() - started,
            "new_batches_this_invocation": new_batches, "new_dual_refits": audit["new_dual_refits"],
            "reused_dual_zero_replicates": audit["reused_dual_zero_replicates"], "unchanged_c1_model_replicates_reused": 3 * args.replicates,
            "new_independent_bootstrap_resamples": 0, "new_quantum_shots": 0, "new_noisy_circuit_simulations": 0,
            "old_interval_endpoints_copied": False, "intervals_to_be_aggregated_from_actual_paired_scalar_arrays": True})
        print(f"Complete {output}; dual refits={audit['new_dual_refits']}, dual-control reuse={audit['reused_dual_zero_replicates']}", flush=True)


if __name__ == "__main__":
    run(arguments())
