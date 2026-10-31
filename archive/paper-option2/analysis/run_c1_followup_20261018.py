#!/usr/bin/env python3
"""Saved-count C1 gate matching and five-setting sensitivity at p2=0.01%.

No circuit code is executed and no quantum shots are generated.  Point estimates
are saved before any bootstrap starts.  The immutable original sensitivity
suite/seed helpers and C1 implementation are imported, not modified.  Bootstrap
checkpoints contain the true observed union and all scalar/selection outputs;
every batch is payload-hashed, atomic, and independently resumable.

Examples:
  OPENBLAS_NUM_THREADS=1 python analysis/run_c1_followup_20261018.py --point-only
  OPENBLAS_NUM_THREADS=1 python analysis/run_c1_followup_20261018.py --smoke --replicates 25
  OPENBLAS_NUM_THREADS=1 python analysis/run_c1_followup_20261018.py --replicates 10000 --instance n119_a13
  ... --resume                    # verifies and resumes an interrupted run
  ... --smoke --replicates 50 --max-batches 1  # interruption/resume smoke
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
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
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis_results" / "c1_followup_20261018"
CASES = ("n119_a13", "n119_a50", "n119_a16")
SEED = 20260804
BATCH = 25
RATE = 0.0001
DIRECT_SHOTS = 160000
GATE_SHOTS = 11188
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
METRICS = ("tvd", "normalized_grouped_peak_margin")
DIAGNOSTICS = (
    "negative_mass_before_projection", "negative_coordinate_fraction",
    "normalization_defect", "l1_projection_correction", "l2_projection_correction",
    "postprojection_support_size", "projected_change_from_scale_one_l2",
    "projection_correction_to_change_ratio",
)
QUANTITIES = (*METRICS, "reset_minus_zne_tvd", "zne_minus_reset_normalized_grouped_peak_margin", *DIAGNOSTICS)
SUPPORT_CONTEXTS = ("equal_shot_eight_scales", "equal_shot_six_scales", "equal_gate_eight_scales")
EXPECTED_BVR_HASH = "792445cfd99777c2e6eddb5dc781ec30c58b9dc8bf8650d931757d60a717c541"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json_text(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=CASES)
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Stop cleanly after this many new batches; no final CI claim")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--setting-workers", type=int, choices=(1, 3, 6), default=1,
                        help="Parallel pure setting fits; all RNG draws remain serial")
    args = parser.parse_args(argv)
    if args.point_only and args.smoke:
        parser.error("--point-only and --smoke are mutually exclusive")
    if not args.point_only and not args.smoke and args.replicates != 10000:
        parser.error("Final bootstrap requires exactly 10000 replicates; use --smoke otherwise")
    if not args.point_only and not 0 < args.replicates <= 10000:
        parser.error("replicates must lie in 1..10000")
    if args.smoke and args.replicates > 100:
        parser.error("Smoke is limited to100 replicates; launch production explicitly")
    if args.blas_threads < 1 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("Thread and batch limits must be positive")
    resolved = args.output_dir.resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("Outputs must remain below analysis_results/c1_followup_20261018")
    args.output_dir = resolved
    return args


def make_settings(lambdas, old):
    all_indices = np.arange(len(lambdas), dtype=int)
    six = np.flatnonzero(lambdas <= 2.2 + 5e-12)
    if len(all_indices) != 8 or len(six) != 6:
        raise AssertionError("Expected frozen8scale grid and6scales through2.2")
    specs = (
        ("full_lambda_minus2_ridge_1e-8", "Primary eight-scale C1", all_indices, 2., 1e-8),
        ("le2p2_lambda_minus2_ridge_1e-8", "Six-scale C1 through nominal2.2;120000shots", six, 2., 1e-8),
        ("full_uniform_ridge_1e-8", "Eight-scale C1 with uniform objective weights", all_indices, 0., 1e-8),
        ("full_lambda_minus2_ridge_1e-9", "Eight-scale C1 with ridge1e-9", all_indices, 2., 1e-9),
        ("full_lambda_minus2_ridge_1e-7", "Eight-scale C1 with ridge1e-7", all_indices, 2., 1e-7),
        ("equal_gate_full_lambda_minus2_ridge_1e-8", "Primary C1 with11188draws/scale", all_indices, 2., 1e-8),
    )
    settings = []
    suites = []
    for index, (key, description, indices, exponent, ridge) in enumerate(specs):
        suite = old._make_suite(lambdas[indices], weight_exponent=exponent, ridge=ridge, estimators=ESTIMATORS)
        settings.append({"key": key, "description": description, "indices": indices.tolist(),
                         "effective_scales": lambdas[indices].tolist(), "scale_count": len(indices),
                         "weight_exponent": exponent, "ridge_regularized_models": ridge,
                         "linear_ridge": 0., "per_scale_shots": GATE_SHOTS if index == 5 else 20000,
                         "zne_circuit_shots": len(indices) * (GATE_SHOTS if index == 5 else 20000),
                         "direct_reset_shots": DIRECT_SHOTS, "budget_index": int(index == 5),
                         "support_context_index": 2 if index == 5 else int(index == 1)})
        suites.append(suite)
    return settings, suites


def draw_budget_batch(grid_parent, extension_parent, per_scale_shots, rngs):
    """Exact original sensitivity draw order; always draw a complete25 batch."""
    counts = np.stack([rngs[0].multinomial(per_scale_shots, parent, size=BATCH)
                       for parent in grid_parent], axis=1).astype(np.float64)
    extension = rngs[1].multinomial(DIRECT_SHOTS - per_scale_shots, extension_parent, size=BATCH).astype(np.float64)
    return counts / per_scale_shots, (counts[:, 0] + extension) / DIRECT_SHOTS, {
        "scale_counts_sha256": array_sha(counts), "extension_counts_sha256": array_sha(extension),
        "per_scale_shots": per_scale_shots, "extension_shots": DIRECT_SHOTS - per_scale_shots,
    }


def evaluate(grids, resets, settings, suites, target, groups, denominator, core, bvr, *, keep_vectors=False, setting_workers=1):
    batch = grids[0].shape[0]
    dimension = grids[0].shape[-1]
    scalar = np.empty((6, 4, len(QUANTITIES), batch))
    selected = np.full((6, 4, batch), -1, dtype=np.int64)
    parameters = np.full((6, 4, 2, batch), np.nan)
    reset_metrics = np.stack([np.stack([bvr._post_metrics(reset, target, groups, denominator)[metric]
                                      for metric in METRICS]) for reset in resets])
    masks = np.empty((3, batch, dimension), dtype=bool)
    masks[0] = np.any(grids[0] > 0, axis=1)
    masks[1] = np.any(grids[0][:, settings[1]["indices"]] > 0, axis=1)
    masks[2] = np.any(grids[1] > 0, axis=1)
    vectors = {}

    def evaluate_setting(si):
        setting = settings[si]
        fixed, exponential = suites[si]
        scalar_row = np.empty((4, len(QUANTITIES), batch))
        selected_row = np.full((4, batch), -1, dtype=np.int64)
        parameters_row = np.full((4, 2, batch), np.nan)
        local_vectors = {}
        grid = grids[setting["budget_index"]][:, setting["indices"]]
        mask = masks[setting["support_context_index"]]
        indices = [np.flatnonzero(row) for row in mask]
        observed = [grid[b][:, index] for b, index in enumerate(indices)]
        result = core.analyze_compressed(observed, indices, dimension, fixed, exponential, variants=("C1",))["C1"]
        if result["fit_arithmetic"] != "observed_only":
            raise AssertionError("Non-C1 estimator entered followup")
        for mi, method in enumerate(METHODS):
            before = result["before"][method].materialize()
            after = result["after"][method].materialize()
            if np.any(before[~mask] != 0) or np.any(after[~mask] != 0):
                raise AssertionError("C1 outside-union coordinates must be exactlyzero")
            if np.min(after) < 0 or np.max(np.abs(after.sum(axis=1) - 1)) > 1e-10:
                raise FloatingPointError("C1 postprojection vector invalid")
            post = bvr._post_metrics(after, target, groups, denominator)
            diagnostic = result["diagnostics"][method]
            change_l2 = np.linalg.norm(after - grid[:, 0], axis=1)
            measured = {
                "tvd": post["tvd"],
                "normalized_grouped_peak_margin": post["normalized_grouped_peak_margin"],
                "reset_minus_zne_tvd": reset_metrics[setting["budget_index"], 0] - post["tvd"],
                "zne_minus_reset_normalized_grouped_peak_margin": post["normalized_grouped_peak_margin"] - reset_metrics[setting["budget_index"], 1],
                "negative_mass_before_projection": diagnostic["negative_mass_before_projection"],
                "negative_coordinate_fraction": diagnostic["negative_entry_fraction_before_projection"],
                "normalization_defect": diagnostic["normalization_error_before_projection"],
                "l1_projection_correction": diagnostic["l1_projection_correction"],
                "l2_projection_correction": diagnostic["l2_projection_correction"],
                "postprojection_support_size": diagnostic["support_size_after_projection"],
                "projected_change_from_scale_one_l2": change_l2,
                "projection_correction_to_change_ratio": diagnostic["l2_projection_correction"] / np.maximum(change_l2, np.finfo(float).eps),
            }
            scalar_row[mi] = np.stack([measured[q] for q in QUANTITIES])
            selection = result["selection"].get(method)
            if selection is not None:
                selected_row[mi] = selection["selected_index"]
                p = selection["selected_parameters"]
                parameters_row[mi, :p.shape[1]] = p.T
            if keep_vectors:
                local_vectors[f"{setting['key']}__before__{method}"] = before[0]
                local_vectors[f"{setting['key']}__after__{method}"] = after[0]
        return si, scalar_row, selected_row, parameters_row, local_vectors

    if setting_workers == 1:
        results = list(map(evaluate_setting, range(len(settings))))
    else:
        with ThreadPoolExecutor(max_workers=setting_workers) as pool:
            results = list(pool.map(evaluate_setting, range(len(settings))))
    # Assemble fixed slots in source-setting order, never completion order.
    for si, scalar_row, selected_row, parameters_row, local_vectors in results:
        scalar[si] = scalar_row
        selected[si] = selected_row
        parameters[si] = parameters_row
        vectors.update(local_vectors)
    if not np.all(np.isfinite(scalar)):
        raise FloatingPointError("Nonfinite C1 scalar output")
    contrasts = scalar[1:5, :, :2] - scalar[0:1, :, :2]
    contrasts[:, :, 0] *= -1   # positive favors sensitivity setting
    return {"scalar_draws": scalar, "selected_indices": selected,
            "selected_parameters": parameters, "reset_draws": reset_metrics,
            "packed_observed_support": np.packbits(masks, axis=-1, bitorder="little"),
            "support_sizes": masks.sum(axis=-1), "paired_setting_improvements": contrasts}, vectors


def scalar_rows(instance, settings, arrays, *, summaries=False, points=None):
    records = []
    for si, setting in enumerate(settings):
        for mi, method in enumerate(METHODS):
            record = {"instance": instance, "rate": RATE, "rate_percent": 100 * RATE,
                      "variant": "C1", "setting": setting["key"], "method": method,
                      "per_scale_shots": setting["per_scale_shots"],
                      "scale_count": setting["scale_count"], "zne_circuit_shots": setting["zne_circuit_shots"],
                      "reset_comparator_shots": DIRECT_SHOTS, "prevector_convention": "native_C1_zero_lift"}
            for qi, quantity in enumerate(QUANTITIES):
                values = arrays["scalar_draws"][si, mi, qi]
                record[f"point_{quantity}"] = float((points or arrays)["scalar_draws"][si, mi, qi, 0])
                if summaries:
                    low, median, high = np.quantile(values, [.025, .5, .975])
                    record[f"{quantity}_ci_low"] = float(low)
                    record[f"{quantity}_ci_high"] = float(high)
                    record[f"{quantity}_bootstrap_median"] = float(median)
            if summaries:
                record["replicates"] = arrays["scalar_draws"].shape[-1]
                record["interval_scope"] = "pointwise95percentile_conditional_saved_parents"
            records.append(record)
    return records


def source_provenance(instance, data, bvr, old, core):
    source_paths = [Path(__file__).resolve(), Path(old.__file__).resolve(), Path(core.__file__).resolve(),
                    *sorted((ROOT / "shor_error_mitigation/src/shor_qem").glob("*.py"))]
    baseline = ROOT / "analysis_results/sparse_zne_20261018/bootstrap_10000" / instance / "rate_0p0001/parent_estimates.npz"
    return {"sources": [{"path": str(p), "sha256": sha(p)} for p in source_paths],
            "inputs": bvr._manifest_rows(bvr.INSTANCE_SPECS[instance]),
            "existing_primary_C1": {"path": str(baseline), "sha256": sha(baseline)},
            "python": sys.version, "numpy": np.__version__}


def checkpoint_write(path, arrays, meta):
    meta = {**meta, "array_hashes": {key: array_sha(value) for key, value in arrays.items()}}
    atomic_npz(path, {**arrays, "checkpoint_json": np.asarray(json_text(meta))})


def checkpoint_read(path, identity, expected_start):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["checkpoint_json"]))
        arrays = {key: z[key] for key in z.files if key != "checkpoint_json"}
    if meta["identity"] != identity or meta["start"] != expected_start:
        raise RuntimeError("Checkpoint identity/order mismatch")
    if set(meta["array_hashes"]) != set(arrays):
        raise RuntimeError("Checkpoint array inventory mismatch")
    for key, value in arrays.items():
        if array_sha(value) != meta["array_hashes"][key]:
            raise RuntimeError(f"Checkpoint payload hash mismatch: {key}")
    return arrays, meta


def rng_states(rngs):
    return [[rng.bit_generator.state for rng in budget] for budget in rngs]


def run_case(instance, args, bvr, old, core):
    start_time = time.perf_counter()
    data = bvr._prepare_data(bvr.INSTANCE_SPECS[instance], np.asarray([RATE]), "realized")
    config = data["config"]
    grid = np.asarray(data["reset"][0])
    extension = np.asarray(data["reset_extension"][0])
    lambdas = np.asarray(data["fit_lambdas"])
    if data["lambda_one"] != 0 or bvr.FROZEN_BATCH_SIZE != BATCH:
        raise AssertionError("Frozen scale-one/batch convention changed")
    budget_ratio = GATE_SHOTS * float(lambdas.sum()) / DIRECT_SHOTS
    if budget_ratio > 1 or (GATE_SHOTS + 1) * float(lambdas.sum()) <= DIRECT_SHOTS:
        raise AssertionError("11188 is not the maximal integer equal-gate allocation")
    settings, suites = make_settings(lambdas, old)
    target = bvr.ideal_distribution(config["measurement_bits"], config["order"])
    groups = bvr.main_lobe_groups(config["measurement_bits"], config["order"])
    denominator = bvr.distribution_metrics(target, config["measurement_bits"], config["order"]).grouped_peak_margin
    reset_points = [(n * grid[0] + (DIRECT_SHOTS - n) * extension) / DIRECT_SHOTS for n in (20000, GATE_SHOTS)]
    if not np.allclose(reset_points[0], data["reset_full"][0], rtol=0, atol=5e-15):
        raise AssertionError("Primary Reset parent does not reconstruct160kcounts")
    kind = "point_only" if args.point_only else (f"smoke_B{args.replicates}" if args.smoke else "bootstrap_10000")
    output = args.output_dir / kind / instance
    existed = output.exists()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {output}") from error
        provenance = source_provenance(instance, data, bvr, old, core)
        configuration = {"instance": instance, "N": config["N"], "a": config["a"], "order": config["order"],
                         "L": config["measurement_bits"], "D": len(target), "rate": RATE,
                         "settings": settings, "methods": METHODS, "quantities": QUANTITIES,
                         "fit_lambdas": lambdas, "nominal_lambdas": data["nominal_lambdas"],
                         "seed": SEED, "batch_size": BATCH, "replicates": 0 if args.point_only else args.replicates,
                         "fitting_floor": 1 / len(target), "linear_ridge": 0., "blas_threads": args.blas_threads,
                         "setting_workers": args.setting_workers,
                         "setting_parallelism": "purefits_privatebuffers_orderedslots;serialRNGdraws",
                         "single_gamma_grid": ESTIMATORS.SINGLE_EXPONENTIAL_GRID,
                         "dual_gamma_grid": ESTIMATORS.DUAL_EXPONENTIAL_GRID,
                         "gamma_selection": "native_np_argmax_no_tolerance_or_borrowed_winner",
                         "support_contexts": SUPPORT_CONTEXTS, "packed_support_bitorder": "little",
                         "support_context_for_setting": [s["support_context_index"] for s in settings],
                         "sampling": "original_scale_by_scale_multinomial_draw_order_in_full25_batches",
                         "reset_pairing": {"equal_shot": [20000, 140000], "equal_gate": [11188, 148812]},
                         "setting_pairing": "allfive_sensitivitysettings_reuse_same8scale_counts_andResetextension",
                         "cross_budget_pairing": "NONE; no equalshot-minus-equalgate paired interval is constructed",
                         "gate_parent_center": "same_savedscale_parents_withResetreference_reweighted11188/148812;notnewreducedshotcircuitrecords",
                         "gate_cost_ratio_to_direct": budget_ratio,
                         "new_quantum_shots": 0, "new_noisy_circuit_simulations": 0}
        identity = hashlib.sha256(json_text({"configuration": configuration, "provenance": provenance}).encode()).hexdigest()
        manifest_path = output / "started_manifest.json"
        if existed and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["identity"] != identity:
                raise RuntimeError("Existing run has different source/input/settings identity; use an exclusive new output-dir")
            if not args.resume:
                raise FileExistsError("Existing run requires --resume; outputs will not be silently overwritten")
        elif existed and any(p.name != "run.lock" for p in output.iterdir()):
            raise RuntimeError("Unrecognized incomplete directory; use an exclusive new output-dir")
        else:
            atomic_json(manifest_path, {"schema_version": 1, "identity": identity, "configuration": configuration,
                                       "provenance": provenance, "started_utc": datetime.now(timezone.utc).isoformat()})
        fit_start = time.perf_counter()
        point, vectors = evaluate([grid[None], grid[None]], [r[None] for r in reset_points], settings, suites,
                                  target, groups, denominator, core, bvr, keep_vectors=True, setting_workers=args.setting_workers)
        with np.load(provenance["existing_primary_C1"]["path"], allow_pickle=False) as previous:
            for mi, method in enumerate(METHODS):
                expected = previous[f"C1__after__{method}"]
                error = np.max(np.abs(vectors[f"{settings[0]['key']}__after__{method}"] - expected))
                if error > 1e-11:
                    raise AssertionError(f"Primary C1 parent replay discrepancy: {method}, {error}")
        atomic_npz(output / "parent_estimates.npz", {
            **vectors, **point, "methods": np.asarray(METHODS), "settings": np.asarray([s["key"] for s in settings]),
            "quantities": np.asarray(QUANTITIES), "support_contexts": np.asarray(SUPPORT_CONTEXTS),
            "scale_parents": grid, "reset_extension_parent": extension,
            "reset_full_original": data["reset_full"][0], "raw_full_original": data["raw_full"][0],
            "reset_references": np.stack(reset_points), "target": target, "ideal_gap_denominator": denominator,
            "fit_lambdas": lambdas, "group_mask": np.stack([np.isin(np.arange(len(target)), g) for g in groups]),
            "configuration_json": np.asarray(json_text(configuration)), "identity": np.asarray(identity),
        })
        write_csv(output / "points.csv", scalar_rows(instance, settings, point))
        atomic_json(output / "point_manifest.json", {"identity": identity, "status": "points_complete", "parent_seconds": time.perf_counter() - fit_start,
                    "parent_archive_sha256": sha(output / "parent_estimates.npz"), "point_csv_sha256": sha(output / "points.csv"),
                    "primary_C1_replay_atol": 1e-11, "gate_cost_ratio_to_direct": budget_ratio,
                    "bootstrap_started": False, "new_quantum_shots": 0})
        print(f"[{instance}] POINTS SAVED {output/'points.csv'} (6settings×4models); gate_cost_ratio={budget_ratio:.12g}", flush=True)
        if args.point_only:
            return {"instance": instance, "status": "points_complete", "output": str(output), "seconds": time.perf_counter() - start_time}

        # Two separately defined budget experiments use the original frozen
        # seed hierarchy. Their equal-numbered replicates are NOT contrasted.
        rngs = [[np.random.default_rng(s) for s in bvr._bootstrap_seed_streams(SEED, instance, RATE)[:2]] for _ in range(2)]
        batch_dir = output / "batches"
        batch_dir.mkdir(exist_ok=True)
        completed = 0
        for path in sorted(batch_dir.glob("batch_*.npz")):
            _, meta = checkpoint_read(path, identity, completed)
            if json_text(meta["rng_before"]) != json_text(rng_states(rngs)):
                raise RuntimeError("Checkpoint RNG chain mismatch")
            completed = meta["stop"]
            for budget_i, states in enumerate(meta["rng_after"]):
                for stream_i, state in enumerate(states):
                    rngs[budget_i][stream_i].bit_generator.state = state
        if completed > args.replicates:
            raise RuntimeError("Checkpoint exceeds configured replicate count")
        new_batches = 0
        bootstrap_started = time.perf_counter()
        for start in range(completed, args.replicates, BATCH):
            stop = min(start + BATCH, args.replicates)
            before_states = rng_states(rngs)
            shot_grid, shot_reset, shot_audit = draw_budget_batch(grid, extension, 20000, rngs[0])
            gate_grid, gate_reset, gate_audit = draw_budget_batch(grid, extension, GATE_SHOTS, rngs[1])
            arrays, _ = evaluate([shot_grid, gate_grid], [shot_reset, gate_reset], settings, suites,
                                 target, groups, denominator, core, bvr, setting_workers=args.setting_workers)
            size = stop - start
            for key in arrays:
                # Packed support has replicate axis1; all other arrays use the
                # final axis except support_sizes(context,replicate).
                arrays[key] = arrays[key][:, :size] if key == "packed_observed_support" else arrays[key][..., :size]
            batch_path = batch_dir / f"batch_{start:06d}_{stop:06d}.npz"
            checkpoint_write(batch_path, arrays, {"identity": identity, "start": start, "stop": stop,
                             "full_draw_batch_size": BATCH, "rng_before": before_states, "rng_after": rng_states(rngs),
                             "draw_counts": {"equal_shot": shot_audit, "equal_gate": gate_audit}})
            # Immediately read back every atomic checkpoint and validate all
            # scalar/selection/support payload digests before advancing.
            checkpoint_read(batch_path, identity, start)
            new_batches += 1
            completed = stop
            elapsed = time.perf_counter() - bootstrap_started
            atomic_json(output / "progress.json", {"identity": identity, "status": "running" if stop < args.replicates else "bootstrap_draws_complete",
                        "completed": stop, "total": args.replicates, "last_checkpoint": batch_path.name,
                        "last_checkpoint_sha256": sha(batch_path), "seconds_this_invocation": elapsed,
                        "replicates_per_second_this_invocation": (stop - (completed - size) if new_batches == 1 else new_batches * BATCH) / max(elapsed, 1e-9),
                        "updated_utc": datetime.now(timezone.utc).isoformat()})
            if stop % 500 == 0 or new_batches == 1 or stop == args.replicates:
                print(f"[{instance}] {stop}/{args.replicates}, new_batches={new_batches}, elapsed={elapsed:.3f}s", flush=True)
            if args.max_batches is not None and new_batches >= args.max_batches:
                break
        if completed != args.replicates:
            return {"instance": instance, "status": "checkpointed_partial", "completed": completed, "output": str(output)}
        checkpoint_arrays = []
        checkpoint_hashes = []
        expected = 0
        for path in sorted(batch_dir.glob("batch_*.npz")):
            arrays, meta = checkpoint_read(path, identity, expected)
            expected = meta["stop"]
            checkpoint_arrays.append(arrays)
            checkpoint_hashes.append({"path": path.name, "sha256": sha(path), "start": meta["start"], "stop": meta["stop"]})
        aggregate = {key: np.concatenate([a[key] for a in checkpoint_arrays], axis=1 if key == "packed_observed_support" else -1)
                     for key in checkpoint_arrays[0]}
        atomic_npz(output / "scalar_bootstrap_draws.npz", {**aggregate, "settings": np.asarray([s["key"] for s in settings]),
                   "methods": np.asarray(METHODS), "quantities": np.asarray(QUANTITIES), "support_contexts": np.asarray(SUPPORT_CONTEXTS),
                   "scalar_axis_order": np.asarray(["setting", "method", "quantity", "replicate"]),
                   "selected_parameter_axis_order": np.asarray(["setting", "method", "parameter", "replicate"]),
                   "support_axis_order": np.asarray(["support_context", "replicate", "packed_byte"]),
                   "seed": SEED, "batch_size": BATCH, "replicates": args.replicates,
                   "configuration_json": np.asarray(json_text(configuration)), "identity": np.asarray(identity)})
        write_csv(output / "summary.csv", scalar_rows(instance, settings, aggregate, summaries=True, points=point))
        contrast_rows = []
        for ci, setting in enumerate(settings[1:5]):
            for mi, method in enumerate(METHODS):
                for qi, metric in enumerate(METRICS):
                    low, median, high = np.quantile(aggregate["paired_setting_improvements"][ci, mi, qi], [.025, .5, .975])
                    contrast_rows.append({"instance": instance, "setting": setting["key"], "reference": settings[0]["key"],
                         "method": method, "metric": metric, "positive_favors": "sensitivity_setting",
                         "point": float(point["paired_setting_improvements"][ci, mi, qi, 0]),
                         "ci_low": float(low), "ci_high": float(high), "bootstrap_median": float(median),
                         "replicates": args.replicates, "pairing": "shared_actual8scale_counts;subsetuses6only;notcrossbudget"})
        write_csv(output / "paired_sensitivity_contrasts.csv", contrast_rows)
        files = ["parent_estimates.npz", "points.csv", "scalar_bootstrap_draws.npz", "summary.csv", "paired_sensitivity_contrasts.csv"]
        seconds = time.perf_counter() - bootstrap_started
        atomic_json(output / "manifest.json", {"identity": identity, "status": "complete", "configuration": configuration,
                    "checkpoint_files": checkpoint_hashes, "artifacts": {name: sha(output / name) for name in files},
                    "completed_utc": datetime.now(timezone.utc).isoformat(), "seconds_this_invocation": seconds,
                    "total_wall_seconds_this_invocation": time.perf_counter() - start_time,
                    "new_batches_this_invocation": new_batches, "new_bootstrap_replicates_this_invocation": new_batches * BATCH,
                    "final_10000_contract": not args.smoke and args.replicates == 10000,
                    "no_cross_budget_paired_contrast": True, "new_quantum_shots": 0})
        return {"instance": instance, "status": "complete", "output": str(output),
                "replicates": args.replicates, "new_batches": new_batches, "seconds": seconds}


def main(argv=None):
    args = arguments(argv)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if ROOT.name != "paper-option2" or branch != "option2-metric-locked-revision":
        raise SystemExit("Wrong isolated worktree/branch")
    if not (ROOT / "codex_revision_1_2/prompts/02_option2_metric_locked_no_decoder.txt").is_file():
        raise SystemExit("Binding prompt missing")
    source = ROOT / "shor_error_mitigation/src/shor_qem/bias_variance_revision.py"
    if sha(source) != EXPECTED_BVR_HASH:
        raise SystemExit("Immutable saved-data helper hash changed")
    sys.path.insert(0, str(ROOT / "shor_error_mitigation/src"))
    from shor_qem import bias_variance_revision as bvr
    from shor_qem import estimators
    import run_scale_weight_sensitivity as old
    import sparse_zne_core_20261018 as core
    try:
        from threadpoolctl import threadpool_limits
        thread_context = threadpool_limits(limits=args.blas_threads)
    except ImportError:
        if os.environ.get("OPENBLAS_NUM_THREADS") != str(args.blas_threads):
            raise SystemExit("Without threadpoolctl, launch with OPENBLAS_NUM_THREADS matching --blas-threads")
        thread_context = nullcontext()
    global ESTIMATORS
    ESTIMATORS = estimators
    results = []
    with thread_context:
        for instance in (args.instance,) if args.instance else CASES:
            results.append(run_case(instance, args, bvr, old, core))
    print(json_text(results), flush=True)


if __name__ == "__main__":
    main()
