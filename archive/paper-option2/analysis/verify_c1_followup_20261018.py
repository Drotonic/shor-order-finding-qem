#!/usr/bin/env python3
"""Independent saved-data C1 follow-up verifier; never generate noisy shots.

The restricted toy reference below explicitly fits every candidate and evaluates
its weighted residual plus ridge penalty.  It does not use the production Gram
score or the production simplex implementation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/c1-followup-independent-verifier-mpl")
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE_ORDER = ("n119_a13", "n119_a50", "n119_a16")
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
EXPECTED_CORE_SHA = "78a83983595b34d1d4929cb68f082ed09a18e33419259011bd32bc942d0e0ac1"
SCALAR_NAMES = ("tvd", "normalized_grouped_peak_margin", "reset_minus_zne_tvd",
                "zne_minus_reset_normalized_grouped_peak_margin", "negative_mass_before_projection",
                "negative_coordinate_fraction", "normalization_defect", "l1_projection_correction",
                "l2_projection_correction", "postprojection_support_size", "projected_change_from_scale_one_l2",
                "projection_correction_to_change_ratio")
SUITE_CACHE = {}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(actual, expected, message, atol=0):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=0, equal_nan=True), message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_modules():
    check(ROOT.name == "paper-option2", "Wrong worktree")
    check(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "option2-metric-locked-revision", "Wrong branch")
    core_path = ROOT / "analysis/sparse_zne_core_20261018.py"
    check(sha(core_path) == EXPECTED_CORE_SHA, "Frozen C1 core changed")
    sys.path.insert(0, str(ROOT / "shor_error_mitigation/src"))
    sys.path.insert(0, str(ROOT / "analysis"))
    from shor_qem import estimators
    core = load_module("followup_independent_frozen_core", core_path)
    return estimators, core


def projection_bisection(x):
    """Solve the support-only simplex threshold without sorting/KKT helpers."""
    x = np.asarray(x)
    left, right = float(x.min() - 1), float(x.max())
    for _ in range(120):
        mid = (left + right) / 2
        if np.maximum(x - mid, 0).sum() > 1:
            left = mid
        else:
            right = mid
    result = np.maximum(x - (left + right) / 2, 0)
    return result / result.sum()


def independent_restricted(values, lambdas, *, method, weight_exponent, ridge, family=None):
    """Direct weighted least-squares on observed coordinates, including zeros."""
    dimension = values.shape[-1]
    observed = np.flatnonzero(np.any(values > 0, axis=0))
    p = values[:, observed]
    w = np.asarray(lambdas) ** (-weight_exponent)
    w /= w.mean()
    if method in METHODS[:2]:
        degree = 1 if method == METHODS[0] else 2
        eta = 0 if degree == 1 else ridge
        design = np.column_stack([np.asarray(lambdas) ** k for k in range(degree + 1)])
        beta = np.linalg.solve(design.T @ (w[:, None] * design) + eta * np.eye(degree + 1), design.T @ (w[:, None] * p))
        before = beta[0]
        selected = -1
    else:
        y = p - 1 / dimension
        errors, estimates = [], []
        for parameters in family.parameters:
            parameters = np.asarray(parameters).reshape(-1)
            design = np.exp(-np.asarray(lambdas)[:, None] * parameters[None, :])
            amplitude = np.linalg.solve(design.T @ (w[:, None] * design) + ridge * np.eye(len(parameters)), design.T @ (w[:, None] * y))
            residual = y - design @ amplitude
            errors.append(float((w[:, None] * residual * residual).sum() + ridge * np.square(amplitude).sum()))
            estimates.append(1 / dimension + amplitude.sum(axis=0))
        selected = int(np.argmin(errors))
        before = estimates[selected]
    full_before = np.zeros(dimension)
    full_after = np.zeros(dimension)
    full_before[observed] = before
    full_after[observed] = projection_bisection(before)
    return full_before, full_after, selected


def verify_restricted_toys(estimators, core):
    toy_count = 0
    for dimension in (16, 31):
        for indices in (np.array([0, 3, dimension - 1]), np.arange(dimension)):
            lambdas = np.array([1., 1.35, 1.8, 2.4])
            p = np.zeros((4, dimension))
            ii, jj = np.indices((4, len(indices)))
            values = .1 + ((17 * ii + 7 * jj + dimension + 3) % 29) / 23
            if len(indices) == 3:
                # A zero at one scale must still enter the fit when seen elsewhere.
                values[0, 1] = 0
            p[:, indices] = values / values.sum(axis=1, keepdims=True)
            for exponent in (0., 2.):
                for ridge in (1e-8, .03):
                    fixed = {"linear_zne": estimators.polynomial_intercept_coefficients(lambdas, degree=1, ridge=0, weight_exponent=exponent),
                             "quadratic_zne": estimators.polynomial_intercept_coefficients(lambdas, degree=2, ridge=ridge, weight_exponent=exponent)}
                    families = {
                        "single_exponential_zne": estimators.build_exponential_grid_family(lambdas, components=1, single_grid=np.array([.08, .3, .75, 1.8]), ridge=ridge, weight_exponent=exponent),
                        "dual_exponential_zne": estimators.build_exponential_grid_family(lambdas, components=2, dual_grid=np.array([0., .2, .8, 2.]), ridge=ridge, weight_exponent=exponent),
                    }
                    support = core.ObservedBatch.from_dense(p[None])
                    for method in METHODS:
                        actual = core.infer_model(method, "C1", fixed_coefficients=fixed, exponential_families=families, support=support)
                        before, after, winner = independent_restricted(p, lambdas, method=method, weight_exponent=exponent, ridge=ridge, family=families.get(method))
                        equal(actual["before"].materialize()[0], before, "Independent observed-only prevector mismatch", atol=5e-11)
                        equal(actual["after"].materialize()[0], after, "Independent support projection mismatch", atol=5e-11)
                        if winner >= 0:
                            check(int(actual["selection"]["selected_index"][0]) == winner, "Observed-only candidate winner differs from direct penalized residual")
                        toy_count += 1
    print(f"PASS {toy_count} independent restricted toy fits: mean-normalized weights, fixed1/Doffset, all-scale zeros, native gamma selection and support-only bisection projection", flush=True)


def independent_gap(probability, order):
    d = len(probability)
    z = np.arange(d)
    groups = []
    for s in range(order):
        diff = np.abs(order * z - d * s)
        groups.append(np.flatnonzero(np.minimum(diff, order * d - diff) < order))
    outside = np.ones(d, dtype=bool)
    outside[np.concatenate(groups)] = False
    return min(float(probability[g].sum()) for g in groups) - float(probability[outside].max())


def metrics(probability, target, order):
    return np.array([np.abs(probability - target).sum() / 2,
                     independent_gap(probability, order) / independent_gap(target, order)])


def suite_for(setting, estimators):
    lambdas = np.asarray(setting["effective_scales"])
    exponent, ridge = setting["weight_exponent"], setting["ridge_regularized_models"]
    key = tuple(lambdas), exponent, ridge
    if key not in SUITE_CACHE:
        fixed = {"linear_zne": estimators.polynomial_intercept_coefficients(lambdas, degree=1, ridge=0., weight_exponent=exponent),
                 "quadratic_zne": estimators.polynomial_intercept_coefficients(lambdas, degree=2, ridge=ridge, weight_exponent=exponent)}
        families = {"single_exponential_zne": estimators.build_exponential_grid_family(lambdas, 1, ridge=ridge, weight_exponent=exponent),
                    "dual_exponential_zne": estimators.build_exponential_grid_family(lambdas, 2, ridge=ridge, weight_exponent=exponent)}
        SUITE_CACHE[key] = fixed, families
    return SUITE_CACHE[key]


def scalar_reference(before, after, original_scale_one, reset, target, order):
    post, baseline = metrics(after, target, order), metrics(reset, target, order)
    change = np.linalg.norm(after - original_scale_one)
    correction = np.linalg.norm(after - before)
    return np.array([*post, baseline[0] - post[0], post[1] - baseline[1],
                     -np.minimum(before, 0).sum(), np.count_nonzero(before < 0) / len(before),
                     abs(before.sum() - 1), np.abs(after - before).sum(), correction,
                     np.count_nonzero(after > 0), change, correction / max(change, np.finfo(float).eps)])


def check_settings(config):
    settings = config["settings"]
    check(len(settings) == 6 and config["rate"] == .0001 and config["D"] == 16384, "Wrong six-setting primary-rate scope")
    check(config["seed"] == 20260804 and config["batch_size"] == 25 and config["fitting_floor"] == 1 / 16384, "Seed/batch/offset changed")
    check(config["reset_pairing"] == {"equal_gate": [11188, 148812], "equal_shot": [20000, 140000]}, "Reset coupling convention changed")
    check(config["cross_budget_pairing"].startswith("NONE;"), "Cross-budget paired inference is not authorized by the sampling design")
    equal(config["single_gamma_grid"], np.linspace(.01, 10, 200), "Single rate grid changed")
    equal(config["dual_gamma_grid"], np.linspace(0, 10, 200), "Dual rate grid changed")
    indices = (list(range(8)), list(range(6)), list(range(8)), list(range(8)), list(range(8)), list(range(8)))
    for i, setting in enumerate(settings):
        check(setting["indices"] == indices[i], "Sensitivity scale subset changed")
        check(setting["weight_exponent"] == (0 if i == 2 else 2), "Weight setting changed")
        check(setting["ridge_regularized_models"] == (1e-9 if i == 3 else 1e-7 if i == 4 else 1e-8), "Ridge setting changed")
        check(setting["linear_ridge"] == 0., "Linear must remain unregularized")
        n = 11188 if i == 5 else 20000
        check(setting["per_scale_shots"] == n and setting["zne_circuit_shots"] == n * len(indices[i]), "120k/160k/89504 resource accounting wrong")
        equal(setting["effective_scales"], np.asarray(config["fit_lambdas"])[indices[i]], "Nominal scales substituted")
    check(11188 * sum(config["fit_lambdas"]) <= 160000 < 11189 * sum(config["fit_lambdas"]), "Integer gate-matched allocation wrong")
    return settings


def verify_followup_points(folder, estimators):
    """Validate all24 models/settings against original parents and direct fits."""
    started = json.loads((folder / "started_manifest.json").read_text())
    config, provenance = started["configuration"], started["provenance"]
    settings = check_settings(config)
    for item in provenance["sources"]:
        check(sha(item["path"]) == item["sha256"], "Follow-up source identity changed")
    for item in provenance["inputs"]:
        check(sha(item["path"]) == item["sha256"], "Original follow-up input changed")
    baseline = provenance["existing_primary_C1"]
    check(sha(baseline["path"]) == baseline["sha256"], "Frozen primary C1 cache changed")
    producer = ROOT / "analysis/run_c1_followup_20261018.py"
    tree = ast.parse(producer.read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    check(not any("_fit_zne_batch_detailed" in n or "analyse_instance" in n for n in calls), "Follow-up accidentally calls dense analysis")
    check("core.analyze_compressed" in calls and 'variants=(\'C1\',)' in ast.unparse(tree), "Follow-up route is not explicitly C1-only")
    point_manifest = json.loads((folder / "point_manifest.json").read_text())
    check(sha(folder / "parent_estimates.npz") == point_manifest["parent_archive_sha256"] and sha(folder / "points.csv") == point_manifest["point_csv_sha256"], "Point outputs changed")
    with (folder / "points.csv").open(newline="") as f:
        point_rows = {(r["setting"], r["method"]): r for r in csv.DictReader(f)}
    check(len(point_rows) == 24, "Missing/duplicate point setting rows")
    with np.load(folder / "parent_estimates.npz", allow_pickle=False) as saved, np.load(baseline["path"], allow_pickle=False) as old:
        grid, extension, target = saved["scale_parents"], saved["reset_extension_parent"], saved["target"]
        equal(grid, old["scale_parents"], "Original scale parents altered")
        equal(target, old["target"], "Ideal target altered")
        equal(saved["fit_lambdas"], old["fit_lambdas"], "Effective grid altered")
        equal(saved["reset_full_original"], old["reset_160k"], "Original Reset160k changed")
        equal(saved["raw_full_original"], old["raw_160k"], "Original Raw160k changed")
        expected_ext = (old["reset_160k"] * 160000 - grid[0] * 20000) / 140000
        equal(extension, expected_ext, "Reset extension parent not original160k minus20k", atol=5e-17)
        resets = np.stack([(n * grid[0] + (160000 - n) * extension) / 160000 for n in (20000, 11188)])
        equal(saved["reset_references"], resets, "Gate/equal-shot parent Reset reference incorrectly mixed")
        equal(saved["reset_draws"][:, :, 0], np.stack([metrics(r, target, config["order"]) for r in resets]), "Reset parent metric convention", atol=2e-14)
        check(saved["quantities"].astype(str).tolist() == list(SCALAR_NAMES), "Scalar diagnostic axis changed")
        for si, setting in enumerate(settings):
            train = grid[setting["indices"]]
            observed = np.any(train > 0, axis=0)
            fixed, families = suite_for(setting, estimators)
            w = np.asarray(setting["effective_scales"]) ** -setting["weight_exponent"]
            w /= w.mean()
            for mi, method in enumerate(METHODS):
                before = saved[f"{setting['key']}__before__{method}"]
                after = saved[f"{setting['key']}__after__{method}"]
                check(np.count_nonzero(before[~observed]) == np.count_nonzero(after[~observed]) == 0, "C1 support contaminated by excluded scale/ideal outcomes")
                if method in METHODS[:2]:
                    degree = 1 if mi == 0 else 2
                    ridge = 0 if mi == 0 else setting["ridge_regularized_models"]
                    design = np.column_stack([np.asarray(setting["effective_scales"]) ** j for j in range(degree + 1)])
                    beta = np.linalg.solve(design.T @ (w[:, None] * design) + ridge * np.eye(degree + 1), design.T @ (w[:, None] * train[:, observed]))
                    expected_before = beta[0]
                    check(int(saved["selected_indices"][si, mi, 0]) == -1, "Polynomial has spurious gamma")
                else:
                    family = families[method]
                    _, info = estimators.fit_exponential_grid_batch(train[:, observed], family, uniform_floor=1 / 16384)
                    winner = int(info["selected_index"])
                    check(int(saved["selected_indices"][si, mi, 0]) == winner, "Shared rate chosen from wrong support/objective")
                    gamma = family.parameters[winner]
                    design = np.exp(-np.asarray(setting["effective_scales"])[:, None] * gamma[None, :])
                    normal = design.T @ (w[:, None] * design) + setting["ridge_regularized_models"] * np.eye(len(gamma))
                    amplitude = np.linalg.solve(normal, design.T @ (w[:, None] * (train[:, observed] - 1 / 16384)))
                    expected_before = 1 / 16384 + amplitude.sum(axis=0)
                equal(before[observed], expected_before, "C1 parent restricted weighted fit differs", atol=3e-10)
                equal(after[observed], projection_bisection(before[observed]), "C1 projection not restricted Euclidean simplex", atol=3e-11)
                expected_scalar = scalar_reference(before, after, grid[0], resets[int(si == 5)], target, config["order"])
                equal(saved["scalar_draws"][si, mi, :, 0], expected_scalar, "Parent scalar/diagnostic mismatch", atol=5e-11)
                row = point_rows[setting["key"], method]
                equal([float(row["point_" + q]) for q in SCALAR_NAMES], expected_scalar, "Point CSV not parent plug-in", atol=5e-11)
                check(row["variant"] == "C1" and float(row["rate_percent"]) == .01, "Variant/rate units mislabeled")
                if si in (0, 5):
                    equal(after, old["C1__after__" + method], "Primary C1 plug-in replay differs from approved parent", atol=1e-11)
        print(f"PASS {config['instance']} all24C1parent settings: original inputs, primary replay, native restricted fits/projections and11188+148812 Reset reference", flush=True)
        return config, {k: saved[k].copy() for k in saved.files}


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def verify_followup_bootstrap(folder, config, parent, estimators, core):
    """Check all cached reductions and independently replay the first batch."""
    manifest = json.loads((folder / "manifest.json").read_text())
    check(manifest["status"] == "complete" and manifest["no_cross_budget_paired_contrast"] is True, "Incomplete/cross-budget follow-up")
    for filename, expected in manifest["artifacts"].items():
        check(sha(folder / filename) == expected, "Follow-up aggregate artifact changed")
    settings, instance = config["settings"], config["instance"]
    children = np.random.SeedSequence([20260804, zlib.crc32(instance.encode()), round(.0001 * 1e9)]).spawn(4)
    initial = [[np.random.default_rng(s) for s in children[:2]] for _ in range(2)]
    grid, extension, target = parent["scale_parents"], parent["reset_extension_parent"], parent["target"]
    previous_after, stop = None, 0
    with np.load(folder / "scalar_bootstrap_draws.npz", allow_pickle=False) as archive:
        # NPZ fields are decompressed on every __getitem__; cache each once
        # before comparing400 checkpoints, otherwise this rereads tens ofGB.
        aggregate = {name: archive[name] for name in archive.files}
        total = int(aggregate["replicates"])
        check(total == config["replicates"] and aggregate["scalar_draws"].shape == (6, 4, 12, total), "Bootstrap shape/replicate count")
        check(int(aggregate["seed"]) == 20260804 and int(aggregate["batch_size"]) == 25, "Bootstrap seed/batch changed")
        for batch_index, entry in enumerate(manifest["checkpoint_files"]):
            path = folder / "batches" / entry["path"]
            check(sha(path) == entry["sha256"], "Checkpoint file changed")
            with np.load(path, allow_pickle=False) as batch:
                meta = json.loads(str(batch["checkpoint_json"]))
                check(meta["start"] == stop and meta["stop"] > stop and meta["full_draw_batch_size"] == 25, "Checkpoint gap/duplicate/batch size")
                start, stop = meta["start"], meta["stop"]
                check(meta["identity"] == manifest["identity"], "Checkpoint identity changed")
                for name, digest in meta["array_hashes"].items():
                    check(array_digest(batch[name]) == digest, "Checkpoint payload digest wrong")
                    saved = aggregate[name][:, start:stop] if name == "packed_observed_support" else aggregate[name][..., start:stop]
                    equal(saved, batch[name], "Aggregate/checkpoint array mismatch")
                if previous_after is None:
                    expected_before = [[r.bit_generator.state for r in budget] for budget in initial]
                    check(meta["rng_before"] == expected_before, "Bootstrap initial RNG not frozen seed hierarchy")
                else:
                    check(meta["rng_before"] == previous_after, "Checkpoint RNG chain broken")
                previous_after = meta["rng_after"]
                packed = batch["packed_observed_support"]
                check(np.all(np.isfinite(batch["scalar_draws"])), "Nonfinite bootstrap scalar/diagnostic")
                tvd = batch["scalar_draws"][:, :, 0]
                check(tvd.min() >= -1e-13 and tvd.max() <= 1 + 1e-13, "Postprojection TVD outside probability-distribution range")
                masks = np.unpackbits(packed, axis=-1, count=16384, bitorder="little").astype(bool)
                equal(masks.sum(axis=-1), batch["support_sizes"], "True support counts differ from packed masks")
                # Paired setting differences must be differences of the SAME
                # scalar replicate arrays, never marginal endpoint arithmetic.
                contrasts = batch["scalar_draws"][1:5, :, :2] - batch["scalar_draws"][0:1, :, :2]
                contrasts[:, :, 0] *= -1
                equal(batch["paired_setting_improvements"], contrasts, "Sensitivity pairing corrupted")
                if batch_index != 0:
                    continue
                grids, resets = [], []
                for bi, n in enumerate((20000, 11188)):
                    counts = np.stack([initial[bi][0].multinomial(n, p, size=25) for p in grid], axis=1).astype(float)
                    ext = initial[bi][1].multinomial(160000 - n, extension, size=25).astype(float)
                    audit = meta["draw_counts"]["equal_shot" if bi == 0 else "equal_gate"]
                    check(array_digest(counts) == audit["scale_counts_sha256"] and array_digest(ext) == audit["extension_counts_sha256"], "Independent multinomial count replay mismatch")
                    grids.append(counts / n)
                    resets.append((counts[:, 0] + ext) / 160000)
                check(meta["rng_after"] == [[r.bit_generator.state for r in budget] for budget in initial], "Independent RNG-after replay mismatch")
                size = stop - start
                independent_masks = np.stack([np.any(grids[0] > 0, axis=1), np.any(grids[0][:, :6] > 0, axis=1), np.any(grids[1] > 0, axis=1)])
                equal(masks, independent_masks[:, :size], "Actual draw support not recomputed per setting/budget/replicate")
                for bi, reset in enumerate(resets):
                    measured = np.stack([metrics(p, target, config["order"]) for p in reset], axis=-1)
                    equal(batch["reset_draws"][bi], measured[:, :size], "Reset bootstrap not matched to scale-one11188/20000 plus correct extension", atol=3e-14)
                for si, setting in enumerate(settings):
                    selected_grid = grids[setting["budget_index"]][:, setting["indices"]]
                    observed_masks = np.any(selected_grid > 0, axis=1)
                    ids = [np.flatnonzero(m) for m in observed_masks]
                    values = [selected_grid[b][:, index] for b, index in enumerate(ids)]
                    fixed, families = suite_for(setting, estimators)
                    result = core.analyze_compressed(values, ids, 16384, fixed, families, variants=("C1",))["C1"]
                    for mi, method in enumerate(METHODS):
                        before, after = result["before"][method].materialize(), result["after"][method].materialize()
                        expected_scalars = np.stack([scalar_reference(before[b], after[b], selected_grid[b, 0], resets[setting["budget_index"]][b], target, config["order"]) for b in range(size)], axis=-1)
                        equal(batch["scalar_draws"][si, mi], expected_scalars, "Independent first25 C1 inference/scalar replay mismatch", atol=6e-11)
                        if mi >= 2:
                            equal(batch["selected_indices"][si, mi], result["selection"][method]["selected_index"][:size], "Bootstrap sharedgamma candidate mismatch")
                            chosen = result["selection"][method]["selected_parameters"][:size]
                            equal(batch["selected_parameters"][si, mi, :chosen.shape[1]], chosen.T, "Bootstrap selectedgamma values differ from their candidate indices")
        check(stop == total, "Checkpoint coverage incomplete")
        with (folder / "summary.csv").open(newline="") as f:
            rows = {(r["setting"], r["method"]): r for r in csv.DictReader(f)}
        check(len(rows) == 24, "Missing bootstrap summary rows")
        for si, setting in enumerate(settings):
            for mi, method in enumerate(METHODS):
                row = rows[setting["key"], method]
                for qi, quantity in enumerate(SCALAR_NAMES):
                    expected = [parent["scalar_draws"][si, mi, qi, 0], *np.quantile(aggregate["scalar_draws"][si, mi, qi], [.025, .5, .975])]
                    actual = [float(row["point_" + quantity]), *[float(row[quantity + suffix]) for suffix in ("_ci_low", "_bootstrap_median", "_ci_high")]]
                    equal(actual, expected, "Summary recentered parent or incorrect percentile")
        with (folder / "paired_sensitivity_contrasts.csv").open(newline="") as f:
            rows = {(r["setting"], r["method"], r["metric"]): r for r in csv.DictReader(f)}
        check(len(rows) == 32, "Missing paired sensitivity contrast rows")
        for ci, setting in enumerate(settings[1:5]):
            for mi, method in enumerate(METHODS):
                for qi, metric_name in enumerate(SCALAR_NAMES[:2]):
                    row = rows[setting["key"], method, metric_name]
                    equal([float(row[k]) for k in ("point", "ci_low", "bootstrap_median", "ci_high")],
                          [parent["paired_setting_improvements"][ci, mi, qi, 0], *np.quantile(aggregate["paired_setting_improvements"][ci, mi, qi], [.025, .5, .975])], "Paired contrast not actual same-replicate differences")
    print(f"PASS {instance} all{total}cached draws/checkpoints/paired reductions; first25 independently replay originalcounts,RNG,three true supports,C1 fits,gamma andReset coupling", flush=True)


def require_final_contract(config, manifest):
    check(config["replicates"] == 10000 and manifest.get("final_10000_contract") is True and manifest.get("status") == "complete", "Smoke/partial output cannot be publication10000 evidence")


def verify_artifacts(directory, estimators, core, *, require_final=False):
    directory = Path(directory)
    cases = [c for c in CASE_ORDER if (directory / c / "parent_estimates.npz").is_file()]
    check(cases == list(CASE_ORDER), "All three C1 follow-up cases are required")
    for instance in cases:
        folder = directory / instance
        config, parent = verify_followup_points(folder, estimators)
        if require_final:
            require_final_contract(config, json.loads((folder / "manifest.json").read_text()))
        if (folder / "manifest.json").is_file():
            verify_followup_bootstrap(folder, config, parent, estimators, core)
        else:
            check(config["replicates"] == 0, "Bootstrap case not complete")
    if (directory / "aggregate_summary.json").exists():
        verify_aggregates(directory)


def verify_aggregates(directory):
    """Check the combined72 summaries/96 paired contrasts/24 budget rows."""
    directory = Path(directory)
    report = json.loads((directory / "aggregate_summary.json").read_text())
    check(report["status"] == "complete" and report["cross_budget_paired_inference"] is False, "Combined summary incomplete/incorrect pairing")
    check(report["runner_sha256"] == sha(ROOT / "analysis/run_c1_followup_20261018.py"), "Aggregate source not frozen runner")
    summaries, contrasts = [], []
    for case, config in zip(CASE_ORDER, report["cases"]):
        manifest = json.loads((directory / case / "manifest.json").read_text())
        check(config["instance"] == case and config["identity"] == manifest["identity"] and config["replicates"] == 10000, "Aggregate source identity/order/replicates changed")
        check(config["manifest_sha256"] == sha(directory / case / "manifest.json"), "Aggregate per-case manifest hash wrong")
        with (directory / case / "summary.csv").open(newline="") as f:
            summaries.extend(csv.DictReader(f))
        with (directory / case / "paired_sensitivity_contrasts.csv").open(newline="") as f:
            contrasts.extend(csv.DictReader(f))
    with (directory / "summary_all_cases.csv").open(newline="") as f:
        actual = list(csv.DictReader(f))
    check(actual == summaries and len(actual) == 72, "Combined parent/interval summaries not exact case concatenation")
    with (directory / "paired_sensitivity_all_cases.csv").open(newline="") as f:
        actual = list(csv.DictReader(f))
    check(actual == contrasts and len(actual) == 96, "Combined paired contrasts not exact96case records")
    with (directory / "budget_improvements_all_cases.csv").open(newline="") as f:
        budget = list(csv.DictReader(f))
    check(len(budget) == 24, "Combined budget table coverage")
    tolerance = report["sign_classification_tolerance"]
    check(tolerance == 1e-10, "Classification numerical tolerance changed")
    for row in budget:
        original = next(r for r in summaries if all(r[k] == row[k] for k in ("instance", "setting", "method")))
        for alias, quantity in (("tvd", SCALAR_NAMES[2]), ("gap", SCALAR_NAMES[3])):
            values = [float(original[k]) for k in ("point_" + quantity, quantity + "_ci_low", quantity + "_ci_high")]
            equal([float(row[alias + suffix]) for suffix in ("_improvement_point", "_improvement_ci_low", "_improvement_ci_high")], values, "Budget aggregate not paired-to-specificReset source")
            classification = "positive" if values[1] > tolerance else "negative" if values[2] < -tolerance else "unresolved"
            check(row[alias + "_interval_sign"] == classification, "Aggregate interval sign classification wrong")
    print("PASS combined72setting/model summaries+96pairedsensitivityrows+24budgetrows/sourceidentities/classification rebuiltindependently", flush=True)


def verify_negative_controls(directory):
    """Reject concrete failure modes in memory; no artifact is overwritten."""
    directory = Path(directory)
    rejected = []
    def reject(name, callback):
        try:
            callback()
        except (AssertionError, ValueError):
            rejected.append(name)
        else:
            raise AssertionError("Negative control unexpectedly accepted: " + name)
    folder = directory / CASE_ORDER[0]
    config = json.loads((folder / "started_manifest.json").read_text())["configuration"]
    bad = copy.deepcopy(config)
    bad["settings"][1]["zne_circuit_shots"] = 128000
    reject("six-scale128kclaim", lambda: check_settings(bad))
    bad_pairing = copy.deepcopy(config)
    bad_pairing["cross_budget_pairing"] = "paired equalshot minus equalgate by replicate index"
    reject("crossbudgetpairedinference", lambda: check_settings(bad_pairing))
    smoke = copy.deepcopy(config)
    smoke["replicates"] = 25
    reject("smoke25asfinal10000", lambda: require_final_contract(smoke, {"status": "complete", "final_10000_contract": False}))
    with np.load(folder / "parent_estimates.npz", allow_pickle=False) as parent:
        reject("wronggateResetparentcenter", lambda: equal(parent["reset_full_original"], parent["reset_references"][1], "Gate center must use11188/148812", atol=1e-13))
        first_path = sorted((folder / "batches").glob("batch_*.npz"))[0]
        with np.load(first_path, allow_pickle=False) as first:
            fixed = np.repeat(parent["packed_observed_support"], first["packed_observed_support"].shape[1], axis=1)
            reject("fixedparentUinbootstrap", lambda: equal(fixed, first["packed_observed_support"], "Support must follow each actual resample"))
    # Use an actual historical dense interval, not an invented number. Linear
    # can be invariant, so seek a genuinely discriminating exponential series.
    dense_rejected = False
    for instance in CASE_ORDER:
        with np.load(directory / instance / "scalar_bootstrap_draws.npz", allow_pickle=False) as c1, np.load(ROOT / "analysis_results/equal_gate_realized" / instance / "realized/bootstrap_cache.npz", allow_pickle=False) as dense:
            ri = int(np.flatnonzero(dense["rates"] == .0001)[0])
            dense_routes, dense_metrics = dense["routes"].astype(str).tolist(), dense["metrics"].astype(str).tolist()
            for mi in (2, 3):
                for qi, quantity in enumerate(SCALAR_NAMES[:2]):
                    expected = np.quantile(c1["scalar_draws"][5, mi, qi], [.025, .975])
                    wrong = np.quantile(dense["postprojection_draws"][ri, dense_routes.index(METHODS[mi]), dense_metrics.index(quantity)], [.025, .975])
                    if np.max(np.abs(expected - wrong)) > 1e-10:
                        reject("historicalDenseintervalasC1", lambda: equal(wrong, expected, "Dense interval is not a C1 interval", atol=1e-10))
                        dense_rejected = True
                        break
                if dense_rejected:
                    break
        if dense_rejected:
            break
    check(dense_rejected and len(rejected) == 6, "No discriminating dense-vs-C1 negative control")
    print("PASS6negativecontrols rejected:" + ";".join(rejected), flush=True)


def verify_parallel_review(directory):
    directory = Path(directory)
    report = json.loads((directory / "verification_summary.json").read_text())
    check(sha(ROOT / report["source"]) == report["source_sha256"], "Parallel review source no longer frozen")
    keys = ("scalar_draws", "selected_indices", "selected_parameters", "reset_draws",
            "packed_observed_support", "support_sizes", "paired_setting_improvements")
    case = report["single_case_single_batch_benchmark"]["instance"]
    baseline = directory / "workers1/smoke_B25" / case
    count = 0
    with np.load(baseline / "scalar_bootstrap_draws.npz", allow_pickle=False) as first:
        for workers in (3, 6):
            folder = directory / f"workers{workers}/smoke_B25" / case
            with np.load(folder / "scalar_bootstrap_draws.npz", allow_pickle=False) as other:
                check(json.loads(str(other["configuration_json"]))["setting_workers"] == workers, "Parallel smoke worker count wrong")
                for key in keys:
                    a, b = first[key], other[key]
                    check(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), "Thread workers alter cached numeric bits: " + key)
                    count += 1
            with np.load(baseline / "parent_estimates.npz", allow_pickle=False) as a, np.load(folder / "parent_estimates.npz", allow_pickle=False) as b:
                check(set(a.files) == set(b.files), "Parallel parent inventory changed")
                for key in a.files:
                    if key in ("identity", "configuration_json"):
                        continue
                    check(a[key].tobytes() == b[key].tobytes(), "Parallel parent arrays changed: " + key)
    resumed = ROOT / report["resume_comparison"]["v2_resume_path"]
    reference = ROOT / report["resume_comparison"]["uninterrupted_reference_path"]
    with np.load(resumed / "scalar_bootstrap_draws.npz", allow_pickle=False) as a, np.load(reference / "scalar_bootstrap_draws.npz", allow_pickle=False) as b:
        check(int(a["replicates"]) == int(b["replicates"]) == 50, "Resume smoke replicate contract")
        for key in keys:
            check(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes(), "Resume changes numeric bits: " + key)
            count += 1
    resumed_paths, original_paths = sorted((resumed / "batches").glob("batch_*.npz")), sorted((reference / "batches").glob("batch_*.npz"))
    check(len(resumed_paths) == len(original_paths) == 2, "Resume duplicates/misses checkpoint")
    for apath, bpath in zip(resumed_paths, original_paths):
        with np.load(apath, allow_pickle=False) as a, np.load(bpath, allow_pickle=False) as b:
            am, bm = json.loads(str(a["checkpoint_json"])), json.loads(str(b["checkpoint_json"]))
            for key in ("start", "stop", "rng_before", "rng_after", "draw_counts"):
                check(am[key] == bm[key], "Resume changes RNG/count checkpoint chain")
    print(f"PASS parallel/resume independent bitwise audit:{count}numeric array families; workers1/3/6parents unchanged and interrupted25→50matchesuninterrupted50includingRNG/counts", flush=True)


def verify_sm_v2_figures(directory):
    """Inspect final budget/sensitivity artists without saving a figure."""
    directory = Path(directory).resolve()
    scripts = ROOT / "figures_c1/scripts"
    sys.path.insert(0, str(scripts))
    import matplotlib.pyplot as plt
    budget = load_module("independent_sm_v2_budget", scripts / "plot_supplement_comparisons_v2.py")
    sensitivity = load_module("independent_sm_v2_sensitivity", scripts / "plot_c1_sensitivity_v2.py")
    for case in CASE_ORDER:
        final = json.loads((directory / case / "manifest.json").read_text())
        require_final_contract(final["configuration"], final)
    verify_aggregates(directory)
    budget.apply_style()
    captures = []
    budget.finish = lambda fig, stem, metadata: captures.append((fig, stem, metadata))
    budget.budget_comparison(followup_root=directory)
    check(len(captures) == 3, "Budget companion plus selected core/dual figures required")
    by_stem = {stem: (fig, payload) for fig, stem, payload in captures}
    stem = "comparison_C1_Reset_two_budgets"
    fig, payload = by_stem[stem]
    check(payload["publication_ready"] is True and len(payload["records"]) == 48 and len(fig.axes) == 6, "Final budget figure coverage/publication gate")
    check("no cross-budget paired inference" in payload["budget_sampling_relationship"], "Budget sampling relationship misrepresented")
    saved = json.loads((ROOT / "figures_c1/supplement_v2" / (stem + "_plotdata.json")).read_text())
    check(saved["records"] == payload["records"] and saved["publication_ready"] is True, "Saved budget plot stale/draft")
    summaries = {}
    for case in CASE_ORDER:
        with (directory / case / "summary.csv").open(newline="") as f:
            summaries[case] = {(r["setting"], r["method"]): r for r in csv.DictReader(f)}
    budget_settings = ("full_lambda_minus2_ridge_1e-8", "equal_gate_full_lambda_minus2_ridge_1e-8")
    for ci, case in enumerate(CASE_ORDER):
        for qi, quantity in enumerate(SCALAR_NAMES[2:4]):
            ax = fig.axes[2 * ci + qi]
            check(len(ax.lines) == 33, "Budget panel should contain only zero reference plus eight independent method/budget intervals and points")
            for mi, method in enumerate(METHODS):
                for bi, setting in enumerate(budget_settings):
                    row = summaries[case][setting, method]
                    low, high = float(row[quantity + "_ci_low"]), float(row[quantity + "_ci_high"])
                    point = float(row["point_" + quantity])
                    index = mi * 2 + bi
                    equal(ax.lines[1 + index * 4].get_xdata(), [low, high], "Budget interval not actual paired C1-minus-matchedReset interval")
                    equal(ax.lines[4 + index * 4].get_xdata(), [point], "Budget parent marker shifted/recentered")
                    equal(ax.lines[4 + index * 4].get_ydata(), [3 - mi + (.17 if bi == 0 else -.17)], "Budget method order/encoding changed")
                    check(ax.get_xlim()[0] <= min(low, point) and ax.get_xlim()[1] >= max(high, point), "Budget interval/point clipped")
                    check((ax.lines[4 + index * 4].get_markerfacecolor() == "white") == bool(bi), "Filled/open budget encoding wrong")
    plt.close(fig)
    expected_keys = {(case, method, quantity, budget_name) for case in CASE_ORDER for method in METHODS
                     for quantity in SCALAR_NAMES[:2] for budget_name in ("equal_circuit_shots", "equal_noisy_gate_applications")}
    seen_keys = []
    for selected_stem, expected_count, height, allowed_models in (
        ("comparison_C1_Reset_core_models", 36, 125, METHODS[:3]),
        ("comparison_C1_Reset_dual", 12, 75, METHODS[3:]),
    ):
        fig, payload = by_stem[selected_stem]
        check(payload["publication_ready"] is True and payload["selected_in_v2_preview"] is True and len(payload["records"]) == expected_count, "Selected budget split coverage/publication gate")
        equal(fig.get_size_inches() * 25.4, [180, height], "Selected budget physical size", atol=1e-10)
        saved = json.loads((ROOT / "figures_c1/supplement_v2" / (selected_stem + "_plotdata.json")).read_text())
        check(saved["records"] == payload["records"] and saved["selected_in_v2_preview"] is True, "Saved selected budget figure stale")
        lines = {line.get_gid(): (ax, line) for ax in fig.axes for line in ax.lines if line.get_gid()}
        check(len(lines) == expected_count * 4 and not any(ax.child_axes or ax.images for ax in fig.axes), "Selected budget duplicate/missing point/interval/cap artists or hidden inset")
        for row in payload["records"]:
            key = tuple(row[k] for k in ("instance", "method", "metric", "budget"))
            seen_keys.append(key)
            case, method, metric_name, budget_name = key
            check(method in allowed_models, "Wrong model assigned to selected budget split")
            bi = int(budget_name == "equal_noisy_gate_applications")
            quantity = SCALAR_NAMES[2] if metric_name == SCALAR_NAMES[0] else SCALAR_NAMES[3]
            original = summaries[case][budget_settings[bi], method]
            point, low, high = [float(original[k]) for k in ("point_" + quantity, quantity + "_ci_low", quantity + "_ci_high")]
            equal([row[k] for k in ("point", "ci_low", "ci_high")], [point, low, high], "Selected split changes source point/interval")
            prefix = row["artist_gid"]
            ax, marker = lines[prefix + "|point"]
            _, interval = lines[prefix + "|interval"]
            equal(marker.get_xdata(), [point], "Selected budget plug-in recentered")
            equal(interval.get_xdata(), [low, high], "Selected budget CI clipped/recentered")
            for side, value in (("low", low), ("high", high)):
                equal(lines[prefix + "|" + side][1].get_xdata(), [value, value], "Selected budget interval cap changed")
            y = 2 - (METHODS.index(method) if len(allowed_models) == 3 else CASE_ORDER.index(case)) + (.17 if bi == 0 else -.17)
            equal(marker.get_ydata(), [y], "Selected model/case/budget row ordering wrong")
            check(ax.get_xlim()[0] <= min(low, point) and ax.get_xlim()[1] >= max(high, point), "Selected budget extreme negative CI clipped")
            check(ax.get_xscale() == ax.get_yscale() == "linear", "Selected budget uses undisclosed compression")
        plt.close(fig)
    check(len(seen_keys) == len(set(seen_keys)) == 48 and set(seen_keys) == expected_keys, "Selected core/dual union omits or duplicates method/case/metric/budget result")
    matrices, records, provenance = sensitivity.load_point_contrasts(directory, require_current_runner=True)
    intervals, interval_provenance = sensitivity.load_final_intervals(directory, records, required=True)
    check(len(records) == len(intervals) == 96 and all(p["current_runner_hash_matches"] for p in provenance), "Sensitivity uses old runner/draft intervals")
    fig, payload = sensitivity.build_sensitivity_figure(matrices, records, provenance, intervals_available=True)
    saved = json.loads((ROOT / "figures_c1/supplement_v2/sensitivity_point_improvements_plotdata.json").read_text())
    check(saved["cell_records"] == payload["cell_records"] and saved["publication_ready"] is True, "Saved sensitivity heatmap not final production data")
    for qi, metric_name in enumerate(SCALAR_NAMES[:2]):
        expected = np.empty((12, 4))
        for ci, case in enumerate(CASE_ORDER):
            primary = "full_lambda_minus2_ridge_1e-8"
            for mi, method in enumerate(METHODS):
                ref = float(summaries[case][primary, method]["point_" + metric_name])
                for ai, setting in enumerate(sensitivity.ALTERNATIVES):
                    val = float(summaries[case][setting, method]["point_" + metric_name])
                    expected[ci * 4 + mi, ai] = ref - val if qi == 0 else val - ref
        mesh_axes = [(ax, mesh) for ax in fig.axes for mesh in ax.collections if mesh.get_gid() == "sensitivity_improvement_" + metric_name]
        check(len(mesh_axes) == 1, "Sensitivity vector metric panel missing")
        ax, mesh = mesh_axes[0]
        equal(mesh.get_array(), expected, "Sensitivity native heatmap values changed/rounded")
        check(not ax.images and mesh.get_rasterized() is False, "Sensitivity heatmap must use vector cells, not embedded raster")
        coordinates = mesh.get_coordinates()
        xx, yy = np.meshgrid(np.arange(5) - .5, np.arange(13) - .5)
        equal(coordinates, np.stack([xx, yy], axis=-1), "Sensitivity vector cells placed at wrong model/setting coordinates")
        check(ax.get_ylim()[0] > ax.get_ylim()[1], "Sensitivity row0 must be at the top")
        norm = mesh.norm
        check(norm.vmin == -.6 and norm.vmax == .6 and norm.vcenter == 0 and np.abs(expected).max() <= .6, "Sensitivity color range saturated/nonzero-centered")
        for i in range(12):
            for j in range(4):
                text = next(t for t in ax.texts if t.get_gid() == f"sensitivity_cell_{metric_name}_{i}_{j}")
                equal(text.get_position(), [j, i], "Sensitivity annotation assigned to wrong data cell")
                rounded = "0.000" if round(float(expected[i, j]), 3) == 0 else f"{float(expected[i,j]):+.3f}".replace("-", "−")
                check(text.get_text() == rounded, "Sensitivity label does not use disclosed three-decimal rounding")
    check("120" in payload["shot_disclosure"] and "not an equal-shot" in payload["shot_disclosure"], "Six-scale reduced-shot sensitivity not disclosed")
    check(not any(ax.images for ax in fig.axes) and all(mesh.get_rasterized() is False for ax in fig.axes for mesh in ax.collections if hasattr(mesh, "get_rasterized")), "Sensitivity heatmap/colorbars contain rasterized artists")
    interval_file = ROOT / "figures_c1/supplement_v2/sensitivity_paired_intervals_plotdata.json"
    table_data = json.loads(interval_file.read_text())
    check(table_data["replicates"] == 10000 and table_data["interval_records"] == intervals, "Sensitivity table not actual96pairedproductionintervals")
    plt.close(fig)
    print("PASS finalSMv2liveartists:selected36core+12dualbudgetpoints/intervals formexact48-result union withoutclipping;all4companionpreserved;96full-precision vectorsensitivitycells/nonsaturatingcolors/3-decimaldisplay and96truepairedintervaltableentries", flush=True)


def verify_loso(directory, estimators):
    """Check all 96 folds directly against training-only observations."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    check(manifest["case_order"] == list(CASE_ORDER) and manifest["rows"] == 96, "LOSO case coverage")
    check(manifest["noisy_shots_generated"] == manifest["bootstrap_replicates_generated"] == 0, "LOSO claims new sampling")
    check(manifest["ideal_distribution_used"] is False, "LOSO must not use ideal support")
    check(sha(ROOT / manifest["source"]) == manifest["source_sha256"], "LOSO producer changed")
    for entry in manifest["inputs"] + manifest["helpers"]:
        check(sha(ROOT / entry["path"]) == entry["sha256"], "LOSO input/helper changed")
    for entry in manifest["outputs"]:
        check(sha(directory / entry["file"]) == entry["sha256"], "LOSO output changed")
    with (directory / "leave_one_scale_out_residuals.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["instance"], int(r["held_scale_index"]), r["route"]): r for r in rows}
    check(len(rows) == len(lookup) == 96, "LOSO duplicate/missing rows")
    family_cache = {}
    max_error, held_only_folds = 0., 0
    for instance in CASE_ORDER:
        source_path = ROOT / "analysis_results/sparse_zne_20261018/bootstrap_10000" / instance / "rate_0p0001/parent_estimates.npz"
        with np.load(source_path, allow_pickle=False) as source, np.load(directory / (instance + "_predictions.npz"), allow_pickle=False) as saved:
            p, lambdas = source["scale_parents"], source["fit_lambdas"]
            equal(saved["observations"], p, "LOSO observations differ from original parents")
            equal(saved["lambdas"], lambdas, "LOSO rounded scales substituted")
            check(saved["methods"].astype(str).tolist() == list(METHODS), "LOSO model order")
            for held in range(8):
                train_ids = np.delete(np.arange(8), held)
                train, train_lambda = p[train_ids], lambdas[train_ids]
                observed = np.any(train > 0, axis=0)
                check(np.count_nonzero(p[held, ~observed]) >= 0, "LOSO missing outcome scope")
                held_only_folds += int(np.count_nonzero(p[held, ~observed]) > 0)
                equal(saved["training_indices"][held], train_ids, "Held-out scale leaks into training indices")
                equal(saved["training_support"][held], observed, "Held-only outcome leaks into training support")
                w = train_lambda ** -2
                w /= w.mean()
                equal(saved["training_weights"][held], w, "LOSO weights not normalized per training fold")
                for mi, method in enumerate(METHODS):
                    row = lookup[instance, held, method]
                    if method in METHODS[:2]:
                        degree = 1 if method == METHODS[0] else 2
                        ridge = 0 if degree == 1 else 1e-8
                        design = np.column_stack([train_lambda ** j for j in range(degree + 1)])
                        evaluation = np.array([lambdas[held] ** j for j in range(degree + 1)])
                        selection, unseen = -1, 0.
                        floor = 0.
                    else:
                        components, ridge, floor = (1 if method == METHODS[2] else 2), 1e-8, 1 / p.shape[-1]
                        key = held, method
                        if key not in family_cache:
                            family_cache[key] = estimators.build_exponential_grid_family(train_lambda, components, ridge=ridge, weight_exponent=2.)
                        family = family_cache[key]
                        # The frozen native winner is re-evaluated on the union of
                        # TRAINING coordinates only, with the original D offset.
                        _, info = estimators.fit_exponential_grid_batch(train[:, observed], family, uniform_floor=floor)
                        selection = int(info["selected_index"])
                        chosen = family.parameters[selection]
                        equal(saved["selected_parameters"][held, mi, :components], chosen, "LOSO gamma parameters changed")
                        design = np.exp(-train_lambda[:, None] * chosen[None, :])
                        evaluation = np.exp(-lambdas[held] * chosen)
                    normal = design.T @ (w[:, None] * design) + ridge * np.eye(design.shape[1])
                    # Solve against values and evaluation independently of the
                    # saved operator; this also detects accidental projection.
                    beta = np.linalg.solve(normal, design.T @ (w[:, None] * (train[:, observed] - floor)))
                    on_support = floor + evaluation @ beta
                    operator = evaluation @ np.linalg.solve(normal, design.T * w)
                    unseen = floor * (1 - operator.sum())
                    native = np.zeros(p.shape[-1])
                    full = np.full(p.shape[-1], unseen)
                    native[observed], full[observed] = on_support, on_support
                    actual_native = saved["native_predictions"][held, mi]
                    actual_full = saved["full_model_predictions"][held, mi]
                    equal(actual_native, native, "LOSO native signed prediction changed/projected/leaks held support", atol=3e-10)
                    equal(actual_full, full, "LOSO full-model continuation incorrectly dense-refitted/zeroed", atol=3e-10)
                    equal(saved["prediction_operators"][held, mi], operator, "LOSO held-lambda operator incorrect", atol=3e-10)
                    check(int(saved["selected_indices"][held, mi]) == selection, "LOSO gamma uses held data or full-outcome objective")
                    check(np.count_nonzero(actual_native[~observed]) == 0, "LOSO native unseen prediction not zero")
                    residual, full_residual = actual_native - p[held], actual_full - p[held]
                    expected = {
                        "native_signed_prediction_l1_residual": np.abs(residual).sum(),
                        "native_signed_prediction_l2_residual": np.linalg.norm(residual),
                        "native_signed_prediction_squared_l2_residual": np.square(residual).sum(),
                        "full_model_signed_prediction_l1_residual": np.abs(full_residual).sum(),
                        "full_model_signed_prediction_l2_residual": np.linalg.norm(full_residual),
                        "full_model_signed_prediction_squared_l2_residual": np.square(full_residual).sum(),
                        "held_probability_mass_outside_training_support": p[held, ~observed].sum(),
                        "training_support_size": observed.sum(),
                        "held_positive_outcomes_unseen_in_training": np.count_nonzero(p[held, ~observed]),
                    }
                    for name, value in expected.items():
                        equal(float(row[name]), value, "LOSO summary residual/convention mismatch: " + name, atol=3e-12)
                    check(row["prediction_stage"] == "signed_preprojection_no_probability_TVD", "LOSO signed diagnostic mislabeled probability TVD")
                    check(int(row["training_total_record_shots"]) == 140000 and int(row["held_record_shots"]) == 20000, "LOSO training/held shots wrong")
                    max_error = max(max_error, float(np.max(np.abs(actual_native - native))))
    check(held_only_folds > 0, "LOSO lacks a real held-only support discrimination")
    print(f"PASS all96LOSO folds: training-seven support only, held-lambda fit/native-zero versus model continuation, actual sharedgamma selection, signed residuals and140k/20k accounting; maxindependentpredicterror{max_error:.3g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--toys-only", action="store_true")
    parser.add_argument("--loso", type=Path)
    parser.add_argument("--parallel-review", type=Path)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Require final SMv2 budget/sensitivity artifacts and inspect their live artists")
    args = parser.parse_args()
    estimators, core = frozen_modules()
    verify_restricted_toys(estimators, core)
    if args.loso is not None:
        verify_loso(args.loso, estimators)
    if args.artifacts is not None:
        verify_artifacts(args.artifacts, estimators, core, require_final=args.require_final)
        if args.negative_controls:
            verify_negative_controls(args.artifacts)
        if args.figures:
            verify_sm_v2_figures(args.artifacts)
    if args.parallel_review is not None:
        verify_parallel_review(args.parallel_review)
    print("PASS independent C1 follow-up checks; no files written/no quantum shots generated.", flush=True)


if __name__ == "__main__":
    main()
