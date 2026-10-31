#!/usr/bin/env python3
"""Independent, read-only checks for the approved mixed-ridge C1 bootstrap.

Only exact-seed replay of saved empirical-parent draws is allowed. No quantum
circuits, new noise trajectories, tuning, or output files are generated.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zlib

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis_results/c1_primary_fixed_20261018"
METHODS = ("linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne")
RIDGES = dict(zip(METHODS, (0., 0., 0., 1e-6)))
CASES = ("n119_a13", "n119_a50", "n119_a16")
D = 16384
TOL = 1e-10
CORE_SHA = "78a83983595b34d1d4929cb68f082ed09a18e33419259011bd32bc942d0e0ac1"
CONTROL_ROUTES = ("dual_exponential_zne_ridge0_control", "dual_exponential_zne_ridge1e-8_control")
SUITES = {}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def equal(actual, expected, message, atol=0, rtol=0):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    result = hashlib.sha256(str(a.dtype).encode())
    result.update(json.dumps(a.shape).encode())
    result.update(a.tobytes())
    return result.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def modules_and_config():
    sys.path.insert(0, str(ROOT / "analysis"))
    sys.path.insert(0, str(ROOT / "shor_error_mitigation/src"))
    from shor_qem import estimators, bias_variance_revision as bvr
    check(sha(ROOT / "analysis/sparse_zne_core_20261018.py") == CORE_SHA, "Frozen C1 core changed")
    check(sha(Path(estimators.__file__)) == "65cb00d076c871ccb1cb63ab9dc05a2f9ed5c0cd8d589a4e5f5fa5d35097b993", "Frozen native estimator changed")
    check(sha(Path(bvr.__file__)) == "792445cfd99777c2e6eddb5dc781ec30c58b9dc8bf8650d931757d60a717c541", "Frozen original draw/input helper changed")
    core = module("primary_verify_frozen_core", ROOT / "analysis/sparse_zne_core_20261018.py")
    reference = module("primary_verify_restricted_reference", ROOT / "analysis/verify_c1_followup_20261018.py")
    config = json.loads((ROOT / "analysis/c1_primary_fixed_20261018.json").read_text())
    check(config["estimator"] == "C1" and config["weight_exponent"] == 2 and config["ridges"] == RIDGES, "Approved per-model ridge/weight map changed")
    check(config["dimension"] == D and config["exponential_offset"] == 1 / D and config["shots_per_scale"] == 20000 and config["direct_shots"] == 160000, "Approved offset/dimension/budget changed")
    check(config["single_gamma_grid"] == {"minimum": .01, "maximum": 10., "count": 200}, "Approved single grid changed")
    check(config["dual_gamma_grid"] == {"minimum": 0., "maximum": 10., "count": 200, "pair_rule": "gamma1 < gamma2", "pairs": 19900}, "Approved dual grid changed")
    return estimators, bvr, core, reference, config


def independent_suite(estimators, lambdas, *, toy=False, dual_ridge=1e-6):
    fixed = {method: estimators.polynomial_intercept_coefficients(lambdas, degree=i + 1, ridge=0., weight_exponent=2.) for i, method in enumerate(METHODS[:2])}
    families = {
        METHODS[2]: estimators.build_exponential_grid_family(lambdas, 1, ridge=0., weight_exponent=2., **({"single_grid": np.array([.08, .3, .75, 1.8])} if toy else {})),
        METHODS[3]: estimators.build_exponential_grid_family(lambdas, 2, ridge=dual_ridge, weight_exponent=2., **({"dual_grid": np.array([0., .2, .8, 2.])} if toy else {})),
    }
    return fixed, families


def verify_approved_toys(estimators, core, reference):
    count = 0
    for dimension in (16, 31):
        for observed in (np.array([0, 3, dimension - 1]), np.arange(dimension)):
            lambdas = np.array([1., 1.35, 1.8, 2.4])
            p = np.zeros((4, dimension))
            ii, jj = np.indices((4, len(observed)))
            values = .1 + ((17 * ii + 7 * jj + dimension + 3) % 29) / 23
            values[0, 1] = 0
            p[:, observed] = values / values.sum(axis=1, keepdims=True)
            for eta in (1e-6, 0., 1e-8):
                fixed, family = independent_suite(estimators, lambdas, toy=True, dual_ridge=eta)
                methods = METHODS if eta == 1e-6 else (METHODS[3],)
                for method in methods:
                    actual = core.infer_model(method, "C1", fixed_coefficients=fixed, exponential_families=family, support=core.ObservedBatch.from_dense(p[None]))
                    ridge = eta if method == METHODS[3] else 0.
                    before, after, selected = reference.independent_restricted(p, lambdas, method=method, weight_exponent=2., ridge=ridge, family=family.get(method))
                    equal(actual["before"].materialize()[0], before, "Mixed-ridge toy differs from direct observed-only weighted LS", atol=5e-11)
                    equal(actual["after"].materialize()[0], after, "Mixed-ridge toy differs from independent support projection", atol=5e-11)
                    if selected >= 0:
                        check(int(actual["selection"]["selected_index"][0]) == selected, "Mixed-ridge toy native gamma differs from explicit penalized residual minimum")
                    count += 1
    print(f"PASS {count} direct restricted-LS/projection toys for approved map and both dual controls; cross-scale zeros and fixed1/D offset retained", flush=True)


def metric_rows(probability, target, order):
    p = np.atleast_2d(probability)
    d = p.shape[1]
    groups = []
    for s in range(order):
        q, rem = divmod(d * s, order)
        groups.append(np.array([q] if rem == 0 else [q, q + 1]))
    outside = np.ones(d, dtype=bool)
    outside[np.concatenate(groups)] = False
    denominator = min(float(target[g].sum()) for g in groups) - float(target[outside].max())
    gap = np.min(np.stack([p[:, g].sum(axis=1) for g in groups]), axis=0) - p[:, outside].max(axis=1)
    if d == D and order == 6:
        check(abs(denominator - .13535877187470172) < 2e-15, "Order-six ideal normalization changed")
    return np.stack([np.abs(p - target).sum(axis=1) / 2, gap / denominator], axis=1)


def simplex_sorted(x):
    ordered = np.sort(x)[::-1]
    cumulative = np.cumsum(ordered) - 1.
    active = np.flatnonzero(ordered - cumulative / np.arange(1, len(x) + 1) > 0)
    check(len(active) > 0, "Empty independent C1 projection")
    rho = active[-1]
    tau = cumulative[rho] / (rho + 1)
    p = np.maximum(x - tau, 0)
    return p / p.sum()


def independent_inference(grid, fixed, families, method):
    """Observed-only Gram selection and separately implemented projection."""
    n, scales, dimension = grid.shape
    ids = [np.flatnonzero(np.any(g > 0, axis=0)) for g in grid]
    width = max(map(len, ids))
    values = np.zeros((n, scales, width))
    valid = np.zeros((n, width), dtype=bool)
    for b, observed in enumerate(ids):
        values[b, :, :len(observed)] = grid[b][:, observed]
        valid[b, :len(observed)] = True
    selected = np.full(n, -1, dtype=np.int64)
    parameters = np.full((n, 2), np.nan)
    if method in fixed:
        observed_before = np.einsum("l,blk->bk", fixed[method], values, optimize=True)
        h = np.broadcast_to(fixed[method], (n, scales))
        condition = np.full(n, np.nan)
        score_gap = np.full(n, np.nan)
    else:
        family = families[method]
        centered = (values - 1 / dimension) * valid[:, None]
        gram = np.einsum("blk,bmk->blm", centered, centered, optimize=True)
        scores = gram.reshape(n, -1) @ family.score_matrices_flat.T
        selected = np.argmax(scores, axis=1)
        h = family.intercept_coefficients[selected]
        observed_before = 1 / dimension + np.einsum("bl,blk->bk", h, values - 1 / dimension, optimize=True)
        chosen = family.parameters[selected]
        parameters[:, :chosen.shape[1]] = chosen
        condition = family.normal_condition_numbers[selected]
        score_gap = scores[np.arange(n), selected] - np.partition(scores, -2, axis=1)[:, -2]
    before, after = np.zeros((n, dimension)), np.zeros((n, dimension))
    for b, observed in enumerate(ids):
        native = observed_before[b, :len(observed)]
        before[b, observed] = native
        after[b, observed] = simplex_sorted(native)
    return {"before": before, "after": after, "selected_indices": selected,
            "selected_parameters": parameters, "coefficients": h,
            "condition": condition, "score_gap": score_gap, "support_ids": ids}


def scalar_dictionary(before, after, scale_one, reset, target, order):
    measured = metric_rows(after, target, order)
    direct = metric_rows(reset, target, order)
    correction = after - before
    change = np.linalg.norm(after - scale_one, axis=1)
    result = {"tvd": measured[:, 0], "normalized_grouped_peak_margin": measured[:, 1],
        "reset_minus_zne_tvd": direct[:, 0] - measured[:, 0],
        "zne_minus_reset_normalized_grouped_peak_margin": measured[:, 1] - direct[:, 1],
        "negative_mass_before_projection": -np.minimum(before, 0).sum(axis=1),
        "negative_coordinate_fraction": np.mean(before < 0, axis=1),
        "normalization_defect": np.abs(before.sum(axis=1) - 1),
        "l1_projection_correction": np.abs(correction).sum(axis=1),
        "l2_projection_correction": np.linalg.norm(correction, axis=1),
        "postprojection_support_size": np.count_nonzero(after > 0, axis=1),
        "projected_change_from_scale_one_l2": change,
        "projection_correction_to_change_ratio": np.linalg.norm(correction, axis=1) / np.maximum(change, np.finfo(float).eps)}
    return result


def exact_seed_generators(seed, instance, rate):
    children = np.random.SeedSequence([int(seed), zlib.crc32(instance.encode()), int(round(float(rate) * 1e9))]).spawn(4)
    return [np.random.default_rng(child) for child in children]


def independently_draw(parent, per_scale, rngs, *, raw=True):
    counts = np.stack([rngs[0].multinomial(per_scale, p, size=25) for p in parent["grid"]], axis=1).astype(np.float64)
    extension = rngs[1].multinomial(160000 - per_scale, parent["reset_extension"], size=25).astype(np.float64)
    result = {"grid": counts / per_scale, "reset": (counts[:, 0] + extension) / 160000,
        "scale_counts": counts, "reset_extension_counts": extension}
    if raw:
        raw_initial = rngs[2].multinomial(20000, parent["raw_initial"], size=25).astype(np.float64)
        raw_extension = rngs[3].multinomial(140000, parent["raw_extension"], size=25).astype(np.float64)
        result.update(raw=(raw_initial + raw_extension) / 160000,
                      raw_20k=raw_initial / 20000, reset_20k=counts[:, 0] / per_scale,
                      raw_initial_counts=raw_initial, raw_extension_counts=raw_extension)
    return result


def verify_fixed_point_pack(estimators, config):
    manifest = json.loads((BASE / "manifest.json").read_text())
    check(manifest["configuration"] == config and manifest["point_rows"] == 148, "Approved point pack/config identity changed")
    for path, digest in manifest["sources"].items():
        check(sha(ROOT / path) == digest, "Approved point-pack source changed")
    for path, digest in manifest["outputs"].items():
        check(sha(BASE / path) == digest, "Approved golden point table changed")
    records = rows(BASE / "point_estimates.csv")
    check(len(records) == len({(r["method"], r["case_id"]) for r in records}) == 148, "Approved golden point coverage changed")
    check(all(r["new_bootstrap_available"] == "False" for r in records), "Point-only golden pack mislabeled as bootstrap result")
    return {(r["method"], r["case_id"]): r for r in records}


def rate_key(rate):
    return format(float(rate), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def original_parent_data(bvr, instance, rate):
    data = bvr._prepare_data(bvr.INSTANCE_SPECS[instance], np.array([float(rate)]), "realized")
    check(data["lambda_one"] == 0, "Scale-one index changed")
    old_dir = ROOT / "analysis_results/sparse_zne_20261018/bootstrap_10000" / instance / ("rate_" + rate_key(rate))
    with np.load(old_dir / "parent_estimates.npz", allow_pickle=False) as saved:
        equal(data["reset"][0], saved["scale_parents"], "Original count-grid parent differs from frozen original")
        equal(data["reset_full"][0], saved["reset_160k"], "Original Reset160k parent changed")
        equal(data["raw_full"][0], saved["raw_160k"], "Original Raw160k parent changed")
        equal(data["fit_lambdas"], saved["fit_lambdas"], "Original effective scales changed")
        target = saved["target"].copy()
    return {"grid": data["reset"][0], "reset_extension": data["reset_extension"][0],
            "raw_initial": data["raw_initial"][0], "raw_extension": data["raw_extension"][0],
            "raw_full": data["raw_full"][0], "reset_full": data["reset_full"][0],
            "lambdas": data["fit_lambdas"], "target": target,
            "order": data["config"]["order"], "original": data, "old_dir": old_dir}


def legacy_series(parent, instance, budget):
    """Only reference arrays generated under the exact same seed/budget."""
    if budget == "equal_shot":
        with np.load(parent["old_dir"] / "scalar_bootstrap_draws.npz", allow_pickle=False) as old:
            check(int(old["seed"]) == 20260722 and int(old["batch_size"]) == 25 and int(old["replicates"]) == 10000, "Legacy equal-shot cache not frozen seed/batch")
            variants, methods = old["variants"].astype(str).tolist(), old["methods"].astype(str).tolist()
            quantities = old["quantities"].astype(str).tolist()
            # Decompress each NPZ field ONCE, then retain only needed slices.
            all_values = old["draws"]
            linear = all_values[variants.index("C1"), methods.index(METHODS[0]), [quantities.index("tvd"), quantities.index("normalized_grouped_peak_margin")]].copy()
            dual = all_values[variants.index("C1"), methods.index(METHODS[3]), [quantities.index("tvd"), quantities.index("normalized_grouped_peak_margin")]].copy()
            dual_indices = old["selected_indices"][variants.index("C1"), methods.index(METHODS[3])].copy()
            dual_parameters = old["selected_parameters"][variants.index("C1"), methods.index(METHODS[3])].copy()
            direct = old["direct_draws"].copy()
            names = old["direct_routes"].astype(str).tolist()
            support = old["support_sizes"].copy()
        return {"linear": linear, "dual_ridge1e-8": dual, "dual_ridge1e-8_indices": dual_indices,
                "dual_ridge1e-8_parameters": dual_parameters,
                "direct": {r: direct[i] for i, r in enumerate(names)}, "support_sizes": support}
    check(budget == "gate_matched", "Unknown resource allocation")
    old_dir = ROOT / "analysis_results/c1_followup_20261018/bootstrap_10000" / instance
    with np.load(old_dir / "scalar_bootstrap_draws.npz", allow_pickle=False) as old:
        check(int(old["seed"]) == 20260804 and int(old["batch_size"]) == 25 and int(old["replicates"]) == 10000, "Legacy gate cache not frozen seed/batch")
        settings, methods = old["settings"].astype(str).tolist(), old["methods"].astype(str).tolist()
        configuration = json.loads(str(old["configuration_json"]))
        si = next(i for i, setting in enumerate(configuration["settings"]) if setting["per_scale_shots"] == 11188)
        values = old["scalar_draws"]
        linear = values[si, methods.index(METHODS[0]), :2].copy()
        reset = old["reset_draws"][1].copy()
        support = old["support_sizes"][2].copy()
    return {"linear": linear, "direct": {"reset_160k": reset}, "support_sizes": support}


def route_reference(grid, reset, parent, spec, estimators):
    method, ridge = spec["method"], spec["ridge"]
    key = tuple(parent["lambdas"]), float(ridge if method == METHODS[3] else 1e-6)
    if key not in SUITES:
        SUITES[key] = independent_suite(estimators, parent["lambdas"], dual_ridge=key[1])
    fixed, families = SUITES[key]
    fit = independent_inference(grid, fixed, families, method)
    s = scalar_dictionary(fit["before"], fit["after"], grid[:, 0], reset, parent["target"], parent["order"])
    h = fit["coefficients"]
    if method in METHODS[:2]:
        degree = 1 if method == METHODS[0] else 2
        design = np.vander(parent["lambdas"], N=degree + 1, increasing=True)
        weights = np.asarray(parent["lambdas"]) ** -2
        weights /= weights.mean()
        condition = np.full(grid.shape[0], np.linalg.cond(design.T @ (weights[:, None] * design)))
        unseen = np.zeros(grid.shape[0])
    else:
        condition = fit["condition"]
        unseen = (1 - h.sum(axis=1)) / D
    unseen_count = np.array([D - len(ids) for ids in fit["support_ids"]])
    h2 = np.square(h).sum(axis=1)
    s.update(normal_condition_number=condition, intercept_l1_norm=np.abs(h).sum(axis=1),
             intercept_l2_norm=np.sqrt(h2), intercept_l2_norm_squared=h2,
             full_model_l2_projection_correction=np.sqrt(np.square(s["l2_projection_correction"]) + unseen_count * np.square(unseen)),
             full_model_normalization_defect=np.abs(fit["before"].sum(axis=1) + unseen_count * unseen - 1))
    return fit, s


def check_job_configuration(c, approved):
    budget, instance, rate = c["budget"], c["instance"], float(c["rate"])
    check(budget in ("equal_shot", "gate_matched") and instance in CASES, "Unknown job budget/case")
    check(budget != "gate_matched" or rate == .0001, "Gate budget expanded beyond approved representative scope")
    n = 20000 if budget == "equal_shot" else 11188
    check(c["configuration_id"] == approved["configuration_id"] and c["ridges"] == RIDGES and c["weight_exponent"] == 2, "Job uses legacy/global ridge or a tuned weight")
    equal(c["fit_lambdas"], approved["effective_scales"], "Job uses rounded effective scales")
    check(c["dimension"] == D and c["fitting_floor"] == 1 / D and c["per_scale_shots"] == n and c["zne_circuit_shots"] == 8 * n and c["direct_shots"] == 160000 and c["reset_extension_shots"] == 160000 - n, "Job offset/gate/sample allocation changed")
    check(c["seed"] == (20260722 if budget == "equal_shot" else 20260804) and c["batch_size"] == 25, "Wrong seed or historical multinomial batch order")
    check(c["single_gamma_grid"] == approved["single_gamma_grid"] and c["dual_gamma_grid"] == approved["dual_gamma_grid"] and c["native_argmax"] is True, "Job gamma grid/native-first-argmax changed")
    check(c["observed_support_reselected"] is True and c["gamma_refitted_each_replicate"] is True and c["packed_support_bitorder"] == "little", "Job freezes parent support or gamma")
    check(c["no_cross_budget_paired_interval"] is True and "postselected fixed settings" in c["coverage"], "Job implies cross-budget pairing or post-selection coverage")
    controls = budget == "equal_shot" and rate == .0001
    expected_routes = list(METHODS) + (list(CONTROL_ROUTES) if controls else [])
    check(c["routes"] == expected_routes and len(c["route_specifications"]) == len(expected_routes), "Control scope or primary route ordering changed")
    for i, spec in enumerate(c["route_specifications"]):
        expected_method = METHODS[i] if i < 4 else METHODS[3]
        expected_ridge = RIDGES[expected_method] if i < 4 else (0. if i == 4 else 1e-8)
        check(spec == {"route": expected_routes[i], "method": expected_method, "ridge": expected_ridge,
                       "role": "approved_primary" if i < 4 else "same_draw_dual_control"}, "Model-specific ridge/control specification wrong")
    expected_direct = ["raw_160k", "reset_160k", "raw_20k", "reset_20k"] if budget == "equal_shot" else ["reset_160k"]
    check(c["direct_routes"] == expected_direct and c["direct_metrics"] == ["tvd", "normalized_grouped_peak_margin"], "Gate Raw invented or direct route/metric changed")
    check(c["scientific_direct_routes"] == (["raw_160k", "reset_160k"] if budget == "equal_shot" else ["reset_160k"]), "20k replay-only route misrepresented as primary scientific comparator")
    check(c["paired_reset_improvement_signs"] == {"tvd": "Reset minus model", "normalized_grouped_peak_margin": "model minus Reset", "positive_favors": "model"}, "Paired Reset sign convention changed")
    check(c["paired_control_improvement_signs"] == {"tvd": "primary dual minus control", "normalized_grouped_peak_margin": "control minus primary dual", "positive_favors": "control"}, "Paired control sign convention reversed")
    check(c["control_reference_route"] == METHODS[3], "Dual controls use another primary reference")
    check(c["replicates"] > 0 and (c["replicates"] <= 100 if c["smoke"] else c["replicates"] == 10000), "Smoke/production replicate contract wrong")
    return controls


def verify_point_artifacts(folder, estimators, bvr, approved, golden):
    started = json.loads((folder / "started_manifest.json").read_text())
    c, provenance = started["configuration"], started["provenance"]
    expected_identity = hashlib.sha256((json.dumps({"configuration": c, "provenance": provenance}, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    check(started["identity"] == expected_identity, "Configuration/provenance identity not actually hashed")
    check_job_configuration(c, approved)
    for item in provenance["sources"] + provenance["inputs"]:
        check(sha(item["path"]) == item["sha256"], "Runner/frozen numerical/input source changed during run")
    for key in ("approved_config", "replay_reference"):
        item = provenance[key]
        check(sha(item["path"]) == item["sha256"], "Approved configuration or legacy replay reference changed")
    pack = provenance["frozen_point_pack"]
    check(sha(pack["path"]) == pack["sha256"] and sha(pack["points_path"]) == pack["points_sha256"], "Golden approved point pack changed")
    point_manifest = json.loads((folder / "point_manifest.json").read_text())
    check(point_manifest["identity"] == started["identity"] and point_manifest["status"] == "points_complete", "Point manifest identity/status wrong")
    for name, digest in point_manifest["artifacts"].items():
        check(sha(folder / name) == digest, "Parent payload/point table changed")
    parent = original_parent_data(bvr, c["instance"], c["rate"])
    n = c["per_scale_shots"]
    reset = (n * parent["grid"][0] + (160000 - n) * parent["reset_extension"]) / 160000
    case_id = c["instance"] + "__rate_" + rate_key(c["rate"])
    with np.load(folder / "parent_estimates.npz", allow_pickle=False) as z:
        payload = {k: z[k] for k in z.files}
    check(str(payload["identity"]) == started["identity"] and json.loads(str(payload["configuration_json"])) == c, "Parent configuration identity wrong")
    equal(payload["scale_parents"], parent["grid"], "Parent frequencies changed for gate matching")
    equal(payload["reset_extension_parent"], parent["reset_extension"], "Independent Reset extension parent changed")
    equal(payload["reset_reference"], reset, "Gate Reset parent center not weighted11188+148812")
    equal(payload["target"], parent["target"], "Ideal metric reference changed")
    equal(payload["fit_lambdas"], parent["lambdas"], "Effective scales changed")
    direct = {"reset_160k": reset[None]}
    if c["budget"] == "equal_shot":
        direct.update(raw_160k=parent["raw_full"][None], raw_20k=parent["raw_initial"][None], reset_20k=parent["grid"][None, 0])
    point_rows = rows(folder / "points.csv")
    points = {(r["route"], r["quantity"]): r for r in point_rows}
    check(len(points) == len(point_rows) == len(c["routes"]) * len(c["quantities"]) + 2 * len(c["direct_routes"]), "Parent point rows missing or duplicate")
    mask = np.any(parent["grid"] > 0, axis=0)
    equal(np.unpackbits(payload["packed_observed_support"], axis=1, count=D, bitorder="little").astype(bool), mask[None], "Parent support not original eight-record union")
    for ri, spec in enumerate(c["route_specifications"]):
        fit, scalar = route_reference(parent["grid"][None], reset[None], parent, spec, estimators)
        equal(payload["before__" + spec["route"]], fit["before"][0], "Parent native signed prevector not approved C1 fit", atol=5e-10, rtol=2e-12)
        equal(payload["after__" + spec["route"]], fit["after"][0], "Parent actual distribution not independent C1 projection", atol=6e-11)
        equal(payload["selected_indices"][ri], fit["selected_indices"], "Parent gamma winner not native observed-support refit")
        equal(payload["selected_parameters"][ri], fit["selected_parameters"].T, "Parent gamma parameters not actual grid indices")
        for qi, quantity in enumerate(c["quantities"]):
            check(quantity in scalar, "Unverified parent quantity: " + quantity)
            equal(payload["scalar_draws"][ri, qi], scalar[quantity], "Parent scalar/diagnostic incorrect: " + quantity, atol=6e-10, rtol=2e-11)
            equal(float(points[spec["route"], quantity]["point"]), scalar[quantity][0], "Point CSV not true parent estimate", atol=6e-10, rtol=2e-11)
        if ri < 4:
            reference = golden[spec["route"], case_id]
            equal(payload["scalar_draws"][ri, :2, 0], [float(reference["tvd"]), float(reference["normalized_grouped_peak_margin"])], "Approved golden148 point replay failed", atol=1e-11)
        check(np.all(payload["after__" + spec["route"]][~mask] == 0) and np.all(payload["after__" + spec["route"]] >= 0), "Parent C1 escaped support or became negative")
    for di, route in enumerate(c["direct_routes"]):
        values = metric_rows(direct[route], parent["target"], parent["order"])[0]
        equal(payload["direct_draws"][di, :, 0], values, "Direct parent metrics wrong", atol=3e-14)
    print(f"PASS {c['budget']}/{c['instance']}/{c['rate']:g} actual parent vectors/all16 diagnostics/golden points and budget-specific Reset center", flush=True)
    return started, parent, payload


def verify_paired_arrays(arrays):
    scalar, direct = arrays["scalar_draws"], arrays["direct_draws"]
    # Direct routes are supplied separately to preserve exact named lookup.
    check(np.isfinite(scalar).all() and np.isfinite(direct).all(), "Nonfinite bootstrap output silently retained/dropped")
    check(scalar[:, 0].min() >= -1e-13 and scalar[:, 0].max() <= 1 + 1e-13, "Projected distribution TVD outside0..1")
    contrast = np.empty((max(0, scalar.shape[0] - 4), 2, scalar.shape[-1]))
    for ci in range(len(contrast)):
        contrast[ci, 0] = scalar[3, 0] - scalar[ci + 4, 0]
        contrast[ci, 1] = scalar[ci + 4, 1] - scalar[3, 1]
    equal(arrays["paired_control_improvements"], contrast, "Dual control contrasts not same-draw control-vs-primary differences")


def verify_first_batch(arrays, meta, c, parent, estimators):
    rngs = exact_seed_generators(c["seed"], c["instance"], c["rate"])
    if c["budget"] == "gate_matched":
        rngs = rngs[:2]
    check(meta["rng_before"] == [r.bit_generator.state for r in rngs], "First checkpoint seed hierarchy changed")
    drawn = independently_draw(parent, c["per_scale_shots"], rngs, raw=c["budget"] == "equal_shot")
    expected_audit = {"scale_counts_sha256": array_sha(drawn["scale_counts"]), "extension_counts_sha256": array_sha(drawn["reset_extension_counts"])}
    if c["budget"] == "equal_shot":
        expected_audit.update(raw_20k_counts_sha256=array_sha(drawn["raw_initial_counts"]), raw_extension_counts_sha256=array_sha(drawn["raw_extension_counts"]))
    for key, digest in expected_audit.items():
        check(meta["draw_counts"][key] == digest, "Independent original-count replay failed: " + key)
    check(meta["rng_after"] == [r.bit_generator.state for r in rngs], "First checkpoint RNG-after changed")
    n = meta["stop"] - meta["start"]
    mask = np.any(drawn["grid"][:n] > 0, axis=1)
    equal(np.unpackbits(arrays["packed_observed_support"], axis=1, count=D, bitorder="little").astype(bool), mask, "Support frozen at parent instead of reselected from actual replicate")
    equal(arrays["support_sizes"], mask.sum(axis=1), "Bootstrap support-size diagnostic wrong")
    direct = {"reset_160k": drawn["reset"]}
    if c["budget"] == "equal_shot":
        direct.update(raw_160k=drawn["raw"], raw_20k=drawn["raw_20k"], reset_20k=drawn["reset_20k"])
    for di, route in enumerate(c["direct_routes"]):
        equal(arrays["direct_draws"][di], metric_rows(direct[route], parent["target"], parent["order"])[:n].T, "Raw/Reset draw does not preserve original overlap and extension allocation", atol=3e-14)
    for ri, spec in enumerate(c["route_specifications"]):
        fit, scalar = route_reference(drawn["grid"], drawn["reset"], parent, spec, estimators)
        equal(arrays["selected_indices"][ri], fit["selected_indices"][:n], "Bootstrap freezes gamma or uses wrong observed/dense objective")
        equal(arrays["selected_parameters"][ri], fit["selected_parameters"][:n].T, "Bootstrap gamma parameters differ from grid winner")
        for qi, quantity in enumerate(c["quantities"]):
            equal(arrays["scalar_draws"][ri, qi], scalar[quantity][:n], "Independent first-batch native fit/projection/metric mismatch: " + quantity, atol=8e-10 if qi >= 2 else 6e-11, rtol=2e-11 if qi >= 2 else 0)


def verify_bootstrap_artifacts(folder, started, parent, point, estimators, *, require_final=False):
    c = started["configuration"]
    total = c["replicates"]
    complete_path = folder / "manifest.json"
    check(complete_path.is_file(), "Incomplete job has no final aggregate; resume before claiming readiness")
    manifest = json.loads(complete_path.read_text())
    check(manifest["identity"] == started["identity"] and manifest["configuration"] == c and manifest["provenance"] == started["provenance"], "Bootstrap identity/configuration/source chain changed")
    check(manifest["status"] == "complete" and manifest["actual_bootstrap_started"] is True and manifest["actual_bootstrap_completed"] is True, "Bootstrap not actually complete")
    check(manifest["new_quantum_shots"] == manifest["new_noisy_circuit_simulations"] == 0 and manifest["no_old_interval_reuse"] is True, "New circuit data or relabeled old intervals")
    if require_final:
        check(total == 10000 and c["smoke"] is False and manifest["final_10000_contract"] is True, "Smoke results substituted for final10000 evidence")
    else:
        check(manifest["final_10000_contract"] == (total == 10000 and not c["smoke"]), "Smoke/final status incorrect")
    for name, digest in manifest["artifacts"].items():
        check(sha(folder / name) == digest, "Bootstrap aggregate artifact changed: " + name)
    with np.load(folder / "scalar_bootstrap_draws.npz", allow_pickle=False) as z:
        aggregate = {k: z[k] for k in z.files}
    check(int(aggregate["seed"]) == c["seed"] and int(aggregate["batch_size"]) == 25 and int(aggregate["replicates"]) == total, "Aggregate seed/batch/size changed")
    check(aggregate["routes"].astype(str).tolist() == c["routes"] and aggregate["quantities"].astype(str).tolist() == c["quantities"] and aggregate["direct_routes"].astype(str).tolist() == c["direct_routes"], "Aggregate route/quantity order changed")
    check(aggregate["scalar_draws"].shape == (len(c["routes"]), len(c["quantities"]), total) and aggregate["direct_draws"].shape == (len(c["direct_routes"]), 2, total), "Aggregate has missing or extra replicates/routes")
    for ri, spec in enumerate(c["route_specifications"]):
        method = spec["method"]
        indices = aggregate["selected_indices"][ri]
        parameters = aggregate["selected_parameters"][ri]
        if method in METHODS[:2]:
            equal(indices, np.full(total, -1), "Polynomial bootstrap assigned gamma selection")
            check(np.isnan(parameters).all(), "Polynomial gamma-parameter sentinel changed")
        else:
            grid = np.linspace(.01 if method == METHODS[2] else 0., 10., 200)
            check(np.all((indices >= 0) & (indices < (200 if method == METHODS[2] else 19900))), "Selected gamma index outside unchanged native grid")
            if method == METHODS[2]:
                equal(parameters[0], grid[indices], "Single gamma values do not map to selected native indices")
                check(np.isnan(parameters[1]).all(), "Single fit invents second decay rate")
            else:
                first, second = np.triu_indices(200, 1)
                equal(parameters, np.stack([grid[first[indices]], grid[second[indices]]]), "Dual gamma values do not map to native increasing pairs")
    legacy = legacy_series(parent, c["instance"], c["budget"])
    equal(aggregate["scalar_draws"][0, :2], legacy["linear"][:, :total], "Unchanged original C1 linear cache replay failed", atol=1e-11)
    for di, route in enumerate(c["direct_routes"]):
        equal(aggregate["direct_draws"][di], legacy["direct"][route][:, :total], "Original direct Raw/Reset cache replay failed", atol=1e-11)
    equal(aggregate["support_sizes"], legacy["support_sizes"][:total], "Original resampled support replay failed")
    if len(c["routes"]) == 6:
        equal(aggregate["scalar_draws"][5, :2], legacy["dual_ridge1e-8"][:, :total], "Original C1 dual1e-8 cache replay failed", atol=1e-11)
        equal(aggregate["selected_indices"][5], legacy["dual_ridge1e-8_indices"][:total], "Original C1 dual1e-8 native gamma cache replay failed")
        equal(aggregate["selected_parameters"][5], legacy["dual_ridge1e-8_parameters"][:, :total], "Original C1 dual1e-8 parameters replay failed")
    previous_hash, previous_rng, next_start = None, None, 0
    keys = ("scalar_draws", "direct_draws", "selected_indices", "selected_parameters", "support_sizes", "packed_observed_support", "paired_reset_improvements", "paired_control_improvements")
    for i, entry in enumerate(manifest["checkpoint_files"]):
        path = folder / "batches" / entry["path"]
        check(sha(path) == entry["sha256"], "Checkpoint file changed")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["checkpoint_json"]))
            arrays = {k: z[k] for k in z.files if k != "checkpoint_json"}
        check(set(arrays) == set(meta["array_hashes"]) == set(keys), "Checkpoint array inventory changed")
        check(meta["identity"] == started["identity"] and meta["start"] == next_start and meta["stop"] > next_start and meta["full_draw_batch_size"] == 25, "Checkpoint gap/overlap/identity changed")
        check(entry["start"] == meta["start"] and entry["stop"] == meta["stop"] and meta["stop"] - meta["start"] <= 25, "Checkpoint inventory start/stop differs from actual payload")
        check(meta["previous_checkpoint_sha256"] == previous_hash and (previous_rng is None or meta["rng_before"] == previous_rng), "Checkpoint hash/RNG chain broken")
        for key in keys:
            check(array_sha(arrays[key]) == meta["array_hashes"][key], "Checkpoint payload changed: " + key)
            expected = aggregate[key][meta["start"]:meta["stop"]] if key in ("support_sizes", "packed_observed_support") else aggregate[key][..., meta["start"]:meta["stop"]]
            equal(arrays[key], expected, "Aggregate not exact checkpoint concatenation")
        verify_paired_arrays(arrays)
        reset = arrays["direct_draws"][c["direct_routes"].index("reset_160k")]
        expected_reset = arrays["scalar_draws"][:, :2].copy()
        expected_reset[:, 0] = reset[0] - expected_reset[:, 0]
        expected_reset[:, 1] -= reset[1]
        equal(arrays["paired_reset_improvements"], expected_reset, "Reset comparison from different draws or marginal endpoint subtraction")
        if i == 0:
            verify_first_batch(arrays, meta, c, parent, estimators)
        next_start, previous_hash, previous_rng = meta["stop"], entry["sha256"], meta["rng_after"]
    check(next_start == total and len(manifest["checkpoint_files"]) == (total + 24) // 25, "Incomplete/duplicated checkpoint coverage")
    # Independent percentile/tail reductions, without creating any new CIs.
    for name in ("paired_reset_improvements", "paired_control_improvements"):
        values = aggregate[name]
        if values.size:
            check(np.isfinite(np.quantile(values, [.025, .5, .975], axis=-1)).all(), "Nonfinite paired percentile reduction")
            check(np.all((values < -TOL).sum(axis=-1) + (values > TOL).sum(axis=-1) + (abs(values) <= TOL).sum(axis=-1) == total), "Tail sign accounting loses replicates")
    linear_error = float(np.max(abs(aggregate["scalar_draws"][0, :2] - legacy["linear"][:, :total])))
    print(f"PASS {c['budget']}/{c['instance']}/{c['rate']:g}: all{total} checkpoint/cache/paired rows and legacy replay; first25 original counts/support/gamma/16 diagnostics independently checked; linear error={linear_error:.3g}", flush=True)
    return aggregate


def verify_resume(left, right):
    def jobs(folder):
        return {p.parent.relative_to(folder): p.parent for p in Path(folder).rglob("started_manifest.json")}
    a, b = jobs(left), jobs(right)
    check(a.keys() == b.keys() and len(a) > 0, "Resume comparison job coverage differs")
    arrays_checked = 0
    for name in a:
        with np.load(a[name] / "scalar_bootstrap_draws.npz", allow_pickle=False) as x, np.load(b[name] / "scalar_bootstrap_draws.npz", allow_pickle=False) as y:
            check(set(x.files) == set(y.files), "Resume aggregate schema changed")
            for key in x.files:
                xv, yv = x[key], y[key]
                if xv.dtype.kind in "biufc":
                    check(np.array_equal(xv, yv, equal_nan=True), "Resume differs bitwise from uninterrupted data: " + key)
                    arrays_checked += 1
                else:
                    check(np.array_equal(xv, yv), "Resume metadata differs from uninterrupted data: " + key)
        ap = sorted((a[name] / "batches").glob("batch_*.npz"))
        bp = sorted((b[name] / "batches").glob("batch_*.npz"))
        check(len(ap) == len(bp) == 2, "Resume50 must contain exactly two25 checkpoints")
        for pa, pb in zip(ap, bp):
            with np.load(pa, allow_pickle=False) as x, np.load(pb, allow_pickle=False) as y:
                am, bm = json.loads(str(x["checkpoint_json"])), json.loads(str(y["checkpoint_json"]))
            check(am == bm, "Resume changes seed/count/hash/checkpoint metadata")
    print(f"PASS interrupted25→50 versus uninterrupted50: {arrays_checked} numeric array families bitwise equal, including control gamma/support/paired differences and full RNG/count-hash chain", flush=True)


def verify_artifacts(directory, estimators, bvr, approved, golden, *, require_final=False, allow_partial=False):
    directory = Path(directory)
    files = [directory / "started_manifest.json"] if (directory / "started_manifest.json").is_file() else sorted(directory.rglob("started_manifest.json"))
    check(files, "No started jobs found")
    seen = set()
    incomplete = 0
    for path in files:
        if allow_partial and not (path.parent / "manifest.json").is_file():
            incomplete += 1
            continue
        started, parent, point = verify_point_artifacts(path.parent, estimators, bvr, approved, golden)
        if not seen:
            verify_negative_controls(started["configuration"], approved)
        c = started["configuration"]
        identity = c["budget"], c["instance"], float(c["rate"])
        check(identity not in seen, "Duplicate condition/budget job")
        seen.add(identity)
        verify_bootstrap_artifacts(path.parent, started, parent, point, estimators, require_final=require_final)
    if require_final:
        expected = {("equal_shot", r["instance"], float(r["rate"])) for r in golden.values()}
        expected |= {("gate_matched", case, .0001) for case in CASES}
        check(len(seen) == 40 and seen == expected, "Final scope must be all37 equal-shot plus3 representative gate jobs")
    check(seen, "No completed jobs available for numerical verification")
    if allow_partial:
        print(f"PASS partial audit of{len(seen)} completed jobs; skipped{incomplete} started/in-progress jobs, no final40-job claim", flush=True)
    return seen


def verify_negative_controls(valid_configuration, approved):
    cases = []
    def mutated(label, change):
        bad = copy.deepcopy(valid_configuration)
        change(bad)
        try:
            check_job_configuration(bad, approved)
        except (AssertionError, KeyError):
            cases.append(label)
            return
        raise AssertionError("Negative control unexpectedly accepted: " + label)
    mutated("legacy single ridge", lambda c: c["ridges"].update(single_exponential_zne=1e-8))
    mutated("support-sized offset", lambda c: c.update(fitting_floor=1 / 1234))
    mutated("wrong seed", lambda c: c.update(seed=c["seed"] + 1))
    mutated("fixed parent support", lambda c: c.update(observed_support_reselected=False))
    mutated("frozen parent gamma", lambda c: c.update(gamma_refitted_each_replicate=False))
    mutated("wrong per-scale budget", lambda c: c.update(per_scale_shots=19999))
    mutated("cross-budget interval", lambda c: c.update(no_cross_budget_paired_interval=False))
    mutated("reversed control sign", lambda c: c["paired_control_improvement_signs"].update(positive_favors="primary"))
    aggregator = module("primary_verify_aggregate_negative_controls", ROOT / "analysis/summarize_c1_primary_bootstrap_20261018.py")
    spike = np.zeros(10000)
    spike[:249], spike[249:304] = -.7, .13
    stats = aggregator.distribution_summary(spike, 0.)
    check(stats["ci_low"] == stats["ci_high"] == 0 and stats["numerical_negative_count"] == 249 and stats["numerical_positive_count"] == 55 and stats["minimum"] < 0 < stats["maximum"] and stats["pointwise_interval_sign"] == "unresolved", "Central[0,0] summary conceals genuine nonzero tails or claims equivalence")
    rejected = False
    try:
        aggregator.distribution_summary(np.array([0., np.nan]), 0.)
    except ValueError:
        rejected = True
    check(rejected, "Nonfinite locked-metric samples silently dropped from intervals")
    print(f"PASS {len(cases)} wrong-ridge/offset/seed/support/gamma/budget/sign negative controls plus[0,0]-tail and nonfinite-metric checks", flush=True)


def summary_reference(values, point):
    values = np.asarray(values, dtype=float).reshape(-1)
    check(np.isfinite(values).all() and np.isfinite(point), "Summary source has dropped/nonfinite draws")
    q = np.quantile(values, [.001, .01, .025, .5, .975, .99, .999])
    return {"point": float(point), "replicates": len(values), "finite_replicates": len(values), "nonfinite_replicates": 0,
        "numerical_tolerance": TOL, "ci_low": q[2], "ci_high": q[4], "bootstrap_mean": values.mean(),
        "bootstrap_median": q[3], "minimum": values.min(), "maximum": values.max(), "q001": q[0], "q01": q[1], "q99": q[5], "q999": q[6],
        "exact_negative_count": int(np.sum(values < 0)), "exact_zero_count": int(np.sum(values == 0)), "exact_positive_count": int(np.sum(values > 0)),
        "numerical_negative_count": int(np.sum(values < -TOL)), "numerical_zero_count": int(np.sum(abs(values) <= TOL)), "numerical_positive_count": int(np.sum(values > TOL)),
        "below_central_interval_count": int(np.sum(values < q[2])), "above_central_interval_count": int(np.sum(values > q[4])),
        "point_outside_interval": bool(point < q[2] or point > q[4]),
        "pointwise_interval_sign": "positive" if q[2] > TOL else "negative" if q[4] < -TOL else "unresolved"}


def verify_summary_row(row, values, point, *, prefix=""):
    for name, expected in summary_reference(values, point).items():
        actual = row[prefix + name]
        if isinstance(expected, (bool, np.bool_)):
            check(actual == str(bool(expected)), "Summary changed center/interval membership")
        elif isinstance(expected, str):
            check(actual == expected, "Pointwise sign classified as equivalence or wrong direction")
        elif isinstance(expected, (int, np.integer)):
            check(int(actual) == expected, "Summary sign/tail/count accounting wrong: " + prefix + name)
        else:
            equal(float(actual), expected, "Independent paired/marginal percentile or parent-center mismatch: " + prefix + name, atol=2e-12, rtol=2e-13)


def verify_aggregate(directory, *, require_final=False):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    check(summary["approved_config_sha256"] == sha(ROOT / "analysis/c1_primary_fixed_20261018.json"), "Aggregate uses stale primary configuration")
    check(summary["generator_sha256"] == sha(ROOT / "analysis/summarize_c1_primary_bootstrap_20261018.py"), "Aggregate generator changed")
    for filename, digest in summary["output_csv_sha256"].items():
        check(sha(directory / filename) == digest, "Aggregate table hash changed/raced: " + filename)
    check(summary["new_quantum_shots"] == 0 and summary["metric_definitions_changed"] is False and summary["cross_budget_paired_interval"] is False, "Aggregate relabels budgets or adds unsupported physical inference")
    n_jobs = summary["completed_jobs"]
    final = n_jobs == 40 and summary["replicates_per_completed_job"] == 10000
    check(summary["publication_ready"] == final and summary["status"] == ("complete" if final else "partial"), "Partial/smoke aggregation mislabeled publication-ready")
    check(not require_final or final, "Incomplete aggregate cannot be final evidence")
    all_scalars = rows(directory / "all_scalar_summaries.csv")
    absolute = rows(directory / "absolute_locked_metrics.csv")
    paired = rows(directory / "paired_comparisons_all.csv")
    controls = rows(directory / "paired_dual_controls.csv")
    support = rows(directory / "support_summaries.csv")
    identities = rows(directory / "verified_job_identities.csv")
    check(len(identities) == len(support) == n_jobs, "Aggregate job/support coverage missing")
    expected_scalar_count = expected_absolute_count = expected_pair_count = expected_control_count = 0
    seen_jobs = set()
    for job in identities:
        key = job["budget"], job["instance"], float(job["rate"])
        check(key not in seen_jobs, "Duplicate aggregate job")
        seen_jobs.add(key)
        path = Path(job["manifest"])
        if not path.is_absolute():
            path = ROOT / path
        check(sha(path) == job["manifest_sha256"], "Aggregate links wrong completed manifest")
        m = json.loads(path.read_text())
        check(m["identity"] == job["identity"] and m["status"] == "complete", "Aggregate uses incomplete/wrong cache identity")
        c = m["configuration"]
        with np.load(path.with_name("scalar_bootstrap_draws.npz"), allow_pickle=False) as z, np.load(path.with_name("parent_estimates.npz"), allow_pickle=False) as p:
            cache, parent = {k: z[k] for k in z.files}, {k: p[k] for k in p.files}
        select = lambda records: [r for r in records if (r["budget"], r["instance"], float(r["rate"])) == key]
        scalar_rows = select(all_scalars)
        scalar_lookup = {(r["route"], r["quantity"]): r for r in scalar_rows}
        check(len(scalar_lookup) == len(scalar_rows), "Aggregate scalar row duplicated")
        for ri, route in enumerate(c["routes"]):
            for qi, quantity in enumerate(c["quantities"]):
                record = scalar_lookup[route, quantity]
                check(record["identity"] == m["identity"] and float(record["rate_percent"]) == 100 * key[2], "Aggregate scalar source/rate identity wrong")
                verify_summary_row(record, cache["scalar_draws"][ri, qi], parent["scalar_draws"][ri, qi, 0])
                expected_scalar_count += 1
        for ri, route in enumerate(c["direct_routes"]):
            for qi, quantity in enumerate(c["direct_metrics"]):
                record = scalar_lookup[route, quantity]
                verify_summary_row(record, cache["direct_draws"][ri, qi], parent["direct_draws"][ri, qi, 0])
                check(record["role"] == ("context_only_20k" if route.endswith("_20k") else "direct_160k"), "Replay-only20k baseline used as scientific160k comparison")
                expected_scalar_count += 1
        absolute_rows = select(absolute)
        expected_abs = {(r, q) for r in c["routes"] for q in c["direct_metrics"]} | {(r, q) for r in c["direct_routes"] if r.endswith("_160k") for q in c["direct_metrics"]}
        check(len(absolute_rows) == len(expected_abs) and {(r["route"], r["metric"]) for r in absolute_rows} == expected_abs, "Absolute metric table omits primary/control or includes20k comparison")
        for row in absolute_rows:
            for field, value in scalar_lookup[row["route"], row["metric"]].items():
                check(row[field] == value, "Absolute metric row not exact scalar-summary subset")
            check(int(row["circuit_shots"]) == (160000 if row["route"] in c["direct_routes"] or key[0] == "equal_shot" else 89504), "Absolute metric resource label wrong")
        expected_absolute_count += len(expected_abs)
        reset_i = c["direct_routes"].index("reset_160k")
        control_rows = select(controls)
        expected_controls = {(r, q) for r in c["routes"][4:] for q in c["direct_metrics"]}
        check(len(control_rows) == len(expected_controls) and {(r["route"], r["metric"]) for r in control_rows} == expected_controls, "Control summary coverage changed")
        for row in select(paired):
            qi = c["direct_metrics"].index(row["metric"])
            if row["reference"] == "reset_160k":
                ri = c["routes"].index(row["route"])
                sample = cache["paired_reset_improvements"][ri, qi]
                point = parent["paired_reset_improvements"][ri, qi, 0]
            elif row["reference"] == "raw_160k":
                raw_i = c["direct_routes"].index("raw_160k")
                sample = cache["direct_draws"][raw_i, qi] - cache["direct_draws"][reset_i, qi]
                point = parent["direct_draws"][raw_i, qi, 0] - parent["direct_draws"][reset_i, qi, 0]
                if qi == 1:
                    sample, point = -sample, -point
            else:
                check(row["reference"] == METHODS[3] and row["route"] in CONTROL_ROUTES, "Unknown/cross-budget paired inference")
                ci = c["routes"][4:].index(row["route"])
                sample = cache["paired_control_improvements"][ci, qi]
                point = parent["paired_control_improvements"][ci, qi, 0]
            check(row["positive_favors"] == row["route"], "Paired positive-direction label changed")
            verify_summary_row(row, sample, point)
            expected_pair_count += 1
        expected_pairs = len(c["routes"]) * 2 + (2 if "raw_160k" in c["direct_routes"] else 0) + len(expected_controls)
        check(len(select(paired)) == expected_pairs, "Paired comparison rows missing or duplicated")
        for row in control_rows:
            ci = c["routes"][4:].index(row["route"])
            qi = c["direct_metrics"].index(row["metric"])
            sample, point = cache["paired_control_improvements"][ci, qi], parent["paired_control_improvements"][ci, qi, 0]
            verify_summary_row(row, sample, point)
            verify_summary_row(row, -sample, -point, prefix="primary_improvement_")
            expected_control_count += 1
        sr = select(support)
        check(len(sr) == 1, "Support summary coverage missing")
        verify_summary_row(sr[0], cache["support_sizes"], np.count_nonzero(np.any(parent["scale_parents"] > 0, axis=0)))
    check(len(all_scalars) == expected_scalar_count == summary["all_scalar_summary_rows"] and len(absolute) == expected_absolute_count == summary["absolute_locked_metric_rows"] and len(paired) == expected_pair_count == summary["paired_comparison_rows"] and len(controls) == expected_control_count == summary["dual_control_rows"], "Aggregate count mismatch")
    for filename, expected in (("paired_reset_improvements.csv", [r for r in paired if r["reference"] == "reset_160k"]),
                               ("paired_raw_reset_improvements.csv", [r for r in paired if r["reference"] == "raw_160k"]),
                               ("primary_rate_results.csv", [r for r in absolute if float(r["rate"]) == .0001])):
        check(rows(directory / filename) == expected, "Aggregate subset table mislabeled or altered: " + filename)
    if require_final:
        check(len(seen_jobs) == 40 and expected_scalar_count == 2958 and expected_absolute_count == 486 and expected_pair_count == 418 and expected_control_count == 12, "Final40-job model/budget/control accounting wrong")
    print(f"PASS independent aggregate reduction: {n_jobs} jobs/{len(all_scalars)} scalar rows/{len(paired)} paired rows, parent centers, all percentiles/signs/tails and both control directions; status={summary['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume-comparison", nargs=2, type=Path)
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    check(not (args.allow_partial and args.require_final), "Partial and final contract options are mutually exclusive")
    estimators, bvr, core, reference, config = modules_and_config()
    verify_approved_toys(estimators, core, reference)
    golden = verify_fixed_point_pack(estimators, config)
    if args.artifacts:
        verify_artifacts(args.artifacts, estimators, bvr, config, golden, require_final=args.require_final, allow_partial=args.allow_partial)
    if args.resume_comparison:
        verify_resume(*args.resume_comparison)
    if args.aggregate:
        verify_aggregate(args.aggregate, require_final=args.require_final)
    check(not args.require_final or args.artifacts is not None, "Final scope verification requires actual completed artifacts")
    if not (args.artifacts or args.aggregate or args.resume_comparison):
        print("PASS restricted-reference/configuration/golden-hash checks only; no bootstrap artifact verification requested.", flush=True)
        return
    print("PASS independent approved-C1 bootstrap checks; no production launch, new circuit data or output writes.", flush=True)


if __name__ == "__main__":
    main()
