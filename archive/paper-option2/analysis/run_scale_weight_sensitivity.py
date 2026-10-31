#!/usr/bin/env python3
"""Run the Option 2 scale/weight/ridge sensitivity at the common 0.01% rate.

The analysis uses saved count parents only.  Each setting reuses the same
multinomial bootstrap draw, so differences among settings are not confounded by
new resampling noise.  The only output-performance quantities are TVD, the
normalized main-lobe group margin, and their permitted direct differences.
All other reported quantities describe fitting or simplex projection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_HASH = (
    "792445cfd99777c2e6eddb5dc781ec30c58b9dc8bf8650d931757d60a717c541"
)
COMMON_RATE = 1.0e-4
ROUTE_LABELS = {
    "linear_zne": "Reset + Linear ZNE",
    "quadratic_zne": "Reset + Quadratic ZNE",
    "single_exponential_zne": "Reset + Single-exponential ZNE",
    "dual_exponential_zne": "Reset + Dual-exponential ZNE",
}
DIAGNOSTICS = (
    "negative_mass_before_projection",
    "negative_coordinate_fraction",
    "normalization_defect",
    "l1_projection_correction",
    "l2_projection_correction",
    "postprojection_support_size",
    "projected_change_from_scale_one_l2",
    "projection_correction_to_change_ratio",
)
BOOTSTRAP_QUANTITIES = (
    "tvd",
    "normalized_group_margin",
    "reset_minus_zne_tvd",
    "zne_minus_reset_normalized_group_margin",
    *DIAGNOSTICS,
)
BOOTSTRAP_CACHE_NAME = "scalar_bootstrap_draws.npz"
BOOTSTRAP_CACHE_SCHEMA_VERSION = 1
BOOTSTRAP_CACHE_AXIS_ORDER = ("setting", "route", "quantity", "replicate")


@dataclass(frozen=True)
class Setting:
    key: str
    description: str
    indices: np.ndarray
    weight_exponent: float
    ridge: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=WORKTREE_ROOT / "shor_error_mitigation",
    )
    parser.add_argument(
        "--instance",
        required=True,
        choices=("n119_a13", "n119_a50", "n119_a16"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Permit a nonfinal replicate count for a temporary smoke test.",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--per-scale-shots",
        type=int,
        default=20_000,
        help=(
            "Multinomial shots drawn from every saved scale parent.  The "
            "default is the equal-circuit-shot design; 11188 is the audited "
            "equal-gate allocation."
        ),
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Run only the primary full-grid setting.  This is used to obtain "
            "complete projection diagnostics for the equal-gate budget."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    low, median, high = np.quantile(
        np.asarray(values, dtype=float), [tail, 0.5, 1.0 - tail]
    )
    return {
        "bootstrap_median": float(median),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(
    lambdas: np.ndarray,
    *,
    weight_exponent: float,
    ridge: float,
    estimators: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fixed = {
        "linear_zne": estimators.polynomial_intercept_coefficients(
            lambdas, degree=1, ridge=0.0, weight_exponent=weight_exponent
        ),
        "quadratic_zne": estimators.polynomial_intercept_coefficients(
            lambdas, degree=2, ridge=ridge, weight_exponent=weight_exponent
        ),
    }
    exponential = {
        "single_exponential_zne": estimators.build_exponential_grid_family(
            lambdas,
            components=1,
            single_grid=estimators.SINGLE_EXPONENTIAL_GRID,
            ridge=ridge,
            weight_exponent=weight_exponent,
        ),
        "dual_exponential_zne": estimators.build_exponential_grid_family(
            lambdas,
            components=2,
            dual_grid=estimators.DUAL_EXPONENTIAL_GRID,
            ridge=ridge,
            weight_exponent=weight_exponent,
        ),
    }
    return fixed, exponential


def _projection_diagnostics_rows(
    before: np.ndarray, after: np.ndarray, scale_one: np.ndarray
) -> dict[str, np.ndarray]:
    correction = after - before
    change = after - scale_one
    correction_l2 = np.linalg.norm(correction, axis=1)
    change_l2 = np.linalg.norm(change, axis=1)
    return {
        "negative_mass_before_projection": -np.minimum(before, 0.0).sum(axis=1),
        "negative_coordinate_fraction": np.mean(before < 0.0, axis=1),
        "normalization_defect": np.abs(before.sum(axis=1) - 1.0),
        "l1_projection_correction": np.abs(correction).sum(axis=1),
        "l2_projection_correction": correction_l2,
        "postprojection_support_size": np.count_nonzero(after > 0.0, axis=1).astype(float),
        "projected_change_from_scale_one_l2": change_l2,
        "projection_correction_to_change_ratio": correction_l2
        / np.maximum(change_l2, np.finfo(np.float64).eps),
    }


def _plugin_metric(
    distribution: np.ndarray, config: dict[str, Any], metrics: Any
) -> tuple[float, float]:
    value = metrics.distribution_metrics(
        distribution,
        int(config["measurement_bits"]),
        int(config["order"]),
    )
    return float(value.tvd), float(value.normalized_grouped_peak_margin)


def _polynomial_prediction(
    train_lambdas: np.ndarray,
    train_distributions: np.ndarray,
    held_lambda: float,
    *,
    degree: int,
    ridge: float,
    weight_exponent: float,
    estimators: Any,
) -> np.ndarray:
    weights = estimators.objective_weights(
        train_lambdas, exponent=weight_exponent
    )
    design = np.vander(train_lambdas, N=degree + 1, increasing=True)
    normal = design.T @ (weights[:, None] * design)
    if ridge:
        normal += ridge * np.eye(degree + 1)
    evaluation = np.asarray([held_lambda**power for power in range(degree + 1)])
    operator = evaluation @ np.linalg.solve(normal, design.T * weights)
    return operator @ train_distributions


def _exponential_prediction(
    train_lambdas: np.ndarray,
    train_distributions: np.ndarray,
    held_lambda: float,
    *,
    components: int,
    ridge: float,
    weight_exponent: float,
    estimators: Any,
) -> np.ndarray:
    family = estimators.build_exponential_grid_family(
        train_lambdas,
        components=components,
        single_grid=estimators.SINGLE_EXPONENTIAL_GRID,
        dual_grid=estimators.DUAL_EXPONENTIAL_GRID,
        ridge=ridge,
        weight_exponent=weight_exponent,
    )
    _, diagnostics = estimators.fit_exponential_grid_batch(
        train_distributions, family
    )
    selected = int(np.asarray(diagnostics["selected_index"]).reshape(-1)[0])
    rates = family.parameters[selected]
    weights = family.objective_weights
    design = np.exp(-train_lambdas[:, None] * rates[None, :])
    normal = design.T @ (weights[:, None] * design)
    if ridge:
        normal += ridge * np.eye(components)
    evaluation = np.exp(-held_lambda * rates)
    operator = evaluation @ np.linalg.solve(normal, design.T * weights)
    floor = 1.0 / train_distributions.shape[1]
    return floor + operator @ (train_distributions - floor)


def main() -> int:
    args = _parse_args()
    if args.replicates != 10_000 and not args.allow_smoke:
        raise SystemExit("final Option 2 sensitivity requires exactly B=10,000")
    if args.per_scale_shots not in (20_000, 11_188) and not args.allow_smoke:
        raise SystemExit(
            "final Option 2 outputs require 20000 equal-shot or 11188 equal-gate "
            "draws per scale"
        )
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("confidence must lie between zero and one")
    if (
        not args.allow_smoke
        and not np.isclose(args.confidence, 0.95, rtol=0.0, atol=1.0e-15)
    ):
        raise SystemExit("final Option 2 sensitivity outputs require confidence 0.95")
    analysis_root = args.analysis_root.resolve()
    source_path = analysis_root / "src" / "shor_qem" / "bias_variance_revision.py"
    source_hash = _sha256(source_path)
    if source_hash != EXPECTED_SOURCE_HASH:
        raise SystemExit(f"audited source hash changed: {source_hash}")
    sys.path.insert(0, str(analysis_root / "src"))

    from shor_qem import bias_variance_revision as bvr  # noqa: PLC0415
    from shor_qem import estimators, metrics  # noqa: PLC0415
    from shor_qem.bootstrap_analysis import (  # noqa: PLC0415
        _fit_zne_batch_detailed,
    )

    spec = bvr.INSTANCE_SPECS[args.instance]
    data = bvr._prepare_data(spec, np.asarray([COMMON_RATE]), "realized")
    config = data["config"]
    lambdas = np.asarray(data["fit_lambdas"], dtype=float)
    grid_parent = np.asarray(data["reset"][0], dtype=float)
    scale_one_index = int(data["lambda_one"])
    target = metrics.ideal_distribution(
        int(config["measurement_bits"]), int(config["order"])
    )
    groups = metrics.main_lobe_groups(
        int(config["measurement_bits"]), int(config["order"])
    )
    ideal_margin = metrics.distribution_metrics(
        target, int(config["measurement_bits"]), int(config["order"])
    ).grouped_peak_margin

    all_indices = np.arange(lambdas.size, dtype=int)
    subset_indices = np.flatnonzero(lambdas <= 2.2 + 5.0e-12)
    settings: tuple[Setting, ...] = (
        Setting(
            "full_lambda_minus2_ridge_1e-8",
            "eight effective scales, weights proportional to lambda^-2, ridge 1e-8",
            all_indices,
            2.0,
            1.0e-8,
        ),
        Setting(
            "le2p2_lambda_minus2_ridge_1e-8",
            "effective scales through nominal 2.2, weights proportional to lambda^-2, ridge 1e-8",
            subset_indices,
            2.0,
            1.0e-8,
        ),
        Setting(
            "full_uniform_ridge_1e-8",
            "eight effective scales, uniform weights, ridge 1e-8",
            all_indices,
            0.0,
            1.0e-8,
        ),
        Setting(
            "full_lambda_minus2_ridge_1e-9",
            "eight effective scales, weights proportional to lambda^-2, ridge 1e-9",
            all_indices,
            2.0,
            1.0e-9,
        ),
        Setting(
            "full_lambda_minus2_ridge_1e-7",
            "eight effective scales, weights proportional to lambda^-2, ridge 1e-7",
            all_indices,
            2.0,
            1.0e-7,
        ),
    )
    if args.primary_only:
        settings = settings[:1]
    suites = {
        setting.key: _make_suite(
            lambdas[setting.indices],
            weight_exponent=setting.weight_exponent,
            ridge=setting.ridge,
            estimators=estimators,
        )
        for setting in settings
    }

    result_names = BOOTSTRAP_QUANTITIES
    draws = {
        setting.key: {
            route: {
                name: np.empty(args.replicates, dtype=float) for name in result_names
            }
            for route in ROUTE_LABELS
        }
        for setting in settings
    }

    extension_shots = bvr.DIRECT_SHOTS - args.per_scale_shots
    plugin_reset = (
        args.per_scale_shots * grid_parent[scale_one_index]
        + extension_shots * np.asarray(data["reset_extension"][0], dtype=float)
    ) / bvr.DIRECT_SHOTS
    if args.per_scale_shots == 20_000 and not np.allclose(
        plugin_reset,
        np.asarray(data["reset_full"][0], dtype=float),
        rtol=0.0,
        atol=5.0e-15,
    ):
        raise RuntimeError(
            "equal-shot empirical-parent mixture does not reconstruct Reset-160k"
        )
    plugin_reset_tvd, plugin_reset_margin = _plugin_metric(
        plugin_reset, config, metrics
    )
    plugin: dict[str, dict[str, dict[str, float]]] = {}
    for setting in settings:
        indices = setting.indices
        fixed, exponential = suites[setting.key]
        projected, _, unconstrained, _ = _fit_zne_batch_detailed(
            grid_parent[indices][None, :, :],
            fixed_coefficients=fixed,
            exponential_families=exponential,
        )
        plugin[setting.key] = {}
        for route in ROUTE_LABELS:
            after = projected[route][0]
            before = unconstrained[route][0]
            tvd, margin = _plugin_metric(after, config, metrics)
            diagnostic = _projection_diagnostics_rows(
                before[None, :], after[None, :], grid_parent[scale_one_index][None, :]
            )
            plugin[setting.key][route] = {
                "tvd": tvd,
                "normalized_group_margin": margin,
                "reset_minus_zne_tvd": plugin_reset_tvd - tvd,
                "zne_minus_reset_normalized_group_margin": margin
                - plugin_reset_margin,
                **{name: float(values[0]) for name, values in diagnostic.items()},
            }

    reset_scale_seed, reset_extension_seed, _, _ = bvr._bootstrap_seed_streams(
        args.seed, args.instance, COMMON_RATE
    )
    reset_rng = np.random.default_rng(reset_scale_seed)
    extension_rng = np.random.default_rng(reset_extension_seed)
    for start in range(0, args.replicates, bvr.FROZEN_BATCH_SIZE):
        stop = min(start + bvr.FROZEN_BATCH_SIZE, args.replicates)
        size = stop - start
        reset_counts = np.stack(
            [
                reset_rng.multinomial(args.per_scale_shots, parent, size=size)
                for parent in grid_parent
            ],
            axis=1,
        ).astype(float)
        reset_grid = reset_counts / args.per_scale_shots
        extension_counts = extension_rng.multinomial(
            extension_shots,
            np.asarray(data["reset_extension"][0]),
            size=size,
        ).astype(float)
        reset160 = (
            reset_counts[:, scale_one_index, :] + extension_counts
        ) / bvr.DIRECT_SHOTS
        reset_metrics = bvr._post_metrics(reset160, target, groups, ideal_margin)
        for setting in settings:
            fixed, exponential = suites[setting.key]
            selected_grid = reset_grid[:, setting.indices, :]
            projected, _, unconstrained, _ = _fit_zne_batch_detailed(
                selected_grid,
                fixed_coefficients=fixed,
                exponential_families=exponential,
            )
            for route in ROUTE_LABELS:
                post = projected[route]
                pre = unconstrained[route]
                post_metrics = bvr._post_metrics(post, target, groups, ideal_margin)
                diagnostic = _projection_diagnostics_rows(
                    pre, post, reset_grid[:, scale_one_index, :]
                )
                route_draws = draws[setting.key][route]
                route_draws["tvd"][start:stop] = post_metrics["tvd"]
                route_draws["normalized_group_margin"][start:stop] = post_metrics[
                    "normalized_grouped_peak_margin"
                ]
                route_draws["reset_minus_zne_tvd"][start:stop] = (
                    reset_metrics["tvd"] - post_metrics["tvd"]
                )
                route_draws[
                    "zne_minus_reset_normalized_group_margin"
                ][start:stop] = (
                    post_metrics["normalized_grouped_peak_margin"]
                    - reset_metrics["normalized_grouped_peak_margin"]
                )
                for name, values in diagnostic.items():
                    route_draws[name][start:stop] = values
        if stop == args.replicates or stop % 1000 == 0:
            print(
                f"[{args.instance}] sensitivity bootstrap {stop}/{args.replicates}",
                flush=True,
            )

    summary_rows: list[dict[str, Any]] = []
    for setting in settings:
        for route, route_label in ROUTE_LABELS.items():
            row: dict[str, Any] = {
                "instance": args.instance,
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": int(config["order"]),
                "rate": COMMON_RATE,
                "rate_percent": 100.0 * COMMON_RATE,
                "setting": setting.key,
                "setting_description": setting.description,
                "route": route,
                "route_label": route_label,
                "effective_scales": ";".join(
                    f"{value:.12g}" for value in lambdas[setting.indices]
                ),
                "scale_count": int(setting.indices.size),
                "zne_circuit_shots": int(
                    setting.indices.size * args.per_scale_shots
                ),
                "weight_exponent": setting.weight_exponent,
                "ridge_regularized_models": setting.ridge,
                "bootstrap_replicates": args.replicates,
                "confidence": args.confidence,
            }
            for name in result_names:
                central = plugin[setting.key][route][name]
                interval = _summary(
                    draws[setting.key][route][name], args.confidence
                )
                row[f"plugin_{name}"] = central
                row[f"{name}_bootstrap_median"] = interval["bootstrap_median"]
                row[f"{name}_ci_low"] = interval["ci_low"]
                row[f"{name}_ci_high"] = interval["ci_high"]
            summary_rows.append(row)

    loo_rows: list[dict[str, Any]] = []
    for held_i, held_lambda in enumerate(lambdas):
        train_indices = np.delete(all_indices, held_i)
        train_lambdas = lambdas[train_indices]
        train_distributions = grid_parent[train_indices]
        observed = grid_parent[held_i]
        predictions = {
            "linear_zne": _polynomial_prediction(
                train_lambdas,
                train_distributions,
                held_lambda,
                degree=1,
                ridge=0.0,
                weight_exponent=2.0,
                estimators=estimators,
            ),
            "quadratic_zne": _polynomial_prediction(
                train_lambdas,
                train_distributions,
                held_lambda,
                degree=2,
                ridge=1.0e-8,
                weight_exponent=2.0,
                estimators=estimators,
            ),
            "single_exponential_zne": _exponential_prediction(
                train_lambdas,
                train_distributions,
                held_lambda,
                components=1,
                ridge=1.0e-8,
                weight_exponent=2.0,
                estimators=estimators,
            ),
            "dual_exponential_zne": _exponential_prediction(
                train_lambdas,
                train_distributions,
                held_lambda,
                components=2,
                ridge=1.0e-8,
                weight_exponent=2.0,
                estimators=estimators,
            ),
        }
        for route, prediction in predictions.items():
            loo_rows.append(
                {
                    "instance": args.instance,
                    "a": int(config["a"]),
                    "order": int(config["order"]),
                    "rate": COMMON_RATE,
                    "held_out_effective_scale": float(held_lambda),
                    "route": route,
                    "route_label": ROUTE_LABELS[route],
                    "squared_l2_predictive_residual": float(
                        np.square(prediction - observed).sum()
                    ),
                    "prediction_stage": "preprojection",
                    "training_weight_exponent": 2.0,
                    "training_ridge_regularized_models": 1.0e-8,
                    "estimate": "saved-parent plug-in",
                }
            )

    output_dir = args.output_dir.resolve() / args.instance
    output_dir.mkdir(parents=True, exist_ok=True)
    setting_axis = tuple(setting.key for setting in settings)
    route_axis = tuple(ROUTE_LABELS)
    scalar_draws = np.stack(
        [
            np.stack(
                [
                    np.stack(
                        [
                            np.asarray(
                                draws[setting][route][quantity],
                                dtype=np.float64,
                            )
                            for quantity in BOOTSTRAP_QUANTITIES
                        ],
                        axis=0,
                    )
                    for route in route_axis
                ],
                axis=0,
            )
            for setting in setting_axis
        ],
        axis=0,
    )
    expected_cache_shape = (
        len(setting_axis),
        len(route_axis),
        len(BOOTSTRAP_QUANTITIES),
        args.replicates,
    )
    if scalar_draws.dtype != np.float64 or scalar_draws.shape != expected_cache_shape:
        raise RuntimeError(
            "unexpected scalar bootstrap cache representation: "
            f"dtype={scalar_draws.dtype}, shape={scalar_draws.shape}"
        )
    if not np.all(np.isfinite(scalar_draws)):
        raise RuntimeError("scalar bootstrap cache contains nonfinite draws")
    np.savez_compressed(
        output_dir / BOOTSTRAP_CACHE_NAME,
        schema_version=np.asarray(BOOTSTRAP_CACHE_SCHEMA_VERSION, dtype=np.int64),
        axis_order=np.asarray(BOOTSTRAP_CACHE_AXIS_ORDER),
        instance=np.asarray(args.instance),
        settings=np.asarray(setting_axis),
        routes=np.asarray(route_axis),
        quantities=np.asarray(BOOTSTRAP_QUANTITIES),
        scalar_draws=scalar_draws,
        replicates=np.asarray(args.replicates, dtype=np.int64),
        confidence=np.asarray(args.confidence, dtype=np.float64),
        seed=np.asarray(args.seed, dtype=np.int64),
        rate=np.asarray(COMMON_RATE, dtype=np.float64),
        per_scale_shots=np.asarray(args.per_scale_shots, dtype=np.int64),
        primary_only=np.asarray(args.primary_only, dtype=np.bool_),
    )
    _write_csv(output_dir / "scale_weight_ridge_sensitivity.csv", summary_rows)
    _write_csv(output_dir / "leave_one_scale_out_residuals.csv", loo_rows)
    metadata = {
        "analysis": (
            "Option 2 primary-setting equal-gate diagnostics"
            if args.primary_only and args.per_scale_shots == 11_188
            else "Option 2 scale, weight, and ridge sensitivity"
        ),
        "instance": args.instance,
        "rate": COMMON_RATE,
        "effective_scales": lambdas.tolist(),
        "settings": [
            {
                "key": setting.key,
                "description": setting.description,
                "indices": setting.indices.tolist(),
                "weight_exponent": setting.weight_exponent,
                "ridge": setting.ridge,
                "scale_count": int(setting.indices.size),
                "zne_circuit_shots": int(
                    setting.indices.size * args.per_scale_shots
                ),
            }
            for setting in settings
        ],
        "bootstrap_replicates": args.replicates,
        "per_scale_shots": args.per_scale_shots,
        "zne_circuit_shots_primary_full_grid": 8 * args.per_scale_shots,
        "direct_reset_shots": int(bvr.DIRECT_SHOTS),
        "primary_only": args.primary_only,
        "seed": args.seed,
        "confidence": args.confidence,
        "bootstrap_draw_cache": BOOTSTRAP_CACHE_NAME,
        "bootstrap_draw_cache_schema_version": BOOTSTRAP_CACHE_SCHEMA_VERSION,
        "central_estimate": "saved-parent plug-in",
        "reset_plugin_comparator": (
            f"{args.per_scale_shots}/160000 times the saved scale-one parent plus "
            f"{extension_shots}/160000 times the saved Reset extension parent"
        ),
        "leave_one_scale_out_stage": "preprojection",
        "interval_scope": "pointwise percentile, conditional on saved empirical parents",
        "audited_analysis_source_sha256": source_hash,
        "new_quantum_simulation": False,
        "final_replicate_contract": (
            args.replicates == 10_000
            and args.per_scale_shots in (20_000, 11_188)
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
