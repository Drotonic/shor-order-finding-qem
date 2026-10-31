"""Conditional finite-shot bootstrap for the N=119 experiment.

The resampling unit is one measured shot within each saved distribution.  Each
noise-scale distribution is resampled independently, except that the same
resampled lambda=1 reset distribution is used for Reset and all four ZNE
estimators.  The no-reset baseline is resampled independently.  The inference
scope follows the supplied parent records: the frozen revision uses the latest
shot-wise random-folding parents, whereas legacy callers may still use older
fixed-pattern parents and must label that distinction in their metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    PROJECT_ROOT,
    build_processed_counts,
    load_config,
    load_processed_counts,
)
from .estimators import (
    DEFAULT_RIDGE,
    DUAL_EXPONENTIAL_GRID,
    SINGLE_EXPONENTIAL_GRID,
    ExponentialGridFamily,
    build_exponential_grid_family,
    fit_exponential_grid_batch,
    objective_weights,
    polynomial_intercept_coefficients,
)
from .metrics import (
    distribution_metrics,
    grouped_peak_statistics_rows,
    ideal_distribution,
    main_lobe_groups,
    simplex_projection_rows,
)


ZNE_METHODS = (
    "linear_zne",
    "quadratic_zne",
    "single_exponential_zne",
    "dual_exponential_zne",
)
METHODS = ("raw_no_reset", "reset_lambda_1", *ZNE_METHODS)
METHOD_LABELS = {
    "raw_no_reset": "Raw (no reset)",
    "reset_lambda_1": "Reset ($\\lambda=1$)",
    "linear_zne": "Linear ZNE",
    "quadratic_zne": "Quadratic ZNE",
    "single_exponential_zne": "Single-exp. ZNE",
    "dual_exponential_zne": "Dual-exp. ZNE",
}
BOOTSTRAP_METRICS = (
    "tvd",
    "grouped_peak_margin",
    "grouped_peak_mass",
    "minimum_group_probability",
    "maximum_outside_group_probability",
)


def _row_metrics(
    probabilities: np.ndarray,
    *,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized metrics for a batch of full outcome distributions."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != target.size:
        raise ValueError("probability batch has an unexpected shape")
    if groups is None:
        if support is None:
            raise ValueError("support or groups is required")
        groups = tuple(
            np.asarray([index], dtype=np.int64)
            for index in np.asarray(support, dtype=np.int64)
        )
    grouped = grouped_peak_statistics_rows(values, groups)
    minimum_group = grouped["minimum_group_probability"]
    maximum_outside = grouped["maximum_outside_group_probability"]
    return {
        "tvd": 0.5 * np.abs(values - target[None, :]).sum(axis=1),
        "grouped_peak_margin": grouped["margin"],
        "grouped_peak_mass": grouped["grouped_probability_mass"],
        "minimum_group_probability": minimum_group,
        "maximum_outside_group_probability": maximum_outside,
    }


def _normalize_counts(counts: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(counts, dtype=np.int64)
    totals = values.sum(axis=-1)
    if not np.all(totals == totals.flat[0]):
        raise ValueError("count vectors do not all have the same shot total")
    shots = int(totals.flat[0])
    if shots <= 0 or np.any(values < 0):
        raise ValueError("counts must be nonnegative with a positive total")
    probabilities = values.astype(np.float64) / shots
    # Multinomial is strict about floating-point totals.  Renormalize each row.
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities, shots


def _build_estimator_suite(
    lambdas: np.ndarray,
    *,
    weight_exponent: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, ExponentialGridFamily]]:
    fixed_coefficients = {
        "linear_zne": polynomial_intercept_coefficients(
            lambdas,
            degree=1,
            ridge=0.0,
            weight_exponent=weight_exponent,
        ),
        "quadratic_zne": polynomial_intercept_coefficients(
            lambdas,
            degree=2,
            ridge=DEFAULT_RIDGE,
            weight_exponent=weight_exponent,
        ),
    }
    exponential_families = {
        "single_exponential_zne": build_exponential_grid_family(
            lambdas,
            components=1,
            single_grid=SINGLE_EXPONENTIAL_GRID,
            ridge=DEFAULT_RIDGE,
            weight_exponent=weight_exponent,
        ),
        "dual_exponential_zne": build_exponential_grid_family(
            lambdas,
            components=2,
            dual_grid=DUAL_EXPONENTIAL_GRID,
            ridge=DEFAULT_RIDGE,
            weight_exponent=weight_exponent,
        ),
    }
    return fixed_coefficients, exponential_families


def _fit_zne_batch_detailed(
    reset_distributions: np.ndarray,
    *,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
]:
    """Fit all four estimators and expose preprojection vectors.

    The historical public helper returns only projected distributions and
    diagnostics.  The revision analysis also needs the unconstrained vectors
    for a conditional L2 bias--variance decomposition.  Keeping this detailed
    path separate avoids changing existing callers while guaranteeing that
    both analyses use exactly the same fit and projection implementation.
    """

    values = np.asarray(reset_distributions, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("reset_distributions must have shape (batch, lambdas, outcomes)")

    unconstrained: dict[str, np.ndarray] = {}
    selection: dict[str, dict[str, np.ndarray]] = {}
    for method, coefficients in fixed_coefficients.items():
        unconstrained[method] = np.einsum(
            "l,blk->bk", coefficients, values, optimize=True
        )

    floor = 1.0 / values.shape[2]
    centered = values - floor
    cross_products = np.einsum(
        "blk,bmk->blm", centered, centered, optimize=True
    )
    for method, family in exponential_families.items():
        estimate, fit_diagnostics = fit_exponential_grid_batch(
            values,
            family,
            uniform_floor=floor,
            cross_products=cross_products,
        )
        unconstrained[method] = estimate
        selection[method] = fit_diagnostics

    projected: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for method in ZNE_METHODS:
        before = unconstrained[method]
        after = simplex_projection_rows(before)
        projected[method] = after
        method_diagnostics: dict[str, np.ndarray] = {
            "normalization_error_before_projection": np.abs(
                before.sum(axis=1) - 1.0
            ),
            "negative_mass_before_projection": -np.minimum(before, 0.0).sum(axis=1),
            "negative_entry_count_before_projection": np.count_nonzero(
                before < 0.0, axis=1
            ).astype(np.float64),
            "negative_entry_fraction_before_projection": np.mean(
                before < 0.0, axis=1
            ),
            "l2_projection_correction": np.linalg.norm(after - before, axis=1),
            "projection_active": np.any(
                np.abs(after - before) > 1.0e-14, axis=1
            ).astype(np.float64),
        }
        if method == "single_exponential_zne":
            for name in (
                "selected_index",
                "selected_intercept_l1_norm",
                "selected_intercept_l2_norm_squared",
                "selected_normal_condition_number",
                "runner_up_score_gap",
                "near_tie_candidate_count",
                "penalized_weighted_residual",
            ):
                method_diagnostics[name] = np.asarray(
                    selection[method][name], dtype=np.float64
                )
            method_diagnostics["selected_gamma"] = selection[method][
                "selected_parameters"
            ][:, 0]
            method_diagnostics["grid_lower_boundary"] = selection[method][
                "lower_boundary"
            ].astype(np.float64)
            method_diagnostics["grid_upper_boundary"] = selection[method][
                "upper_boundary"
            ].astype(np.float64)
            method_diagnostics["grid_any_boundary"] = selection[method][
                "any_boundary"
            ].astype(np.float64)
        elif method == "dual_exponential_zne":
            for name in (
                "selected_index",
                "selected_intercept_l1_norm",
                "selected_intercept_l2_norm_squared",
                "selected_normal_condition_number",
                "selected_minimum_rate_separation",
                "runner_up_score_gap",
                "near_tie_candidate_count",
                "penalized_weighted_residual",
            ):
                method_diagnostics[name] = np.asarray(
                    selection[method][name], dtype=np.float64
                )
            method_diagnostics["selected_gamma_1"] = selection[method][
                "selected_parameters"
            ][:, 0]
            method_diagnostics["selected_gamma_2"] = selection[method][
                "selected_parameters"
            ][:, 1]
            method_diagnostics["grid_lower_boundary"] = selection[method][
                "lower_boundary"
            ].astype(np.float64)
            method_diagnostics["grid_upper_boundary"] = selection[method][
                "upper_boundary"
            ].astype(np.float64)
            method_diagnostics["grid_any_boundary"] = selection[method][
                "any_boundary"
            ].astype(np.float64)
        diagnostics[method] = method_diagnostics
    return projected, diagnostics, unconstrained, selection


def _fit_zne_batch(
    reset_distributions: np.ndarray,
    *,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """Fit and project all four ZNE estimators for one bootstrap batch."""

    projected, diagnostics, _, _ = _fit_zne_batch_detailed(
        reset_distributions,
        fixed_coefficients=fixed_coefficients,
        exponential_families=exponential_families,
    )
    return projected, diagnostics


def _point_rows(
    *,
    config: dict[str, Any],
    rates: np.ndarray,
    reset_probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    measurement_bits = int(config["measurement_bits"])
    order = int(config["order"])
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for rate_i, rate in enumerate(rates):
        zne_distributions, zne_diagnostics = _fit_zne_batch(
            reset_probabilities[rate_i][None, :, :],
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
        )
        distributions = {
            "raw_no_reset": raw_probabilities[rate_i],
            "reset_lambda_1": reset_probabilities[rate_i, 0],
            **{method: values[0] for method, values in zne_distributions.items()},
        }
        for method, distribution in distributions.items():
            metrics = distribution_metrics(distribution, measurement_bits, order)
            rows.append(
                {
                    "instance": config["instance"],
                    "N": int(config["N"]),
                    "a": int(config["a"]),
                    "order": order,
                    "rate": float(rate),
                    "method": method,
                    "tvd_point": metrics.tvd,
                    "grouped_peak_margin_point": metrics.grouped_peak_margin,
                    "normalized_grouped_peak_margin_point": (
                        metrics.normalized_grouped_peak_margin
                    ),
                    "main_lobe_to_outside_ratio_point": (
                        metrics.main_lobe_to_outside_ratio
                    ),
                    "grouped_peak_mass_point": metrics.grouped_peak_mass,
                    "minimum_group_probability_point": (
                        metrics.minimum_group_probability
                    ),
                    "maximum_outside_group_probability_point": (
                        metrics.maximum_outside_group_probability
                    ),
                }
            )
        for method in ZNE_METHODS:
            diagnostic_row: dict[str, Any] = {
                "instance": config["instance"],
                "rate": float(rate),
                "method": method,
            }
            for name, values in zne_diagnostics[method].items():
                diagnostic_row[f"{name}_point"] = float(values[0])
            diagnostic_row["fit_failure_point"] = False
            diagnostics.append(diagnostic_row)
    return rows, diagnostics


def _bootstrap_rate(
    *,
    reset_probabilities: np.ndarray,
    raw_probabilities: np.ndarray,
    shots: int,
    raw_shots: int | None = None,
    reset_baseline_probabilities: np.ndarray | None = None,
    reset_baseline_shots: int | None = None,
    reset_extension_probabilities: np.ndarray | None = None,
    reset_extension_shots: int = 0,
    lambda_one_index: int = 0,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
    replicates: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if shots <= 0:
        raise ValueError("shots must be positive")
    raw_draw_shots = shots if raw_shots is None else int(raw_shots)
    reset_draw_shots = (
        shots if reset_baseline_shots is None else int(reset_baseline_shots)
    )
    if raw_draw_shots <= 0 or reset_draw_shots <= 0:
        raise ValueError("baseline shot counts must be positive")
    if not 0 <= lambda_one_index < int(reset_probabilities.shape[0]):
        raise ValueError("lambda_one_index is outside the reset scale grid")
    if reset_extension_probabilities is not None:
        if reset_extension_shots <= 0:
            raise ValueError(
                "reset_extension_shots must be positive when an extension is supplied"
            )
        if reset_draw_shots != shots + int(reset_extension_shots):
            raise ValueError(
                "reset baseline shots must equal ZNE lambda=1 shots plus extension shots"
            )
        if reset_baseline_probabilities is not None:
            raise ValueError(
                "pass either reset_baseline_probabilities or a paired extension, not both"
            )
    elif reset_extension_shots != 0:
        raise ValueError(
            "reset_extension_shots requires reset_extension_probabilities"
        )
    elif reset_baseline_probabilities is None and reset_draw_shots != shots:
        raise ValueError(
            "a distinct reset baseline distribution is required when its shot count differs"
        )

    raw_seed, reset_seed = seed_sequence.spawn(2)
    raw_rng = np.random.default_rng(raw_seed)
    reset_rng = np.random.default_rng(reset_seed)
    reset_baseline_rng = np.random.default_rng(reset_seed.spawn(1)[0])

    sampled = {
        method: {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in BOOTSTRAP_METRICS
        }
        for method in METHODS
    }
    diagnostics = {
        method: {
            name: np.empty(replicates, dtype=np.float64)
            for name in (
                (
                    "normalization_error_before_projection",
                    "negative_mass_before_projection",
                    "negative_entry_count_before_projection",
                    "negative_entry_fraction_before_projection",
                    "l2_projection_correction",
                    "projection_active",
                    "selected_index",
                    "selected_intercept_l1_norm",
                    "selected_intercept_l2_norm_squared",
                    "selected_normal_condition_number",
                    "runner_up_score_gap",
                    "near_tie_candidate_count",
                    "penalized_weighted_residual",
                    "selected_gamma",
                    "grid_lower_boundary",
                    "grid_upper_boundary",
                    "grid_any_boundary",
                )
                if method == "single_exponential_zne"
                else (
                    "normalization_error_before_projection",
                    "negative_mass_before_projection",
                    "negative_entry_count_before_projection",
                    "negative_entry_fraction_before_projection",
                    "l2_projection_correction",
                    "projection_active",
                    "selected_index",
                    "selected_intercept_l1_norm",
                    "selected_intercept_l2_norm_squared",
                    "selected_normal_condition_number",
                    "selected_minimum_rate_separation",
                    "runner_up_score_gap",
                    "near_tie_candidate_count",
                    "penalized_weighted_residual",
                    "selected_gamma_1",
                    "selected_gamma_2",
                    "grid_lower_boundary",
                    "grid_upper_boundary",
                    "grid_any_boundary",
                )
                if method == "dual_exponential_zne"
                else (
                    "normalization_error_before_projection",
                    "negative_mass_before_projection",
                    "negative_entry_count_before_projection",
                    "negative_entry_fraction_before_projection",
                    "l2_projection_correction",
                    "projection_active",
                )
            )
        }
        for method in ZNE_METHODS
    }

    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        raw_batch = raw_rng.multinomial(
            raw_draw_shots, raw_probabilities, size=size
        ).astype(np.float64)
        raw_batch /= raw_draw_shots

        reset_batch = np.stack(
            [
                reset_rng.multinomial(shots, distribution, size=size)
                for distribution in reset_probabilities
            ],
            axis=1,
        ).astype(np.float64)
        reset_batch /= shots

        # The original lambda=1 block is shared between ZNE and the high-shot
        # Reset baseline.  When extra baseline shots are available, only the
        # independent extension is resampled separately before the two blocks
        # are combined.  This preserves the actual overlap between estimators.
        reset_lambda_one = reset_batch[:, lambda_one_index, :]
        if reset_extension_probabilities is not None:
            extension_counts = reset_baseline_rng.multinomial(
                int(reset_extension_shots),
                reset_extension_probabilities,
                size=size,
            ).astype(np.float64)
            reset_baseline_batch = (
                reset_lambda_one * shots + extension_counts
            ) / reset_draw_shots
        elif reset_baseline_probabilities is not None:
            reset_baseline_batch = reset_baseline_rng.multinomial(
                reset_draw_shots,
                reset_baseline_probabilities,
                size=size,
            ).astype(np.float64)
            reset_baseline_batch /= reset_draw_shots
        else:
            reset_baseline_batch = reset_lambda_one
        zne_batches, zne_diagnostics = _fit_zne_batch(
            reset_batch,
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
        )

        method_batches = {
            "raw_no_reset": raw_batch,
            "reset_lambda_1": reset_baseline_batch,
            **zne_batches,
        }
        for method, values in method_batches.items():
            metrics = _row_metrics(
                values, target=target, support=support, groups=groups
            )
            for metric in BOOTSTRAP_METRICS:
                sampled[method][metric][start:stop] = metrics[metric]

        for method, method_diagnostics in zne_diagnostics.items():
            for name, values in method_diagnostics.items():
                diagnostics[method][name][start:stop] = values
    return sampled, diagnostics


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float, float]:
    tail = (1.0 - confidence) / 2.0
    low, median, high = np.quantile(values, [tail, 0.5, 1.0 - tail])
    return float(low), float(median), float(high)


def analyse_instance(
    config_path: str | Path,
    *,
    replicates: int,
    seed: int,
    confidence: float,
    batch_size: int,
    weight_exponent: float = 2.0,
    overwrite_counts: bool = False,
) -> dict[str, Any]:
    config, resolved_config = load_config(config_path)
    count_path = build_processed_counts(resolved_config, overwrite=overwrite_counts)
    data = load_processed_counts(count_path)
    rates = np.asarray(data["rates"], dtype=np.float64)
    lambdas = np.asarray(data["lambdas"], dtype=np.float64)
    reset_probabilities, reset_shots = _normalize_counts(data["reset_counts"])
    raw_probabilities, raw_shots = _normalize_counts(data["no_reset_counts"])
    if reset_shots != raw_shots:
        raise ValueError("reset and no-reset shot counts differ")

    fixed_coefficients, exponential_families = _build_estimator_suite(
        lambdas,
        weight_exponent=weight_exponent,
    )
    rows, point_diagnostics = _point_rows(
        config=config,
        rates=rates,
        reset_probabilities=reset_probabilities,
        raw_probabilities=raw_probabilities,
        fixed_coefficients=fixed_coefficients,
        exponential_families=exponential_families,
    )
    row_lookup = {(row["rate"], row["method"]): row for row in rows}
    diagnostic_lookup = {
        (row["rate"], row["method"]): row for row in point_diagnostics
    }
    comparisons: list[dict[str, Any]] = []

    measurement_bits = int(config["measurement_bits"])
    order = int(config["order"])
    target = ideal_distribution(measurement_bits, order)
    groups = main_lobe_groups(measurement_bits, order)
    instance_code = zlib.crc32(config["instance"].encode("utf-8"))

    for rate_i, rate in enumerate(rates):
        print(
            f"[{config['instance']}] bootstrap {rate_i + 1}/{rates.size}: "
            f"p={rate:g}, B={replicates}",
            flush=True,
        )
        rate_seed = np.random.SeedSequence([seed, instance_code, rate_i])
        sampled, sampled_diagnostics = _bootstrap_rate(
            reset_probabilities=reset_probabilities[rate_i],
            raw_probabilities=raw_probabilities[rate_i],
            shots=reset_shots,
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
            replicates=replicates,
            batch_size=batch_size,
            seed_sequence=rate_seed,
            target=target,
            groups=groups,
        )
        for method in METHODS:
            row = row_lookup[(float(rate), method)]
            for metric in BOOTSTRAP_METRICS:
                low, median, high = _interval(sampled[method][metric], confidence)
                row[f"{metric}_bootstrap_median"] = median
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            row["grouped_peak_margin_positive_frequency"] = float(
                np.mean(sampled[method]["grouped_peak_margin"] > 0.0)
            )
            row["shots"] = reset_shots
            row["bootstrap_replicates"] = replicates
            row["confidence_level"] = confidence
            row["bootstrap_seed"] = seed
            row["conditional_on_fixed_folding_pattern"] = True

        comparison_specs = [
            ("reset_minus_raw", "reset_lambda_1", "raw_no_reset"),
            *[
                (f"{method}_minus_reset", method, "reset_lambda_1")
                for method in ZNE_METHODS
            ],
            *[
                (f"{method}_minus_linear_zne", method, "linear_zne")
                for method in ZNE_METHODS[1:]
            ],
        ]
        for comparison, improved_method, reference_method in comparison_specs:
            improved_row = row_lookup[(float(rate), improved_method)]
            reference_row = row_lookup[(float(rate), reference_method)]
            margin_samples = (
                sampled[improved_method]["grouped_peak_margin"]
                - sampled[reference_method]["grouped_peak_margin"]
            )
            tvd_reduction_samples = (
                sampled[reference_method]["tvd"] - sampled[improved_method]["tvd"]
            )
            margin_low, margin_median, margin_high = _interval(
                margin_samples, confidence
            )
            tvd_low, tvd_median, tvd_high = _interval(
                tvd_reduction_samples, confidence
            )
            comparisons.append(
                {
                    "instance": config["instance"],
                    "rate": float(rate),
                    "comparison": comparison,
                    "improved_method": improved_method,
                    "reference_method": reference_method,
                    "grouped_peak_margin_improvement_point": (
                        improved_row["grouped_peak_margin_point"]
                        - reference_row["grouped_peak_margin_point"]
                    ),
                    "grouped_peak_margin_improvement_bootstrap_median": margin_median,
                    "grouped_peak_margin_improvement_ci_low": margin_low,
                    "grouped_peak_margin_improvement_ci_high": margin_high,
                    "grouped_peak_margin_improvement_positive_frequency": float(
                        np.mean(margin_samples > 0.0)
                    ),
                    "tvd_reduction_point": (
                        reference_row["tvd_point"] - improved_row["tvd_point"]
                    ),
                    "tvd_reduction_bootstrap_median": tvd_median,
                    "tvd_reduction_ci_low": tvd_low,
                    "tvd_reduction_ci_high": tvd_high,
                    "tvd_reduction_probability": float(
                        np.mean(tvd_reduction_samples > 0.0)
                    ),
                    "shots": reset_shots,
                    "bootstrap_replicates": replicates,
                    "confidence_level": confidence,
                    "bootstrap_seed": seed,
                    "conditional_on_fixed_folding_pattern": True,
                }
            )

        for method in ZNE_METHODS:
            diagnostic_row = diagnostic_lookup[(float(rate), method)]
            diagnostic_row["fit_failure_bootstrap_probability"] = 0.0
            for name, values in sampled_diagnostics[method].items():
                low, median, high = _interval(values, confidence)
                diagnostic_row[f"{name}_bootstrap_median"] = median
                diagnostic_row[f"{name}_ci_low"] = low
                diagnostic_row[f"{name}_ci_high"] = high
                if name.startswith("grid_"):
                    diagnostic_row[f"{name}_bootstrap_probability"] = float(
                        np.mean(values > 0.5)
                    )

    return {
        "instance": config["instance"],
        "config": str(resolved_config.relative_to(PROJECT_ROOT)),
        "processed_counts": str(count_path.relative_to(PROJECT_ROOT)),
        "rates": rates.tolist(),
        "lambdas": lambdas.tolist(),
        "wls_weight_exponent": float(weight_exponent),
        "wls_objective_weights": objective_weights(
            lambdas, exponent=weight_exponent
        ).tolist(),
        "wls_intercept_coefficients": {
            method: coefficients.tolist()
            for method, coefficients in fixed_coefficients.items()
        },
        "model_specification": {
            "linear_zne": {
                "form": "a + b*lambda",
                "ridge": 0.0,
            },
            "quadratic_zne": {
                "form": "a + b*lambda + c*lambda^2",
                "ridge": DEFAULT_RIDGE,
            },
            "single_exponential_zne": {
                "form": "A + B_z*exp(-gamma*lambda)",
                "uniform_floor_A": 1.0 / (1 << measurement_bits),
                "shared_rate_grid": SINGLE_EXPONENTIAL_GRID.tolist(),
                "candidate_count": exponential_families[
                    "single_exponential_zne"
                ].candidate_count,
                "ridge": DEFAULT_RIDGE,
            },
            "dual_exponential_zne": {
                "form": (
                    "A + B_z*exp(-gamma_1*lambda) "
                    "+ C_z*exp(-gamma_2*lambda)"
                ),
                "uniform_floor_A": 1.0 / (1 << measurement_bits),
                "shared_ordered_rate_grid": DUAL_EXPONENTIAL_GRID.tolist(),
                "constraint": "gamma_2 > gamma_1",
                "candidate_count": exponential_families[
                    "dual_exponential_zne"
                ].candidate_count,
                "ridge": DEFAULT_RIDGE,
            },
            "rate_selection": (
                "one shared rate or ordered rate pair minimizes the summed "
                "penalized weighted residual over all outcomes and is reselected "
                "inside every bootstrap replicate"
            ),
        },
        "rows": rows,
        "comparisons": comparisons,
        "projection_diagnostics": point_diagnostics,
        "raw_metadata": data["metadata"],
    }


def _strict_json_value(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def _plot_summary(
    instances: list[dict[str, Any]],
    output_dir: Path,
    *,
    methods: tuple[str, ...],
    basename: str,
) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "raw_no_reset": "#777777",
        "reset_lambda_1": "#3274A1",
        "linear_zne": "#D64F4F",
        "quadratic_zne": "#6A3D9A",
        "single_exponential_zne": "#E17C05",
        "dual_exponential_zne": "#138A72",
    }
    markers = {
        "raw_no_reset": "o",
        "reset_lambda_1": "s",
        "linear_zne": "^",
        "quadratic_zne": "D",
        "single_exponential_zne": "v",
        "dual_exponential_zne": "P",
    }
    metrics = (
        ("grouped_peak_margin", "Main-lobe group margin", True),
        ("tvd", "Total variation distance", False),
    )
    figure, axes = plt.subplots(
        len(instances),
        2,
        figsize=(7.2, (2.85 if len(methods) > 4 else 2.65) * len(instances)),
        squeeze=False,
        sharex=True,
    )
    for row_i, instance in enumerate(instances):
        rates = np.asarray(instance["rates"], dtype=float)
        rows = instance["rows"]
        for col_i, (metric, label, zero_line) in enumerate(metrics):
            axis = axes[row_i, col_i]
            for method in methods:
                selected = sorted(
                    (row for row in rows if row["method"] == method),
                    key=lambda row: row["rate"],
                )
                central = np.asarray(
                    [row[f"{metric}_bootstrap_median"] for row in selected]
                )
                low = np.asarray([row[f"{metric}_ci_low"] for row in selected])
                high = np.asarray([row[f"{metric}_ci_high"] for row in selected])
                axis.plot(
                    rates,
                    central,
                    color=colors[method],
                    marker=markers[method],
                    markersize=3.5,
                    linewidth=1.4,
                    label=METHOD_LABELS[method],
                )
                axis.fill_between(
                    rates,
                    low,
                    high,
                    color=colors[method],
                    alpha=0.10 if len(methods) > 4 else 0.17,
                )
            if zero_line:
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_xscale("log")
            axis.grid(alpha=0.23, linewidth=0.5)
            axis.set_ylabel(label)
            axis.set_title(
                rf"$N={instance['raw_metadata']['N']},\ a={instance['raw_metadata']['a']},"
                rf"\ r={instance['raw_metadata']['order']}$"
            )
            if row_i == len(instances) - 1:
                axis.set_xlabel("Two-qubit Pauli error rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3 if len(methods) > 4 else len(methods),
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    top = 0.89 if len(methods) > 4 else 0.95
    figure.tight_layout(rect=(0.0, 0.0, 1.0, top))
    figure.savefig(output_dir / f"{basename}.pdf", bbox_inches="tight")
    figure.savefig(
        output_dir / f"{basename}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--weight-exponent",
        type=float,
        default=2.0,
        help="WLS exponent q in weights proportional to lambda**(-q)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "bootstrap",
    )
    parser.add_argument("--overwrite-counts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must lie strictly between 0 and 1")
    if not np.isfinite(args.weight_exponent):
        raise SystemExit("--weight-exponent must be finite")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = [
        analyse_instance(
            config,
            replicates=args.replicates,
            seed=args.seed,
            confidence=args.confidence,
            batch_size=args.batch_size,
            weight_exponent=args.weight_exponent,
            overwrite_counts=args.overwrite_counts,
        )
        for config in args.configs
    ]
    result = {
        "schema_version": 2,
        "analysis": "conditional multinomial shot bootstrap for four ZNE models",
        "resampling_statement": (
            "Each saved lambda distribution is resampled independently; the "
            "lambda=1 reset draw is shared by Reset and all four ZNE methods, "
            "while the no-reset baseline is resampled separately. Exponential "
            "decay rates are reselected within every replicate."
        ),
        "inference_scope": (
            "Finite-shot channel-and-measurement randomness conditional on the "
            "single saved folding realization per lambda and QPE round."
        ),
        "outcome_space": "all 16384 outcomes; no frequency filtering",
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "confidence_level": args.confidence,
        "wls_weight_exponent": args.weight_exponent,
        "instances": instances,
    }
    with (output_dir / "bootstrap_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_strict_json_value(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    all_rows = [row for instance in instances for row in instance["rows"]]
    all_diagnostics = [
        row for instance in instances for row in instance["projection_diagnostics"]
    ]
    all_comparisons = [
        row for instance in instances for row in instance["comparisons"]
    ]
    _write_csv(output_dir / "bootstrap_summary.csv", all_rows)
    _write_csv(output_dir / "bootstrap_comparisons.csv", all_comparisons)
    _write_csv(output_dir / "projection_diagnostics.csv", all_diagnostics)
    _plot_summary(
        instances,
        output_dir,
        methods=METHODS,
        basename="conditional_bootstrap_metrics",
    )
    _plot_summary(
        instances,
        output_dir,
        methods=ZNE_METHODS,
        basename="conditional_bootstrap_model_sensitivity",
    )
    print(f"Wrote analysis outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
