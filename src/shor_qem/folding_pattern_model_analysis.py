"""Bootstrap all four ZNE models for shot-wise random-folding data."""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap_analysis import (
    BOOTSTRAP_METRICS,
    METHODS,
    METHOD_LABELS,
    ZNE_METHODS,
    _bootstrap_rate,
    _build_estimator_suite,
    _fit_zne_batch,
    _interval,
)
from .data import load_config
from .folding_pattern_analysis import (
    _assemble_shotwise_grid,
    _load_no_reset_counts,
    _load_shotwise_records,
)
from .metrics import distribution_metrics, ideal_distribution, main_lobe_groups


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "raw_no_reset": "#777777",
    "reset_lambda_1": "#3274A1",
    "linear_zne": "#D64F4F",
    "quadratic_zne": "#6A3D9A",
    "single_exponential_zne": "#E17C05",
    "dual_exponential_zne": "#138A72",
}
MARKERS = {
    "raw_no_reset": "o",
    "reset_lambda_1": "s",
    "linear_zne": "^",
    "quadratic_zne": "D",
    "single_exponential_zne": "v",
    "dual_exponential_zne": "P",
}


def _counts_from_outcomes(outcomes: np.ndarray, outcome_count: int) -> np.ndarray:
    values = np.asarray(outcomes, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= outcome_count):
        raise ValueError("baseline archive contains invalid outcomes")
    return np.bincount(values, minlength=outcome_count).astype(np.int64)


def _load_cost_matched_baselines(
    *,
    baseline_root: Path,
    config: dict[str, Any],
    rates: np.ndarray,
    reset_zne_counts: np.ndarray,
    raw_initial_counts: np.ndarray,
    lambda_one_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load 160k Raw/Reset baselines and preserve their shared 20k block.

    The merged follow-up archives concatenate the original lambda=1 shots
    before the independent extension.  The bootstrap can therefore share the
    resampled original Reset block with ZNE and resample only the extension
    separately, matching the actual overlap between the two estimators.
    """

    instance = str(config["instance"])
    instance_root = baseline_root.resolve() / instance
    outcome_count = 1 << int(config["measurement_bits"])
    zne_shots = int(reset_zne_counts[0, lambda_one_index].sum(dtype=np.int64))
    raw_full = np.empty((rates.size, outcome_count), dtype=np.int64)
    reset_full = np.empty_like(raw_full)
    reset_extension = np.empty_like(raw_full)
    baseline_shots: int | None = None
    extension_shots: int | None = None

    for mode, destination, initial_counts in (
        ("raw", raw_full, raw_initial_counts),
        ("reset", reset_full, reset_zne_counts[:, lambda_one_index]),
    ):
        directory = instance_root / mode
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        records: list[tuple[float, Path]] = []
        for path in sorted(directory.glob("lambda1_p*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                records.append((float(archive["rate"]), path))

        for rate_i, rate in enumerate(rates):
            matches = [
                path
                for saved_rate, path in records
                if np.isclose(saved_rate, rate, rtol=0.0, atol=5.0e-15)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{instance}/{mode}: expected one high-shot baseline for rate={rate}"
                )
            path = matches[0]
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "instance",
                    "N",
                    "a",
                    "order",
                    "mode",
                    "rate",
                    "lambda",
                    "shots",
                    "measurement_bits",
                    "outcomes",
                    "counts",
                }
                missing = sorted(required.difference(archive.files))
                if missing:
                    raise ValueError(f"{path}: missing arrays {missing}")
                if str(archive["instance"]) != instance or str(archive["mode"]) != mode:
                    raise ValueError(f"{path}: baseline identity differs from request")
                if (
                    int(archive["N"]) != int(config["N"])
                    or int(archive["a"]) != int(config["a"])
                    or int(archive["order"]) != int(config["order"])
                    or int(archive["measurement_bits"])
                    != int(config["measurement_bits"])
                ):
                    raise ValueError(f"{path}: circuit metadata differs from config")
                if not np.isclose(float(archive["lambda"]), 1.0):
                    raise ValueError(f"{path}: high-shot baseline must use lambda=1")
                outcomes = np.asarray(archive["outcomes"], dtype=np.int64)
                shots = int(archive["shots"])
                counts = np.asarray(archive["counts"], dtype=np.int64)
                if outcomes.size != shots or counts.shape != (outcome_count,):
                    raise ValueError(f"{path}: invalid high-shot baseline shape")
                if not np.array_equal(counts, _counts_from_outcomes(outcomes, outcome_count)):
                    raise ValueError(f"{path}: counts do not match saved outcomes")
                first_counts = _counts_from_outcomes(outcomes[:zne_shots], outcome_count)
                if not np.array_equal(first_counts, np.asarray(initial_counts[rate_i])):
                    raise ValueError(
                        f"{path}: first {zne_shots} outcomes differ from the ZNE source block"
                    )
                if baseline_shots is None:
                    baseline_shots = shots
                    extension_shots = shots - zne_shots
                elif shots != baseline_shots:
                    raise ValueError("high-shot baseline totals are inconsistent")
                if shots <= zne_shots:
                    raise ValueError("high-shot baseline has no independent extension")
                destination[rate_i] = counts
                if mode == "reset":
                    reset_extension[rate_i] = _counts_from_outcomes(
                        outcomes[zne_shots:], outcome_count
                    )

    assert baseline_shots is not None and extension_shots is not None
    if not np.all(reset_extension.sum(axis=1, dtype=np.int64) == extension_shots):
        raise ValueError("reset extension shot totals are inconsistent")
    return raw_full, reset_full, reset_extension, baseline_shots, extension_shots


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


def _summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    low, median, high = _interval(values, confidence)
    return {
        "bootstrap_median": median,
        "ci_low": low,
        "ci_high": high,
        "sd": float(np.std(values, ddof=1)),
    }


def analyse(
    *,
    config_path: Path,
    run_dir: Path,
    no_reset_run_dir: Path,
    baseline_root: Path | None,
    rates: np.ndarray,
    lambdas: np.ndarray | None,
    replicates: int,
    batch_size: int,
    confidence: float,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    config, resolved_config = load_config(config_path)
    configured_lambdas = np.asarray(config["lambdas"], dtype=np.float64)
    if lambdas is None:
        lambdas = configured_lambdas
    else:
        lambdas = np.asarray(lambdas, dtype=np.float64)
        if (
            lambdas.size == 0
            or np.unique(lambdas).size != lambdas.size
            or any(
                not np.any(np.isclose(value, configured_lambdas))
                for value in lambdas
            )
        ):
            raise ValueError(
                "lambdas must be a distinct nonempty subset of the config grid"
            )
    records = _load_shotwise_records(run_dir, config)
    reset_counts, sources = _assemble_shotwise_grid(
        historical=None,
        records=records,
        rates=rates,
        lambdas=lambdas,
    )
    if np.any(sources != "shotwise"):
        raise ValueError("all model-bootstrap distributions must be fresh shotwise data")
    no_reset_counts = _load_no_reset_counts(no_reset_run_dir, config, rates)

    reset_totals = reset_counts.sum(axis=-1, dtype=np.int64)
    no_reset_totals = no_reset_counts.sum(axis=-1, dtype=np.int64)
    all_totals = np.concatenate([reset_totals.ravel(), no_reset_totals.ravel()])
    if np.unique(all_totals).size != 1:
        raise ValueError(f"all distributions must use equal shots: {all_totals}")
    shots = int(all_totals[0])
    lambda_one_i = int(np.argmin(np.abs(lambdas - 1.0)))
    if not np.isclose(lambdas[lambda_one_i], 1.0, rtol=0.0, atol=5.0e-15):
        raise ValueError("the selected ZNE scale grid must contain lambda=1")

    reset_extension_counts: np.ndarray | None = None
    reset_extension_shots = 0
    if baseline_root is not None:
        (
            baseline_raw_counts,
            baseline_reset_counts,
            reset_extension_counts,
            baseline_shots,
            reset_extension_shots,
        ) = _load_cost_matched_baselines(
            baseline_root=baseline_root,
            config=config,
            rates=rates,
            reset_zne_counts=reset_counts,
            raw_initial_counts=no_reset_counts,
            lambda_one_index=lambda_one_i,
        )
    else:
        baseline_raw_counts = no_reset_counts.astype(np.int64)
        baseline_reset_counts = reset_counts[:, lambda_one_i].astype(np.int64)
        baseline_shots = shots

    reset_probabilities = reset_counts.astype(np.float64)
    reset_probabilities /= reset_totals[..., None]
    no_reset_probabilities = baseline_raw_counts.astype(np.float64)
    no_reset_probabilities /= baseline_raw_counts.sum(axis=-1, keepdims=True)
    reset_baseline_probabilities = baseline_reset_counts.astype(np.float64)
    reset_baseline_probabilities /= baseline_reset_counts.sum(
        axis=-1, keepdims=True
    )
    reset_extension_probabilities = None
    if reset_extension_counts is not None:
        reset_extension_probabilities = reset_extension_counts.astype(np.float64)
        reset_extension_probabilities /= reset_extension_counts.sum(
            axis=-1, keepdims=True
        )

    fixed_coefficients, exponential_families = _build_estimator_suite(lambdas)
    point_zne, point_diagnostics = _fit_zne_batch(
        reset_probabilities,
        fixed_coefficients=fixed_coefficients,
        exponential_families=exponential_families,
    )
    point_distributions = {
        "raw_no_reset": no_reset_probabilities,
        "reset_lambda_1": reset_baseline_probabilities,
        **point_zne,
    }
    method_total_shots = {
        "raw_no_reset": baseline_shots,
        "reset_lambda_1": baseline_shots,
        **{method: shots * int(lambdas.size) for method in ZNE_METHODS},
    }
    method_shots_per_scale = {
        "raw_no_reset": baseline_shots,
        "reset_lambda_1": baseline_shots,
        **{method: shots for method in ZNE_METHODS},
    }

    measurement_bits = int(config["measurement_bits"])
    order = int(config["order"])
    target = ideal_distribution(measurement_bits, order)
    groups = main_lobe_groups(measurement_bits, order)
    ideal_group_margin = distribution_metrics(
        target, measurement_bits, order
    ).grouped_peak_margin
    instance_code = zlib.crc32(str(config["instance"]).encode("utf-8"))

    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    draw_array = np.empty(
        (rates.size, len(METHODS), len(BOOTSTRAP_METRICS), replicates),
        dtype=np.float64,
    )

    for rate_i, rate in enumerate(rates):
        print(
            f"[{config['instance']}] model bootstrap {rate_i + 1}/{rates.size}: "
            f"p={rate:g}, B={replicates}",
            flush=True,
        )
        sampled, sampled_diagnostics = _bootstrap_rate(
            reset_probabilities=reset_probabilities[rate_i],
            raw_probabilities=no_reset_probabilities[rate_i],
            shots=shots,
            raw_shots=baseline_shots,
            reset_baseline_shots=baseline_shots,
            reset_extension_probabilities=(
                None
                if reset_extension_probabilities is None
                else reset_extension_probabilities[rate_i]
            ),
            reset_extension_shots=reset_extension_shots,
            lambda_one_index=lambda_one_i,
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
            replicates=replicates,
            batch_size=batch_size,
            seed_sequence=np.random.SeedSequence(
                [seed, instance_code, rate_i]
            ),
            target=target,
            groups=groups,
        )

        point_metrics: dict[str, Any] = {}
        for method_i, method in enumerate(METHODS):
            metric = distribution_metrics(
                point_distributions[method][rate_i],
                measurement_bits,
                order,
            )
            point_metrics[method] = metric
            row: dict[str, Any] = {
                "instance": config["instance"],
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": order,
                "rate": float(rate),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "shots": method_total_shots[method],
                "shots_per_scale": method_shots_per_scale[method],
                "number_of_scales": (
                    int(lambdas.size) if method in ZNE_METHODS else 1
                ),
                "bootstrap_replicates": replicates,
                "confidence_level": confidence,
                "tvd_point": metric.tvd,
                "grouped_peak_margin_point": metric.grouped_peak_margin,
                "normalized_grouped_peak_margin_point": (
                    metric.normalized_grouped_peak_margin
                ),
            }
            for metric_i, metric_name in enumerate(BOOTSTRAP_METRICS):
                values = sampled[method][metric_name]
                draw_array[rate_i, method_i, metric_i] = values
                for statistic, value in _summary(values, confidence).items():
                    row[f"{metric_name}_{statistic}"] = value
                if metric_name == "grouped_peak_margin":
                    for statistic, value in _summary(
                        values / ideal_group_margin, confidence
                    ).items():
                        row[
                            f"normalized_grouped_peak_margin_{statistic}"
                        ] = value
            row["grouped_peak_margin_positive_frequency"] = float(
                np.mean(sampled[method]["grouped_peak_margin"] > 0.0)
            )
            rows.append(row)

        for method in ZNE_METHODS:
            for reference_method, comparison_label in (
                ("reset_lambda_1", "vs_reset"),
                ("raw_no_reset", "vs_no_reset"),
            ):
                tvd_reduction = (
                    sampled[reference_method]["tvd"]
                    - sampled[method]["tvd"]
                )
                margin_improvement = (
                    sampled[method]["grouped_peak_margin"]
                    - sampled[reference_method]["grouped_peak_margin"]
                )
                comparison: dict[str, Any] = {
                    "instance": config["instance"],
                    "rate": float(rate),
                    "method": method,
                    "reference_method": reference_method,
                    "comparison": comparison_label,
                    "tvd_reduction_point": (
                        point_metrics[reference_method].tvd
                        - point_metrics[method].tvd
                    ),
                    "grouped_peak_margin_improvement_point": (
                        point_metrics[method].grouped_peak_margin
                        - point_metrics[reference_method].grouped_peak_margin
                    ),
                }
                for prefix, values in (
                    ("tvd_reduction", tvd_reduction),
                    ("grouped_peak_margin_improvement", margin_improvement),
                ):
                    for statistic, value in _summary(
                        values, confidence
                    ).items():
                        comparison[f"{prefix}_{statistic}"] = value
                    comparison[f"{prefix}_positive_frequency"] = float(
                        np.mean(values > 0.0)
                    )
                    if prefix == "grouped_peak_margin_improvement":
                        for statistic, value in _summary(
                            values / ideal_group_margin, confidence
                        ).items():
                            comparison[
                                f"normalized_grouped_peak_margin_improvement_{statistic}"
                            ] = value
                comparisons.append(comparison)

        for method in ZNE_METHODS:
            diagnostic: dict[str, Any] = {
                "instance": config["instance"],
                "rate": float(rate),
                "method": method,
            }
            for name, values in point_diagnostics[method].items():
                diagnostic[f"{name}_point"] = float(values[rate_i])
            for name, values in sampled_diagnostics[method].items():
                for statistic, value in _summary(
                    values, confidence
                ).items():
                    diagnostic[f"{name}_{statistic}"] = value
                if name.startswith("grid_"):
                    diagnostic[f"{name}_positive_frequency"] = float(
                        np.mean(values > 0.5)
                    )
            diagnostics.append(diagnostic)

    arrays = {
        "rates": rates,
        "lambdas": lambdas,
        "methods": np.asarray(METHODS),
        "metrics": np.asarray(BOOTSTRAP_METRICS),
        "draws": draw_array,
        "ideal_grouped_peak_margin": np.asarray(ideal_group_margin),
        "method_total_shots": np.asarray(
            [method_total_shots[method] for method in METHODS], dtype=np.int64
        ),
        "method_shots_per_scale": np.asarray(
            [method_shots_per_scale[method] for method in METHODS], dtype=np.int64
        ),
    }
    metadata = {
        "schema_version": 1,
        "analysis": "all-model bootstrap for per-shot random folding",
        "config": str(resolved_config),
        "run_dir": str(run_dir),
        "no_reset_run_dir": str(no_reset_run_dir),
        "baseline_root": None if baseline_root is None else str(baseline_root),
        "rates": rates.tolist(),
        "lambdas": lambdas.tolist(),
        "zne_shots_per_scale": shots,
        "baseline_shots": baseline_shots,
        "method_total_shots": method_total_shots,
        "reset_baseline_extension_shots": reset_extension_shots,
        "bootstrap_replicates": replicates,
        "batch_size": batch_size,
        "confidence_level": confidence,
        "seed": seed,
        "ideal_target": "exact_finite_resolution_qpe_distribution",
        "main_lobe_groups": [group.tolist() for group in groups],
        "ideal_grouped_peak_margin": ideal_group_margin,
        "exponential_rate_selection": (
            "repeated independently inside every bootstrap replicate"
        ),
        "folding_randomness": (
            "each saved shot used an independently seeded folding pattern; "
            "empirical shot resampling therefore includes its shot-level variation"
        ),
    }
    return rows, comparisons, diagnostics, arrays, metadata


def _plot(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    confidence: float,
) -> None:
    import matplotlib.pyplot as plt

    rates_percent = 100.0 * arrays["rates"]
    methods = [str(method) for method in arrays["methods"]]
    metrics = [str(metric) for metric in arrays["metrics"]]
    tvd_i = metrics.index("tvd")
    reset_i = methods.index("reset_lambda_1")
    draws = arrays["draws"]

    row_lookup = {
        (float(row["rate"]), str(row["method"])): row for row in rows
    }
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.3),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    for method_i, method in enumerate(methods):
        medians = np.asarray(
            [
                row_lookup[(float(rate), method)]["tvd_bootstrap_median"]
                for rate in arrays["rates"]
            ]
        )
        lows = np.asarray(
            [
                row_lookup[(float(rate), method)]["tvd_ci_low"]
                for rate in arrays["rates"]
            ]
        )
        highs = np.asarray(
            [
                row_lookup[(float(rate), method)]["tvd_ci_high"]
                for rate in arrays["rates"]
            ]
        )
        axes[0].plot(
            rates_percent,
            medians,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.55,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
        axes[0].fill_between(
            rates_percent, lows, highs, color=COLORS[method], alpha=0.12
        )
        if method in ZNE_METHODS:
            improvements = draws[:, reset_i, tvd_i] - draws[:, method_i, tvd_i]
            low, median, high = np.quantile(
                improvements,
                [(1.0 - confidence) / 2.0, 0.5, 1.0 - (1.0 - confidence) / 2.0],
                axis=1,
            )
            axes[1].plot(
                rates_percent,
                median,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.55,
                markersize=4.5,
                label=METHOD_LABELS[method],
            )
            axes[1].fill_between(
                rates_percent, low, high, color=COLORS[method], alpha=0.14
            )

    axes[0].set_ylabel("TVD to exact ideal QPE distribution")
    axes[0].set_ylim(0.0, 1.03)
    axes[0].legend(frameon=False, ncol=2, fontsize=8.5)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("TVD reduction vs reset (positive is better)")
    axes[1].set_xlabel("Physical two-qubit error rate (%)")
    axes[1].legend(frameon=False, ncol=2, fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].set_xticks(rates_percent)
    axes[1].set_xticklabels([f"{value:g}" for value in rates_percent])
    for axis in axes:
        axis.grid(alpha=0.23, linewidth=0.55)
    shots = int(rows[0]["shots"])
    N = int(rows[0]["N"])
    a = int(rows[0]["a"])
    order = int(rows[0]["order"])
    figure.suptitle(
        rf"$N={N},\ a={a},\ r={order}$: {shots:,}-shot model bootstrap",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(
        output_dir / "model_bootstrap_tvd.png",
        dpi=240,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "model_bootstrap_tvd.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    margin_i = metrics.index("grouped_peak_margin")
    ideal_group_margin = float(arrays["ideal_grouped_peak_margin"])
    margin_figure, margin_axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 0.85]},
    )
    margin_axis, probability_axis = margin_axes
    for method_i, method in enumerate(methods):
        normalized = draws[:, method_i, margin_i] / ideal_group_margin
        tail = (1.0 - confidence) / 2.0
        low, median, high = np.quantile(
            normalized, [tail, 0.5, 1.0 - tail], axis=1
        )
        margin_axis.plot(
            rates_percent,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.55,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
        margin_axis.fill_between(
            rates_percent, low, high, color=COLORS[method], alpha=0.13
        )
        probability_axis.plot(
            rates_percent,
            np.mean(normalized > 0.0, axis=1),
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.55,
            markersize=4.5,
            label=METHOD_LABELS[method],
        )
    margin_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    margin_axis.axhline(
        1.0, color="#777777", linestyle=":", linewidth=0.8, alpha=0.8
    )
    margin_axis.set_xscale("log")
    margin_axis.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    margin_axis.set_ylabel("Normalized main-lobe group margin")
    margin_axis.grid(alpha=0.23, linewidth=0.55)
    margin_axis.legend(frameon=False, ncol=2, fontsize=8.3)
    probability_axis.set_xscale("log")
    probability_axis.set_xticks(rates_percent)
    probability_axis.set_xticklabels([f"{value:g}" for value in rates_percent])
    probability_axis.set_ylim(-0.03, 1.03)
    probability_axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    probability_axis.set_ylabel(
        r"$\Pr^*(\Delta_{\mathrm{group}}>0)$"
    )
    probability_axis.set_xlabel("Physical two-qubit error rate (%)")
    probability_axis.grid(alpha=0.23, linewidth=0.55)
    margin_figure.suptitle(
        rf"$N={N},\ a={a},\ r={order}$: "
        f"{shots:,}-shot main-lobe group-margin bootstrap"
    )
    margin_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    margin_figure.savefig(
        output_dir / "model_bootstrap_grouped_margin.png",
        dpi=240,
        bbox_inches="tight",
    )
    margin_figure.savefig(
        output_dir / "model_bootstrap_grouped_margin.pdf",
        bbox_inches="tight",
    )
    plt.close(margin_figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "n21_a2_r6.json",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-reset-run-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help=(
            "Optional merged baseline_160k root. Raw and Reset use these "
            "cost-matched high-shot records while ZNE retains the saved "
            "per-scale shot count."
        ),
    )
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        help="Optional subset of the configured scale grid.",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replicates <= 1 or args.batch_size <= 0:
        raise ValueError("replicates must exceed one and batch size must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    rates = np.asarray(args.rates, dtype=np.float64)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, comparisons, diagnostics, arrays, metadata = analyse(
        config_path=args.config.resolve(),
        run_dir=args.run_dir.resolve(),
        no_reset_run_dir=args.no_reset_run_dir.resolve(),
        baseline_root=(
            None if args.baseline_root is None else args.baseline_root.resolve()
        ),
        rates=rates,
        lambdas=(
            None
            if args.lambdas is None
            else np.asarray(args.lambdas, dtype=np.float64)
        ),
        replicates=int(args.replicates),
        batch_size=int(args.batch_size),
        confidence=float(args.confidence),
        seed=int(args.seed),
    )
    _write_csv(output_dir / "model_bootstrap_summary.csv", rows)
    _write_csv(output_dir / "model_bootstrap_comparisons.csv", comparisons)
    _write_csv(output_dir / "model_bootstrap_diagnostics.csv", diagnostics)
    np.savez_compressed(output_dir / "model_bootstrap_draws.npz", **arrays)
    with (output_dir / "model_bootstrap_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    _plot(rows, arrays, output_dir, float(args.confidence))
    print(f"wrote model-bootstrap outputs to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
