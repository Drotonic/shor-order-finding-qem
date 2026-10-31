"""Bootstrap analysis for the configured shot-wise random-folding grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .data import load_config, processed_path, resolve_project_path, sha256_file
from .folding import (
    FOLDING_ARCHIVE_SCHEMA_VERSION,
    FOLDING_PATTERN_SCOPE,
    FOLDING_SEED_FIELDS,
)
from .metrics import (
    distribution_metrics,
    grouped_peak_statistics_rows,
    ideal_distribution,
    ideal_indices,
    main_lobe_groups,
    simplex_projection_rows,
    weighted_linear_intercept_coefficients,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "n119_a13_matched.json",
    PROJECT_ROOT / "configs" / "n119_a50_matched.json",
    PROJECT_ROOT / "configs" / "n119_a16_matched.json",
)
ENDPOINTS = (1.0, 3.0)


def _shared_rate_grid(config_paths: list[Path]) -> np.ndarray:
    grids = []
    for path in config_paths:
        config, _ = load_config(path)
        grids.append(tuple(map(float, config["rates"])))
    if not grids:
        raise ValueError("at least one configuration is required")
    if len(set(grids)) != 1:
        raise ValueError("configs use different rate grids; pass --rates")
    return np.asarray(grids[0], dtype=np.float64)


def _nearest_index(value: float, grid: np.ndarray, label: str) -> int:
    index = int(np.argmin(np.abs(grid - float(value))))
    if not np.isclose(grid[index], value, rtol=0.0, atol=5e-12):
        raise ValueError(f"unexpected {label}={value}; grid={grid.tolist()}")
    return index


def _load_historical(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = processed_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"processed historical counts not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {
            "rates": np.asarray(archive["rates"], dtype=np.float64),
            "lambdas": np.asarray(archive["lambdas"], dtype=np.float64),
            "reset_counts": np.asarray(archive["reset_counts"], dtype=np.int32),
        }


def _load_shotwise_records(
    run_dir: Path, config: dict[str, Any]
) -> dict[tuple[float, float], np.ndarray]:
    instance = str(config["instance"])
    cache_path = resolve_project_path(config["ops_cache"])
    expected_cache_hash = sha256_file(cache_path)
    expected_variant = str(config.get("circuit_variant", "historical_opt3"))
    expected_two_qubit_count = int(config.get("two_qubit_gate_count", -1))
    files = sorted((run_dir / instance / "shotwise").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no shot-wise folding archives for {instance}")
    records: dict[tuple[float, float], np.ndarray] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "instance",
                "N",
                "a",
                "schema_version",
                "folding_pattern_scope",
                "folding_seed_fields",
                "ops_cache_sha256",
                "circuit_variant",
                "two_qubit_gate_count",
                "rates",
                "lambdas",
                "counts",
                "shots",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"{path}: missing arrays {missing}")
            if str(archive["instance"]) != instance:
                raise ValueError(f"{path}: wrong instance")
            if int(archive["N"]) != int(config["N"]) or int(archive["a"]) != int(
                config["a"]
            ):
                raise ValueError(f"{path}: wrong circuit instance")
            if int(archive["schema_version"]) != FOLDING_ARCHIVE_SCHEMA_VERSION:
                raise ValueError(f"{path}: incompatible folding archive version")
            if str(archive["ops_cache_sha256"]) != expected_cache_hash:
                raise ValueError(f"{path}: operation-cache hash differs from config")
            if str(archive["circuit_variant"]) != expected_variant:
                raise ValueError(f"{path}: circuit variant differs from config")
            if int(archive["two_qubit_gate_count"]) != expected_two_qubit_count:
                raise ValueError(f"{path}: two-qubit gate count differs from config")
            if str(archive["folding_pattern_scope"]) != FOLDING_PATTERN_SCOPE:
                raise ValueError(
                    f"{path}: expected {FOLDING_PATTERN_SCOPE} folding scope"
                )
            if not np.array_equal(
                np.asarray(archive["folding_seed_fields"]),
                np.asarray(FOLDING_SEED_FIELDS),
            ):
                raise ValueError(f"{path}: incompatible folding seed hierarchy")
            rates = np.asarray(archive["rates"], dtype=np.float64)
            lambdas = np.asarray(archive["lambdas"], dtype=np.float64)
            counts = np.asarray(archive["counts"], dtype=np.int32)
            shots = int(archive["shots"])
            if counts.shape[:2] != (rates.size, lambdas.size):
                raise ValueError(f"{path}: invalid count shape {counts.shape}")
            if not np.all(counts.sum(axis=-1) == shots):
                raise ValueError(f"{path}: count totals differ from shots")
            for rate_i, rate in enumerate(rates):
                for lambda_i, lam in enumerate(lambdas):
                    key = (float(rate), float(lam))
                    if key in records:
                        raise ValueError(f"duplicate record {key} in {path}")
                    records[key] = counts[rate_i, lambda_i].copy()
    return records


def _load_no_reset_counts(
    run_dir: Path, config: dict[str, Any], rates: np.ndarray
) -> np.ndarray:
    records: dict[float, np.ndarray] = {}
    pattern = (
        f"safe_random_noareset_N{int(config['N'])}_a{int(config['a'])}_p*.pkl"
    )
    for path in sorted(run_dir.glob(pattern)):
        with path.open("rb") as handle:
            archive = pickle.load(handle)
        if not isinstance(archive, dict):
            raise ValueError(f"{path}: expected a dictionary archive")
        for key, bits in archive.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"{path}: invalid no-reset key {key!r}")
            rate, lam = map(float, key)
            if not np.isclose(lam, 1.0, rtol=0.0, atol=5e-12):
                raise ValueError(f"{path}: no-reset baseline must use lambda=1")
            bit_array = np.asarray(bits)
            measurement_bits = int(config["measurement_bits"])
            if (
                bit_array.ndim != 2
                or bit_array.shape[1] != measurement_bits
                or not np.all((bit_array == 0) | (bit_array == 1))
            ):
                raise ValueError(f"{path}: invalid measured-bit array")
            weights = np.left_shift(
                np.uint64(1), np.arange(measurement_bits, dtype=np.uint64)
            )
            outcomes = (bit_array.astype(np.uint64) @ weights).astype(np.int64)
            counts = np.bincount(
                outcomes, minlength=1 << measurement_bits
            ).astype(np.int32)
            if rate in records:
                raise ValueError(f"duplicate no-reset record for rate={rate}")
            records[rate] = counts
    if not records:
        raise FileNotFoundError(f"no no-reset archives found under {run_dir}")
    grid = np.empty((rates.size, 1 << int(config["measurement_bits"])), np.int32)
    for rate_i, rate in enumerate(rates):
        matches = [
            counts
            for saved_rate, counts in records.items()
            if np.isclose(saved_rate, rate, rtol=0.0, atol=5e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one no-reset record for rate={rate}")
        grid[rate_i] = matches[0]
    return grid


def _assemble_shotwise_grid(
    *,
    historical: dict[str, np.ndarray] | None,
    records: dict[tuple[float, float], np.ndarray],
    rates: np.ndarray,
    lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if historical is None:
        outcome_sizes = {int(np.asarray(counts).size) for counts in records.values()}
        if len(outcome_sizes) != 1:
            raise ValueError("shot-wise records use inconsistent outcome spaces")
        outcomes = outcome_sizes.pop()
    else:
        outcomes = int(historical["reset_counts"].shape[-1])
    grid = np.empty((rates.size, lambdas.size, outcomes), dtype=np.int32)
    sources = np.empty((rates.size, lambdas.size), dtype="<U10")
    for rate_i, rate in enumerate(rates):
        historical_rate_i = (
            _nearest_index(rate, historical["rates"], "rate")
            if historical is not None
            else None
        )
        for lambda_i, lam in enumerate(lambdas):
            key = (float(rate), float(lam))
            if key in records:
                grid[rate_i, lambda_i] = records[key]
                sources[rate_i, lambda_i] = "shotwise"
            elif historical is not None and any(
                np.isclose(lam, endpoint) for endpoint in ENDPOINTS
            ):
                historical_lambda_i = _nearest_index(
                    lam, historical["lambdas"], "lambda"
                )
                grid[rate_i, lambda_i] = historical["reset_counts"][
                    int(historical_rate_i), historical_lambda_i
                ]
                sources[rate_i, lambda_i] = "historical"
            else:
                raise ValueError(f"missing shot-wise record for rate={rate}, lambda={lam}")
    if np.any(grid.sum(axis=-1) <= 0):
        raise ValueError("every rate and lambda requires a positive shot count")
    return grid, sources


def _batch_metrics(
    probabilities: np.ndarray,
    *,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tvd = 0.5 * np.abs(probabilities - target[None, :]).sum(axis=1)
    if groups is None:
        if support is None:
            raise ValueError("support or groups is required")
        groups = tuple(
            np.asarray([index], dtype=np.int64)
            for index in np.asarray(support, dtype=np.int64)
        )
    margin = grouped_peak_statistics_rows(probabilities, groups)["margin"]
    return tvd, margin


def _bootstrap_linear_and_baseline_metrics(
    counts: np.ndarray,
    *,
    lambdas: np.ndarray,
    replicates: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shots = counts.sum(axis=1, dtype=np.int64)
    if np.any(shots <= 0):
        raise ValueError("each lambda requires a positive shot count")
    probabilities = counts.astype(np.float64) / shots[:, None]
    coefficients = weighted_linear_intercept_coefficients(lambdas)
    baseline_i = _nearest_index(1.0, lambdas, "baseline lambda")
    rng = np.random.default_rng(seed_sequence)
    linear_tvd_samples = np.empty(replicates, dtype=np.float64)
    linear_margin_samples = np.empty(replicates, dtype=np.float64)
    baseline_tvd_samples = np.empty(replicates, dtype=np.float64)
    baseline_margin_samples = np.empty(replicates, dtype=np.float64)

    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        sampled_lambdas = []
        for lambda_i, distribution in enumerate(probabilities):
            lambda_shots = int(shots[lambda_i])
            sample = rng.multinomial(
                lambda_shots, distribution, size=size
            ).astype(np.float64)
            sampled_lambdas.append(sample / lambda_shots)
        sampled = np.stack(sampled_lambdas, axis=1)
        unconstrained = np.einsum(
            "l,blo->bo", coefficients, sampled, optimize=True
        )
        projected = simplex_projection_rows(unconstrained)
        linear_tvd, linear_margin = _batch_metrics(
            projected, target=target, support=support, groups=groups
        )
        baseline_tvd, baseline_margin = _batch_metrics(
            sampled[:, baseline_i],
            target=target,
            support=support,
            groups=groups,
        )
        linear_tvd_samples[start:stop] = linear_tvd
        linear_margin_samples[start:stop] = linear_margin
        baseline_tvd_samples[start:stop] = baseline_tvd
        baseline_margin_samples[start:stop] = baseline_margin
    return (
        linear_tvd_samples,
        linear_margin_samples,
        baseline_tvd_samples,
        baseline_margin_samples,
    )


def _bootstrap_linear_metrics(
    counts: np.ndarray,
    *,
    lambdas: np.ndarray,
    replicates: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible linear-ZNE bootstrap view."""
    linear_tvd, linear_margin, _, _ = _bootstrap_linear_and_baseline_metrics(
        counts,
        lambdas=lambdas,
        replicates=replicates,
        batch_size=batch_size,
        seed_sequence=seed_sequence,
        target=target,
        support=support,
        groups=groups,
    )
    return linear_tvd, linear_margin


def _bootstrap_distribution_metrics(
    counts: np.ndarray,
    *,
    replicates: int,
    batch_size: int,
    seed_sequence: np.random.SeedSequence,
    target: np.ndarray,
    support: np.ndarray | None = None,
    groups: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    shots = int(np.sum(counts, dtype=np.int64))
    if shots <= 0:
        raise ValueError("distribution requires a positive shot count")
    probability = np.asarray(counts, dtype=np.float64) / shots
    rng = np.random.default_rng(seed_sequence)
    tvd_samples = np.empty(replicates, dtype=np.float64)
    margin_samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        sampled = rng.multinomial(
            shots, probability, size=stop - start
        ).astype(np.float64)
        sampled /= shots
        tvd, margin = _batch_metrics(
            sampled, target=target, support=support, groups=groups
        )
        tvd_samples[start:stop] = tvd
        margin_samples[start:stop] = margin
    return tvd_samples, margin_samples


def _summary(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "median": float(median),
        "ci_low": float(low),
        "ci_high": float(high),
        "variance": float(np.var(values, ddof=1)),
        "sd": float(np.std(values, ddof=1)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def analyse_instance(
    config_path: Path,
    *,
    run_dir: Path,
    rates: np.ndarray,
    replicates: int,
    batch_size: int,
    seed: int,
    no_reset_run_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    config, _ = load_config(config_path)
    historical = (
        None
        if config.get("require_all_lambdas_from_shotwise", False)
        else _load_historical(config)
    )
    lambdas = np.asarray(config["lambdas"], dtype=np.float64)
    records = _load_shotwise_records(run_dir, config)
    count_grid, sources = _assemble_shotwise_grid(
        historical=historical,
        records=records,
        rates=rates,
        lambdas=lambdas,
    )
    no_reset_grid = (
        None
        if no_reset_run_dir is None
        else _load_no_reset_counts(no_reset_run_dir, config, rates)
    )
    if config.get("require_all_lambdas_from_shotwise", False) and np.any(
        sources != "shotwise"
    ):
        missing = [
            (float(rates[rate_i]), float(lambdas[lambda_i]))
            for rate_i, lambda_i in np.argwhere(sources != "shotwise")
        ]
        raise ValueError(
            "gate-matched analysis requires fresh shot-wise records at every "
            f"rate and lambda; missing {missing}"
        )
    coefficients = weighted_linear_intercept_coefficients(lambdas)
    measurement_bits = int(config["measurement_bits"])
    order = int(config["order"])
    target = ideal_distribution(measurement_bits, order)
    support = ideal_indices(measurement_bits, order)
    groups = main_lobe_groups(measurement_bits, order)
    ideal_group_margin = distribution_metrics(
        target, measurement_bits, order
    ).grouped_peak_margin
    grid_aligned = (1 << measurement_bits) % order == 0
    instance_code = zlib.crc32(config["instance"].encode("utf-8"))
    rows: list[dict[str, Any]] = []
    metric_names = np.asarray(
        [
            "linear_zne_tvd",
            "linear_zne_grouped_peak_margin",
            "lambda1_tvd",
            "lambda1_grouped_peak_margin",
            "tvd_improvement",
            "grouped_peak_margin_improvement",
            "no_reset_tvd",
            "no_reset_grouped_peak_margin",
            "reset_tvd_improvement_vs_no_reset",
            "reset_grouped_peak_margin_improvement_vs_no_reset",
            "zne_tvd_improvement_vs_no_reset",
            "zne_grouped_peak_margin_improvement_vs_no_reset",
        ]
    )
    draws = np.empty(
        (rates.size, metric_names.size, replicates), dtype=np.float64
    )

    for rate_i, rate in enumerate(rates):
        counts = count_grid[rate_i]
        shots = counts.sum(axis=1, dtype=np.int64)
        probabilities = counts.astype(np.float64) / shots[:, None]
        projected = simplex_projection_rows(coefficients @ probabilities)
        point = distribution_metrics(
            projected, measurement_bits, order
        )
        baseline_i = _nearest_index(1.0, lambdas, "baseline lambda")
        baseline_point = distribution_metrics(
            probabilities[baseline_i], measurement_bits, order
        )
        (
            tvd,
            margin,
            baseline_tvd,
            baseline_margin,
        ) = _bootstrap_linear_and_baseline_metrics(
            counts,
            lambdas=lambdas,
            replicates=replicates,
            batch_size=batch_size,
            seed_sequence=np.random.SeedSequence(
                [seed, instance_code, rate_i]
            ),
            target=target,
            groups=groups,
        )
        tvd_improvement = baseline_tvd - tvd
        margin_improvement = margin - baseline_margin
        draws[rate_i, 0] = tvd
        draws[rate_i, 1] = margin
        draws[rate_i, 2] = baseline_tvd
        draws[rate_i, 3] = baseline_margin
        draws[rate_i, 4] = tvd_improvement
        draws[rate_i, 5] = margin_improvement
        tvd_summary = _summary(tvd)
        margin_summary = _summary(margin)
        baseline_tvd_summary = _summary(baseline_tvd)
        baseline_margin_summary = _summary(baseline_margin)
        tvd_improvement_summary = _summary(tvd_improvement)
        margin_improvement_summary = _summary(margin_improvement)
        row: dict[str, Any] = {
            "instance": config["instance"],
            "a": int(config["a"]),
            "order": order,
            "order_grid_aligned": grid_aligned,
            "ideal_target": (
                "delta_peaks"
                if grid_aligned
                else "exact_finite_resolution_qpe_distribution"
            ),
            "primary_peak_indices": ";".join(map(str, support.tolist())),
            "main_lobe_groups": ";".join(
                ",".join(map(str, group.tolist())) for group in groups
            ),
            "ideal_grouped_peak_margin": ideal_group_margin,
            "rate": float(rate),
            "folding_scope": FOLDING_PATTERN_SCOPE,
            "shots_per_lambda": ";".join(map(str, shots.tolist())),
            "lambda_sources": ";".join(sources[rate_i].tolist()),
            "linear_zne_tvd_point": point.tvd,
            "linear_zne_grouped_peak_margin_point": point.grouped_peak_margin,
            "linear_zne_normalized_grouped_peak_margin_point": (
                point.normalized_grouped_peak_margin
            ),
            "lambda1_tvd_point": baseline_point.tvd,
            "lambda1_grouped_peak_margin_point": (
                baseline_point.grouped_peak_margin
            ),
            "lambda1_normalized_grouped_peak_margin_point": (
                baseline_point.normalized_grouped_peak_margin
            ),
            "tvd_improvement_point": baseline_point.tvd - point.tvd,
            "grouped_peak_margin_improvement_point": (
                point.grouped_peak_margin - baseline_point.grouped_peak_margin
            ),
            "no_reset_tvd_point": float("nan"),
            "no_reset_grouped_peak_margin_point": float("nan"),
            "no_reset_normalized_grouped_peak_margin_point": float("nan"),
            "reset_tvd_improvement_vs_no_reset_point": float("nan"),
            "reset_grouped_peak_margin_improvement_vs_no_reset_point": float("nan"),
            "zne_tvd_improvement_vs_no_reset_point": float("nan"),
            "zne_grouped_peak_margin_improvement_vs_no_reset_point": float("nan"),
        }
        for label, summary in (
            ("tvd", tvd_summary),
            ("grouped_peak_margin", margin_summary),
            ("lambda1_tvd", baseline_tvd_summary),
            ("lambda1_grouped_peak_margin", baseline_margin_summary),
            ("tvd_improvement", tvd_improvement_summary),
            ("grouped_peak_margin_improvement", margin_improvement_summary),
        ):
            for statistic, value in summary.items():
                row[f"{label}_{statistic}"] = value
        row["tvd_improvement_positive_frequency"] = float(
            np.mean(tvd_improvement > 0.0)
        )
        row["grouped_peak_margin_improvement_positive_frequency"] = float(
            np.mean(margin_improvement > 0.0)
        )
        for label, values in (
            ("normalized_grouped_peak_margin", margin / ideal_group_margin),
            (
                "lambda1_normalized_grouped_peak_margin",
                baseline_margin / ideal_group_margin,
            ),
        ):
            for statistic, value in _summary(values).items():
                row[f"{label}_{statistic}"] = value
        if no_reset_grid is not None:
            no_reset_probabilities = no_reset_grid[rate_i].astype(np.float64)
            no_reset_probabilities /= no_reset_probabilities.sum()
            no_reset_point = distribution_metrics(
                no_reset_probabilities, measurement_bits, order
            )
            no_reset_tvd, no_reset_margin = _bootstrap_distribution_metrics(
                no_reset_grid[rate_i],
                replicates=replicates,
                batch_size=batch_size,
                seed_sequence=np.random.SeedSequence(
                    [seed, instance_code, rate_i, 1]
                ),
                target=target,
                groups=groups,
            )
            reset_tvd_improvement = no_reset_tvd - baseline_tvd
            reset_margin_improvement = baseline_margin - no_reset_margin
            zne_tvd_improvement = no_reset_tvd - tvd
            zne_margin_improvement = margin - no_reset_margin
            draws[rate_i, 6] = no_reset_tvd
            draws[rate_i, 7] = no_reset_margin
            draws[rate_i, 8] = reset_tvd_improvement
            draws[rate_i, 9] = reset_margin_improvement
            draws[rate_i, 10] = zne_tvd_improvement
            draws[rate_i, 11] = zne_margin_improvement
            row.update(
                {
                    "no_reset_tvd_point": no_reset_point.tvd,
                    "no_reset_grouped_peak_margin_point": (
                        no_reset_point.grouped_peak_margin
                    ),
                    "no_reset_normalized_grouped_peak_margin_point": (
                        no_reset_point.normalized_grouped_peak_margin
                    ),
                    "reset_tvd_improvement_vs_no_reset_point": (
                        no_reset_point.tvd - baseline_point.tvd
                    ),
                    "reset_grouped_peak_margin_improvement_vs_no_reset_point": (
                        baseline_point.grouped_peak_margin
                        - no_reset_point.grouped_peak_margin
                    ),
                    "zne_tvd_improvement_vs_no_reset_point": (
                        no_reset_point.tvd - point.tvd
                    ),
                    "zne_grouped_peak_margin_improvement_vs_no_reset_point": (
                        point.grouped_peak_margin
                        - no_reset_point.grouped_peak_margin
                    ),
                }
            )
            for label, values in (
                ("no_reset_tvd", no_reset_tvd),
                ("no_reset_grouped_peak_margin", no_reset_margin),
                ("reset_tvd_improvement_vs_no_reset", reset_tvd_improvement),
                (
                    "reset_grouped_peak_margin_improvement_vs_no_reset",
                    reset_margin_improvement,
                ),
                ("zne_tvd_improvement_vs_no_reset", zne_tvd_improvement),
                (
                    "zne_grouped_peak_margin_improvement_vs_no_reset",
                    zne_margin_improvement,
                ),
            ):
                for statistic, value in _summary(values).items():
                    row[f"{label}_{statistic}"] = value
                if "improvement" in label:
                    row[f"{label}_positive_frequency"] = float(
                        np.mean(values > 0.0)
                    )
        else:
            draws[rate_i, 6:] = np.nan
        rows.append(row)

    arrays = {
        "rates": rates,
        "lambdas": lambdas,
        "draws": draws,
        "metric_names": metric_names,
        "folding_scope": np.asarray(FOLDING_PATTERN_SCOPE),
    }
    return rows, arrays


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap linear ZNE with shot-wise random folding."
    )
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "simulator_runs"
        / "folding_patterns_matched_11rates",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        help="Override the common physical-error grid from the config files.",
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--no-reset-run-dir",
        type=Path,
        help=(
            "Optional directory containing lambda=1 no-reset simulator archives "
            "for a full no-reset, reset, and reset+ZNE comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "folding_patterns_matched_11rates",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replicates <= 1 or args.batch_size <= 0:
        raise ValueError("replicates must exceed one and batch-size must be positive")
    rates = (
        _shared_rate_grid(list(args.configs))
        if args.rates is None
        else np.asarray(args.rates, dtype=np.float64)
    )
    if (
        rates.size == 0
        or np.unique(rates).size != rates.size
        or np.min(rates) < 0.0
    ):
        raise ValueError("rates must be distinct and nonnegative")
    all_rows: list[dict[str, Any]] = []
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.configs:
        rows, arrays = analyse_instance(
            config_path.resolve(),
            run_dir=args.run_dir.resolve(),
            rates=rates,
            replicates=int(args.replicates),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            no_reset_run_dir=(
                None
                if args.no_reset_run_dir is None
                else args.no_reset_run_dir.resolve()
            ),
        )
        all_rows.extend(rows)
        np.savez_compressed(
            output_dir / f"{rows[0]['instance']}_shotwise_bootstrap_draws.npz",
            **arrays,
        )

    summary_path = output_dir / "shotwise_linear_zne_metrics.csv"
    _write_csv(summary_path, all_rows)
    metadata = {
        "analysis": "linear-ZNE bootstrap with per-shot random folding",
        "folding_randomness": "included in each empirical shot distribution",
        "configs": [str(path.resolve()) for path in args.configs],
        "run_dir": str(args.run_dir.resolve()),
        "no_reset_run_dir": (
            None
            if args.no_reset_run_dir is None
            else str(args.no_reset_run_dir.resolve())
        ),
        "rates": rates.tolist(),
        "replicates": int(args.replicates),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "output": str(summary_path),
    }
    temporary = output_dir / "analysis_metadata.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output_dir / "analysis_metadata.json")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
