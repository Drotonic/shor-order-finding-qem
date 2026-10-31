"""Frozen eight-scale, four-model bias--variance revision analysis.

This module resamples existing shot records only.  It never constructs a
circuit, samples a quantum trajectory, or invokes the simulator.  The
implementation preserves the shared lambda-one block between Reset and ZNE,
uses one common eight-scale bootstrap grid for all four extrapolation models,
and accumulates preprojection vector moments without storing every vector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap_analysis import (
    ZNE_METHODS,
    _build_estimator_suite,
    _fit_zne_batch_detailed,
    _row_metrics,
)
from .data import load_config
from .estimators import DEFAULT_RIDGE, objective_weights
from .folding_pattern_analysis import (
    _assemble_shotwise_grid,
    _load_no_reset_counts,
    _load_shotwise_records,
)
from .folding_pattern_model_analysis import _load_cost_matched_baselines
from .metrics import distribution_metrics, ideal_distribution, main_lobe_groups


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP_ROOT = (
    PROJECT_ROOT
    / "a100_followup_downloads"
    / "n119-a100-followup-20260726-v2"
)
H100_ROOT = (
    PROJECT_ROOT
    / "h100_downloads"
    / "n119-g7e-notebook-20260724"
    / "extracted"
    / "run"
    / "simulator"
)
BASELINE_ROOT = FOLLOWUP_ROOT / "merged" / "baseline_160k"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "bias_variance_revision"

ZNE_SHOTS_PER_SCALE = 20_000
DIRECT_SHOTS = 160_000
DIRECT_EXTENSION_SHOTS = DIRECT_SHOTS - ZNE_SHOTS_PER_SCALE
FROZEN_BATCH_SIZE = 25
VARIANCE_MC_BLOCK_SIZE = 500
PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE = 1.0e-12
L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE = 5.0e-12
PLUGIN_EMPIRICAL_MC_Z_WARNING = 4.0
NOMINAL_LAMBDAS = np.asarray(
    [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0], dtype=np.float64
)
ROUTES = (
    "raw_20k",
    "raw_160k",
    "reset_20k",
    "reset_160k",
    *ZNE_METHODS,
)
MAIN_ROUTES = ("raw_160k", "reset_160k", *ZNE_METHODS)
POST_METRICS = ("tvd", "grouped_peak_margin", "normalized_grouped_peak_margin")
DIRECT_PAIRS = (("raw_20k", "raw_160k"), ("reset_20k", "reset_160k"))
ROUTE_LABELS = {
    "raw_20k": "Raw 20k",
    "raw_160k": "Raw 160k",
    "reset_20k": "Reset 20k",
    "reset_160k": "Reset 160k",
    "linear_zne": "Linear ZNE 160k",
    "quadratic_zne": "Quadratic ZNE 160k",
    "single_exponential_zne": "Single-exp. ZNE 160k",
    "dual_exponential_zne": "Dual-exp. ZNE 160k",
}


@dataclass(frozen=True)
class InstanceSpec:
    key: str
    config: Path
    run_dir: Path
    no_reset_dir: Path


@dataclass
class _StreamingVectorMoments:
    """Stable Chan--Welford moments for vector-valued bootstrap draws."""

    count: int
    mean: np.ndarray
    centered_sum_squares: float

    @classmethod
    def empty(cls, dimension: int) -> "_StreamingVectorMoments":
        if dimension <= 0:
            raise ValueError("streaming-moment dimension must be positive")
        return cls(0, np.zeros(int(dimension), dtype=np.float64), 0.0)

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.size:
            raise ValueError("streaming vector batch has an unexpected shape")
        if batch.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch)):
            raise FloatingPointError("streaming vector batch contains nonfinite values")
        batch_count = int(batch.shape[0])
        batch_mean = np.mean(batch, axis=0)
        centered = batch - batch_mean[None, :]
        batch_m2 = float(np.einsum("bi,bi->", centered, centered, optimize=True))
        if self.count == 0:
            self.count = batch_count
            self.mean[...] = batch_mean
            self.centered_sum_squares = batch_m2
            return
        old_count = self.count
        new_count = old_count + batch_count
        delta = batch_mean - self.mean
        correction = (
            float(np.dot(delta, delta)) * old_count * batch_count / new_count
        )
        self.mean += delta * (batch_count / new_count)
        self.centered_sum_squares += batch_m2 + correction
        self.count = new_count

    def population_variance_trace(self, *, negative_tolerance: float) -> float:
        if self.count <= 0:
            raise ValueError("cannot finalize empty streaming moments")
        m2 = float(self.centered_sum_squares)
        if m2 < -abs(float(negative_tolerance)):
            raise FloatingPointError(
                f"materially negative centered sum of squares: {m2:.17g}"
            )
        if m2 < 0.0:
            m2 = 0.0
        return m2 / self.count

    def unbiased_variance_trace(self, *, negative_tolerance: float) -> float:
        if self.count <= 1:
            return math.nan
        population = self.population_variance_trace(
            negative_tolerance=negative_tolerance
        )
        return population * self.count / (self.count - 1)


INSTANCE_SPECS = {
    "n119_a13": InstanceSpec(
        "n119_a13",
        PROJECT_ROOT / "configs" / "n119_a13_matched.json",
        H100_ROOT / "shotwise",
        H100_ROOT / "no_reset" / "n119_a13",
    ),
    "n119_a50": InstanceSpec(
        "n119_a50",
        PROJECT_ROOT / "configs" / "n119_a50_matched.json",
        H100_ROOT / "shotwise",
        H100_ROOT / "no_reset" / "n119_a50",
    ),
    "n119_a16": InstanceSpec(
        "n119_a16",
        PROJECT_ROOT / "configs" / "n119_a16_augmented.json",
        FOLLOWUP_ROOT / "merged" / "a16_augmented_20k" / "shotwise",
        FOLLOWUP_ROOT
        / "merged"
        / "a16_augmented_20k"
        / "no_reset"
        / "n119_a16",
    ),
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


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "sd": math.nan,
            "q95": math.nan,
            "q99": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
        }
    tail = (1.0 - confidence) / 2.0
    low, median, high, q95, q99 = np.quantile(
        finite, [tail, 0.5, 1.0 - tail, 0.95, 0.99]
    )
    return {
        "mean": float(np.mean(finite)),
        "median": float(median),
        "ci_low": float(low),
        "ci_high": float(high),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q95": float(q95),
        "q99": float(q99),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _positive_denominator_ratio(
    numerator: float, denominator: float, *, context: str
) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or numerator < 0.0:
        raise FloatingPointError(f"{context}: invalid numerator {numerator!r}")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ZeroDivisionError(f"{context}: denominator must be positive")
    return numerator / denominator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(spec: InstanceSpec) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = [("config", spec.config)]
    shot_root = spec.run_dir / spec.key / "shotwise"
    files.extend(("shotwise_scale_archive", path) for path in sorted(shot_root.glob("*.npz")))
    files.extend(("no_reset_20k", path) for path in sorted(spec.no_reset_dir.glob("*.pkl")))
    for mode in ("raw", "reset"):
        files.extend(
            (f"baseline_160k_{mode}", path)
            for path in sorted((BASELINE_ROOT / spec.key / mode).glob("*.npz"))
        )
    return files


def _manifest_rows(spec: InstanceSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, path in _source_files(spec):
        rows.append(
            {
                "instance": spec.key,
                "role": role,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _rate_indices(requested: np.ndarray, configured: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for rate in requested:
        index = int(np.argmin(np.abs(configured - rate)))
        if not np.isclose(configured[index], rate, rtol=0.0, atol=5.0e-12):
            raise ValueError(f"requested rate {rate} is absent")
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def _realized_lambdas(spec: InstanceSpec, nominal: np.ndarray) -> np.ndarray:
    root = spec.run_dir / spec.key / "shotwise"
    values: list[float] = []
    for lam in nominal:
        matches: list[Path] = []
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                saved = np.asarray(archive["lambdas"], dtype=np.float64)
                if saved.size == 1 and np.isclose(saved[0], lam, atol=5e-12):
                    matches.append(path)
        if len(matches) != 1:
            raise ValueError(f"expected one archive for nominal lambda={lam}")
        with np.load(matches[0], allow_pickle=False) as archive:
            folded = np.asarray(archive["folded_gate_counts_by_round"], dtype=np.int64)
            if folded.shape[0] != 1:
                raise ValueError("scale archive must contain one lambda")
            base = int(archive["two_qubit_gate_count"])
            values.append(1.0 + 2.0 * float(folded.sum()) / base)
    return np.asarray(values, dtype=np.float64)


def _polynomial_score_matrix(
    lambdas: np.ndarray, degree: int, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    weights = objective_weights(lambdas, exponent=2.0)
    design = np.vander(lambdas, N=degree + 1, increasing=True)
    normal = design.T @ (weights[:, None] * design)
    if ridge:
        normal = normal + ridge * np.eye(degree + 1)
    weighted_design = weights[:, None] * design
    score = weighted_design @ np.linalg.solve(normal, weighted_design.T)
    return weights, score


def _polynomial_residual(
    distributions: np.ndarray, weights: np.ndarray, score: np.ndarray
) -> np.ndarray:
    gram = np.einsum("blk,bmk->blm", distributions, distributions, optimize=True)
    base = np.einsum("l,bll->b", weights, gram, optimize=True)
    explained = np.einsum("lm,blm->b", score, gram, optimize=True)
    return np.maximum(base - explained, 0.0)


def _post_metrics(
    values: np.ndarray,
    target: np.ndarray,
    groups: tuple[np.ndarray, ...],
    ideal_margin: float,
) -> dict[str, np.ndarray]:
    metrics = _row_metrics(values, target=target, groups=groups)
    return {
        "tvd": metrics["tvd"],
        "grouped_peak_margin": metrics["grouped_peak_margin"],
        "normalized_grouped_peak_margin": (
            metrics["grouped_peak_margin"] / ideal_margin
        ),
    }


def _draw_paired_batch(
    *,
    reset_parents: np.ndarray,
    reset_extension_parent: np.ndarray,
    raw_initial_parent: np.ndarray,
    raw_extension_parent: np.ndarray,
    reset_rng: np.random.Generator,
    reset_extension_rng: np.random.Generator,
    raw_rng: np.random.Generator,
    raw_extension_rng: np.random.Generator,
    size: int,
    lambda_one_index: int,
) -> dict[str, np.ndarray]:
    """Generate one frozen paired-bootstrap batch.

    The returned ``reset_grid`` is the sole eight-scale draw passed to every
    ZNE model.  ``reset_20k`` is a view of its lambda-one block, and
    ``reset_160k`` combines exactly that block with only an independent 140k
    extension.  Raw20/Raw160 preserve the analogous verified first-block
    overlap while remaining independent of Reset and ZNE.
    """

    if size <= 0:
        raise ValueError("bootstrap batch size must be positive")
    reset_counts = np.stack(
        [
            reset_rng.multinomial(ZNE_SHOTS_PER_SCALE, parent, size=size)
            for parent in np.asarray(reset_parents, dtype=np.float64)
        ],
        axis=1,
    ).astype(np.float64)
    reset_grid = reset_counts / ZNE_SHOTS_PER_SCALE
    reset20 = reset_grid[:, int(lambda_one_index)]
    reset_extension_counts = reset_extension_rng.multinomial(
        DIRECT_EXTENSION_SHOTS,
        np.asarray(reset_extension_parent, dtype=np.float64),
        size=size,
    ).astype(np.float64)
    reset160 = (
        reset20 * ZNE_SHOTS_PER_SCALE + reset_extension_counts
    ) / DIRECT_SHOTS

    raw20_counts = raw_rng.multinomial(
        ZNE_SHOTS_PER_SCALE,
        np.asarray(raw_initial_parent, dtype=np.float64),
        size=size,
    ).astype(np.float64)
    raw20 = raw20_counts / ZNE_SHOTS_PER_SCALE
    raw_extension_counts = raw_extension_rng.multinomial(
        DIRECT_EXTENSION_SHOTS,
        np.asarray(raw_extension_parent, dtype=np.float64),
        size=size,
    ).astype(np.float64)
    raw160 = (raw20_counts + raw_extension_counts) / DIRECT_SHOTS
    return {
        "reset_grid": reset_grid,
        "reset_20k": reset20,
        "reset_160k": reset160,
        "reset_extension_counts": reset_extension_counts,
        "raw_20k": raw20,
        "raw_160k": raw160,
        "raw_20k_counts": raw20_counts,
        "raw_extension_counts": raw_extension_counts,
    }


def _bootstrap_seed_streams(
    seed: int, instance_key: str, rate: float
) -> tuple[np.random.SeedSequence, ...]:
    """Return stable rate-keyed streams for the four parent blocks."""

    instance_code = zlib.crc32(instance_key.encode("utf-8"))
    rate_code = int(round(float(rate) * 1.0e9))
    return tuple(
        np.random.SeedSequence([int(seed), instance_code, rate_code]).spawn(4)
    )


def _prepare_data(
    spec: InstanceSpec, requested_rates: np.ndarray | None, scale_mode: str
) -> dict[str, Any]:
    config, resolved = load_config(spec.config)
    configured_rates = np.asarray(config["rates"], dtype=np.float64)
    if requested_rates is None:
        rates = configured_rates
    else:
        indices = _rate_indices(requested_rates, configured_rates)
        rates = configured_rates[indices]
    nominal = np.asarray(config["lambdas"], dtype=np.float64)
    if not np.allclose(nominal, NOMINAL_LAMBDAS, rtol=0.0, atol=5e-12):
        raise ValueError("configuration does not use the frozen eight-scale grid")
    fit_lambdas = nominal if scale_mode == "nominal" else _realized_lambdas(spec, nominal)

    records = _load_shotwise_records(spec.run_dir, config)
    reset_counts, sources = _assemble_shotwise_grid(
        historical=None, records=records, rates=rates, lambdas=nominal
    )
    if np.any(sources != "shotwise"):
        raise ValueError("all ZNE inputs must be latest shot-wise archives")
    raw_initial = _load_no_reset_counts(spec.no_reset_dir, config, rates)
    lambda_one = int(np.argmin(np.abs(nominal - 1.0)))
    (
        raw_full,
        reset_full,
        reset_extension,
        baseline_shots,
        extension_shots,
    ) = _load_cost_matched_baselines(
        baseline_root=BASELINE_ROOT,
        config=config,
        rates=rates,
        reset_zne_counts=reset_counts,
        raw_initial_counts=raw_initial,
        lambda_one_index=lambda_one,
    )
    if baseline_shots != DIRECT_SHOTS or extension_shots != DIRECT_EXTENSION_SHOTS:
        raise ValueError("baseline shot totals differ from the frozen specification")
    totals = reset_counts.sum(axis=-1, dtype=np.int64)
    if not np.all(totals == ZNE_SHOTS_PER_SCALE):
        raise ValueError("each ZNE scale must contain exactly 20,000 shots")
    raw_extension = raw_full - raw_initial.astype(np.int64)
    if np.any(raw_extension < 0) or not np.all(
        raw_extension.sum(axis=1) == DIRECT_EXTENSION_SHOTS
    ):
        raise ValueError("Raw 160k archives do not preserve the verified first-20k block")

    def probabilities(counts: np.ndarray) -> np.ndarray:
        values = counts.astype(np.float64)
        return values / values.sum(axis=-1, keepdims=True)

    return {
        "config": config,
        "resolved_config": resolved,
        "rates": rates,
        "nominal_lambdas": nominal,
        "fit_lambdas": fit_lambdas,
        "reset": probabilities(reset_counts),
        "raw_initial": probabilities(raw_initial),
        "raw_full": probabilities(raw_full),
        "raw_extension": probabilities(raw_extension),
        "reset_full": probabilities(reset_full),
        "reset_extension": probabilities(reset_extension),
        "lambda_one": lambda_one,
    }


def _coefficient_rows(
    config: dict[str, Any], fit_lambdas: np.ndarray, fixed: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("linear_zne", "quadratic_zne"):
        coefficients = fixed[method]
        degree = 1 if method == "linear_zne" else 2
        moments = [float(np.dot(coefficients, fit_lambdas**power)) for power in range(4)]
        norm2 = float(np.dot(coefficients, coefficients))
        for index, (lam, coefficient) in enumerate(zip(fit_lambdas, coefficients, strict=True)):
            rows.append(
                {
                    "instance": config["instance"],
                    "method": method,
                    "scale_index": index,
                    "fit_lambda": float(lam),
                    "coefficient": float(coefficient),
                    "coefficient_l1_norm": float(np.abs(coefficients).sum()),
                    "coefficient_l2_norm_squared": norm2,
                    "equal_covariance_effective_shots": ZNE_SHOTS_PER_SCALE / norm2,
                    "equal_covariance_variance_ratio_vs_reset_20k": norm2,
                    "equal_covariance_variance_ratio_vs_reset_160k": 8.0 * norm2,
                    "moment_0": moments[0],
                    "moment_1": moments[1],
                    "moment_2": moments[2],
                    "moment_3": moments[3],
                    "degree": degree,
                    "ridge": 0.0 if degree == 1 else DEFAULT_RIDGE,
                }
            )
    return rows


def _plugin_variance_rows(
    *,
    config: dict[str, Any],
    rates: np.ndarray,
    reset: np.ndarray,
    reset_extension: np.ndarray,
    fixed: dict[str, np.ndarray],
    replicates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rate_i, rate in enumerate(rates):
        reset20_trace = (
            1.0 - float(np.square(reset[rate_i, 0]).sum())
        ) / ZNE_SHOTS_PER_SCALE
        reset160_trace = (
            ZNE_SHOTS_PER_SCALE
            * (1.0 - float(np.square(reset[rate_i, 0]).sum()))
            + DIRECT_EXTENSION_SHOTS
            * (1.0 - float(np.square(reset_extension[rate_i]).sum()))
        ) / (DIRECT_SHOTS**2)
        for method in ("linear_zne", "quadratic_zne"):
            coefficients = fixed[method]
            trace = float(
                np.sum(
                    np.square(coefficients)
                    * (
                        1.0
                        - np.square(reset[rate_i]).sum(axis=1)
                    )
                    / ZNE_SHOTS_PER_SCALE
                )
            )
            rows.append(
                {
                    "instance": config["instance"],
                    "rate": float(rate),
                    "rate_percent": 100.0 * float(rate),
                    "method": method,
                    "variance_source": "multinomial_plugin_preprojection_trace",
                    "preprojection_covariance_trace": trace,
                    "reset_20k_covariance_trace": reset20_trace,
                    "reset_160k_covariance_trace": reset160_trace,
                    "variance_ratio_vs_reset_20k": _positive_denominator_ratio(
                        trace,
                        reset20_trace,
                        context=f"{config['instance']}/{rate:g}/{method}/Reset20k",
                    ),
                    "variance_ratio_vs_reset_160k": _positive_denominator_ratio(
                        trace,
                        reset160_trace,
                        context=f"{config['instance']}/{rate:g}/{method}/Reset160k",
                    ),
                    "bootstrap_replicates": replicates,
                    "bootstrap_replicates_applicable": False,
                }
            )
    return rows


def analyse_instance(
    spec: InstanceSpec,
    *,
    output_dir: Path,
    replicates: int,
    batch_size: int,
    confidence: float,
    seed: int,
    requested_rates: np.ndarray | None,
    scale_mode: str,
) -> dict[str, Any]:
    if int(batch_size) != FROZEN_BATCH_SIZE:
        raise ValueError(
            f"the frozen analysis requires batch_size={FROZEN_BATCH_SIZE}, "
            f"got {batch_size}"
        )
    data = _prepare_data(spec, requested_rates, scale_mode)
    config = data["config"]
    rates = data["rates"]
    fit_lambdas = data["fit_lambdas"]
    reset = data["reset"]
    outcome_count = 1 << int(config["measurement_bits"])
    target = ideal_distribution(int(config["measurement_bits"]), int(config["order"]))
    groups = main_lobe_groups(int(config["measurement_bits"]), int(config["order"]))
    ideal_margin = distribution_metrics(
        target, int(config["measurement_bits"]), int(config["order"])
    ).grouped_peak_margin
    fixed, exponential = _build_estimator_suite(fit_lambdas, weight_exponent=2.0)
    linear_weights, linear_score = _polynomial_score_matrix(fit_lambdas, 1, 0.0)
    quadratic_weights, quadratic_score = _polynomial_score_matrix(
        fit_lambdas, 2, DEFAULT_RIDGE
    )

    route_index = {route: index for index, route in enumerate(ROUTES)}
    zne_method_index = {
        method: index for index, method in enumerate(ZNE_METHODS)
    }
    metric_index = {metric: index for index, metric in enumerate(POST_METRICS)}
    draws = np.empty(
        (rates.size, len(ROUTES), len(POST_METRICS), replicates), dtype=np.float64
    )
    preprojection_mean_vectors = np.empty(
        (rates.size, len(ROUTES), outcome_count), dtype=np.float64
    )
    preprojection_mean_squared_norms = np.empty(
        (rates.size, len(ROUTES)), dtype=np.float64
    )
    preprojection_centered_sum_squares = np.empty(
        (rates.size, len(ROUTES)), dtype=np.float64
    )
    preprojection_variance_traces = np.empty(
        (rates.size, len(ROUTES)), dtype=np.float64
    )
    maximum_mc_blocks = int(math.ceil(replicates / VARIANCE_MC_BLOCK_SIZE))
    preprojection_block_unbiased_variance_traces = np.full(
        (rates.size, len(ROUTES), maximum_mc_blocks),
        np.nan,
        dtype=np.float64,
    )
    preprojection_block_counts = np.zeros(
        (rates.size, maximum_mc_blocks), dtype=np.int64
    )
    preprojection_l2_squared_to_ideal = np.empty(
        (rates.size, len(ZNE_METHODS), replicates), dtype=np.float64
    )
    postprojection_l2_squared_to_ideal = np.empty_like(
        preprojection_l2_squared_to_ideal
    )
    post_minus_pre_l2_squared_to_ideal = np.empty_like(
        preprojection_l2_squared_to_ideal
    )
    dual_gamma_1_zero = np.empty((rates.size, replicates), dtype=np.float64)
    curve_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    bias_variance_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    empirical_variance_rows: list[dict[str, Any]] = []

    for rate_i, rate in enumerate(rates):
        print(
            f"[{spec.key}/{scale_mode}] {rate_i + 1}/{rates.size} "
            f"rate={rate:g}, B={replicates}",
            flush=True,
        )
        # Key streams by the physical rate rather than its position in a
        # requested subset.  A boundary-only B=20k confirmation therefore
        # extends the same deterministic sequence used by the full B=10k run.
        reset_scale_seed, reset_extension_seed, raw_scale_seed, raw_extension_seed = (
            _bootstrap_seed_streams(seed, spec.key, float(rate))
        )
        reset_rng = np.random.default_rng(reset_scale_seed)
        reset_extension_rng = np.random.default_rng(reset_extension_seed)
        raw_rng = np.random.default_rng(raw_scale_seed)
        raw_extension_rng = np.random.default_rng(raw_extension_seed)

        route_moments = [
            _StreamingVectorMoments.empty(outcome_count) for _ in ROUTES
        ]
        block_moments = [
            _StreamingVectorMoments.empty(outcome_count) for _ in ROUTES
        ]
        completed_block_count = 0
        diagnostic_draws: dict[str, dict[str, np.ndarray]] = {
            method: {} for method in ZNE_METHODS
        }
        pre_tvd = {method: np.empty(replicates) for method in ZNE_METHODS}
        pre_margin = {method: np.empty(replicates) for method in ZNE_METHODS}
        post_tvd = {method: np.empty(replicates) for method in ZNE_METHODS}
        post_margin = {method: np.empty(replicates) for method in ZNE_METHODS}

        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            size = stop - start
            paired_batch = _draw_paired_batch(
                reset_parents=reset[rate_i],
                reset_extension_parent=data["reset_extension"][rate_i],
                raw_initial_parent=data["raw_initial"][rate_i],
                raw_extension_parent=data["raw_extension"][rate_i],
                reset_rng=reset_rng,
                reset_extension_rng=reset_extension_rng,
                raw_rng=raw_rng,
                raw_extension_rng=raw_extension_rng,
                size=size,
                lambda_one_index=data["lambda_one"],
            )
            reset_grid = paired_batch["reset_grid"]
            reset20 = paired_batch["reset_20k"]
            reset160 = paired_batch["reset_160k"]
            raw20 = paired_batch["raw_20k"]
            raw160 = paired_batch["raw_160k"]

            projected, fit_diagnostics, unconstrained, _ = _fit_zne_batch_detailed(
                reset_grid,
                fixed_coefficients=fixed,
                exponential_families=exponential,
            )
            fit_diagnostics["linear_zne"]["penalized_weighted_residual"] = (
                _polynomial_residual(reset_grid, linear_weights, linear_score)
            )
            fit_diagnostics["quadratic_zne"]["penalized_weighted_residual"] = (
                _polynomial_residual(reset_grid, quadratic_weights, quadratic_score)
            )

            direct = {
                "raw_20k": raw20,
                "raw_160k": raw160,
                "reset_20k": reset20,
                "reset_160k": reset160,
            }
            pre_routes = {**direct, **unconstrained}
            post_routes = {**direct, **projected}

            for route in ROUTES:
                route_i = route_index[route]
                pre = pre_routes[route]
                post = post_routes[route]
                route_moments[route_i].update(pre)
                metrics = _post_metrics(post, target, groups, ideal_margin)
                for metric in POST_METRICS:
                    draws[rate_i, route_i, metric_index[metric], start:stop] = metrics[
                        metric
                    ]

            # Maintain independent fixed-size block traces for a Monte Carlo
            # diagnostic of the empirical covariance-trace estimate.  The
            # production batch size divides the block size, but the split loop
            # keeps the update correct if a final partial batch is encountered.
            piece_start = 0
            while piece_start < size:
                block_remaining = (
                    VARIANCE_MC_BLOCK_SIZE - block_moments[0].count
                )
                take = min(block_remaining, size - piece_start)
                piece_stop = piece_start + take
                for route in ROUTES:
                    route_i = route_index[route]
                    block_moments[route_i].update(
                        pre_routes[route][piece_start:piece_stop]
                    )
                piece_start = piece_stop
                if block_moments[0].count == VARIANCE_MC_BLOCK_SIZE:
                    preprojection_block_counts[
                        rate_i, completed_block_count
                    ] = VARIANCE_MC_BLOCK_SIZE
                    for route_i, moments in enumerate(block_moments):
                        preprojection_block_unbiased_variance_traces[
                            rate_i, route_i, completed_block_count
                        ] = moments.unbiased_variance_trace(
                            negative_tolerance=(
                                PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
                            )
                        )
                    completed_block_count += 1
                    block_moments = [
                        _StreamingVectorMoments.empty(outcome_count)
                        for _ in ROUTES
                    ]

            for method in ZNE_METHODS:
                method_i = zne_method_index[method]
                pre_l2_squared = np.square(
                    unconstrained[method] - target[None, :]
                ).sum(axis=1)
                post_l2_squared = np.square(
                    projected[method] - target[None, :]
                ).sum(axis=1)
                l2_excess = post_l2_squared - pre_l2_squared
                l2_violation = (
                    l2_excess > L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE
                ).astype(np.float64)
                preprojection_l2_squared_to_ideal[
                    rate_i, method_i, start:stop
                ] = pre_l2_squared
                postprojection_l2_squared_to_ideal[
                    rate_i, method_i, start:stop
                ] = post_l2_squared
                post_minus_pre_l2_squared_to_ideal[
                    rate_i, method_i, start:stop
                ] = l2_excess
                fit_diagnostics[method][
                    "preprojection_l2_squared_to_ideal"
                ] = pre_l2_squared
                fit_diagnostics[method][
                    "postprojection_l2_squared_to_ideal"
                ] = post_l2_squared
                fit_diagnostics[method][
                    "post_minus_pre_l2_squared_to_ideal"
                ] = l2_excess
                fit_diagnostics[method][
                    "l2_nonexpansiveness_violation"
                ] = l2_violation
                if method == "dual_exponential_zne":
                    fit_diagnostics[method]["selected_gamma_1_zero"] = np.isclose(
                        fit_diagnostics[method]["selected_gamma_1"],
                        0.0,
                        rtol=0.0,
                        atol=0.0,
                    ).astype(np.float64)
                    dual_gamma_1_zero[rate_i, start:stop] = fit_diagnostics[method][
                        "selected_gamma_1_zero"
                    ]
                pre_metrics = _post_metrics(
                    unconstrained[method], target, groups, ideal_margin
                )
                post_metrics = _post_metrics(projected[method], target, groups, ideal_margin)
                pre_tvd[method][start:stop] = pre_metrics["tvd"]
                pre_margin[method][start:stop] = pre_metrics[
                    "normalized_grouped_peak_margin"
                ]
                post_tvd[method][start:stop] = post_metrics["tvd"]
                post_margin[method][start:stop] = post_metrics[
                    "normalized_grouped_peak_margin"
                ]
                for name, values in fit_diagnostics[method].items():
                    if name not in diagnostic_draws[method]:
                        diagnostic_draws[method][name] = np.empty(
                            replicates, dtype=np.float64
                        )
                    diagnostic_draws[method][name][start:stop] = np.asarray(
                        values, dtype=np.float64
                    )

        if block_moments[0].count > 0:
            preprojection_block_counts[
                rate_i, completed_block_count
            ] = block_moments[0].count
            for route_i, moments in enumerate(block_moments):
                preprojection_block_unbiased_variance_traces[
                    rate_i, route_i, completed_block_count
                ] = moments.unbiased_variance_trace(
                    negative_tolerance=PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
                )
            completed_block_count += 1

        for route in ROUTES:
            route_i = route_index[route]
            shots = ZNE_SHOTS_PER_SCALE if route.endswith("20k") else DIRECT_SHOTS
            if route in ZNE_METHODS:
                shots = len(fit_lambdas) * ZNE_SHOTS_PER_SCALE
            row: dict[str, Any] = {
                "instance": spec.key,
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": int(config["order"]),
                "rate": float(rate),
                "rate_percent": 100.0 * float(rate),
                "route": route,
                "route_label": ROUTE_LABELS[route],
                "shots": shots,
                "diagnostic_only": route.endswith("20k"),
                "bootstrap_replicates": replicates,
                "confidence_level": confidence,
            }
            for metric in POST_METRICS:
                values = draws[rate_i, route_i, metric_index[metric]]
                for statistic, value in _summary(values, confidence).items():
                    row[f"{metric}_{statistic}"] = value
                if metric == "normalized_grouped_peak_margin":
                    row["normalized_grouped_peak_margin_positive_frequency"] = float(
                        np.mean(values > 0.0)
                    )
                    row["resolved_pointwise_95"] = bool(
                        row["normalized_grouped_peak_margin_ci_low"] > 0.0
                    )
            curve_rows.append(row)

            moments = route_moments[route_i]
            if moments.count != replicates:
                raise RuntimeError(
                    f"{spec.key}/{rate:g}/{route}: accumulated {moments.count} "
                    f"replicates, expected {replicates}"
                )
            mean_vector = moments.mean.copy()
            variance = moments.population_variance_trace(
                negative_tolerance=PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
            )
            mean_norm_squared = float(np.dot(mean_vector, mean_vector))
            mean_squared_norm = mean_norm_squared + variance
            bias_squared = float(np.square(mean_vector - target).sum())
            preprojection_mean_vectors[rate_i, route_i] = mean_vector
            preprojection_mean_squared_norms[rate_i, route_i] = mean_squared_norm
            preprojection_centered_sum_squares[
                rate_i, route_i
            ] = moments.centered_sum_squares
            preprojection_variance_traces[rate_i, route_i] = variance
            bias_variance_rows.append(
                {
                    "instance": spec.key,
                    "rate": float(rate),
                    "rate_percent": 100.0 * float(rate),
                    "route": route,
                    "shots": shots,
                    "preprojection_bias_squared_l2": bias_squared,
                    "preprojection_variance_l2": variance,
                    "preprojection_mse_l2": bias_squared + variance,
                    "bootstrap_replicates": replicates,
                    "conditional_parent_definition": "empirical_saved_shot_records",
                    "streaming_algorithm": "parallel_chan_welford_float64",
                    "variance_negative_tolerance": (
                        PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
                    ),
                }
            )

        reset20_variance = bias_variance_rows[-len(ROUTES) + route_index["reset_20k"]][
            "preprojection_variance_l2"
        ]
        reset160_variance = bias_variance_rows[-len(ROUTES) + route_index["reset_160k"]][
            "preprojection_variance_l2"
        ]
        for method in ZNE_METHODS:
            method_row = bias_variance_rows[-len(ROUTES) + route_index[method]]
            empirical_variance_rows.append(
                {
                    "instance": spec.key,
                    "rate": float(rate),
                    "rate_percent": 100.0 * float(rate),
                    "method": method,
                    "variance_source": "bootstrap_preprojection_covariance_trace",
                    "preprojection_covariance_trace": method_row[
                        "preprojection_variance_l2"
                    ],
                    "reset_20k_covariance_trace": reset20_variance,
                    "reset_160k_covariance_trace": reset160_variance,
                    "variance_ratio_vs_reset_20k": _positive_denominator_ratio(
                        method_row["preprojection_variance_l2"],
                        reset20_variance,
                        context=f"{spec.key}/{rate:g}/{method}/empirical/Reset20k",
                    ),
                    "variance_ratio_vs_reset_160k": _positive_denominator_ratio(
                        method_row["preprojection_variance_l2"],
                        reset160_variance,
                        context=f"{spec.key}/{rate:g}/{method}/empirical/Reset160k",
                    ),
                    "bootstrap_replicates": replicates,
                    "bootstrap_replicates_applicable": True,
                }
            )

        for method in ZNE_METHODS:
            for reference in ("reset_20k", "reset_160k"):
                reference_i = route_index[reference]
                method_i = route_index[method]
                tvd_gain = (
                    draws[rate_i, reference_i, metric_index["tvd"]]
                    - draws[rate_i, method_i, metric_index["tvd"]]
                )
                margin_gain = (
                    draws[
                        rate_i,
                        method_i,
                        metric_index["normalized_grouped_peak_margin"],
                    ]
                    - draws[
                        rate_i,
                        reference_i,
                        metric_index["normalized_grouped_peak_margin"],
                    ]
                )
                paired: dict[str, Any] = {
                    "instance": spec.key,
                    "rate": float(rate),
                    "rate_percent": 100.0 * float(rate),
                    "method": method,
                    "reference": reference,
                    "reference_shots": (
                        ZNE_SHOTS_PER_SCALE if reference.endswith("20k") else DIRECT_SHOTS
                    ),
                    "zne_total_shots": DIRECT_SHOTS,
                    "equal_total_shot_comparison": reference == "reset_160k",
                    "bootstrap_replicates": replicates,
                }
                for name, values in (
                    ("tvd_reduction", tvd_gain),
                    ("normalized_margin_improvement", margin_gain),
                ):
                    for statistic, value in _summary(values, confidence).items():
                        paired[f"{name}_{statistic}"] = value
                    paired[f"{name}_positive_frequency"] = float(
                        np.mean(values > 0.0)
                    )
                paired_rows.append(paired)

        for small, large in DIRECT_PAIRS:
            for metric in ("tvd", "normalized_grouped_peak_margin"):
                small_values = draws[
                    rate_i, route_index[small], metric_index[metric]
                ]
                large_values = draws[
                    rate_i, route_index[large], metric_index[metric]
                ]
                small_summary = _summary(small_values, confidence)
                large_summary = _summary(large_values, confidence)
                sd_ratio = _positive_denominator_ratio(
                    small_summary["sd"],
                    large_summary["sd"],
                    context=f"{spec.key}/{rate:g}/{small}/sd",
                )
                small_ci_width = small_summary["ci_high"] - small_summary["ci_low"]
                large_ci_width = large_summary["ci_high"] - large_summary["ci_low"]
                ci_width_ratio = _positive_denominator_ratio(
                    small_ci_width,
                    large_ci_width,
                    context=f"{spec.key}/{rate:g}/{small}/ci_width",
                )
                precision_rows.append(
                    {
                        "instance": spec.key,
                        "rate": float(rate),
                        "rate_percent": 100.0 * float(rate),
                        "direct_method": small.split("_")[0],
                        "metric": metric,
                        "small_shots": ZNE_SHOTS_PER_SCALE,
                        "large_shots": DIRECT_SHOTS,
                        "sd_20k": small_summary["sd"],
                        "sd_160k": large_summary["sd"],
                        "sd_ratio_20k_over_160k": sd_ratio,
                        "variance_ratio_20k_over_160k": sd_ratio**2,
                        "ci_width_20k": small_ci_width,
                        "ci_width_160k": large_ci_width,
                        "ci_width_ratio_20k_over_160k": ci_width_ratio,
                        "bootstrap_replicates": replicates,
                        "confidence_level": confidence,
                        "seed": seed,
                    }
                )

        for method in ZNE_METHODS:
            projection_l1_change = post_tvd[method] - pre_tvd[method]
            l2_excess = diagnostic_draws[method][
                "post_minus_pre_l2_squared_to_ideal"
            ]
            l2_violation = diagnostic_draws[method][
                "l2_nonexpansiveness_violation"
            ]
            violation_count = int(np.count_nonzero(l2_violation > 0.5))
            if violation_count:
                raise FloatingPointError(
                    f"{spec.key}/{rate:g}/{method}: {violation_count} simplex "
                    "projection L2 nonexpansiveness violations above tolerance "
                    f"{L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE:g}"
                )
            derived = {
                # The preprojection vector can be signed and unnormalised, so
                # 0.5*L1 is a discrepancy with the TVD algebraic form, not a
                # TVD between probability distributions.
                "post_tvd_minus_pre_half_l1_discrepancy": projection_l1_change,
                # Retained as a machine-compatibility alias for historical
                # verifier versions.  Manuscript/report code must use the
                # accurately named field above.
                "post_minus_pre_tvd": projection_l1_change,
                "post_minus_pre_normalized_margin": (
                    post_margin[method] - pre_margin[method]
                ),
            }
            all_diagnostics = {**diagnostic_draws[method], **derived}
            row = {
                "instance": spec.key,
                "rate": float(rate),
                "rate_percent": 100.0 * float(rate),
                "method": method,
                "bootstrap_replicates": replicates,
            }
            for name, values in all_diagnostics.items():
                for statistic, value in _summary(values, confidence).items():
                    row[f"{name}_{statistic}"] = value
                if name in {
                    "projection_active",
                    "grid_lower_boundary",
                    "grid_upper_boundary",
                    "grid_any_boundary",
                    "l2_nonexpansiveness_violation",
                    "selected_gamma_1_zero",
                }:
                    row[f"{name}_frequency"] = float(np.mean(values > 0.5))
            row["l2_nonexpansiveness_tolerance"] = (
                L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE
            )
            row["l2_nonexpansiveness_max_excess"] = float(np.max(l2_excess))
            row["l2_nonexpansiveness_violation_count"] = violation_count
            row["l2_nonexpansiveness_violation_frequency"] = float(
                violation_count / replicates
            )
            row["l2_nonexpansiveness_verified"] = violation_count == 0
            row["completed_without_abort"] = True
            row["completion_semantics"] = (
                "the complete bootstrap/refit run finished; any nonfinite fit, "
                "linear-algebra error, materially negative variance, or L2 "
                "nonexpansiveness violation above tolerance aborts before output"
            )
            projection_rows.append(row)

    instance_dir = output_dir / spec.key / scale_mode
    instance_dir.mkdir(parents=True, exist_ok=True)
    coefficient_rows = _coefficient_rows(config, fit_lambdas, fixed)
    plugin_rows = _plugin_variance_rows(
        config=config,
        rates=rates,
        reset=reset,
        reset_extension=data["reset_extension"],
        fixed=fixed,
        replicates=replicates,
    )
    plugin_lookup = {
        (int(round(float(row["rate"]) * 1.0e12)), str(row["method"])): row
        for row in plugin_rows
    }
    for row in plugin_rows:
        row["monte_carlo_diagnostic_applicable"] = False
        row["monte_carlo_diagnostic_available"] = False
        row["monte_carlo_diagnostic_interpretation"] = (
            "analytic multinomial plug-in trace; no bootstrap Monte Carlo error"
        )
    for row in empirical_variance_rows:
        method = str(row["method"])
        row["monte_carlo_diagnostic_applicable"] = method in {
            "linear_zne",
            "quadratic_zne",
        }
        if not row["monte_carlo_diagnostic_applicable"]:
            row["monte_carlo_diagnostic_available"] = False
            row["monte_carlo_diagnostic_interpretation"] = (
                "no fixed-coefficient plug-in target exists for a replicate-wise "
                "selected nonlinear estimator"
            )
            continue
        rate_key = int(round(float(row["rate"]) * 1.0e12))
        plugin_row = plugin_lookup[(rate_key, method)]
        rate_matches = np.flatnonzero(
            np.isclose(rates, float(row["rate"]), rtol=0.0, atol=5.0e-15)
        )
        if rate_matches.size != 1:
            raise RuntimeError(f"cannot locate variance diagnostic rate {row['rate']}")
        rate_i = int(rate_matches[0])
        route_i = route_index[method]
        block_values = preprojection_block_unbiased_variance_traces[
            rate_i, route_i
        ]
        block_values = block_values[np.isfinite(block_values)]
        expected_block_count = int(
            math.ceil(replicates / VARIANCE_MC_BLOCK_SIZE)
        )
        if block_values.size != expected_block_count:
            raise RuntimeError(
                "variance diagnostic block-count invariant failed: "
                f"observed={block_values.size}, expected={expected_block_count}"
            )
        if block_values.size < 2:
            mcse = math.nan
        else:
            mcse = float(
                np.std(block_values, ddof=1) / math.sqrt(block_values.size)
            )
        diagnostic_available = bool(math.isfinite(mcse) and mcse > 0.0)
        if block_values.size >= 2 and not diagnostic_available:
            raise FloatingPointError(
                "multi-block variance diagnostic requires a finite positive "
                f"Monte Carlo standard error; observed {mcse!r}"
            )
        empirical_population = float(row["preprojection_covariance_trace"])
        empirical_unbiased = empirical_population * replicates / (replicates - 1)
        plugin_trace = float(plugin_row["preprojection_covariance_trace"])
        difference = empirical_unbiased - plugin_trace
        standardized = (
            difference / mcse if math.isfinite(mcse) and mcse > 0.0 else math.nan
        )
        row.update(
            {
                "monte_carlo_diagnostic_available": diagnostic_available,
                "empirical_unbiased_covariance_trace": empirical_unbiased,
                "plugin_reference_trace": plugin_trace,
                "empirical_minus_plugin_trace": difference,
                "monte_carlo_block_size": VARIANCE_MC_BLOCK_SIZE,
                "monte_carlo_block_count": int(block_values.size),
                "monte_carlo_standard_error": mcse,
                "monte_carlo_standardized_difference": standardized,
                "monte_carlo_warning_threshold_abs_z": (
                    PLUGIN_EMPIRICAL_MC_Z_WARNING
                ),
                "monte_carlo_diagnostic_interpretation": (
                    (
                        "the fixed-model plug-in trace is the conditional covariance "
                        "target; the finite-B empirical trace is compared after "
                        "B/(B-1) correction using the standard error across "
                        "independent 500-replicate block trace estimates; this is a "
                        "diagnostic, not an equality requirement"
                    )
                    if diagnostic_available
                    else (
                        "the fixed-model plug-in trace and finite-B empirical trace "
                        "are reported, but a Monte Carlo standard error requires at "
                        "least two independent 500-replicate blocks; this is a "
                        "diagnostic, not an equality requirement"
                    )
                ),
            }
        )
    variance_rows = plugin_rows + empirical_variance_rows
    _write_csv(instance_dir / "all_route_curves.csv", curve_rows)
    _write_csv(
        instance_dir / "main_160k_curves.csv",
        [row for row in curve_rows if row["route"] in MAIN_ROUTES],
    )
    _write_csv(instance_dir / "paired_differences.csv", paired_rows)
    _write_csv(instance_dir / "direct_precision_diagnostic.csv", precision_rows)
    _write_csv(instance_dir / "preprojection_bias_variance.csv", bias_variance_rows)
    _write_csv(instance_dir / "projection_conditioning.csv", projection_rows)
    _write_csv(instance_dir / "variance_amplification.csv", variance_rows)
    _write_csv(instance_dir / "polynomial_coefficients.csv", coefficient_rows)
    _write_csv(instance_dir / "input_manifest.csv", _manifest_rows(spec))
    np.savez_compressed(
        instance_dir / "preprojection_sufficient_statistics.npz",
        schema_version=np.asarray(1, dtype=np.int64),
        rates=rates,
        routes=np.asarray(ROUTES),
        target=target,
        replicates=np.asarray(replicates, dtype=np.int64),
        mean_vectors=preprojection_mean_vectors,
        mean_squared_norms=preprojection_mean_squared_norms,
        centered_sum_squares=preprojection_centered_sum_squares,
        population_variance_traces=preprojection_variance_traces,
        block_unbiased_variance_traces=(
            preprojection_block_unbiased_variance_traces
        ),
        block_counts=preprojection_block_counts,
        variance_mc_block_size=np.asarray(VARIANCE_MC_BLOCK_SIZE, dtype=np.int64),
        variance_negative_tolerance=np.asarray(
            PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE, dtype=np.float64
        ),
        streaming_algorithm=np.asarray("parallel_chan_welford_float64"),
    )
    np.savez_compressed(
        instance_dir / "bootstrap_cache.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        rates=rates,
        nominal_lambdas=data["nominal_lambdas"],
        fit_lambdas=fit_lambdas,
        routes=np.asarray(ROUTES),
        metrics=np.asarray(POST_METRICS),
        postprojection_draws=draws,
        ideal_grouped_peak_margin=np.asarray(ideal_margin),
        zne_methods=np.asarray(ZNE_METHODS),
        preprojection_l2_squared_to_ideal=preprojection_l2_squared_to_ideal,
        postprojection_l2_squared_to_ideal=postprojection_l2_squared_to_ideal,
        post_minus_pre_l2_squared_to_ideal=(
            post_minus_pre_l2_squared_to_ideal
        ),
        dual_gamma_1_zero=dual_gamma_1_zero,
        l2_nonexpansiveness_tolerance=np.asarray(
            L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE, dtype=np.float64
        ),
        batch_size=np.asarray(FROZEN_BATCH_SIZE, dtype=np.int64),
        replicates=np.asarray(replicates, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )

    metadata = {
        "schema_version": 2,
        "analysis": "frozen four-model equal-shot bias-variance revision",
        "instance": spec.key,
        "config": str(data["resolved_config"]),
        "scale_mode": scale_mode,
        "nominal_lambdas": data["nominal_lambdas"],
        "fit_lambdas": fit_lambdas,
        "weight_exponent": 2.0,
        "weight_interpretation": "fixed locality weight, not inverse variance",
        "rates": rates,
        "bootstrap_replicates": replicates,
        "batch_size": FROZEN_BATCH_SIZE,
        "batch_size_frozen": True,
        "confidence_level": confidence,
        "seed": seed,
        "seed_policy": (
            "SeedSequence([global_seed, crc32(instance), round(rate*1e9)]), "
            "with four child streams for reset scales, reset extension, Raw20k, "
            "and Raw extension"
        ),
        "seed_prefix_contract": (
            "for a fixed software environment and frozen batch_size=25, standalone "
            "B=2,000, 5,000, and 10,000 runs are bit-exact prefixes of B=20,000"
        ),
        "routes": ROUTES,
        "main_routes": MAIN_ROUTES,
        "shots": {
            "direct_20k": ZNE_SHOTS_PER_SCALE,
            "direct_160k": DIRECT_SHOTS,
            "zne_per_scale": ZNE_SHOTS_PER_SCALE,
            "zne_total": DIRECT_SHOTS,
            "reset_extension": DIRECT_EXTENSION_SHOTS,
        },
        "dependency": {
            "reset_20k": "same lambda-one draw used by all four ZNE models",
            "reset_160k": "shared 20k reset draw plus independent 140k extension",
            "raw_20k": "verified preserved first-20k Raw block",
            "raw_160k": "shared Raw20k diagnostic draw plus independent 140k Raw extension",
            "zne_models": "same eight-scale draws within every replicate",
        },
        "nonlinear_selection_policy": (
            "single- and dual-exponential candidate selection is repeated inside "
            "every bootstrap replicate before simplex projection"
        ),
        "inference_scope": (
            "conditional finite-shot bootstrap over the saved empirical parents; "
            "not selection-adjusted or an independent trajectory validation"
        ),
        "preprojection_sufficient_statistics": {
            "filename": "preprojection_sufficient_statistics.npz",
            "schema_version": 1,
            "streaming_algorithm": "parallel_chan_welford_float64",
            "variance_negative_tolerance": (
                PREPROJECTION_VARIANCE_NEGATIVE_TOLERANCE
            ),
            "monte_carlo_block_size": VARIANCE_MC_BLOCK_SIZE,
        },
        "projection_l2_nonexpansiveness": {
            "tolerance": L2_PROJECTION_NONEXPANSIVENESS_TOLERANCE,
            "required_violation_count": 0,
            "reference": "P_ideal",
        },
        "variance_plugin_empirical_diagnostic": {
            "warning_threshold_abs_z": PLUGIN_EMPIRICAL_MC_Z_WARNING,
            "interpretation": (
                "Monte Carlo diagnostic only: the fixed-model plug-in conditional "
                "trace and finite-B empirical covariance trace are distinct "
                "estimands and are not required to be equal"
            ),
        },
        "completion_semantics": (
            "output is written only after the complete bootstrap/refit run finishes "
            "without a nonfinite fit, linear-algebra abort, materially negative "
            "variance, or L2 nonexpansiveness violation above tolerance"
        ),
        "new_quantum_simulation": False,
    }
    with (instance_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metadata), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {
        "instance": spec.key,
        "output_dir": str(instance_dir),
        "curve_rows": len(curve_rows),
        "paired_rows": len(paired_rows),
        "replicates": replicates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances", nargs="+", choices=tuple(INSTANCE_SPECS), default=tuple(INSTANCE_SPECS)
    )
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=FROZEN_BATCH_SIZE)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--scale-mode", choices=("nominal", "realized"), default="nominal")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replicates <= 1:
        raise SystemExit("replicates must exceed one")
    if args.batch_size != FROZEN_BATCH_SIZE:
        raise SystemExit(
            f"the frozen analysis requires --batch-size {FROZEN_BATCH_SIZE}"
        )
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("confidence must lie in (0,1)")
    requested = None if args.rates is None else np.asarray(args.rates, dtype=np.float64)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key in args.instances:
        summaries.append(
            analyse_instance(
                INSTANCE_SPECS[key],
                output_dir=output,
                replicates=int(args.replicates),
                batch_size=int(args.batch_size),
                confidence=float(args.confidence),
                seed=int(args.seed),
                requested_rates=requested,
                scale_mode=str(args.scale_mode),
            )
        )
    run_label = (
        "all"
        if tuple(args.instances) == tuple(INSTANCE_SPECS)
        else "_".join(str(key) for key in args.instances)
    )
    with (output / f"run_summary_{args.scale_mode}_{run_label}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_value(summaries), handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
