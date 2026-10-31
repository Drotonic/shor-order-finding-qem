"""Estimators and full-outcome performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionMetrics:
    """Metrics for a full phase-register probability distribution.

    ``support_margin`` and the other support-named properties are retained as
    read-only compatibility aliases for earlier analysis scripts.  New output
    files use the main-lobe group terminology.
    """

    tvd: float
    grouped_peak_margin: float
    normalized_grouped_peak_margin: float
    grouped_peak_mass: float
    minimum_group_probability: float
    maximum_outside_group_probability: float
    main_lobe_to_outside_ratio: float

    @property
    def support_margin(self) -> float:
        return self.grouped_peak_margin

    @property
    def ideal_to_spurious_ratio(self) -> float:
        return self.main_lobe_to_outside_ratio

    @property
    def ideal_support_mass(self) -> float:
        return self.grouped_peak_mass

    @property
    def minimum_ideal_probability(self) -> float:
        return self.minimum_group_probability

    @property
    def maximum_spurious_probability(self) -> float:
        return self.maximum_outside_group_probability


def ideal_indices(measurement_bits: int, order: int) -> np.ndarray:
    """Nearest output-bin indices to the ``order`` ideal QPE phases."""

    outcomes = 1 << int(measurement_bits)
    if order <= 0 or order > outcomes:
        raise ValueError("order must lie in [1, 2**measurement_bits]")
    phases = np.arange(order, dtype=np.float64) * (outcomes / float(order))
    indices = np.floor(phases + 0.5).astype(np.int64) % outcomes
    if np.unique(indices).size != order:
        raise ValueError("measurement register is too small to resolve every phase")
    return indices


def main_lobe_groups(
    measurement_bits: int, order: int
) -> tuple[np.ndarray, ...]:
    """Integer QPE outcomes within cyclic distance one of each phase center."""

    outcomes = 1 << int(measurement_bits)
    if order <= 0 or order > outcomes:
        raise ValueError("order must lie in [1, 2**measurement_bits]")
    groups: list[np.ndarray] = []
    for phase_index in range(order):
        quotient, remainder = divmod(outcomes * phase_index, order)
        if remainder == 0:
            group = np.asarray([quotient % outcomes], dtype=np.int64)
        else:
            group = np.asarray(
                [quotient % outcomes, (quotient + 1) % outcomes],
                dtype=np.int64,
            )
        groups.append(group)
    flattened = np.concatenate(groups)
    if np.unique(flattened).size != flattened.size:
        raise ValueError(
            "main-lobe groups overlap; increase the measurement resolution"
        )
    return tuple(groups)


def grouped_peak_statistics_rows(
    probabilities: np.ndarray, groups: tuple[np.ndarray, ...]
) -> dict[str, np.ndarray]:
    """Vectorized grouped-peak statistics for one or more distributions."""

    values = np.asarray(probabilities, dtype=np.float64)
    squeezed = values.ndim == 1
    if squeezed:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("probabilities must have shape (batch, outcomes)")
    if not groups:
        raise ValueError("at least one main-lobe group is required")
    group_masses = np.stack(
        [values[:, np.asarray(group, dtype=np.int64)].sum(axis=1) for group in groups],
        axis=1,
    )
    grouped_indices = np.unique(np.concatenate(groups))
    outside_mask = np.ones(values.shape[1], dtype=bool)
    outside_mask[grouped_indices] = False
    minimum_group = group_masses.min(axis=1)
    maximum_outside = (
        values[:, outside_mask].max(axis=1)
        if np.any(outside_mask)
        else np.zeros(values.shape[0], dtype=np.float64)
    )
    result = {
        "margin": minimum_group - maximum_outside,
        "minimum_group_probability": minimum_group,
        "maximum_outside_group_probability": maximum_outside,
        "grouped_probability_mass": values[:, grouped_indices].sum(axis=1),
    }
    if squeezed:
        return {name: value[0] for name, value in result.items()}
    return result


def ideal_distribution(measurement_bits: int, order: int) -> np.ndarray:
    """Exact ideal QPE distribution averaged over all order eigenphases."""

    outcomes = 1 << int(measurement_bits)
    ideal_indices(measurement_bits, order)
    if outcomes % order == 0:
        result = np.zeros(outcomes, dtype=np.float64)
        result[np.arange(order, dtype=np.int64) * (outcomes // order)] = (
            1.0 / order
        )
        return result

    eigenphases = np.arange(order, dtype=np.float64) / float(order)
    output_phases = np.arange(outcomes, dtype=np.float64) / float(outcomes)
    delta = eigenphases[:, None] - output_phases[None, :]
    # sin(pi*Q*delta)/(Q*sin(pi*delta)) written with np.sinc for the
    # removable delta=0 singularity.
    amplitudes = np.sinc(outcomes * delta) / np.sinc(delta)
    result = np.mean(amplitudes * amplitudes, axis=0)
    result = np.maximum(result, 0.0)
    result /= result.sum()
    return result


def simplex_projection(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection of one vector onto the probability simplex."""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("simplex_projection expects a one-dimensional vector")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("input must be a non-empty finite vector")

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0.0
    rho = int(np.flatnonzero(positive)[-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    # Correct only machine-level summation drift, without changing ordering.
    projected /= projected.sum()
    return projected


def simplex_projection_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim == 1:
        return simplex_projection(values)
    if values.ndim != 2:
        raise ValueError("simplex_projection_rows expects one or two dimensions")
    if values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("input must have non-empty finite rows")

    # Vectorized form of the same Euclidean simplex projection used above.
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    positive = ordered - cumulative / indices[None, :] > 0.0
    rho = positive.sum(axis=1) - 1
    theta = cumulative[np.arange(values.shape[0]), rho] / (rho + 1)
    projected = np.maximum(values - theta[:, None], 0.0)
    projected /= projected.sum(axis=1, keepdims=True)
    return projected


def weighted_linear_intercept_coefficients(lambdas: np.ndarray) -> np.ndarray:
    """Return h such that the fixed WLS zero-noise intercept is h @ y.

    The objective is ``sum_i lambda_i**-2 * (y_i-b0-b1*lambda_i)**2``.
    """

    x_values = np.asarray(lambdas, dtype=np.float64)
    if x_values.ndim != 1 or x_values.size < 2:
        raise ValueError("at least two one-dimensional lambda values are required")
    if not np.all(np.isfinite(x_values)) or np.any(x_values <= 0):
        raise ValueError("lambda values must be finite and positive")
    if np.unique(x_values).size < 2:
        raise ValueError("lambda values must not all be equal")

    # Import locally to keep this public compatibility wrapper while all model
    # construction lives in one implementation.
    from .estimators import polynomial_intercept_coefficients

    return polynomial_intercept_coefficients(x_values, degree=1, ridge=0.0)


def linear_zne(
    distributions: np.ndarray, lambdas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply fixed WLS extrapolation and full-simplex projection."""

    probabilities = np.asarray(distributions, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("distributions must have shape (lambdas, outcomes)")
    coefficients = weighted_linear_intercept_coefficients(lambdas)
    if probabilities.shape[0] != coefficients.size:
        raise ValueError("number of distributions and lambdas differs")
    unconstrained = coefficients @ probabilities
    projected = simplex_projection(unconstrained)
    diagnostics = {
        "normalization_error_before_projection": float(abs(unconstrained.sum() - 1.0)),
        "negative_mass_before_projection": float(-np.minimum(unconstrained, 0).sum()),
        "l2_projection_correction": float(np.linalg.norm(projected - unconstrained)),
        "unconstrained_sum": float(unconstrained.sum()),
    }
    return projected, unconstrained, diagnostics


def distribution_metrics(
    distribution: np.ndarray, measurement_bits: int, order: int
) -> DistributionMetrics:
    probabilities = np.asarray(distribution, dtype=np.float64)
    expected_size = 1 << int(measurement_bits)
    if probabilities.shape != (expected_size,):
        raise ValueError(
            f"distribution has shape {probabilities.shape}, expected ({expected_size},)"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("distribution contains non-finite values")

    groups = main_lobe_groups(measurement_bits, order)
    grouped = grouped_peak_statistics_rows(probabilities, groups)
    minimum_group = float(grouped["minimum_group_probability"])
    maximum_outside = float(grouped["maximum_outside_group_probability"])
    ratio = (
        float("inf")
        if maximum_outside <= 0.0
        else float(minimum_group / maximum_outside)
    )
    target = ideal_distribution(measurement_bits, order)
    ideal_grouped = grouped_peak_statistics_rows(target, groups)
    ideal_margin = float(ideal_grouped["margin"])
    normalized_margin = (
        float(grouped["margin"]) / ideal_margin
        if ideal_margin > 0.0
        else float("nan")
    )
    return DistributionMetrics(
        tvd=float(0.5 * np.abs(probabilities - target).sum()),
        grouped_peak_margin=float(grouped["margin"]),
        normalized_grouped_peak_margin=normalized_margin,
        grouped_peak_mass=float(grouped["grouped_probability_mass"]),
        minimum_group_probability=minimum_group,
        maximum_outside_group_probability=maximum_outside,
        main_lobe_to_outside_ratio=ratio,
    )
