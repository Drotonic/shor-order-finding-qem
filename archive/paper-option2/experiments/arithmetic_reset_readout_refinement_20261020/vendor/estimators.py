"""Distribution-level ZNE estimators used by the bootstrap analysis.

The nonlinear model definitions intentionally follow the historical analysis:
the asymptotic floor is fixed to the uniform distribution and one shared set of
decay rates is selected for the complete distribution by minimizing the summed
weighted residual error.  Unlike the historical notebooks, this module always
operates on the complete outcome space and never applies a count threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_RIDGE = 1.0e-8
SINGLE_EXPONENTIAL_GRID = np.linspace(0.01, 10.0, 200)
DUAL_EXPONENTIAL_GRID = np.linspace(0.0, 10.0, 200)


@dataclass(frozen=True)
class ExponentialGridFamily:
    """Precomputed linear algebra for a shared-rate exponential grid search."""

    parameters: np.ndarray
    intercept_coefficients: np.ndarray
    score_matrices_flat: np.ndarray
    lower_boundary: np.ndarray
    upper_boundary: np.ndarray
    objective_weights: np.ndarray
    intercept_l1_norms: np.ndarray
    intercept_l2_norms_squared: np.ndarray
    normal_condition_numbers: np.ndarray
    minimum_rate_separations: np.ndarray
    ridge: float
    weight_exponent: float

    @property
    def components(self) -> int:
        return int(self.parameters.shape[1])

    @property
    def candidate_count(self) -> int:
        return int(self.parameters.shape[0])


def objective_weights(
    lambdas: np.ndarray, *, exponent: float = 2.0
) -> np.ndarray:
    """Normalized WLS weights proportional to ``lambda**(-exponent)``.

    The default ``exponent=2`` preserves the frozen analysis.  Exposing the
    exponent permits explicitly labelled auxiliary checks without changing
    the four-model comparison used in the manuscript.
    """

    values = np.asarray(lambdas, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two one-dimensional lambda values are required")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("lambda values must be finite and positive")
    exponent = float(exponent)
    if not np.isfinite(exponent):
        raise ValueError("weight exponent must be finite")
    weights = np.power(values, -exponent)
    return weights / weights.mean()


def polynomial_intercept_coefficients(
    lambdas: np.ndarray,
    degree: int,
    *,
    ridge: float = 0.0,
    weight_exponent: float = 2.0,
) -> np.ndarray:
    """Return the fixed WLS coefficient vector for a polynomial intercept."""

    values = np.asarray(lambdas, dtype=np.float64)
    if degree < 0 or values.size < degree + 1:
        raise ValueError("the polynomial degree is incompatible with the lambda grid")
    weights = objective_weights(values, exponent=weight_exponent)
    design = np.vander(values, N=degree + 1, increasing=True)
    normal = design.T @ (weights[:, None] * design)
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if ridge:
        normal = normal + float(ridge) * np.eye(degree + 1)
    selector = np.zeros(degree + 1, dtype=np.float64)
    selector[0] = 1.0
    return selector @ np.linalg.solve(normal, design.T * weights)


def _parameter_combinations(
    components: int,
    single_grid: np.ndarray,
    dual_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if components == 1:
        grid = np.asarray(single_grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError("single-exponential grid must be non-empty and 1D")
        parameters = grid[:, None]
        lower = np.isclose(grid, grid.min())
        upper = np.isclose(grid, grid.max())
        return parameters, lower, upper
    if components == 2:
        grid = np.asarray(dual_grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("dual-exponential grid must have at least two values")
        pairs = [(first, second) for first in grid for second in grid if second > first]
        parameters = np.asarray(pairs, dtype=np.float64)
        lower = np.isclose(parameters[:, 0], grid.min())
        upper = np.isclose(parameters[:, 1], grid.max())
        return parameters, lower, upper
    raise ValueError("only one- and two-component exponential models are supported")


def build_exponential_grid_family(
    lambdas: np.ndarray,
    components: int,
    *,
    single_grid: np.ndarray = SINGLE_EXPONENTIAL_GRID,
    dual_grid: np.ndarray = DUAL_EXPONENTIAL_GRID,
    ridge: float = DEFAULT_RIDGE,
    weight_exponent: float = 2.0,
) -> ExponentialGridFamily:
    """Precompute candidate scores and zero-noise coefficient vectors.

    For each candidate rate vector, the amplitudes are outcome-specific linear
    coefficients.  The rate vector itself is shared by all outcomes and is
    chosen by the summed penalized WLS residual, matching the notebook logic.
    """

    values = np.asarray(lambdas, dtype=np.float64)
    weights = objective_weights(values, exponent=weight_exponent)
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    parameters, lower, upper = _parameter_combinations(
        components, single_grid, dual_grid
    )
    candidate_count = parameters.shape[0]
    intercepts = np.empty((candidate_count, values.size), dtype=np.float64)
    score_matrices = np.empty(
        (candidate_count, values.size, values.size), dtype=np.float64
    )
    condition_numbers = np.empty(candidate_count, dtype=np.float64)
    minimum_separations = np.full(candidate_count, np.nan, dtype=np.float64)
    zero_noise_basis = np.ones(components, dtype=np.float64)

    for candidate_i, rates in enumerate(parameters):
        design = np.exp(-values[:, None] * rates[None, :])
        weighted_design = weights[:, None] * design
        normal = design.T @ weighted_design
        if ridge:
            normal = normal + float(ridge) * np.eye(components)
        condition_numbers[candidate_i] = np.linalg.cond(normal)
        if components == 2:
            minimum_separations[candidate_i] = rates[1] - rates[0]
        inverse = np.linalg.inv(normal)
        amplitude_operator = inverse @ weighted_design.T
        intercepts[candidate_i] = zero_noise_basis @ amplitude_operator
        # The minimized objective differs across candidates only by
        # -trace(H Y Y^T), where H is this symmetric score matrix.
        score_matrices[candidate_i] = (
            weighted_design @ inverse @ weighted_design.T
        )

    return ExponentialGridFamily(
        parameters=parameters,
        intercept_coefficients=intercepts,
        score_matrices_flat=score_matrices.reshape(candidate_count, -1),
        lower_boundary=lower,
        upper_boundary=upper,
        objective_weights=weights,
        intercept_l1_norms=np.abs(intercepts).sum(axis=1),
        intercept_l2_norms_squared=np.square(intercepts).sum(axis=1),
        normal_condition_numbers=condition_numbers,
        minimum_rate_separations=minimum_separations,
        ridge=float(ridge),
        weight_exponent=float(weight_exponent),
    )


def fit_exponential_grid_batch(
    distributions: np.ndarray,
    family: ExponentialGridFamily,
    *,
    uniform_floor: float | None = None,
    cross_products: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit one exponential family to a batch of full distributions.

    Parameters
    ----------
    distributions:
        Array with shape ``(batch, lambdas, outcomes)`` or
        ``(lambdas, outcomes)``.
    family:
        Precomputed candidate family for the same lambda grid.
    uniform_floor:
        Fixed asymptote ``A``.  The default is ``1 / outcomes``.

    Returns
    -------
    unconstrained:
        Zero-noise vectors before simplex projection, shape
        ``(batch, outcomes)``.
    diagnostics:
        Selected grid indices/rates and endpoint-hit flags for every batch row.
    """

    values = np.asarray(distributions, dtype=np.float64)
    squeezed = values.ndim == 2
    if squeezed:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError("distributions must have shape (batch, lambdas, outcomes)")
    if values.shape[1] ** 2 != family.score_matrices_flat.shape[1]:
        raise ValueError("the distribution and family lambda dimensions differ")
    if not np.all(np.isfinite(values)):
        raise ValueError("distributions contain non-finite values")

    floor = 1.0 / values.shape[2] if uniform_floor is None else float(uniform_floor)
    centered = values - floor
    if cross_products is None:
        gram = np.einsum("blk,bmk->blm", centered, centered, optimize=True)
    else:
        gram = np.asarray(cross_products, dtype=np.float64)
        expected = (values.shape[0], values.shape[1], values.shape[1])
        if gram.shape != expected:
            raise ValueError(f"cross_products has shape {gram.shape}, expected {expected}")
    scores = gram.reshape(values.shape[0], -1) @ family.score_matrices_flat.T
    selected_indices = np.argmax(scores, axis=1)
    selected_coefficients = family.intercept_coefficients[selected_indices]
    unconstrained = floor + np.einsum(
        "bl,blk->bk", selected_coefficients, centered, optimize=True
    )
    selected_parameters = family.parameters[selected_indices]
    selected_scores = scores[np.arange(values.shape[0]), selected_indices]
    if family.candidate_count > 1:
        runner_up_scores = np.partition(scores, -2, axis=1)[:, -2]
    else:
        runner_up_scores = np.full(values.shape[0], -np.inf, dtype=np.float64)
    score_gaps = selected_scores - runner_up_scores
    tie_tolerances = 1.0e-12 * np.maximum(1.0, np.abs(selected_scores))
    base_objectives = np.einsum(
        "l,bll->b", family.objective_weights, gram, optimize=True
    )
    diagnostics = {
        "selected_index": selected_indices,
        "selected_parameters": selected_parameters,
        "selected_intercept_l1_norm": family.intercept_l1_norms[
            selected_indices
        ],
        "selected_intercept_l2_norm_squared": family.intercept_l2_norms_squared[
            selected_indices
        ],
        "selected_normal_condition_number": family.normal_condition_numbers[
            selected_indices
        ],
        "selected_minimum_rate_separation": family.minimum_rate_separations[
            selected_indices
        ],
        "runner_up_score_gap": score_gaps,
        "near_tie_candidate_count": np.sum(
            selected_scores[:, None] - scores <= tie_tolerances[:, None], axis=1
        ).astype(np.int64),
        "penalized_weighted_residual": np.maximum(
            base_objectives - selected_scores, 0.0
        ),
        "lower_boundary": family.lower_boundary[selected_indices],
        "upper_boundary": family.upper_boundary[selected_indices],
        "any_boundary": (
            family.lower_boundary[selected_indices]
            | family.upper_boundary[selected_indices]
        ),
    }
    if squeezed:
        unconstrained = unconstrained[0]
        diagnostics = {key: value[0] for key, value in diagnostics.items()}
    return unconstrained, diagnostics
