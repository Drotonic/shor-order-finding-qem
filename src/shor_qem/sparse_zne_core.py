"""Isolated saved-data comparison of dense and observed-support ZNE.

The frozen estimator implementation is imported, never modified.  ``A`` is its
dense fit and dense simplex projection.  ``B`` is an algebraically exact
compression, retaining the multiplicity of outcomes absent at every scale.
``C1`` intentionally omits those outcomes from fitting and projection; ``C2``
changes only the projection support.  Every exponential fit retains floor 1/D.
No ideal distribution, order, peak groups, or performance metric enters here.

``analyze_batch`` is the full-output comparison API.  ``analyze_compressed``
accepts already compressed inputs and allocates no dimension-D output until a
returned :class:`CompressedVectors` object's ``materialize`` method is called.
Native ``np.argmax`` is used throughout: near ties are diagnosed, not resolved
by a tolerance or by borrowing the reference estimator's selected candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from shor_qem.bootstrap_analysis import _fit_zne_batch_detailed
from shor_qem.estimators import ExponentialGridFamily, fit_exponential_grid_batch
from shor_qem.metrics import simplex_projection_rows


VARIANTS = ("A", "B", "C1", "C2")


@dataclass(frozen=True)
class ObservedBatch:
    """Ragged observed support represented by padded arrays of width max |U|."""

    values: np.ndarray
    indices: np.ndarray
    valid: np.ndarray
    dimension: int

    @property
    def support_size(self) -> np.ndarray:
        return self.valid.sum(axis=1)

    @property
    def unseen_count(self) -> np.ndarray:
        return self.dimension - self.support_size

    @classmethod
    def from_dense(cls, values: np.ndarray) -> "ObservedBatch":
        values = _validate_dense(values)
        supports = np.any(values > 0.0, axis=1)
        indices = [np.flatnonzero(row) for row in supports]
        observations = [values[b][:, index] for b, index in enumerate(indices)]
        return cls.from_observed(observations, indices, values.shape[2])

    @classmethod
    def from_observed(
        cls,
        observed_values: Sequence[np.ndarray] | np.ndarray,
        observed_indices: Sequence[np.ndarray] | np.ndarray,
        dimension: int,
    ) -> "ObservedBatch":
        original_dimension = dimension
        dimension = int(dimension)
        if dimension != original_dimension:
            raise ValueError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if isinstance(observed_values, np.ndarray) and observed_values.ndim == 2:
            value_rows = [observed_values]
            index_rows = [np.asarray(observed_indices)]
        else:
            value_rows = list(observed_values)
            index_rows = list(observed_indices)
        if not value_rows or len(value_rows) != len(index_rows):
            raise ValueError("one index vector per nonempty batch row is required")
        cleaned: list[tuple[np.ndarray, np.ndarray]] = []
        scales = None
        for row, indices in zip(value_rows, index_rows):
            row = np.asarray(row, dtype=np.float64)
            indices_array = np.asarray(indices)
            if row.ndim != 2 or indices_array.ndim != 1:
                raise ValueError("observed rows must be scales x support, with 1D indices")
            if row.shape[1] != indices_array.size or row.shape[0] == 0:
                raise ValueError("observed values and indices have incompatible shapes")
            if not np.issubdtype(indices_array.dtype, np.integer):
                raise ValueError("observed indices must be integers")
            indices_array = indices_array.astype(np.int64, copy=False)
            if np.any(indices_array < 0) or np.any(indices_array >= dimension):
                raise ValueError("observed index is outside the full outcome space")
            if np.unique(indices_array).size != indices_array.size:
                raise ValueError("observed indices must be distinct")
            if not np.all(np.isfinite(row)) or np.any(row < 0.0):
                raise ValueError("observations must be finite and nonnegative")
            if scales is not None and row.shape[0] != scales:
                raise ValueError("batch rows must share the same scale grid")
            scales = row.shape[0]
            # Recompute U even for compressed resamples: parent-support entries
            # may have disappeared in this bootstrap replicate.
            positive = np.any(row > 0.0, axis=0)
            row = row[:, positive]
            indices_array = indices_array[positive]
            if not indices_array.size:
                raise ValueError("each batch row must have a nonempty observed support")
            order = np.argsort(indices_array)
            cleaned.append((row[:, order], indices_array[order]))
        width = max(indices.size for _, indices in cleaned)
        packed = np.zeros((len(cleaned), int(scales), width), dtype=np.float64)
        packed_indices = np.zeros((len(cleaned), width), dtype=np.int64)
        valid = np.zeros((len(cleaned), width), dtype=bool)
        for b, (row, indices) in enumerate(cleaned):
            count = indices.size
            packed[b, :, :count] = row
            packed_indices[b, :count] = indices
            valid[b, :count] = True
        return cls(packed, packed_indices, valid, dimension)


@dataclass(frozen=True)
class CompressedVectors:
    """One value per observed outcome plus one repeated unseen value per row."""

    support: ObservedBatch
    observed: np.ndarray
    unseen: np.ndarray

    def materialize(self) -> np.ndarray:
        """Explicitly allocate the full outcome-space batch."""
        result = np.broadcast_to(
            self.unseen[:, None], (self.observed.shape[0], self.support.dimension)
        ).copy()
        for b, count in enumerate(self.support.support_size):
            result[b, self.support.indices[b, :count]] = self.observed[b, :count]
        return result

    def total(self) -> np.ndarray:
        return (self.observed * self.support.valid).sum(axis=1) + (
            self.support.unseen_count * self.unseen
        )


def _validate_dense(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or any(size == 0 for size in values.shape):
        raise ValueError("values must have nonempty shape (batch, scales, outcomes)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("values must be finite and nonnegative")
    return values


def multiplicity_simplex_projection(
    observed: np.ndarray,
    unseen: np.ndarray | float,
    unseen_count: np.ndarray | int,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project onto a full simplex without expanding identical coordinates.

    The objective weights the unseen squared correction by its multiplicity:
    sum_j (q_j-x_j)^2 + m (q_0-x_0)^2, with sum_j q_j+m q_0=1.
    Sorting one entry of weight m is equivalent to sorting m equal entries.
    A block is active precisely when its last repeated entry is active.
    Setting m=0 projects onto the observed-support simplex.
    """
    observed = np.asarray(observed, dtype=np.float64)
    squeezed = observed.ndim == 1
    if squeezed:
        observed = observed[None, :]
    if observed.ndim != 2 or observed.shape[1] == 0:
        raise ValueError("observed must have nonempty one- or two-dimensional rows")
    if valid is None:
        valid = np.ones(observed.shape, dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if squeezed and valid.ndim == 1:
            valid = valid[None, :]
    if valid.shape != observed.shape:
        raise ValueError("valid mask shape must match observed values")
    batch = observed.shape[0]
    unseen = np.broadcast_to(np.asarray(unseen, dtype=np.float64), (batch,))
    counts = np.broadcast_to(np.asarray(unseen_count), (batch,))
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("unseen multiplicities must be nonnegative integers")
    if not np.all(np.isfinite(observed[valid])) or not np.all(np.isfinite(unseen)):
        raise ValueError("projection values must be finite")
    if np.any(valid.sum(axis=1) + counts <= 0):
        raise ValueError("projection requires a nonempty outcome space")
    entries = np.concatenate((observed, unseen[:, None]), axis=1)
    weights = np.concatenate((valid.astype(np.float64), counts[:, None]), axis=1)
    order = np.argsort(np.where(weights > 0, entries, -np.inf), axis=1)[:, ::-1]
    ordered = np.take_along_axis(entries, order, axis=1)
    ordered_weights = np.take_along_axis(weights, order, axis=1)
    cumulative_weight = np.cumsum(ordered_weights, axis=1)
    cumulative_value = np.cumsum(ordered * ordered_weights, axis=1) - 1.0
    thresholds = cumulative_value / np.maximum(cumulative_weight, 1.0)
    positive = (ordered_weights > 0) & (ordered - thresholds > 0.0)
    positions = np.arange(entries.shape[1])[None, :]
    last = np.max(np.where(positive, positions, -1), axis=1)
    if np.any(last < 0):
        raise FloatingPointError("simplex active set could not be determined")
    theta = thresholds[np.arange(batch), last]
    projected = np.maximum(observed - theta[:, None], 0.0) * valid
    projected_unseen = np.where(counts > 0, np.maximum(unseen - theta, 0.0), 0.0)
    total = projected.sum(axis=1) + counts * projected_unseen
    projected /= total[:, None]
    projected_unseen /= total
    if squeezed:
        return projected[0], projected_unseen[0]
    return projected, projected_unseen


def _projection_diagnostics(
    before: CompressedVectors,
    after: CompressedVectors,
    selection: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    valid = before.support.valid
    multiplicity = before.support.unseen_count
    dimension = before.support.dimension
    negative_count = ((before.observed < 0) & valid).sum(axis=1) + (
        multiplicity * (before.unseen < 0)
    )
    change = (after.observed - before.observed) * valid
    unseen_change = after.unseen - before.unseen
    l2 = np.sqrt(np.square(change).sum(axis=1) + multiplicity * unseen_change**2)
    diagnostics = {
        "normalization_error_before_projection": np.abs(before.total() - 1.0),
        "negative_mass_before_projection": (
            -np.minimum(before.observed, 0.0).sum(axis=1)
            - multiplicity * np.minimum(before.unseen, 0.0)
        ),
        "negative_entry_count_before_projection": negative_count.astype(float),
        "negative_entry_fraction_before_projection": negative_count / dimension,
        "l1_projection_correction": np.abs(change).sum(axis=1)
        + multiplicity * np.abs(unseen_change),
        "l2_projection_correction": l2,
        "support_only_l1_projection_correction": np.abs(change).sum(axis=1),
        "support_only_l2_projection_correction": np.linalg.norm(change, axis=1),
        "support_only_normalization_error_before_projection": np.abs(
            (before.observed * valid).sum(axis=1) - 1.0
        ),
        "support_only_negative_mass_before_projection": -np.minimum(
            before.observed, 0.0
        ).sum(axis=1),
        "support_only_negative_entry_fraction_before_projection": (
            ((before.observed < 0) & valid).sum(axis=1) / before.support.support_size
        ),
        "projection_active": (
            np.any(np.abs(change) > 1.0e-14, axis=1)
            | ((multiplicity > 0) & (np.abs(unseen_change) > 1.0e-14))
        ).astype(float),
        "support_size_after_projection": ((after.observed > 0) & valid).sum(axis=1)
        + multiplicity * (after.unseen > 0),
        "unseen_value_before_projection": before.unseen.copy(),
        "unseen_value_after_projection": after.unseen.copy(),
        "unseen_mass_before_projection": multiplicity * before.unseen,
        "unseen_mass_after_projection": multiplicity * after.unseen,
    }
    reference_change = (after.observed - before.support.values[:, 0, :]) * valid
    total_change_l2 = np.sqrt(
        np.square(reference_change).sum(axis=1) + multiplicity * after.unseen**2
    )
    diagnostics["projection_relative_to_total_change"] = l2 / np.maximum(
        total_change_l2, np.finfo(np.float64).eps
    )
    if selection is not None:
        for key, value in selection.items():
            if key not in ("selected_parameters", "lower_boundary", "upper_boundary", "any_boundary"):
                diagnostics[key] = np.asarray(value, dtype=np.float64)
        parameters = selection["selected_parameters"]
        if parameters.shape[1] == 1:
            diagnostics["selected_gamma"] = parameters[:, 0]
        else:
            diagnostics["selected_gamma_1"] = parameters[:, 0]
            diagnostics["selected_gamma_2"] = parameters[:, 1]
        for boundary in ("lower", "upper", "any"):
            diagnostics[f"grid_{boundary}_boundary"] = selection[
                f"{boundary}_boundary"
            ].astype(float)
    return diagnostics


def _compressed_analysis(
    support: ObservedBatch,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
    *,
    variants: tuple[str, ...] = ("B", "C1", "C2"),
    c2_dense_before: dict[str, np.ndarray] | None = None,
    c2_dense_selection: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if not variants or set(variants) - {"B", "C1", "C2"}:
        raise ValueError("compressed variants must be a nonempty subset of B,C1,C2")
    values, valid = support.values, support.valid
    floor = 1.0 / support.dimension
    centered_observed = (values - floor) * valid[:, None, :]
    observed_gram = np.einsum(
        "blk,bmk->blm", centered_observed, centered_observed, optimize=True
    )
    full_gram = observed_gram + support.unseen_count[:, None, None] * floor**2
    fit_before: dict[str, dict[str, CompressedVectors]] = {"full": {}, "observed": {}}
    fit_selection: dict[str, dict[str, dict[str, np.ndarray]]] = {"full": {}, "observed": {}}
    for method, coefficients in fixed_coefficients.items():
        estimate = np.einsum("l,blk->bk", coefficients, values, optimize=True) * valid
        vector = CompressedVectors(support, estimate, np.zeros(values.shape[0]))
        fit_before["full"][method] = vector
        fit_before["observed"][method] = vector
    needed_objectives = []
    if "B" in variants or ("C2" in variants and c2_dense_before is None):
        needed_objectives.append(("full", full_gram))
    if "C1" in variants:
        needed_objectives.append(("observed", observed_gram))
    for objective, gram in needed_objectives:
        for method, family in exponential_families.items():
            estimate, selection = fit_exponential_grid_batch(
                values, family, uniform_floor=floor, cross_products=gram
            )
            coefficients = family.intercept_coefficients[selection["selected_index"]]
            unseen = floor * (1.0 - coefficients.sum(axis=1))
            if objective == "observed":
                unseen = np.zeros(values.shape[0])
            # A non-existent block carries no probability even when its formal
            # candidate formula would be nonzero.
            unseen = np.where(support.unseen_count > 0, unseen, 0.0)
            fit_before[objective][method] = CompressedVectors(
                support, estimate * valid, unseen
            )
            fit_selection[objective][method] = selection
    result: dict[str, Any] = {
        "support_size": support.support_size,
        "unseen_count": support.unseen_count,
        "observed_gram": observed_gram,
        "full_gram": full_gram,
        "fitting_floor": floor,
    }
    for variant in variants:
        objective = "observed" if variant == "C1" else "full"
        before = dict(fit_before[objective])
        selection = dict(fit_selection[objective])
        fit_arithmetic = "observed_only" if variant == "C1" else "compressed_full"
        if variant == "C2" and c2_dense_before is not None:
            before = {
                method: _compress_dense_vectors(vector, support)
                for method, vector in c2_dense_before.items()
            }
            selection = dict(c2_dense_selection or {})
            fit_arithmetic = "frozen_dense"
        after: dict[str, CompressedVectors] = {}
        diagnostics: dict[str, dict[str, np.ndarray]] = {}
        unseen_model_values: dict[str, np.ndarray] = {}
        for method in (*fixed_coefficients, *exponential_families):
            vector = before[method]
            model_unseen = np.zeros(values.shape[0], dtype=np.float64)
            if method in exponential_families:
                family = exponential_families[method]
                coefficients = family.intercept_coefficients[
                    selection[method]["selected_index"]
                ]
                model_unseen = floor * (1.0 - coefficients.sum(axis=1))
            unseen_model_values[method] = model_unseen
            projected, projected_unseen = multiplicity_simplex_projection(
                vector.observed,
                vector.unseen,
                support.unseen_count if variant == "B" else 0,
                valid,
            )
            after[method] = CompressedVectors(support, projected, projected_unseen)
            diagnostics[method] = _projection_diagnostics(
                vector, after[method], selection.get(method)
            )
            # C1's native prevector is zero-lifted.  Also show the correction
            # from the hypothetical full model, without feeding that omitted
            # block into fitting or projection.
            full_model_diagnostics = _projection_diagnostics(
                CompressedVectors(support, vector.observed, model_unseen),
                after[method], None,
            )
            for key in (
                "normalization_error_before_projection",
                "negative_mass_before_projection",
                "negative_entry_count_before_projection",
                "negative_entry_fraction_before_projection",
                "l1_projection_correction", "l2_projection_correction",
                "projection_relative_to_total_change",
            ):
                diagnostics[method][f"full_model_{key}"] = full_model_diagnostics[key]
        result[variant] = {
            "before": before,
            "after": after,
            "selection": selection,
            "diagnostics": diagnostics,
            "unseen_model_value": unseen_model_values,
            "fit_arithmetic": fit_arithmetic,
            "fitting_floor": floor,
        }
    return result


def _compress_dense_vectors(values: np.ndarray, support: ObservedBatch) -> CompressedVectors:
    observed = np.take_along_axis(values, support.indices, axis=1) * support.valid
    unseen = np.zeros(values.shape[0], dtype=np.float64)
    for b, count in enumerate(support.support_size):
        if count < support.dimension:
            # Sorted U: the first absent index is the first departure from
            # 0,1,...; this avoids an additional dimension-D mask.
            indices = support.indices[b, :count]
            different = np.flatnonzero(indices != np.arange(count))
            first_unseen = int(different[0]) if different.size else int(count)
            unseen[b] = values[b, first_unseen]
    return CompressedVectors(support, observed, unseen)


def analyze_compressed(
    observed_values: Sequence[np.ndarray] | np.ndarray,
    observed_indices: Sequence[np.ndarray] | np.ndarray,
    dimension: int,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
    *,
    variants: tuple[str, ...] = ("B", "C1", "C2"),
) -> dict[str, Any]:
    """Infer from observed coordinates without allocating full-D outputs.

    Values may be a single scales x support array or a sequence of differently
    sized arrays.  Indices identify actual outcomes, not peak/ideal support.
    C2 here uses the full *compressed* fitting objective, explicitly labeled
    ``compressed_full``.  Use ``analyze_batch`` for C2 with frozen dense floating
    arithmetic, and for a native-argmax comparison against A.
    """
    support = ObservedBatch.from_observed(observed_values, observed_indices, dimension)
    return _compressed_analysis(
        support, fixed_coefficients, exponential_families, variants=variants
    )


def infer_model(
    method: str,
    variant: str,
    *,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
    dense: np.ndarray | None = None,
    support: ObservedBatch | None = None,
) -> dict[str, Any]:
    """Inference-only, single-model path for honest per-variant benchmarks.

    No other model, diagnostic, or full-D output is computed.  A and C2 use
    frozen dense fitting arithmetic; B/C1 operate only on precompressed support.
    A returns arrays, the other variants return ``CompressedVectors``.  The
    C2 dense intermediate is intrinsic to its advertised dense fitting route.
    Input loading/compression and optional materialization are separate costs.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    polynomial = method in fixed_coefficients
    if not polynomial and method not in exponential_families:
        raise ValueError(f"unknown method: {method}")
    selection: dict[str, np.ndarray] = {}
    if variant in ("A", "C2"):
        if dense is None:
            raise ValueError("A and C2 require full dense observations")
        values = _validate_dense(dense)
        if polynomial:
            before_dense = np.einsum(
                "l,blk->bk", fixed_coefficients[method], values, optimize=True
            )
        else:
            before_dense, selection = fit_exponential_grid_batch(
                values, exponential_families[method], uniform_floor=1.0 / values.shape[2]
            )
        if variant == "A":
            return {"before": before_dense, "after": simplex_projection_rows(before_dense),
                    "selection": selection, "fit_arithmetic": "frozen_dense"}
        if support is None:
            support = ObservedBatch.from_dense(values)
        before = _compress_dense_vectors(before_dense, support)
    else:
        if support is None:
            raise ValueError("B and C1 require an ObservedBatch")
        values = support.values
        floor = 1.0 / support.dimension
        unseen = np.zeros(values.shape[0], dtype=np.float64)
        if polynomial:
            observed = np.einsum(
                "l,blk->bk", fixed_coefficients[method], values, optimize=True
            )
        else:
            centered = (values - floor) * support.valid[:, None, :]
            gram = np.einsum("blk,bmk->blm", centered, centered, optimize=True)
            if variant == "B":
                gram += support.unseen_count[:, None, None] * floor**2
            family = exponential_families[method]
            observed, selection = fit_exponential_grid_batch(
                values, family, uniform_floor=floor, cross_products=gram
            )
            if variant == "B":
                coefficients = family.intercept_coefficients[selection["selected_index"]]
                unseen = floor * (1.0 - coefficients.sum(axis=1))
                unseen = np.where(support.unseen_count > 0, unseen, 0.0)
        before = CompressedVectors(support, observed * support.valid, unseen)
    projected, projected_unseen = multiplicity_simplex_projection(
        before.observed, before.unseen,
        support.unseen_count if variant == "B" else 0, support.valid,
    )
    return {
        "before": before,
        "after": CompressedVectors(support, projected, projected_unseen),
        "selection": selection,
        "fit_arithmetic": "frozen_dense" if variant == "C2" else (
            "compressed_full" if variant == "B" else "observed_only"
        ),
    }


def analyze_batch(
    values: np.ndarray,
    fixed_coefficients: dict[str, np.ndarray],
    exponential_families: dict[str, ExponentialGridFamily],
    *,
    include_a: bool = True,
    a_result: tuple | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare A/B/C1/C2 on one batch, recomputing observed U for each row.

    Variant records have ``before``, ``after``, ``selection``, and ``diagnostics``
    dictionaries keyed by method; before/after arrays have shape (batch,D).
    ``a_result`` may be the frozen helper's four-tuple or a prior A record.
    ``include_a=False`` omits the A output but does not replace C2's dense fit
    with B's fit.  Native selected-index disagreements are always exposed.
    """
    values = _validate_dense(values)
    support = ObservedBatch.from_dense(values)
    if a_result is None:
        projected, diagnostics, unconstrained, selection = _fit_zne_batch_detailed(
            values,
            fixed_coefficients=fixed_coefficients,
            exponential_families=exponential_families,
        )
        dense = {
            "before": unconstrained, "after": projected,
            "selection": selection, "diagnostics": diagnostics,
            "fit_arithmetic": "frozen_dense", "fitting_floor": 1.0 / values.shape[2],
        }
    elif isinstance(a_result, dict):
        dense = a_result
    else:
        projected, diagnostics, unconstrained, selection = a_result
        dense = {
            "before": unconstrained, "after": projected,
            "selection": selection, "diagnostics": diagnostics,
            "fit_arithmetic": "frozen_dense", "fitting_floor": 1.0 / values.shape[2],
        }
    compressed = _compressed_analysis(
        support, fixed_coefficients, exponential_families,
        c2_dense_before=dense["before"], c2_dense_selection=dense["selection"],
    )
    result: dict[str, Any] = {
        "support": np.any(values > 0.0, axis=1),
        "support_size": support.support_size,
        "unseen_count": support.unseen_count,
        "fitting_floor": 1.0 / values.shape[2],
    }
    if include_a:
        # Add comparable diagnostics without altering frozen outputs/diagnostics.
        augmented_diagnostics = {}
        for method in dense["before"]:
            extra = _projection_diagnostics(
                _compress_dense_vectors(dense["before"][method], support),
                _compress_dense_vectors(dense["after"][method], support),
                dense["selection"].get(method),
            )
            extra.update(dense["diagnostics"][method])
            for key in (
                "normalization_error_before_projection",
                "negative_mass_before_projection",
                "negative_entry_count_before_projection",
                "negative_entry_fraction_before_projection",
                "l1_projection_correction", "l2_projection_correction",
                "projection_relative_to_total_change",
            ):
                extra[f"full_model_{key}"] = extra[key]
            augmented_diagnostics[method] = extra
        result["A"] = dict(dense, diagnostics=augmented_diagnostics)
        result["A"]["unseen_model_value"] = compressed["C2"]["unseen_model_value"]
    for variant in ("B", "C1", "C2"):
        record = compressed[variant]
        result[variant] = {
            **record,
            "before": dense["before"] if variant == "C2" else {
                method: vector.materialize() for method, vector in record["before"].items()
            },
            "after": {method: vector.materialize() for method, vector in record["after"].items()},
        }
    # C2's preprojection vector is literally the frozen dense result.  The
    # compressed representation above is used only for its constrained fit.
    result["C2"]["before"] = dense["before"]
    centered = values - result["fitting_floor"]
    dense_gram = np.einsum("blk,bmk->blm", centered, centered, optimize=True)
    gram_delta = compressed["full_gram"] - dense_gram
    audit: dict[str, Any] = {
        "gram_max_abs_difference": np.max(np.abs(gram_delta), axis=(1, 2)),
    }
    for method, family in exponential_families.items():
        a_selection = dense["selection"][method]
        b_selection = result["B"]["selection"][method]
        # A useful absolute roundoff envelope without allocating all scores.
        # This is an upper bound on |<H, Gram_B-Gram_A>|, not a tie rule.
        score_error_bound = np.abs(gram_delta).reshape(values.shape[0], -1) @ (
            np.max(np.abs(family.score_matrices_flat), axis=0)
        )
        audit[method] = {
            "a_selected_index": a_selection["selected_index"],
            "b_selected_index": b_selection["selected_index"],
            "selected_index_equal": a_selection["selected_index"] == b_selection["selected_index"],
            "a_runner_up_score_gap": a_selection["runner_up_score_gap"],
            "b_runner_up_score_gap": b_selection["runner_up_score_gap"],
            "gram_score_error_bound": score_error_bound,
            "max_abs_before_difference": np.max(
                np.abs(result["B"]["before"][method] - dense["before"][method]), axis=1
            ),
            "max_abs_after_difference": np.max(
                np.abs(result["B"]["after"][method] - dense["after"][method]), axis=1
            ),
        }
    result["audit"] = audit
    return result
