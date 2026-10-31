"""Portable CPU-only C1 reconstruction, fresh paired bootstrap, and S6 decoder.

Counts NPZ schema (no pickle): ``reset_scale_counts`` (8,D),
``reset_extension_counts`` (D,), ``raw_initial_counts`` (D,),
``raw_extension_counts`` (D,), and ``effective_lambdas`` (8,). Each scale and
the raw initial block contain 20,000 shots; each extension contains 140,000.
The first reset scale is lambda=1. Merged 160k baselines cannot substitute for
their two parent blocks. D must be a power of two.

The estimator is frozen: C1 support is the union of the eight reset scales,
beta=2 mean-one weights, all ridges zero, exponential offset 1/D, and the
original 200 single-rate / 19,900 ordered dual-rate candidates. Order is used
only for evaluation, never for fitting or support selection. No simulator,
quantum backend, GPU, scheduling, or external data discovery is used.

``bootstrap`` is an explicitly requested fresh conditional bootstrap, NOT a
bitwise replay of the historical archived bootstrap. Reproducibility requires
the same seed, batch size, NumPy/BLAS environment, and input arrays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .estimators import (
    ExponentialGridFamily,
    build_exponential_grid_family,
    objective_weights,
    polynomial_intercept_coefficients,
)
from .metrics import grouped_peak_statistics_rows, ideal_distribution, main_lobe_groups
from .sparse_zne_core import analyze_compressed


METHODS = (
    "linear_zne", "quadratic_zne", "single_exponential_zne", "dual_exponential_zne"
)
DIRECT_ROUTES = ("raw_20k", "raw_160k", "reset_20k", "reset_160k")
METRICS = ("tvd", "grouped_peak_margin", "normalized_grouped_peak_margin")
SCALE_SHOTS = 20_000
EXTENSION_SHOTS = 140_000
DIRECT_SHOTS = SCALE_SHOTS + EXTENSION_SHOTS
FRESH_BOOTSTRAP_LABEL = "portable fresh bootstrap NOT bitwise historical archived replay"
COUNT_KEYS = (
    "reset_scale_counts", "reset_extension_counts", "raw_initial_counts",
    "raw_extension_counts", "effective_lambdas",
)


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result != value or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _counts(value: Any, name: str, shape: tuple[int, ...], total: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite integer counts")
    if np.any(array < 0) or np.any(array > total) or np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain nonnegative integer counts <= {total}")
    result = array.astype(np.int64, copy=True)
    if not np.all(result.sum(axis=-1, dtype=np.int64) == total):
        raise ValueError(f"{name} must total exactly {total} shots per parent")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class CountParents:
    """Validated distinct parent blocks; construction never splits merged data."""

    reset_scale_counts: np.ndarray
    reset_extension_counts: np.ndarray
    raw_initial_counts: np.ndarray
    raw_extension_counts: np.ndarray
    effective_lambdas: np.ndarray

    def __post_init__(self) -> None:
        grid = np.asarray(self.reset_scale_counts)
        if grid.ndim != 2 or grid.shape[0] != 8:
            raise ValueError("reset_scale_counts must have shape (8,D)")
        dimension = grid.shape[1]
        if dimension < 2 or dimension & (dimension - 1):
            raise ValueError("D must be a power of two >= 2")
        for name, shape, total in (
            ("reset_scale_counts", (8, dimension), SCALE_SHOTS),
            ("reset_extension_counts", (dimension,), EXTENSION_SHOTS),
            ("raw_initial_counts", (dimension,), SCALE_SHOTS),
            ("raw_extension_counts", (dimension,), EXTENSION_SHOTS),
        ):
            object.__setattr__(self, name, _counts(getattr(self, name), name, shape, total))
        original_lambdas = np.asarray(self.effective_lambdas)
        if original_lambdas.dtype.kind not in "iuf":
            raise ValueError("effective_lambdas must be real numbers")
        lambdas = original_lambdas.astype(np.float64, copy=True)
        if (lambdas.shape != (8,) or not np.all(np.isfinite(lambdas))
                or not np.isclose(lambdas[0], 1.0, rtol=0, atol=1e-12)
                or np.any(np.diff(lambdas) <= 0)):
            raise ValueError("effective_lambdas must be eight finite increasing scales starting at 1")
        lambdas.flags.writeable = False
        object.__setattr__(self, "effective_lambdas", lambdas)

    @property
    def dimension(self) -> int:
        return int(self.reset_scale_counts.shape[1])

    @property
    def bits(self) -> int:
        return self.dimension.bit_length() - 1

    @property
    def reset_parents(self) -> np.ndarray:
        return self.reset_scale_counts.astype(np.float64) / SCALE_SHOTS

    def direct_probabilities(self) -> dict[str, np.ndarray]:
        return {
            "raw_20k": self.raw_initial_counts / SCALE_SHOTS,
            "raw_160k": (self.raw_initial_counts + self.raw_extension_counts) / DIRECT_SHOTS,
            "reset_20k": self.reset_scale_counts[0] / SCALE_SHOTS,
            "reset_160k": (self.reset_scale_counts[0] + self.reset_extension_counts) / DIRECT_SHOTS,
        }


def load_counts(path: str | Path) -> CountParents:
    """Read the strict five-key schema; unrelated metadata keys are ignored."""
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = set(COUNT_KEYS) - set(archive.files)
        if missing:
            raise ValueError("missing separate parent arrays: " + ", ".join(sorted(missing)))
        return CountParents(**{name: archive[name] for name in COUNT_KEYS})


@lru_cache(maxsize=4)
def _cached_suite(lambdas: tuple[float, ...]) -> tuple[
    dict[str, np.ndarray], dict[str, ExponentialGridFamily]
]:
    values = np.asarray(lambdas)
    fixed = {
        method: polynomial_intercept_coefficients(values, degree, ridge=0.0, weight_exponent=2.0)
        for method, degree in zip(METHODS[:2], (1, 2))
    }
    families = {
        method: build_exponential_grid_family(
            values, components, ridge=0.0, weight_exponent=2.0,
            single_grid=np.linspace(0.01, 10.0, 200), dual_grid=np.linspace(0.0, 10.0, 200),
        )
        for method, components in zip(METHODS[2:], (1, 2))
    }
    return fixed, families


def build_suite(lambdas: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, ExponentialGridFamily]]:
    """Build the unchanged beta=2, all-zero-ridge four-model suite."""
    return _cached_suite(tuple(np.asarray(lambdas, dtype=np.float64)))


def _reference(bits: int, order: int) -> tuple[np.ndarray, tuple[np.ndarray, ...], float]:
    order = _integer(order, "order")
    target = ideal_distribution(bits, order)
    groups = main_lobe_groups(bits, order)
    denominator = float(grouped_peak_statistics_rows(target, groups)["margin"])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("the ideal grouped-peak margin must be positive")
    return target, groups, denominator


def _metrics(values: np.ndarray, reference: tuple) -> np.ndarray:
    target, groups, denominator = reference
    grouped = grouped_peak_statistics_rows(values, groups)
    return np.stack((
        0.5 * np.abs(values - target[None, :]).sum(axis=1),
        grouped["margin"], grouped["margin"] / denominator,
    ))


def evaluate_batch(
    grid: np.ndarray, direct: Mapping[str, np.ndarray], suite: tuple,
    reference: tuple, *, keep_vectors: bool = False,
) -> dict[str, np.ndarray]:
    """Refit every model on each row's actual eight-reset-scale observed union."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[1] != 8:
        raise ValueError("grid must have shape (batch,8,D)")
    support = np.any(grid > 0, axis=1)
    indices = [np.flatnonzero(row) for row in support]
    observed = [grid[b][:, index] for b, index in enumerate(indices)]
    result = analyze_compressed(observed, indices, grid.shape[-1], *suite, variants=("C1",))["C1"]
    scalar = []
    before_vectors, after_vectors = [], []
    selected = np.full((len(METHODS), len(grid)), -1, dtype=np.int64)
    parameters = np.full((len(METHODS), 2, len(grid)), np.nan)
    for index, method in enumerate(METHODS):
        after = result["after"][method].materialize()
        if np.any(after[~support] != 0):
            raise AssertionError("C1 probability escaped the reset-only observed support")
        if np.any(after < 0) or not np.allclose(after.sum(axis=1), 1, rtol=0, atol=1e-11):
            raise FloatingPointError("invalid C1 simplex output")
        scalar.append(_metrics(after, reference))
        if method in result["selection"]:
            selection = result["selection"][method]
            selected[index] = selection["selected_index"]
            parameters[index, :selection["selected_parameters"].shape[1]] = selection["selected_parameters"].T
        if keep_vectors:
            before_vectors.append(result["before"][method].materialize())
            after_vectors.append(after)
    direct_scalar = np.stack([_metrics(np.asarray(direct[route]), reference) for route in DIRECT_ROUTES])
    scalar_array = np.stack(scalar)
    reset = direct_scalar[DIRECT_ROUTES.index("reset_160k")]
    improvements = scalar_array - reset[None, :, :]
    improvements[:, 0] *= -1  # TVD: Reset-ZNE; both gap metrics: ZNE-Reset.
    arrays = {
        "scalar_draws": scalar_array, "direct_draws": direct_scalar,
        "paired_reset_improvements": improvements, "selected_indices": selected,
        "selected_parameters": parameters, "support_sizes": support.sum(axis=1),
        "packed_observed_support": np.packbits(support, axis=1, bitorder="little"),
    }
    if keep_vectors:
        for index, method in enumerate(METHODS):
            arrays["before__" + method] = before_vectors[index][0]
            arrays["after__" + method] = after_vectors[index][0]
    return arrays


def _point_json(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    points = {
        route: {metric: float(arrays[key][ri, mi, 0]) for mi, metric in enumerate(METRICS)}
        for key, routes in (("scalar_draws", METHODS), ("direct_draws", DIRECT_ROUTES))
        for ri, route in enumerate(routes)
    }
    improvements = {
        route: {metric: float(arrays["paired_reset_improvements"][ri, mi, 0])
                for mi, metric in enumerate(METRICS)}
        for ri, route in enumerate(METHODS)
    }
    return {"points": points, "paired_reset_improvements": improvements}


def reconstruct(parents: CountParents, order: int) -> dict[str, Any]:
    """Return original-parent point metrics and vectors, not bootstrap means."""
    suite = build_suite(parents.effective_lambdas)
    reference = _reference(parents.bits, order)
    direct = {name: values[None, :] for name, values in parents.direct_probabilities().items()}
    arrays = evaluate_batch(parents.reset_parents[None, :, :], direct, suite, reference, keep_vectors=True)
    arrays.update({
        "routes": np.asarray(METHODS), "direct_routes": np.asarray(DIRECT_ROUTES),
        "metrics": np.asarray(METRICS), "effective_lambdas": parents.effective_lambdas,
        "objective_weights": objective_weights(parents.effective_lambdas, exponent=2.0),
        "scale_parents": parents.reset_parents,
        "probability_routes": np.asarray((*METHODS, *DIRECT_ROUTES)),
        "probabilities": np.stack([
            *(arrays["after__" + name] for name in METHODS),
            *(direct[name][0] for name in DIRECT_ROUTES),
        ]),
    })
    metadata = {
        "analysis": "portable CPU C1 original-parent reconstruction", "budget": "equal_shot",
        "dimension": parents.dimension, "bits": parents.bits, "order_evaluation_only": int(order),
        "weight_exponent": 2.0, "weights_normalization": "mean_one",
        "ridges": dict.fromkeys(METHODS, 0.0), "exponential_offset": 1.0 / parents.dimension,
        "candidate_counts": {METHODS[2]: 200, METHODS[3]: 19900},
        "support_rule": "union of positive counts in eight reset scales only; recomputed per replicate",
        "paired_difference_signs": {"tvd": "reset_160k minus ZNE", "gap_metrics": "ZNE minus reset_160k"},
        "gap_definition": "grouped_peak_margin; normalized_grouped_peak_margin divides by ideal margin",
        "gate_matched": "not supported by this portable equal-shot workflow",
        "point_definition": "original parent estimates, never bootstrap means",
    }
    return {"metadata": metadata, **_point_json(arrays), "arrays": arrays}


def bootstrap_seed_streams(seed: int) -> tuple[np.random.Generator, ...]:
    """Four independent parent-block streams in a documented stable order."""
    seed = _integer(seed, "seed", minimum=0)
    return tuple(np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4))


def draw_paired_batch(parents: CountParents, rngs: Sequence[np.random.Generator], size: int) -> dict[str, np.ndarray]:
    """Literal equal-shot equivalent of frozen bvr._draw_paired_batch.

    RNG order: reset eight-scale grid, reset 140k extension, raw 20k initial,
    raw 140k extension. The sole grid is shared by all four models. Scale loops
    match the original helper; batch size is part of reproducibility metadata.
    """
    size = _integer(size, "batch size")
    if len(rngs) != 4:
        raise ValueError("exactly four parent-block RNG streams are required")
    reset_rng, reset_extension_rng, raw_rng, raw_extension_rng = rngs
    reset_counts = np.stack([
        reset_rng.multinomial(SCALE_SHOTS, parent, size=size)
        for parent in parents.reset_parents
    ], axis=1).astype(np.float64)
    reset_grid = reset_counts / SCALE_SHOTS
    reset20 = reset_grid[:, 0]
    reset_extension_counts = reset_extension_rng.multinomial(
        EXTENSION_SHOTS, parents.reset_extension_counts / EXTENSION_SHOTS, size=size
    ).astype(np.float64)
    raw20_counts = raw_rng.multinomial(
        SCALE_SHOTS, parents.raw_initial_counts / SCALE_SHOTS, size=size
    ).astype(np.float64)
    raw_extension_counts = raw_extension_rng.multinomial(
        EXTENSION_SHOTS, parents.raw_extension_counts / EXTENSION_SHOTS, size=size
    ).astype(np.float64)
    return {
        "reset_grid": reset_grid, "reset_20k": reset20,
        "reset_160k": (reset20 * SCALE_SHOTS + reset_extension_counts) / DIRECT_SHOTS,
        "reset_extension_counts": reset_extension_counts,
        "raw_20k": raw20_counts / SCALE_SHOTS,
        "raw_160k": (raw20_counts + raw_extension_counts) / DIRECT_SHOTS,
        "raw_20k_counts": raw20_counts, "raw_extension_counts": raw_extension_counts,
    }


def _interval(values: np.ndarray, point: float, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0}


def bootstrap(
    parents: CountParents, order: int, *, replicates: int = 10_000,
    seed: int = 20260722, batch_size: int = 25, confidence: float = 0.95,
    progress: Any = None,
) -> dict[str, Any]:
    """Run only when called explicitly; never resume or replay historical data."""
    replicates = _integer(replicates, "replicates")
    batch_size = _integer(batch_size, "batch_size")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    rngs = bootstrap_seed_streams(seed)
    point = reconstruct(parents, order)
    suite, reference = build_suite(parents.effective_lambdas), _reference(parents.bits, order)
    pieces: dict[str, list[np.ndarray]] = {}
    for first in range(0, replicates, batch_size):
        size = min(batch_size, replicates - first)
        draw = draw_paired_batch(parents, rngs, size)
        evaluated = evaluate_batch(draw["reset_grid"], draw, suite, reference)
        for key, value in evaluated.items():
            pieces.setdefault(key, []).append(value)
        if progress is not None:
            progress(first + size, replicates)
    draws = {key: np.concatenate(value, axis=0 if key in ("support_sizes", "packed_observed_support") else -1)
             for key, value in pieces.items()}
    intervals = {
        "absolute": {
            route: {metric: _interval(draws[key][ri, mi], point["points"][route][metric], confidence)
                    for mi, metric in enumerate(METRICS)}
            for key, routes in (("scalar_draws", METHODS), ("direct_draws", DIRECT_ROUTES))
            for ri, route in enumerate(routes)
        },
        "paired_reset_improvements": {
            route: {metric: _interval(draws["paired_reset_improvements"][ri, mi],
                                      point["paired_reset_improvements"][route][metric], confidence)
                    for mi, metric in enumerate(METRICS)}
            for ri, route in enumerate(METHODS)
        },
    }
    metadata = {
        **point["metadata"], "analysis": FRESH_BOOTSTRAP_LABEL,
        "replicates": replicates, "seed": int(seed), "batch_size": batch_size,
        "confidence": float(confidence), "interval_method": "paired percentile",
        "rng": "NumPy default_rng, SeedSequence(seed).spawn(4)",
        "rng_stream_order": ["reset_scales", "reset_extension", "raw_initial", "raw_extension"],
        "numpy_version": np.__version__, "historical_replay": False,
        "warning": "Fresh conditional resampling; numerical intervals are not claimed identical to archived production intervals.",
    }
    draws.update({"routes": np.asarray(METHODS), "direct_routes": np.asarray(DIRECT_ROUTES),
                  "metrics": np.asarray(METRICS), "seed": np.asarray(seed),
                  "batch_size": np.asarray(batch_size), "replicates": np.asarray(replicates)})
    return {"point": point, "metadata": metadata, "intervals": intervals, "arrays": draws}


def decoder_selected_orders(N: int, a: int, bits: int) -> np.ndarray:
    """Strict S6 one-candidate decoder; zero means rejection.

    Select ``Fraction(y, 2**bits).limit_denominator(N)`` exactly ONCE. Reject
    its zero numerator or a failed ``pow(a,q,N)==1`` check. Return the selected
    denominator AS IS, including verified nonminimal multiples. Never scan
    alternative convergents, try multiples, prime-reduce, or use a true order.
    """
    N, a, bits = _integer(N, "N", minimum=3), _integer(a, "a"), _integer(bits, "bits")
    if a >= N or math.gcd(a, N) != 1:
        raise ValueError("a must lie in [1,N) and be coprime to N")
    selected = np.zeros(1 << bits, dtype=np.int64)
    for outcome in range(selected.size):
        candidate = Fraction(outcome, selected.size).limit_denominator(N)
        if candidate.numerator != 0 and pow(a, candidate.denominator, N) == 1:
            selected[outcome] = candidate.denominator
    return selected


def evaluation_order_mask(selected_orders: np.ndarray, true_order: int) -> np.ndarray:
    """Apply the supplied truth label only AFTER decoding."""
    true_order = _integer(true_order, "true_order")
    selected = np.asarray(selected_orders)
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or np.any(selected < 0):
        raise ValueError("selected_orders must be a nonnegative integer vector")
    return selected == true_order


def exact_order_mask(N: int, a: int, bits: int, *, true_order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded q and the evaluation-only exact-order success mask."""
    selected = decoder_selected_orders(N, a, bits)
    return selected, evaluation_order_mask(selected, true_order)


def _probability_rows(path: Path, dimension: int, keys: Sequence[str] | None) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if keys:
            if any(key.startswith("before__") for key in keys):
                raise ValueError("before__ signed preprojection vectors are not probability distributions")
            names = list(keys)
            values = np.stack([archive[key] for key in keys])
        elif "probabilities" in archive.files:
            values = np.asarray(archive["probabilities"])
            if values.ndim == 1:
                values = values[None, :]
            names = (archive["probability_routes"].astype(str).tolist()
                     if "probability_routes" in archive.files else [f"distribution_{i}" for i in range(len(values))])
        else:
            names = sorted(key for key in archive.files if key.startswith("after__"))
            if not names:
                raise ValueError("probability NPZ requires probabilities (+ optional probability_routes), after__* arrays, or --key")
            values = np.stack([archive[key] for key in names])
    if (values.ndim != 2 or values.shape[1] != dimension or len(names) != len(values)
            or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values))
            or np.any(values < 0) or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError(f"probabilities must be finite nonnegative normalized rows with D={dimension}")
    if len(set(names)) != len(names):
        raise ValueError("probability route names must be unique")
    return names, values.astype(np.float64)


def _new_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_points(output: Path, result: Mapping[str, Any]) -> None:
    _write_json(output / "points.json", {key: result[key] for key in ("metadata", "points", "paired_reset_improvements")})
    _write_npz(output / "parent_estimates.npz", result["arrays"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("reconstruct", "bootstrap", "decode"):
        child = commands.add_parser(command)
        child.add_argument("--input", required=True, type=Path)
        child.add_argument("--output-dir", "--outputdir", required=True, type=Path)
        child.add_argument("--order", required=True, type=int, help="true-order evaluation label; never used for fitting/decoding")
        if command != "decode":
            child.add_argument("--budget", choices=("equal_shot",), default="equal_shot",
                               help="only equal_shot supported; gate_matched is deliberately unsupported")
        else:
            child.add_argument("--N", required=True, type=int)
            child.add_argument("--a", required=True, type=int)
            child.add_argument("--bits", required=True, type=int)
            child.add_argument("--key", action="append", help="explicit 1D probability array key; repeat for multiple routes")
        if command == "bootstrap":
            child.add_argument("--replicates", type=int, default=10_000)
            child.add_argument("--seed", type=int, default=20260722)
            child.add_argument("--batch-size", type=int, default=25)
            child.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"output directory must be new: {args.output_dir}")
        input_hash = _sha256(args.input)
        if args.command == "decode":
            selected, mask = exact_order_mask(args.N, args.a, args.bits, true_order=args.order)
            names, probabilities = _probability_rows(args.input, selected.size, args.key)
            result = {
                "decoder": "S6 strict single candidate Fraction(y,2**bits).limit_denominator(N); zero numerator/modular failure rejected; q retained as is",
                "N": args.N, "a": args.a, "bits": args.bits, "order_evaluation_only": args.order,
                "true_order_used_during_decoding": False, "accepted_bins": int(mask.sum()),
                "P_exact": dict(zip(names, (probabilities @ mask).astype(float).tolist())),
                "arbitrary_denominator_multiples": False, "multi_shot_lcm": False,
                "prime_factor_reduction": False, "alternative_convergents": False,
            }
            output = _new_output(args.output_dir)
            _write_json(output / "decoder.json", result)
            _write_npz(output / "decoder.npz", {"selected_orders": selected, "exact_order_mask": mask,
                                               "probability_routes": np.asarray(names), "P_exact": probabilities @ mask})
        else:
            parents = load_counts(args.input)
            if args.command == "reconstruct":
                point = reconstruct(parents, args.order)
                output = _new_output(args.output_dir)
                _write_points(output, point)
            else:
                # Validate before claiming the new directory; claiming it before the
                # computation also prevents concurrent invocations from sharing it.
                _integer(args.replicates, "replicates")
                _integer(args.batch_size, "batch_size")
                _integer(args.seed, "seed", minimum=0)
                if not 0 < args.confidence < 1:
                    raise ValueError("confidence must lie strictly between zero and one")
                _reference(parents.bits, args.order)
                output = _new_output(args.output_dir)
                result = bootstrap(parents, args.order, replicates=args.replicates, seed=args.seed,
                                   batch_size=args.batch_size, confidence=args.confidence,
                                   progress=lambda done, total: print(f"Fresh CPU bootstrap: {done}/{total}", flush=True))
                _write_points(output, result["point"])
                _write_json(output / "intervals.json", {"metadata": result["metadata"], "intervals": result["intervals"]})
                _write_npz(output / "scalar_bootstrap_draws.npz", result["arrays"])
        artifacts = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
        _write_json(output / "manifest.json", {
            "command": args.command, "input_sha256": input_hash, "artifacts": artifacts,
            "numpy_version": np.__version__, "cpu_only": True,
            "historical_replay": False, "no_file_overwrite": True,
        })
        print(f"Wrote {args.command} outputs to {output}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
