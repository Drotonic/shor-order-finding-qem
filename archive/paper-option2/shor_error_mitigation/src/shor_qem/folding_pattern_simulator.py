"""Shot-wise randomized-folding multi-GPU experiment over a configured grid.

This driver reuses the validated numerical kernels in :mod:`shor_qem.simulator`
but changes the experimental hierarchy:

* every shot selects an independent random-folding realization;
* the realization is independent across physical error rates and noise scales;
* every shot has a distinct deterministic stochastic-noise/measurement stream;
* all visible CUDA GPUs are detected and used unless explicitly restricted;
* completed task archives are validated and reused after an interrupted run.

The default run uses 20,000 shot-wise patterns at every physical error rate
listed in the selected configuration files. Lambda=1 contains no folded gates
and lambda=3 folds every eligible gate. Gate-matched configurations require
fresh samples at both endpoints; historical configurations can reuse their
saved endpoint data unless the optional endpoint flags are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import pickle
import platform
import queue
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .folding import (
    FOLDING_ARCHIVE_SCHEMA_VERSION,
    FOLDING_PATTERN_SCOPE,
    FOLDING_SEED_FIELDS,
    FOLDING_SEED_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = (
    PROJECT_ROOT / "configs" / "n119_a13_matched.json",
    PROJECT_ROOT / "configs" / "n119_a50_matched.json",
    PROJECT_ROOT / "configs" / "n119_a16_matched.json",
)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    with resolved.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "instance",
        "N",
        "a",
        "order",
        "measurement_bits",
        "global_seed",
        "rates",
        "lambdas",
        "shots_per_lambda",
        "ops_cache",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"{resolved}: missing fields {', '.join(missing)}")
    config = dict(config)
    config["config_path"] = str(resolved)
    config["ops_cache_path"] = str(_resolve_project_path(config["ops_cache"]))
    return config


def _shared_config_grid(
    configs: list[dict[str, Any]], field: str, override_flag: str
) -> list[float]:
    grids = [tuple(map(float, config[field])) for config in configs]
    if not grids:
        raise ValueError("at least one configuration is required")
    if len(set(grids)) != 1:
        raise ValueError(
            f"configs use different {field} grids; pass {override_flag}"
        )
    return list(grids[0])


def _load_operation_map(cache_path: Path) -> dict[int, list[Any]]:
    """Load one trusted local cache and reject count-only placeholders."""

    with cache_path.open("rb") as handle:
        operation_map = pickle.load(handle)
    if not isinstance(operation_map, dict) or not operation_map:
        raise ValueError(f"{cache_path}: operation cache is empty or malformed")
    for k_pow, operations in operation_map.items():
        if not isinstance(operations, list):
            raise ValueError(f"{cache_path}: round {k_pow} is not an operation list")
        for operation_i, operation in enumerate(operations):
            if not isinstance(operation, tuple) or len(operation) != 3:
                raise ValueError(
                    f"{cache_path}: malformed operation {operation_i} in round {k_pow}"
                )
            name, _, parameters = operation
            if name == "unitary":
                matrix = (
                    np.asarray(parameters[0])
                    if isinstance(parameters, (list, tuple)) and parameters
                    else np.asarray(None)
                )
                if matrix.shape != (2, 2):
                    raise ValueError(
                        f"{cache_path}: fused one-qubit operation {operation_i} "
                        f"in round {k_pow} has no executable 2x2 matrix"
                    )
    return operation_map


def _lambda_token(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "p")


def _task_path(output_dir: Path, task: dict[str, Any]) -> Path:
    lambdas = task["lambdas"]
    token = f"lambda_{_lambda_token(lambdas[0])}-{_lambda_token(lambdas[-1])}"
    return (
        output_dir
        / task["instance"]
        / "shotwise"
        / f"{token}.npz"
    )


def build_tasks(
    configs: list[dict[str, Any]],
    *,
    rates: list[float],
    lambdas: list[float],
    shots: int,
    lambda_group_size: int,
    output_dir: Path,
    simulate_shared_lambda_one: bool,
    simulate_shared_lambda_three: bool,
) -> list[dict[str, Any]]:
    if shots <= 0 or lambda_group_size <= 0:
        raise ValueError("shots and lambda_group_size must be positive")
    randomized_lambdas = [
        value
        for value in lambdas
        if not np.isclose(value, 1.0) and not np.isclose(value, 3.0)
    ]
    tasks: list[dict[str, Any]] = []
    for config in configs:
        master_seed = int(config["global_seed"])
        require_fresh_endpoints = bool(
            config.get("require_all_lambdas_from_shotwise", False)
        )
        for start in range(0, len(randomized_lambdas), lambda_group_size):
            group = randomized_lambdas[start : start + lambda_group_size]
            task = {
                "instance": str(config["instance"]),
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": int(config["order"]),
                "measurement_bits": int(config["measurement_bits"]),
                "ops_cache_path": str(config["ops_cache_path"]),
                "ops_cache_sha256": str(
                    config.get("ops_cache_sha256", "unrecorded")
                ),
                "circuit_variant": str(
                    config.get("circuit_variant", "historical_opt3")
                ),
                "two_qubit_gate_count": int(
                    config.get("two_qubit_gate_count", -1)
                ),
                "rates": list(map(float, rates)),
                "lambdas": list(map(float, group)),
                "shots": int(shots),
                "folding_seed": master_seed,
            }
            task["output_path"] = str(_task_path(output_dir, task))
            tasks.append(task)

        if (simulate_shared_lambda_one or require_fresh_endpoints) and any(
            np.isclose(value, 1.0) for value in lambdas
        ):
            baseline_task = {
                "instance": str(config["instance"]),
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": int(config["order"]),
                "measurement_bits": int(config["measurement_bits"]),
                "ops_cache_path": str(config["ops_cache_path"]),
                "ops_cache_sha256": str(
                    config.get("ops_cache_sha256", "unrecorded")
                ),
                "circuit_variant": str(
                    config.get("circuit_variant", "historical_opt3")
                ),
                "two_qubit_gate_count": int(
                    config.get("two_qubit_gate_count", -1)
                ),
                "rates": list(map(float, rates)),
                "lambdas": [1.0],
                "shots": int(shots),
                "folding_seed": master_seed,
            }
            baseline_task["output_path"] = str(
                _task_path(output_dir, baseline_task)
            )
            tasks.append(baseline_task)
        if (simulate_shared_lambda_three or require_fresh_endpoints) and any(
            np.isclose(value, 3.0) for value in lambdas
        ):
            endpoint_task = {
                "instance": str(config["instance"]),
                "N": int(config["N"]),
                "a": int(config["a"]),
                "order": int(config["order"]),
                "measurement_bits": int(config["measurement_bits"]),
                "ops_cache_path": str(config["ops_cache_path"]),
                "ops_cache_sha256": str(
                    config.get("ops_cache_sha256", "unrecorded")
                ),
                "circuit_variant": str(
                    config.get("circuit_variant", "historical_opt3")
                ),
                "two_qubit_gate_count": int(
                    config.get("two_qubit_gate_count", -1)
                ),
                "rates": list(map(float, rates)),
                "lambdas": [3.0],
                "shots": int(shots),
                "folding_seed": master_seed,
            }
            endpoint_task["output_path"] = str(
                _task_path(output_dir, endpoint_task)
            )
            tasks.append(endpoint_task)
    return tasks


def _bits_to_outcomes(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits)
    if values.ndim != 2 or values.shape[1] > 16:
        raise ValueError(f"unexpected measurement array shape {values.shape}")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("measurement array is not binary")
    powers = np.left_shift(np.uint32(1), np.arange(values.shape[1], dtype=np.uint32))
    outcomes = np.sum(
        values.astype(np.uint32, copy=False) * powers[None, :],
        axis=1,
        dtype=np.uint32,
    )
    return outcomes.astype(np.uint16, copy=False)


def _counts_from_outcomes(outcomes: np.ndarray, measurement_bits: int) -> np.ndarray:
    return np.bincount(
        np.asarray(outcomes, dtype=np.int64), minlength=1 << measurement_bits
    ).astype(np.int32, copy=False)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.stem + "_", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fold_plan_metadata(
    ops_map: dict[int, list[tuple[str, list[int], list[Any]]]],
    *,
    N: int,
    a: int,
    rates: list[float],
    lambdas: list[float],
    folding_seed: int,
    shots: int,
    noisy_gate_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Hash the shot-wise plan recipe and record per-round fold counts."""

    num_rounds = 2 * int(N).bit_length()
    digests = np.empty((len(rates), len(lambdas)), dtype="<U64")
    counts = np.zeros((len(lambdas), num_rounds), dtype=np.int32)
    for lambda_i, lam in enumerate(lambdas):
        lambda_code = int(round(float(lam) * 10.0))
        round_recipe = hashlib.sha256()
        for round_i in range(num_rounds):
            k_pow = 2 ** (num_rounds - 1 - round_i)
            ops = ops_map[k_pow]
            targets = [
                op_i for op_i, (name, _, _) in enumerate(ops)
                if name in noisy_gate_names
            ]
            fold_count = int(round((float(lam) - 1.0) * len(targets) / 2.0))
            fold_count = min(max(fold_count, 0), len(targets))
            counts[lambda_i, round_i] = fold_count
            round_recipe.update(
                np.asarray(
                    [round_i, lambda_code, len(targets), fold_count],
                    dtype=np.int32,
                ).tobytes()
            )
        for rate_i, rate in enumerate(rates):
            digest = hashlib.sha256()
            digest.update(b"shotwise_exact_cardinality_v2")
            digest.update(
                np.asarray(
                    [
                        FOLDING_SEED_VERSION,
                        folding_seed,
                        N,
                        a,
                        shots,
                        int(round(float(rate) * 1_000_000_000.0)),
                        lambda_code,
                    ],
                    dtype=np.uint64,
                ).tobytes()
            )
            digest.update(round_recipe.digest())
            digests[rate_i, lambda_i] = digest.hexdigest()
    return digests, counts


def validate_task_output(path: Path, task: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "rates",
        "lambdas",
        "outcomes",
        "counts",
        "folding_seed",
        "folding_pattern_scope",
        "folding_seed_fields",
        "ops_cache_sha256",
        "circuit_variant",
        "two_qubit_gate_count",
        "shots",
        "measurement_bits",
        "fold_plan_sha256",
        "folded_gate_counts_by_round",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        rates = np.asarray(archive["rates"], dtype=np.float64)
        lambdas = np.asarray(archive["lambdas"], dtype=np.float64)
        outcomes = np.asarray(archive["outcomes"])
        counts = np.asarray(archive["counts"])
        plan_hashes = np.asarray(archive["fold_plan_sha256"])
        folded_counts = np.asarray(archive["folded_gate_counts_by_round"])
        shots = int(archive["shots"])
        measurement_bits = int(archive["measurement_bits"])
        if int(archive["schema_version"]) != FOLDING_ARCHIVE_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema version differs; rerun with --overwrite"
            )
        if not np.array_equal(rates, np.asarray(task["rates"], dtype=np.float64)):
            raise ValueError(f"{path}: rate grid differs from task")
        if not np.array_equal(lambdas, np.asarray(task["lambdas"], dtype=np.float64)):
            raise ValueError(f"{path}: lambda grid differs from task")
        if int(archive["folding_seed"]) != int(task["folding_seed"]):
            raise ValueError(f"{path}: folding_seed differs from task")
        if str(archive["ops_cache_sha256"]) != str(task["ops_cache_sha256"]):
            raise ValueError(f"{path}: operation-cache hash differs from task")
        if str(archive["circuit_variant"]) != str(task["circuit_variant"]):
            raise ValueError(f"{path}: circuit variant differs from task")
        if int(archive["two_qubit_gate_count"]) != int(
            task["two_qubit_gate_count"]
        ):
            raise ValueError(f"{path}: two-qubit gate count differs from task")
        if str(archive["folding_pattern_scope"]) != FOLDING_PATTERN_SCOPE:
            raise ValueError(
                f"{path}: folding pattern scope differs; rerun with --overwrite"
            )
        if not np.array_equal(
            np.asarray(archive["folding_seed_fields"]),
            np.asarray(FOLDING_SEED_FIELDS),
        ):
            raise ValueError(
                f"{path}: folding seed hierarchy differs; rerun with --overwrite"
            )
        if shots != int(task["shots"]) or measurement_bits != int(task["measurement_bits"]):
            raise ValueError(f"{path}: shot or measurement-bit metadata differs")
        expected_outcome_shape = (rates.size, lambdas.size, shots)
        expected_count_shape = (rates.size, lambdas.size, 1 << measurement_bits)
        if outcomes.shape != expected_outcome_shape:
            raise ValueError(f"{path}: outcomes shape {outcomes.shape}, expected {expected_outcome_shape}")
        if counts.shape != expected_count_shape:
            raise ValueError(f"{path}: counts shape {counts.shape}, expected {expected_count_shape}")
        if plan_hashes.shape != (rates.size, lambdas.size):
            raise ValueError(
                f"{path}: fold-plan hash shape {plan_hashes.shape} is invalid"
            )
        expected_rounds = 2 * int(task["N"]).bit_length()
        if folded_counts.shape != (lambdas.size, expected_rounds):
            raise ValueError(
                f"{path}: folded-gate count shape {folded_counts.shape} is invalid"
            )
        if not np.all(counts.sum(axis=-1) == shots):
            raise ValueError(f"{path}: one or more count rows have the wrong total")
        for rate_i in range(rates.size):
            for lambda_i in range(lambdas.size):
                rebuilt = _counts_from_outcomes(
                    outcomes[rate_i, lambda_i], measurement_bits
                )
                if not np.array_equal(rebuilt, counts[rate_i, lambda_i]):
                    raise ValueError(f"{path}: outcomes and counts disagree")


def _summarize_fold_plan_design(
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    summary: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for task in tasks:
        with np.load(Path(task["output_path"]), allow_pickle=False) as archive:
            rates = np.asarray(archive["rates"], dtype=np.float64)
            lambdas = np.asarray(archive["lambdas"], dtype=np.float64)
            hashes = np.asarray(archive["fold_plan_sha256"])
            shots = int(archive["shots"])
        for rate_i, rate in enumerate(rates):
            rate_summary = summary.setdefault(task["instance"], {}).setdefault(
                str(float(rate)), {}
            )
            for lambda_i, lam in enumerate(lambdas):
                rate_summary[str(float(lam))] = {
                    "scope": FOLDING_PATTERN_SCOPE,
                    "shots": shots,
                    "recipe_sha256": str(hashes[rate_i, lambda_i]),
                }
    return summary


def _run_task(
    task: dict[str, Any], core: Any, *, ops_map: dict[int, list[Any]] | None = None
) -> Path:
    cache_path = Path(task["ops_cache_path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"operation cache not found: {cache_path}")
    core.OPS_CACHE_DIR = str(cache_path.parent)
    core.GLOBAL_SEED = int(task["folding_seed"])

    if ops_map is None:
        ops_map = _load_operation_map(cache_path)
    plan_hashes, folded_counts = _fold_plan_metadata(
        ops_map,
        N=int(task["N"]),
        a=int(task["a"]),
        rates=list(task["rates"]),
        lambdas=list(task["lambdas"]),
        folding_seed=int(task["folding_seed"]),
        shots=int(task["shots"]),
        noisy_gate_names=tuple(core.NOISY_2Q_GATES),
    )

    # Clear retained pools between independent tasks so the VRAM estimator sees
    # the memory that can actually be reused.
    core.smart_mem_gc(force=True, tag=f"before {task['instance']} shotwise")
    results = core._run_gpu_engine_chunked_shotwise(
        "random",
        list(task["rates"]),
        list(task["lambdas"]),
        int(task["N"]),
        int(task["a"]),
        int(task["shots"]),
        ops_map,
        restrict_folding_to_step=None,
        shot_offset=0,
        reset_ancilla=True,
        folding_pattern_per_shot=True,
    )

    configs = [
        (rate, lam) for rate in task["rates"] for lam in task["lambdas"]
    ]
    indexed = {int(index): data for index, data in results}
    measurement_bits = int(task["measurement_bits"])
    outcome_grid = np.empty(
        (len(task["rates"]), len(task["lambdas"]), int(task["shots"])),
        dtype=np.uint16,
    )
    count_grid = np.empty(
        (len(task["rates"]), len(task["lambdas"]), 1 << measurement_bits),
        dtype=np.int32,
    )
    for config_i, (rate, lam) in enumerate(configs):
        if config_i not in indexed:
            raise RuntimeError(f"missing simulator result for rate={rate}, lambda={lam}")
        rate_i = task["rates"].index(rate)
        lambda_i = task["lambdas"].index(lam)
        outcomes = _bits_to_outcomes(indexed[config_i])
        outcome_grid[rate_i, lambda_i] = outcomes
        count_grid[rate_i, lambda_i] = _counts_from_outcomes(
            outcomes, measurement_bits
        )

    output_path = Path(task["output_path"])
    _atomic_npz(
        output_path,
        schema_version=np.int32(FOLDING_ARCHIVE_SCHEMA_VERSION),
        instance=np.asarray(task["instance"]),
        N=np.int32(task["N"]),
        a=np.int32(task["a"]),
        order=np.int32(task["order"]),
        measurement_bits=np.int32(measurement_bits),
        rates=np.asarray(task["rates"], dtype=np.float64),
        lambdas=np.asarray(task["lambdas"], dtype=np.float64),
        shots=np.int32(task["shots"]),
        folding_seed=np.uint32(task["folding_seed"]),
        folding_pattern_scope=np.asarray(FOLDING_PATTERN_SCOPE),
        folding_seed_fields=np.asarray(FOLDING_SEED_FIELDS),
        ops_cache_sha256=np.asarray(task["ops_cache_sha256"]),
        circuit_variant=np.asarray(task["circuit_variant"]),
        two_qubit_gate_count=np.int32(task["two_qubit_gate_count"]),
        outcomes=outcome_grid,
        counts=count_grid,
        fold_plan_sha256=plan_hashes,
        folded_gate_counts_by_round=folded_counts,
    )
    validate_task_output(output_path, task)
    return output_path


def _worker(
    gpu_id: int,
    task_queue: Any,
    result_queue: Any,
    tuning: dict[str, float | int],
) -> None:
    try:
        from . import simulator as core

        core.MAX_BATCH_COLS_PER_CHUNK = int(tuning["max_batch_columns"])
        core.VRAM_FREE_FRACTION = float(tuning["vram_free_fraction"])
        core.VRAM_SAFETY_FACTOR = float(tuning["vram_safety_factor"])
        core.USE_FUSED_2Q_KERNELS = bool(tuning["fused_2q_kernels"])
        with core.cp.cuda.Device(gpu_id):
            print(f"[GPU {gpu_id}] worker ready", flush=True)
            operation_maps: dict[tuple[str, int, int], dict[int, list[Any]]] = {}
            while True:
                task = task_queue.get()
                if task is None:
                    break
                try:
                    operation_key = (
                        str(Path(task["ops_cache_path"])),
                        int(task["N"]),
                        int(task["a"]),
                    )
                    if operation_key not in operation_maps:
                        operation_maps[operation_key] = _load_operation_map(
                            Path(operation_key[0])
                        )
                    path = _run_task(
                        task, core, ops_map=operation_maps[operation_key]
                    )
                    result_queue.put(
                        {"ok": True, "gpu_id": gpu_id, "path": str(path)}
                    )
                    print(f"[GPU {gpu_id}] completed {path}", flush=True)
                except Exception:
                    result_queue.put(
                        {
                            "ok": False,
                            "gpu_id": gpu_id,
                            "task": task,
                            "traceback": traceback.format_exc(),
                        }
                    )
    except Exception:
        result_queue.put(
            {"ok": False, "gpu_id": gpu_id, "traceback": traceback.format_exc()}
        )
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _parse_gpu_ids(value: str | None, available: int) -> list[int]:
    if value is None:
        return list(range(available))
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("--gpu-ids must contain distinct comma-separated IDs")
    if min(ids) < 0 or max(ids) >= available:
        raise ValueError(f"GPU IDs must lie in 0..{available - 1}")
    return ids


def _select_worker_gpu_ids(
    value: str | None, *, available: int, pending_tasks: int
) -> list[int]:
    """Use all requested/visible GPUs, capped by the number of pending tasks."""

    if available <= 0:
        raise ValueError("available must be positive")
    if pending_tasks <= 0:
        return []
    selected = _parse_gpu_ids(value, available)
    return selected[: min(len(selected), pending_tasks)]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 20,000 shot-wise random-folding realizations over the "
            "configured physical-error grid on all visible CUDA GPUs."
        )
    )
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--output-dir",
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
    parser.add_argument("--lambdas", nargs="+", type=float)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--lambda-group-size", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        help=(
            "Optional comma-separated logical CUDA IDs. By default, use every "
            "GPU visible through CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument("--max-batch-columns", type=int, default=40000)
    parser.add_argument("--vram-free-fraction", type=float, default=0.90)
    parser.add_argument("--vram-safety-factor", type=float, default=1.45)
    parser.add_argument(
        "--legacy-2q-kernels",
        action="store_true",
        help=(
            "Use the slower reference two-qubit kernels. Intended only for "
            "numerical-equivalence benchmarks."
        ),
    )
    parser.add_argument(
        "--simulate-shared-lambda-one",
        action="store_true",
        help="Simulate one shared lambda=1 baseline instead of reusing the historical data.",
    )
    parser.add_argument(
        "--simulate-shared-lambda-three",
        action="store_true",
        help="Simulate one shared lambda=3 endpoint instead of reusing the historical data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace matching task archives; otherwise valid archives are resumed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = [_load_config(path) for path in args.configs]
    if args.shots <= 0:
        raise ValueError("shots must be positive")
    rates = (
        _shared_config_grid(configs, "rates", "--rates")
        if args.rates is None
        else list(map(float, args.rates))
    )
    if not rates or len(rates) != len(set(rates)) or min(rates) < 0.0:
        raise ValueError("rates must be distinct and nonnegative")
    if args.lambdas is None:
        lambdas = _shared_config_grid(configs, "lambdas", "--lambdas")
    else:
        lambdas = list(map(float, args.lambdas))
    if (
        not lambdas
        or len(lambdas) != len(set(lambdas))
        or min(lambdas) < 1.0
        or max(lambdas) > 3.0
    ):
        raise ValueError("lambdas must be distinct and lie in [1,3]")
    if not any(1.0 < value < 3.0 for value in lambdas):
        raise ValueError("at least one lambda with a nontrivial random subset is required")
    if not 0.0 < args.vram_free_fraction <= 1.0:
        raise ValueError("--vram-free-fraction must lie in (0,1]")
    if args.vram_safety_factor < 1.0:
        raise ValueError("--vram-safety-factor must be at least 1")

    for config in configs:
        cache_path = Path(config["ops_cache_path"])
        if not cache_path.is_file():
            raise FileNotFoundError(f"saved operation cache not found: {cache_path}")
        config["ops_cache_sha256"] = _sha256_file(cache_path)

    output_dir = args.output_dir.resolve()
    tasks = build_tasks(
        configs,
        rates=rates,
        lambdas=lambdas,
        shots=int(args.shots),
        lambda_group_size=int(args.lambda_group_size),
        output_dir=output_dir,
        simulate_shared_lambda_one=bool(args.simulate_shared_lambda_one),
        simulate_shared_lambda_three=bool(args.simulate_shared_lambda_three),
    )

    pending: list[dict[str, Any]] = []
    resumed: list[Path] = []
    for task in tasks:
        path = Path(task["output_path"])
        if path.exists() and not args.overwrite:
            validate_task_output(path, task)
            resumed.append(path)
        else:
            pending.append(task)

    total_configs = sum(len(task["rates"]) * len(task["lambdas"]) for task in tasks)
    print(
        f"instances={len(configs)}, folding_scope={FOLDING_PATTERN_SCOPE}, rates={rates}, "
        f"randomized_lambdas={[value for value in lambdas if 1.0 < value < 3.0]}"
    )
    print(
        f"task archives={len(tasks)}, rate-lambda simulations={total_configs}, "
        f"resume={len(resumed)}, pending={len(pending)}, shots/config={args.shots}"
    )
    print(
        "seed hierarchy: independent exact-cardinality fold plan per "
        "(N,a,rate,lambda,QPE round,shot_id); independent stochastic stream per "
        "(rate,lambda,shot_id)"
    )
    print(
        "two-qubit kernel mode: "
        + ("legacy reference" if args.legacy_2q_kernels else "fused optimized")
    )
    if args.dry_run:
        for task in pending[:10]:
            print("  ", task["output_path"])
        if len(pending) > 10:
            print(f"  ... {len(pending) - 10} more")
        return 0

    available_gpu_count: int | None = None
    if pending:
        from . import simulator as core

        available_gpu_count = int(core.cp.cuda.runtime.getDeviceCount())
        if available_gpu_count == 0:
            raise RuntimeError("No CUDA GPUs found")
        gpu_ids = _select_worker_gpu_ids(
            args.gpu_ids,
            available=available_gpu_count,
            pending_tasks=len(pending),
        )
        print(
            f"detected {available_gpu_count} visible CUDA GPU(s); "
            f"using logical device IDs {gpu_ids}"
        )
        context = mp.get_context("spawn")
        task_queue = context.Queue()
        result_queue = context.Queue()
        rng = np.random.default_rng(int(configs[0]["global_seed"]) + 20260722)
        rng.shuffle(pending)
        for task in pending:
            task_queue.put(task)
        for _ in gpu_ids:
            task_queue.put(None)
        tuning = {
            "max_batch_columns": int(args.max_batch_columns),
            "vram_free_fraction": float(args.vram_free_fraction),
            "vram_safety_factor": float(args.vram_safety_factor),
            "fused_2q_kernels": not bool(args.legacy_2q_kernels),
        }
        processes = [
            context.Process(
                target=_worker,
                args=(gpu_id, task_queue, result_queue, tuning),
            )
            for gpu_id in gpu_ids
        ]
        for process in processes:
            process.start()

        records: list[dict[str, Any]] = []
        while len(records) < len(pending):
            try:
                records.append(result_queue.get(timeout=10.0))
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    break
        for process in processes:
            process.join()
        crashed = [process.exitcode for process in processes if process.exitcode != 0]
        failures = [record for record in records if not record.get("ok")]
        if len(records) != len(pending) or crashed or failures:
            for failure in failures:
                print(failure.get("traceback", failure), file=sys.stderr)
            raise RuntimeError(
                f"GPU run incomplete: results={len(records)}/{len(pending)}, "
                f"worker_exit_codes={crashed}, task_failures={len(failures)}"
            )
    else:
        gpu_ids = []

    output_paths = []
    for task in tasks:
        path = Path(task["output_path"])
        validate_task_output(path, task)
        output_paths.append(path)
    fold_plan_design = _summarize_fold_plan_design(tasks)
    fresh_lambda_one_instances = [
        config["instance"]
        for config in configs
        if any(np.isclose(value, 1.0) for value in lambdas)
        and (
            args.simulate_shared_lambda_one
            or config.get("require_all_lambdas_from_shotwise", False)
        )
    ]
    fresh_lambda_three_instances = [
        config["instance"]
        for config in configs
        if any(np.isclose(value, 3.0) for value in lambdas)
        and (
            args.simulate_shared_lambda_three
            or config.get("require_all_lambdas_from_shotwise", False)
        )
    ]
    metadata = {
        "schema_version": FOLDING_ARCHIVE_SCHEMA_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "shor_qem.folding_pattern_simulator", *sys.argv[1:]],
        "configs": [config["config_path"] for config in configs],
        "instances": [config["instance"] for config in configs],
        "circuit_configs": [
            {
                "instance": config["instance"],
                "variant": config.get("circuit_variant", "historical_opt3"),
                "transpile_optimization_level": config.get(
                    "transpile_optimization_level", 3
                ),
                "transpiler_seed": config.get("transpiler_seed"),
                "two_qubit_gate_count": config.get("two_qubit_gate_count"),
            }
            for config in configs
        ],
        "rates": rates,
        "lambdas": lambdas,
        "shots_per_rate_lambda": int(args.shots),
        "shared_lambda_one": len(fresh_lambda_one_instances) == len(configs),
        "shared_lambda_three": len(fresh_lambda_three_instances) == len(configs),
        "lambda_one_note": (
            f"fresh lambda=1 baseline for {fresh_lambda_one_instances}; "
            "other instances reuse historical lambda=1 counts"
        ),
        "lambda_three_note": (
            f"fresh lambda=3 endpoint for {fresh_lambda_three_instances}; "
            "other instances reuse historical lambda=3 counts"
        ),
        "folding_design": (
            "one exact-cardinality random subset per instance, rate, shot, "
            "lambda, and QPE round"
        ),
        "fold_plan_design": fold_plan_design,
        "folding_seed_version": FOLDING_SEED_VERSION,
        "folding_seed_fields": list(FOLDING_SEED_FIELDS),
        "stochastic_seed_fields": [
            "folding_seed", "physical_error_rate", "lambda", "shot_id"
        ],
        "available_gpu_count": available_gpu_count,
        "gpu_ids": gpu_ids,
        "optimization": {
            "lambda_group_size": int(args.lambda_group_size),
            "max_batch_columns": int(args.max_batch_columns),
            "vram_free_fraction": float(args.vram_free_fraction),
            "vram_safety_factor": float(args.vram_safety_factor),
            "two_qubit_kernel_mode": (
                "legacy_reference"
                if args.legacy_2q_kernels
                else "fused_diagonal_and_sampled_pauli"
            ),
            "dynamic_gpu_queue": True,
            "atomic_resume_archives": True,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": _package_version("cupy-cuda12x") or _package_version("cupy"),
            "qiskit": _package_version("qiskit"),
        },
        "operation_caches": [
            {
                "instance": config["instance"],
                "path": config["ops_cache_path"],
                "sha256": _sha256_file(Path(config["ops_cache_path"])),
            }
            for config in configs
        ],
        "outputs": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in sorted(output_paths)
        ],
    }
    metadata_path = output_dir / "run_metadata.json"
    _atomic_json(metadata_path, metadata)
    print(f"completed; metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
