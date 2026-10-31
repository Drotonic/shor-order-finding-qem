"""Plan, execute, validate, and merge the N=119 A100 follow-up run.

The experiment is deliberately split into atomic shot ranges.  A task's
stochastic stream depends on its physical parameters and global ``shot_id``;
it never depends on a host, GPU, worker process, or queue order.  Consequently
the same manifest can be resumed or redistributed over a different number of
GPUs without repeating or changing a shot.

The production plan has two parts:

* extend Raw and Reset at lambda=1 from 20,000 to 160,000 shots for every
  previously simulated N=119 condition;
* add a=16 rates 0.012%, 0.014%, 0.016%, and 0.018%, with 20,000 shots at all
  eight ZNE scales and 160,000 shots for both lambda=1 baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import queue
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .folding import (
    FOLDING_ARCHIVE_SCHEMA_VERSION,
    FOLDING_PATTERN_SCOPE,
    FOLDING_SEED_FIELDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "configs" / "n119_a100_followup.json"
TASK_ARCHIVE_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
MODE_RAW = "raw"
MODE_RESET = "reset"
VALID_MODES = (MODE_RAW, MODE_RESET)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


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


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    with path.resolve().open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _validate_float_grid(
    values: Iterable[Any], *, name: str, lower: float, upper: float | None = None
) -> list[float]:
    grid = [float(value) for value in values]
    if not grid or len(grid) != len(set(grid)) or min(grid) < lower:
        raise ValueError(f"{name} must be distinct and at least {lower}")
    if upper is not None and max(grid) > upper:
        raise ValueError(f"{name} must not exceed {upper}")
    return grid


def load_experiment(path: Path = DEFAULT_EXPERIMENT) -> dict[str, Any]:
    resolved = path.resolve()
    experiment = _load_json(resolved)
    required = {
        "experiment",
        "instance_configs",
        "existing_rates",
        "a16_additional_rates",
        "zne_lambdas",
        "existing_shots",
        "target_baseline_shots",
        "task_shots",
        "additional_rate_zne_shots",
        "baseline_modes",
    }
    missing = sorted(required.difference(experiment))
    if missing:
        raise ValueError(f"{resolved}: missing fields {missing}")

    existing_rates = _validate_float_grid(
        experiment["existing_rates"], name="existing_rates", lower=0.0
    )
    additional_rates = _validate_float_grid(
        experiment["a16_additional_rates"],
        name="a16_additional_rates",
        lower=0.0,
    )
    if set(existing_rates).intersection(additional_rates):
        raise ValueError("existing and additional rate grids overlap")
    lambdas = _validate_float_grid(
        experiment["zne_lambdas"], name="zne_lambdas", lower=1.0, upper=3.0
    )
    if not any(np.isclose(value, 1.0) for value in lambdas):
        raise ValueError("zne_lambdas must contain lambda=1")

    existing_shots = int(experiment["existing_shots"])
    target_shots = int(experiment["target_baseline_shots"])
    task_shots = int(experiment["task_shots"])
    zne_shots = int(experiment["additional_rate_zne_shots"])
    if min(existing_shots, target_shots, task_shots, zne_shots) <= 0:
        raise ValueError("shot counts must be positive")
    if target_shots <= existing_shots:
        raise ValueError("target_baseline_shots must exceed existing_shots")
    if existing_shots % task_shots or target_shots % task_shots:
        raise ValueError("existing and target shots must be divisible by task_shots")
    if zne_shots != existing_shots:
        raise ValueError(
            "additional_rate_zne_shots must equal existing_shots so all ZNE "
            "scales have the same finite-shot sample size"
        )
    modes = [str(value) for value in experiment["baseline_modes"]]
    if sorted(modes) != sorted(VALID_MODES):
        raise ValueError(f"baseline_modes must be exactly {VALID_MODES}")

    config_paths = [_resolve_project_path(value) for value in experiment["instance_configs"]]
    configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = _load_json(config_path)
        needed = {
            "instance",
            "N",
            "a",
            "order",
            "measurement_bits",
            "global_seed",
            "ops_cache",
            "circuit_variant",
            "two_qubit_gate_count",
        }
        absent = sorted(needed.difference(config))
        if absent:
            raise ValueError(f"{config_path}: missing fields {absent}")
        cache_path = _resolve_project_path(config["ops_cache"])
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        enriched = dict(config)
        enriched["config_path"] = _project_relative(config_path)
        enriched["ops_cache_path"] = _project_relative(cache_path)
        enriched["ops_cache_sha256"] = _sha256_file(cache_path)
        configs.append(enriched)
    if [int(config["a"]) for config in configs].count(16) != 1:
        raise ValueError("exactly one configured instance must have a=16")

    normalized = dict(experiment)
    normalized["experiment_path"] = _project_relative(resolved)
    normalized["existing_rates"] = existing_rates
    normalized["a16_additional_rates"] = additional_rates
    normalized["zne_lambdas"] = lambdas
    normalized["existing_shots"] = existing_shots
    normalized["target_baseline_shots"] = target_shots
    normalized["task_shots"] = task_shots
    normalized["additional_rate_zne_shots"] = zne_shots
    normalized["baseline_modes"] = modes
    normalized["configs"] = configs
    normalized["experiment_sha256"] = _sha256_file(resolved)
    return normalized


def _rate_code(rate: float) -> int:
    return int(round(float(rate) * 1_000_000_000.0))


def _lambda_code(lam: float) -> int:
    return int(round(float(lam) * 10.0))


def _task_id(
    instance: str,
    mode: str,
    rate: float,
    lam: float,
    shot_offset: int,
    shots: int,
) -> str:
    return (
        f"{instance}__{mode}__r{_rate_code(rate):09d}__"
        f"l{_lambda_code(lam):02d}__s{shot_offset:06d}_n{shots:06d}"
    )


def _estimated_work(mode: str, lam: float, shots: int) -> float:
    # Relative units only.  Folding scales approximately with the number of
    # executed two-qubit gates; reset adds measurement/reset work each round.
    reset_overhead = 0.18 if mode == MODE_RESET else 0.0
    return float(shots) * (float(lam) + reset_overhead)


def _make_task(
    config: dict[str, Any],
    *,
    mode: str,
    rate: float,
    lam: float,
    shot_offset: int,
    shots: int,
    purpose: str,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode {mode}")
    if mode == MODE_RAW and not np.isclose(lam, 1.0):
        raise ValueError("Raw tasks are defined only at lambda=1")
    task_id = _task_id(
        str(config["instance"]), mode, rate, lam, shot_offset, shots
    )
    return {
        "task_id": task_id,
        "instance": str(config["instance"]),
        "N": int(config["N"]),
        "a": int(config["a"]),
        "order": int(config["order"]),
        "measurement_bits": int(config["measurement_bits"]),
        "global_seed": int(config["global_seed"]),
        "config_path": str(config["config_path"]),
        "ops_cache_path": str(config["ops_cache_path"]),
        "ops_cache_sha256": str(config["ops_cache_sha256"]),
        "circuit_variant": str(config["circuit_variant"]),
        "two_qubit_gate_count": int(config["two_qubit_gate_count"]),
        "mode": mode,
        "rate": float(rate),
        "lambda": float(lam),
        "shot_offset": int(shot_offset),
        "shots": int(shots),
        "purpose": purpose,
        "estimated_work": _estimated_work(mode, lam, shots),
        "output_relpath": (
            Path("chunks") / str(config["instance"]) / mode / f"{task_id}.npz"
        ).as_posix(),
    }


def build_followup_tasks(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    existing_rates = list(map(float, experiment["existing_rates"]))
    additional_rates = list(map(float, experiment["a16_additional_rates"]))
    lambdas = list(map(float, experiment["zne_lambdas"]))
    existing_shots = int(experiment["existing_shots"])
    target_shots = int(experiment["target_baseline_shots"])
    task_shots = int(experiment["task_shots"])
    zne_shots = int(experiment["additional_rate_zne_shots"])

    tasks: list[dict[str, Any]] = []
    for config in experiment["configs"]:
        is_a16 = int(config["a"]) == 16
        baseline_rates = existing_rates + (additional_rates if is_a16 else [])
        for rate in baseline_rates:
            is_new_rate = is_a16 and rate in additional_rates
            first_offset = 0 if is_new_rate else existing_shots
            for mode in VALID_MODES:
                for shot_offset in range(first_offset, target_shots, task_shots):
                    tasks.append(
                        _make_task(
                            config,
                            mode=mode,
                            rate=rate,
                            lam=1.0,
                            shot_offset=shot_offset,
                            shots=min(task_shots, target_shots - shot_offset),
                            purpose=(
                                "additional_rate_lambda_one"
                                if is_new_rate
                                else "existing_rate_lambda_one_extension"
                            ),
                        )
                    )

        if is_a16:
            for rate in additional_rates:
                for lam in lambdas:
                    if np.isclose(lam, 1.0):
                        continue
                    tasks.append(
                        _make_task(
                            config,
                            mode=MODE_RESET,
                            rate=rate,
                            lam=lam,
                            shot_offset=0,
                            shots=zne_shots,
                            purpose="additional_rate_zne_scale",
                        )
                    )

    task_ids = [str(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("generated duplicate task IDs")
    return sorted(tasks, key=lambda task: str(task["task_id"]))


@dataclass(frozen=True)
class WorkerCapacity:
    name: str
    gpu_count: int


def parse_worker_capacities(values: Iterable[str]) -> list[WorkerCapacity]:
    workers: list[WorkerCapacity] = []
    for value in values:
        name, separator, count_text = value.partition("=")
        if not separator or not name or not count_text.isdigit():
            raise ValueError("workers must use NAME=GPU_COUNT")
        count = int(count_text)
        if count <= 0:
            raise ValueError("worker GPU counts must be positive")
        workers.append(WorkerCapacity(name=name, gpu_count=count))
    if not workers or len({worker.name for worker in workers}) != len(workers):
        raise ValueError("worker names must be distinct")
    return workers


def distribute_tasks(
    tasks: list[dict[str, Any]], workers: list[WorkerCapacity]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    slots: list[dict[str, Any]] = []
    for worker in workers:
        for local_gpu in range(worker.gpu_count):
            slots.append(
                {
                    "worker": worker.name,
                    "local_slot": local_gpu,
                    "load": 0.0,
                    "tasks": [],
                }
            )
    if not slots:
        raise ValueError("at least one GPU slot is required")

    for task in sorted(
        tasks, key=lambda item: (-float(item["estimated_work"]), item["task_id"])
    ):
        slot = min(
            slots,
            key=lambda item: (
                float(item["load"]),
                str(item["worker"]),
                int(item["local_slot"]),
            ),
        )
        assigned = dict(task)
        assigned["assigned_worker"] = str(slot["worker"])
        assigned["assigned_slot"] = int(slot["local_slot"])
        slot["tasks"].append(assigned)
        slot["load"] += float(task["estimated_work"])

    shards = {worker.name: [] for worker in workers}
    slot_summary: list[dict[str, Any]] = []
    for slot in slots:
        shards[str(slot["worker"])].extend(slot["tasks"])
        slot_summary.append(
            {
                "worker": str(slot["worker"]),
                "local_slot": int(slot["local_slot"]),
                "task_count": len(slot["tasks"]),
                "estimated_work": float(slot["load"]),
            }
        )
    for worker_tasks in shards.values():
        worker_tasks.sort(key=lambda task: str(task["task_id"]))
    return shards, slot_summary


def create_plan(
    *,
    experiment_path: Path,
    output_dir: Path,
    workers: list[WorkerCapacity],
) -> dict[str, Any]:
    experiment = load_experiment(experiment_path)
    tasks = build_followup_tasks(experiment)
    shards, slots = distribute_tasks(tasks, workers)
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    global_manifest: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "experiment": str(experiment["experiment"]),
        "experiment_path": str(experiment["experiment_path"]),
        "experiment_sha256": str(experiment["experiment_sha256"]),
        "existing_rates": list(experiment["existing_rates"]),
        "a16_additional_rates": list(experiment["a16_additional_rates"]),
        "zne_lambdas": list(experiment["zne_lambdas"]),
        "existing_shots": int(experiment["existing_shots"]),
        "target_baseline_shots": int(experiment["target_baseline_shots"]),
        "task_shots": int(experiment["task_shots"]),
        "workers": [
            {"name": worker.name, "gpu_count": worker.gpu_count}
            for worker in workers
        ],
        "slots": slots,
        "task_count": len(tasks),
        "tasks": [task for name in sorted(shards) for task in shards[name]],
    }
    global_manifest["task_set_sha256"] = _canonical_sha256(
        [task["task_id"] for task in global_manifest["tasks"]]
    )
    global_path = output / "global_manifest.json"
    _atomic_json(global_path, global_manifest)

    shard_paths: dict[str, str] = {}
    for worker in workers:
        worker_tasks = shards[worker.name]
        shard = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_utc": global_manifest["created_utc"],
            "experiment": global_manifest["experiment"],
            "experiment_sha256": global_manifest["experiment_sha256"],
            "task_set_sha256": global_manifest["task_set_sha256"],
            "worker": worker.name,
            "expected_gpu_count": worker.gpu_count,
            "task_count": len(worker_tasks),
            "estimated_work": float(
                sum(float(task["estimated_work"]) for task in worker_tasks)
            ),
            "tasks": worker_tasks,
        }
        shard_path = output / f"shard_{worker.name}.json"
        _atomic_json(shard_path, shard)
        shard_paths[worker.name] = str(shard_path)

    result = {
        "global_manifest": str(global_path),
        "shards": shard_paths,
        "task_count": len(tasks),
        "worker_task_counts": {
            name: len(worker_tasks) for name, worker_tasks in shards.items()
        },
        "worker_estimated_work": {
            name: float(sum(float(task["estimated_work"]) for task in worker_tasks))
            for name, worker_tasks in shards.items()
        },
    }
    _atomic_json(output / "plan_summary.json", result)
    return result


def _bits_to_outcomes(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits)
    if values.ndim != 2 or values.shape[1] > 16:
        raise ValueError(f"unexpected measurement array shape {values.shape}")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("measurement array is not binary")
    powers = np.left_shift(
        np.uint32(1), np.arange(values.shape[1], dtype=np.uint32)
    )
    return np.sum(
        values.astype(np.uint32, copy=False) * powers[None, :],
        axis=1,
        dtype=np.uint32,
    ).astype(np.uint16, copy=False)


def _outcomes_to_bits(outcomes: np.ndarray, measurement_bits: int) -> np.ndarray:
    values = np.asarray(outcomes, dtype=np.uint16)
    shifts = np.arange(measurement_bits, dtype=np.uint16)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.int32)


def _counts(outcomes: np.ndarray, measurement_bits: int) -> np.ndarray:
    return np.bincount(
        np.asarray(outcomes, dtype=np.int64), minlength=1 << measurement_bits
    ).astype(np.int32, copy=False)


def task_output_path(output_dir: Path, task: dict[str, Any]) -> Path:
    relative = Path(str(task["output_relpath"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe output_relpath in {task['task_id']}")
    return output_dir.resolve() / relative


def validate_task_archive(path: Path, task: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "task_id",
        "instance",
        "N",
        "a",
        "order",
        "measurement_bits",
        "global_seed",
        "mode",
        "rate",
        "lambda",
        "shot_offset",
        "shots",
        "ops_cache_sha256",
        "circuit_variant",
        "two_qubit_gate_count",
        "folding_pattern_scope",
        "folding_seed_fields",
        "outcomes",
        "counts",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        scalar_checks = {
            "schema_version": TASK_ARCHIVE_SCHEMA_VERSION,
            "task_id": str(task["task_id"]),
            "instance": str(task["instance"]),
            "N": int(task["N"]),
            "a": int(task["a"]),
            "order": int(task["order"]),
            "measurement_bits": int(task["measurement_bits"]),
            "global_seed": int(task["global_seed"]),
            "mode": str(task["mode"]),
            "shot_offset": int(task["shot_offset"]),
            "shots": int(task["shots"]),
            "ops_cache_sha256": str(task["ops_cache_sha256"]),
            "circuit_variant": str(task["circuit_variant"]),
            "two_qubit_gate_count": int(task["two_qubit_gate_count"]),
        }
        for name, expected in scalar_checks.items():
            value = archive[name].item()
            if value != expected:
                raise ValueError(f"{path}: {name}={value!r}, expected {expected!r}")
        if not np.isclose(float(archive["rate"]), float(task["rate"]), atol=5e-15):
            raise ValueError(f"{path}: rate differs from task")
        if not np.isclose(
            float(archive["lambda"]), float(task["lambda"]), atol=5e-15
        ):
            raise ValueError(f"{path}: lambda differs from task")
        if str(archive["folding_pattern_scope"]) != FOLDING_PATTERN_SCOPE:
            raise ValueError(f"{path}: folding scope differs")
        if not np.array_equal(
            np.asarray(archive["folding_seed_fields"]),
            np.asarray(FOLDING_SEED_FIELDS),
        ):
            raise ValueError(f"{path}: folding seed fields differ")
        outcomes = np.asarray(archive["outcomes"])
        counts = np.asarray(archive["counts"])
        if outcomes.shape != (int(task["shots"]),) or outcomes.dtype != np.uint16:
            raise ValueError(f"{path}: invalid outcomes {outcomes.shape}/{outcomes.dtype}")
        expected_count_shape = (1 << int(task["measurement_bits"]),)
        if counts.shape != expected_count_shape or int(counts.sum()) != int(task["shots"]):
            raise ValueError(f"{path}: invalid counts")
        if not np.array_equal(counts, _counts(outcomes, int(task["measurement_bits"]))):
            raise ValueError(f"{path}: outcomes and counts disagree")


def _run_atomic_task(
    task: dict[str, Any],
    output_dir: Path,
    core: Any,
    operation_maps: dict[str, dict[int, list[Any]]],
    *,
    shot_subchunk_size: int = 0,
) -> Path:
    cache_path = _resolve_project_path(str(task["ops_cache_path"]))
    if _sha256_file(cache_path) != str(task["ops_cache_sha256"]):
        raise ValueError(f"operation-cache hash mismatch: {cache_path}")
    cache_key = str(cache_path)
    if cache_key not in operation_maps:
        operation_maps[cache_key] = core.load_compiled_ops_cache(cache_path)
    ops_map = operation_maps[cache_key]
    core.OPS_CACHE_DIR = str(cache_path.parent)
    core.GLOBAL_SEED = int(task["global_seed"])
    core.smart_mem_gc(force=True, tag=f"before {task['task_id']}")

    total_shots = int(task["shots"])
    if shot_subchunk_size < 0:
        raise ValueError("shot_subchunk_size must be nonnegative")
    effective_subchunk = (
        total_shots
        if shot_subchunk_size == 0
        else min(int(shot_subchunk_size), total_shots)
    )
    outcome_parts: list[np.ndarray] = []
    engine = ""
    for local_start in range(0, total_shots, effective_subchunk):
        current_shots = min(effective_subchunk, total_shots - local_start)
        common = (
            [float(task["rate"])],
            [float(task["lambda"])],
            int(task["N"]),
            int(task["a"]),
            current_shots,
            ops_map,
        )
        global_offset = int(task["shot_offset"]) + local_start
        if str(task["mode"]) == MODE_RESET:
            results = core._run_gpu_engine_chunked_shotwise(
                "random",
                *common,
                restrict_folding_to_step=None,
                shot_offset=global_offset,
                reset_ancilla=True,
                folding_pattern_per_shot=True,
                fused_2q_kernels=True,
            )
            engine = "shotwise_random_folding_fused"
        elif str(task["mode"]) == MODE_RAW:
            results = core._run_gpu_engine_chunked(
                "random_noareset",
                *common,
                restrict_folding_to_step=None,
                shot_offset=global_offset,
                reset_ancilla=False,
            )
            engine = "historical_no_reset"
        else:
            raise ValueError(f"unsupported mode {task['mode']}")

        indexed = {int(index): np.asarray(bits) for index, bits in results}
        if sorted(indexed) != [0]:
            raise RuntimeError(
                f"{task['task_id']}: simulator returned indices {sorted(indexed)}"
            )
        part = _bits_to_outcomes(indexed[0])
        if part.shape != (current_shots,):
            raise RuntimeError(
                f"{task['task_id']}: subchunk returned {part.shape}, "
                f"expected {(current_shots,)}"
            )
        outcome_parts.append(part)
        del results, indexed, part
        if local_start + current_shots < total_shots:
            core.smart_mem_gc(
                force=True,
                tag=(
                    f"subchunk {local_start + current_shots}/{total_shots} "
                    f"{task['task_id']}"
                ),
            )

    outcomes = (
        outcome_parts[0]
        if len(outcome_parts) == 1
        else np.concatenate(outcome_parts)
    )
    measurement_bits = int(task["measurement_bits"])
    output = task_output_path(output_dir, task)
    _atomic_npz(
        output,
        schema_version=np.int32(TASK_ARCHIVE_SCHEMA_VERSION),
        created_utc=np.asarray(_utc_now()),
        task_id=np.asarray(task["task_id"]),
        instance=np.asarray(task["instance"]),
        N=np.int32(task["N"]),
        a=np.int32(task["a"]),
        order=np.int32(task["order"]),
        measurement_bits=np.int32(measurement_bits),
        global_seed=np.int64(task["global_seed"]),
        mode=np.asarray(task["mode"]),
        rate=np.float64(task["rate"]),
        **{"lambda": np.float64(task["lambda"])},
        shot_offset=np.int64(task["shot_offset"]),
        shots=np.int32(task["shots"]),
        purpose=np.asarray(task["purpose"]),
        engine=np.asarray(engine),
        engine_shot_subchunk_size=np.int32(effective_subchunk),
        engine_shot_subchunk_count=np.int32(len(outcome_parts)),
        ops_cache_sha256=np.asarray(task["ops_cache_sha256"]),
        circuit_variant=np.asarray(task["circuit_variant"]),
        two_qubit_gate_count=np.int32(task["two_qubit_gate_count"]),
        folding_pattern_scope=np.asarray(FOLDING_PATTERN_SCOPE),
        folding_seed_fields=np.asarray(FOLDING_SEED_FIELDS),
        outcomes=outcomes,
        counts=_counts(outcomes, measurement_bits),
    )
    validate_task_archive(output, task)
    return output


def _gpu_worker(
    gpu_id: int,
    tasks: Any,
    results: Any,
    output_dir: str,
    tuning: dict[str, Any],
) -> None:
    try:
        from . import simulator as core

        core.MAX_BATCH_COLS_PER_CHUNK = int(tuning["max_batch_columns"])
        core.VRAM_FREE_FRACTION = float(tuning["vram_free_fraction"])
        core.VRAM_SAFETY_FACTOR = float(tuning["vram_safety_factor"])
        core.USE_FUSED_2Q_KERNELS = True
        operation_maps: dict[str, dict[int, list[Any]]] = {}
        with core.cp.cuda.Device(gpu_id):
            print(f"[GPU {gpu_id}] follow-up worker ready", flush=True)
            while True:
                task = tasks.get()
                if task is None:
                    break
                started = time.monotonic()
                try:
                    path = _run_atomic_task(
                        task,
                        Path(output_dir),
                        core,
                        operation_maps,
                        shot_subchunk_size=int(tuning["shot_subchunk_size"]),
                    )
                    results.put(
                        {
                            "ok": True,
                            "gpu_id": gpu_id,
                            "task_id": task["task_id"],
                            "path": str(path),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    print(f"[GPU {gpu_id}] completed {task['task_id']}", flush=True)
                except Exception:
                    results.put(
                        {
                            "ok": False,
                            "gpu_id": gpu_id,
                            "task_id": task.get("task_id"),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    return
    except Exception:
        results.put(
            {"ok": False, "gpu_id": gpu_id, "traceback": traceback.format_exc()}
        )
        raise


def _parse_gpu_ids(value: str | None, available: int) -> list[int]:
    if value is None:
        return list(range(available))
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("--gpu-ids must contain distinct comma-separated IDs")
    if min(result) < 0 or max(result) >= available:
        raise ValueError(f"GPU IDs must lie in 0..{available - 1}")
    return result


def _write_progress(
    path: Path,
    *,
    worker: str,
    total: int,
    resumed: int,
    records: list[dict[str, Any]],
    status: str,
) -> None:
    successful = [record for record in records if record.get("ok")]
    failed = [record for record in records if not record.get("ok")]
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "updated_utc": _utc_now(),
            "worker": worker,
            "status": status,
            "task_total": total,
            "task_resumed": resumed,
            "task_completed_this_run": len(successful),
            "task_completed_total": resumed + len(successful),
            "task_failed": len(failed),
            "completed_task_ids": [record["task_id"] for record in successful],
            "failures": failed,
        },
    )


def run_shard(
    *,
    manifest_path: Path,
    output_dir: Path,
    gpu_ids_value: str | None,
    max_batch_columns: int,
    vram_free_fraction: float,
    vram_safety_factor: float,
    shot_subchunk_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError("incompatible shard manifest")
    worker = str(manifest["worker"])
    all_tasks = list(manifest["tasks"])
    if len(all_tasks) != int(manifest["task_count"]):
        raise ValueError("shard task_count differs from task list")
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    resumed: list[str] = []
    pending: list[dict[str, Any]] = []
    for task in all_tasks:
        path = task_output_path(output, task)
        if path.is_file():
            validate_task_archive(path, task)
            resumed.append(str(task["task_id"]))
        else:
            pending.append(task)
    progress_path = output / "progress.json"
    _write_progress(
        progress_path,
        worker=worker,
        total=len(all_tasks),
        resumed=len(resumed),
        records=[],
        status="DRY_RUN" if dry_run else "STARTING",
    )
    print(
        f"worker={worker}, tasks={len(all_tasks)}, resumed={len(resumed)}, "
        f"pending={len(pending)}",
        flush=True,
    )
    if dry_run or not pending:
        status = "DRY_RUN" if dry_run else "COMPLETE"
        _write_progress(
            progress_path,
            worker=worker,
            total=len(all_tasks),
            resumed=len(resumed),
            records=[],
            status=status,
        )
        return {
            "worker": worker,
            "status": status,
            "tasks": len(all_tasks),
            "resumed": len(resumed),
            "pending": len(pending),
        }

    if max_batch_columns <= 0:
        raise ValueError("max_batch_columns must be positive")
    if not 0.0 < vram_free_fraction <= 1.0:
        raise ValueError("vram_free_fraction must lie in (0,1]")
    if vram_safety_factor < 1.0:
        raise ValueError("vram_safety_factor must be at least one")
    if shot_subchunk_size < 0:
        raise ValueError("shot_subchunk_size must be nonnegative")

    from . import simulator as core

    available = int(core.cp.cuda.runtime.getDeviceCount())
    if available <= 0:
        raise RuntimeError("No CUDA GPUs found")
    gpu_ids = _parse_gpu_ids(gpu_ids_value, available)
    gpu_ids = gpu_ids[: min(len(gpu_ids), len(pending))]
    print(f"detected {available} CUDA GPUs; using {gpu_ids}", flush=True)

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    # A stable shuffle avoids placing all expensive folded tasks at the end,
    # while task seeds remain independent of this queue order.
    shuffled = list(pending)
    np.random.default_rng(20260726).shuffle(shuffled)
    for task in shuffled:
        task_queue.put(task)
    for _ in gpu_ids:
        task_queue.put(None)
    tuning = {
        "max_batch_columns": int(max_batch_columns),
        "vram_free_fraction": float(vram_free_fraction),
        "vram_safety_factor": float(vram_safety_factor),
        "shot_subchunk_size": int(shot_subchunk_size),
    }
    processes = [
        context.Process(
            target=_gpu_worker,
            args=(gpu_id, task_queue, result_queue, str(output), tuning),
        )
        for gpu_id in gpu_ids
    ]
    for process in processes:
        process.start()

    records: list[dict[str, Any]] = []
    _write_progress(
        progress_path,
        worker=worker,
        total=len(all_tasks),
        resumed=len(resumed),
        records=records,
        status="RUNNING",
    )
    while len(records) < len(pending):
        try:
            record = result_queue.get(timeout=10.0)
            records.append(record)
            _write_progress(
                progress_path,
                worker=worker,
                total=len(all_tasks),
                resumed=len(resumed),
                records=records,
                status="FAILED" if not record.get("ok") else "RUNNING",
            )
            if not record.get("ok"):
                break
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break
    if any(not record.get("ok") for record in records):
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join()

    failures = [record for record in records if not record.get("ok")]
    crashed = [process.exitcode for process in processes if process.exitcode != 0]
    if failures or crashed or len(records) != len(pending):
        _write_progress(
            progress_path,
            worker=worker,
            total=len(all_tasks),
            resumed=len(resumed),
            records=records,
            status="FAILED",
        )
        for failure in failures:
            print(failure.get("traceback", failure), file=sys.stderr)
        raise RuntimeError(
            f"shard incomplete: results={len(records)}/{len(pending)}, "
            f"worker_exit_codes={crashed}"
        )

    for task in all_tasks:
        validate_task_archive(task_output_path(output, task), task)
    _write_progress(
        progress_path,
        worker=worker,
        total=len(all_tasks),
        resumed=len(resumed),
        records=records,
        status="COMPLETE",
    )
    summary = {
        "schema_version": 1,
        "completed_utc": _utc_now(),
        "worker": worker,
        "manifest": str(manifest_path.resolve()),
        "task_set_sha256": str(manifest["task_set_sha256"]),
        "task_count": len(all_tasks),
        "resumed_count": len(resumed),
        "gpu_ids": gpu_ids,
        "shot_subchunk_size": int(shot_subchunk_size),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": core.cp.__version__,
            "cuda_runtime": int(core.cp.cuda.runtime.runtimeGetVersion()),
        },
        "outputs": [
            {
                "task_id": task["task_id"],
                "path": task["output_relpath"],
                "sha256": _sha256_file(task_output_path(output, task)),
            }
            for task in all_tasks
        ],
    }
    _atomic_json(output / "worker_summary.json", summary)
    return summary


def _lookup_rate_index(rates: np.ndarray, rate: float) -> int:
    matches = np.flatnonzero(np.isclose(rates, rate, rtol=0.0, atol=5e-12))
    if matches.size != 1:
        raise ValueError(f"expected one match for rate={rate}, got {matches.tolist()}")
    return int(matches[0])


def _load_existing_reset(
    base_run_dir: Path, instance: str, rate: float, lam: float
) -> np.ndarray:
    token = f"{float(lam):.1f}".replace(".", "p")
    path = base_run_dir / "shotwise" / instance / "shotwise" / f"lambda_{token}-{token}.npz"
    with np.load(path, allow_pickle=False) as archive:
        rates = np.asarray(archive["rates"], dtype=np.float64)
        lambdas = np.asarray(archive["lambdas"], dtype=np.float64)
        rate_i = _lookup_rate_index(rates, rate)
        lambda_i = _lookup_rate_index(lambdas, lam)
        return np.asarray(archive["outcomes"][rate_i, lambda_i], dtype=np.uint16).copy()


def _load_existing_raw(
    base_run_dir: Path,
    config: dict[str, Any],
    rate: float,
) -> np.ndarray:
    path = (
        base_run_dir
        / "no_reset"
        / str(config["instance"])
        / (
            f"safe_random_noareset_N{int(config['N'])}_a{int(config['a'])}_"
            f"p{float(rate):.6f}.pkl"
        )
    )
    with path.open("rb") as handle:
        archive = pickle.load(handle)
    matches = [
        np.asarray(bits)
        for (saved_rate, lam), bits in archive.items()
        if np.isclose(float(saved_rate), rate, rtol=0.0, atol=5e-12)
        and np.isclose(float(lam), 1.0, rtol=0.0, atol=5e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one raw record")
    return _bits_to_outcomes(matches[0])


def _task_key(task: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(task["instance"]),
        str(task["mode"]),
        _rate_code(float(task["rate"])),
        _lambda_code(float(task["lambda"])),
        int(task["shot_offset"]),
    )


def _load_task_outcomes(
    task: dict[str, Any], worker_roots: dict[str, Path]
) -> np.ndarray:
    worker = str(task["assigned_worker"])
    if worker not in worker_roots:
        raise ValueError(f"missing --worker-root for {worker}")
    output_dir = worker_roots[worker].resolve() / "results"
    path = task_output_path(output_dir, task)
    validate_task_archive(path, task)
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["outcomes"], dtype=np.uint16).copy()


def _save_baseline_archive(
    path: Path,
    *,
    config: dict[str, Any],
    mode: str,
    rate: float,
    outcomes: np.ndarray,
    source_ranges: list[dict[str, Any]],
) -> None:
    measurement_bits = int(config["measurement_bits"])
    _atomic_npz(
        path,
        schema_version=np.int32(1),
        created_utc=np.asarray(_utc_now()),
        instance=np.asarray(config["instance"]),
        N=np.int32(config["N"]),
        a=np.int32(config["a"]),
        order=np.int32(config["order"]),
        mode=np.asarray(mode),
        rate=np.float64(rate),
        **{"lambda": np.float64(1.0)},
        shots=np.int32(outcomes.size),
        measurement_bits=np.int32(measurement_bits),
        global_seed=np.int64(config["global_seed"]),
        ops_cache_sha256=np.asarray(config["ops_cache_sha256"]),
        circuit_variant=np.asarray(config["circuit_variant"]),
        two_qubit_gate_count=np.int32(config["two_qubit_gate_count"]),
        source_ranges_json=np.asarray(json.dumps(source_ranges, sort_keys=True)),
        outcomes=np.asarray(outcomes, dtype=np.uint16),
        counts=_counts(outcomes, measurement_bits),
    )


def merge_cluster_outputs(
    *,
    global_manifest_path: Path,
    worker_roots: dict[str, Path],
    base_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _load_json(global_manifest_path)
    if int(manifest.get("schema_version", -1)) != PLAN_SCHEMA_VERSION:
        raise ValueError("incompatible global manifest")
    experiment = load_experiment(_resolve_project_path(manifest["experiment_path"]))
    if str(manifest["experiment_sha256"]) != str(experiment["experiment_sha256"]):
        raise ValueError("experiment file changed after planning")
    tasks = list(manifest["tasks"])
    if _canonical_sha256([task["task_id"] for task in tasks]) != str(
        manifest["task_set_sha256"]
    ):
        raise ValueError("global task-set hash mismatch")
    task_map: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for task in tasks:
        key = _task_key(task)
        if key in task_map:
            raise ValueError(f"duplicate task coordinate {key}")
        # Validate every downloaded archive before constructing merged data.
        _load_task_outcomes(task, worker_roots)
        task_map[key] = task

    output = output_dir.resolve()
    baseline_root = output / "baseline_160k"
    zne_root = output / "a16_augmented_20k"
    existing_rates = list(map(float, experiment["existing_rates"]))
    additional_rates = list(map(float, experiment["a16_additional_rates"]))
    lambdas = list(map(float, experiment["zne_lambdas"]))
    existing_shots = int(experiment["existing_shots"])
    target_shots = int(experiment["target_baseline_shots"])
    task_shots = int(experiment["task_shots"])
    baseline_records: list[dict[str, Any]] = []

    for config in experiment["configs"]:
        instance = str(config["instance"])
        rates = existing_rates + (additional_rates if int(config["a"]) == 16 else [])
        for rate in rates:
            new_rate = int(config["a"]) == 16 and rate in additional_rates
            for mode in VALID_MODES:
                chunks: list[np.ndarray] = []
                source_ranges: list[dict[str, Any]] = []
                if new_rate:
                    first_task = task_map[
                        (instance, mode, _rate_code(rate), _lambda_code(1.0), 0)
                    ]
                    chunks.append(_load_task_outcomes(first_task, worker_roots))
                    source_ranges.append(
                        {"shot_offset": 0, "shots": existing_shots, "source": first_task["task_id"]}
                    )
                else:
                    initial = (
                        _load_existing_reset(base_run_dir, instance, rate, 1.0)
                        if mode == MODE_RESET
                        else _load_existing_raw(base_run_dir, config, rate)
                    )
                    if initial.size != existing_shots:
                        raise ValueError(
                            f"{instance}/{mode}/{rate}: existing shots={initial.size}, "
                            f"expected {existing_shots}"
                        )
                    chunks.append(initial)
                    source_ranges.append(
                        {"shot_offset": 0, "shots": existing_shots, "source": "original_run"}
                    )
                for offset in range(existing_shots, target_shots, task_shots):
                    task = task_map[
                        (instance, mode, _rate_code(rate), _lambda_code(1.0), offset)
                    ]
                    chunk = _load_task_outcomes(task, worker_roots)
                    if chunk.size != min(task_shots, target_shots - offset):
                        raise ValueError(f"{task['task_id']}: wrong merged chunk size")
                    chunks.append(chunk)
                    source_ranges.append(
                        {"shot_offset": offset, "shots": int(chunk.size), "source": task["task_id"]}
                    )
                merged = np.concatenate(chunks)
                if merged.size != target_shots:
                    raise RuntimeError(
                        f"{instance}/{mode}/{rate}: merged {merged.size}, expected {target_shots}"
                    )
                rate_token = f"p{float(rate):.6f}"
                path = baseline_root / instance / mode / f"lambda1_{rate_token}.npz"
                _save_baseline_archive(
                    path,
                    config=config,
                    mode=mode,
                    rate=rate,
                    outcomes=merged,
                    source_ranges=source_ranges,
                )
                baseline_records.append(
                    {
                        "instance": instance,
                        "mode": mode,
                        "rate": rate,
                        "shots": int(merged.size),
                        "path": str(path.relative_to(output)),
                        "sha256": _sha256_file(path),
                    }
                )

    # Build a drop-in 20k dataset for a=16 containing the original eleven
    # rates plus the four new rates.  The 160k baseline files remain separate,
    # so all eight ZNE scales in this directory have equal shot counts.
    a16 = next(config for config in experiment["configs"] if int(config["a"]) == 16)
    augmented_rates = sorted(existing_rates + additional_rates)
    measurement_bits = int(a16["measurement_bits"])
    cache_path = _resolve_project_path(a16["ops_cache_path"])
    from .folding_pattern_simulator import _fold_plan_metadata, _load_operation_map

    ops_map = _load_operation_map(cache_path)
    shotwise_dir = zne_root / "shotwise" / str(a16["instance"]) / "shotwise"
    zne_records: list[dict[str, Any]] = []
    for lam in lambdas:
        outcome_rows: list[np.ndarray] = []
        for rate in augmented_rates:
            if rate in existing_rates:
                row = _load_existing_reset(base_run_dir, str(a16["instance"]), rate, lam)
            else:
                task = task_map[
                    (
                        str(a16["instance"]),
                        MODE_RESET,
                        _rate_code(rate),
                        _lambda_code(lam),
                        0,
                    )
                ]
                row = _load_task_outcomes(task, worker_roots)
            if row.size != existing_shots:
                raise ValueError(f"a=16 rate={rate}, lambda={lam}: expected 20000 shots")
            outcome_rows.append(row)
        outcomes = np.stack(outcome_rows, axis=0)[:, None, :]
        counts = np.stack(
            [_counts(row, measurement_bits) for row in outcome_rows], axis=0
        )[:, None, :]
        plan_hashes, folded_counts = _fold_plan_metadata(
            ops_map,
            N=int(a16["N"]),
            a=int(a16["a"]),
            rates=augmented_rates,
            lambdas=[lam],
            folding_seed=int(a16["global_seed"]),
            shots=existing_shots,
            noisy_gate_names=("cz", "rzz"),
        )
        token = f"{float(lam):.1f}".replace(".", "p")
        path = shotwise_dir / f"lambda_{token}-{token}.npz"
        _atomic_npz(
            path,
            schema_version=np.int32(FOLDING_ARCHIVE_SCHEMA_VERSION),
            instance=np.asarray(a16["instance"]),
            N=np.int32(a16["N"]),
            a=np.int32(a16["a"]),
            order=np.int32(a16["order"]),
            measurement_bits=np.int32(measurement_bits),
            rates=np.asarray(augmented_rates, dtype=np.float64),
            lambdas=np.asarray([lam], dtype=np.float64),
            shots=np.int32(existing_shots),
            folding_seed=np.uint32(a16["global_seed"]),
            folding_pattern_scope=np.asarray(FOLDING_PATTERN_SCOPE),
            folding_seed_fields=np.asarray(FOLDING_SEED_FIELDS),
            ops_cache_sha256=np.asarray(a16["ops_cache_sha256"]),
            circuit_variant=np.asarray(a16["circuit_variant"]),
            two_qubit_gate_count=np.int32(a16["two_qubit_gate_count"]),
            outcomes=outcomes,
            counts=counts,
            fold_plan_sha256=plan_hashes,
            folded_gate_counts_by_round=folded_counts,
        )
        zne_records.append(
            {"lambda": lam, "path": str(path.relative_to(output)), "sha256": _sha256_file(path)}
        )

    raw_dir = zne_root / "no_reset" / str(a16["instance"])
    raw_records: list[dict[str, Any]] = []
    for rate in augmented_rates:
        if rate in existing_rates:
            outcomes = _load_existing_raw(base_run_dir, a16, rate)
        else:
            task = task_map[
                (
                    str(a16["instance"]),
                    MODE_RAW,
                    _rate_code(rate),
                    _lambda_code(1.0),
                    0,
                )
            ]
            outcomes = _load_task_outcomes(task, worker_roots)
        bits = _outcomes_to_bits(outcomes, measurement_bits)
        path = raw_dir / (
            f"safe_random_noareset_N{int(a16['N'])}_a{int(a16['a'])}_"
            f"p{float(rate):.6f}.pkl"
        )
        _atomic_pickle(path, {(float(rate), 1.0): bits})
        raw_records.append(
            {"rate": rate, "path": str(path.relative_to(output)), "sha256": _sha256_file(path)}
        )

    summary = {
        "schema_version": 1,
        "completed_utc": _utc_now(),
        "global_manifest": str(global_manifest_path.resolve()),
        "global_task_set_sha256": str(manifest["task_set_sha256"]),
        "base_run_dir": str(base_run_dir.resolve()),
        "worker_roots": {name: str(path.resolve()) for name, path in worker_roots.items()},
        "baseline_target_shots": target_shots,
        "baseline_records": baseline_records,
        "a16_augmented_rates": augmented_rates,
        "a16_zne_shots_per_scale": existing_shots,
        "a16_zne_records": zne_records,
        "a16_raw_records": raw_records,
    }
    _atomic_json(output / "merge_summary.json", summary)
    return summary


def _parse_worker_roots(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path_text = value.partition("=")
        if not separator or not name or not path_text:
            raise ValueError("--worker-root must use NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate worker root {name}")
        result[name] = Path(path_text).expanduser().resolve()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create global and per-host task manifests")
    plan.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument(
        "--workers",
        nargs="+",
        default=["a100x4=4", "a100x2=2", "a100x1=1"],
        help="Worker capacities as NAME=GPU_COUNT",
    )

    run = subparsers.add_parser("run-shard", help="Run one manifest on local GPUs")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--gpu-ids")
    run.add_argument("--max-batch-columns", type=int, default=40000)
    run.add_argument("--vram-free-fraction", type=float, default=0.90)
    run.add_argument("--vram-safety-factor", type=float, default=1.45)
    run.add_argument(
        "--shot-subchunk-size",
        type=int,
        default=0,
        help="Split each atomic task into this many shots internally (0 disables)",
    )
    run.add_argument("--dry-run", action="store_true")

    merge = subparsers.add_parser("merge", help="Validate and merge downloaded shards")
    merge.add_argument("--global-manifest", type=Path, required=True)
    merge.add_argument(
        "--worker-root",
        action="append",
        required=True,
        help="Downloaded run root as NAME=PATH (repeat for every worker)",
    )
    merge.add_argument("--base-run-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        result = create_plan(
            experiment_path=args.experiment,
            output_dir=args.output_dir,
            workers=parse_worker_capacities(args.workers),
        )
    elif args.command == "run-shard":
        result = run_shard(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            gpu_ids_value=args.gpu_ids,
            max_batch_columns=args.max_batch_columns,
            vram_free_fraction=args.vram_free_fraction,
            vram_safety_factor=args.vram_safety_factor,
            shot_subchunk_size=args.shot_subchunk_size,
            dry_run=args.dry_run,
        )
    elif args.command == "merge":
        result = merge_cluster_outputs(
            global_manifest_path=args.global_manifest,
            worker_roots=_parse_worker_roots(args.worker_root),
            base_run_dir=args.base_run_dir.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    else:  # pragma: no cover - argparse enforces the command
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
